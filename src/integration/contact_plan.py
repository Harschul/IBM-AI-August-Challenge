"""Convert physical orbital snapshots into the canonical temporal ContactPlan.

This is the missing seam between `src/model/` (positions/topology) and both the
contact-graph baseline and RL policy. It also makes the three ground stations
first-class destinations and enables direct science-to-ground windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.models.contact import Contact, ContactPlan

from .config import (
    GROUND_IDS,
    SCIENCE_IDS,
    NUM_NODES,
    PrototypeConfig,
    link_profile_name,
    node_role,
)

C_KM_S = 299_792.458


@dataclass(frozen=True)
class ContactPlanDiagnostics:
    contacts: int
    direct_to_ground_contacts: int
    satellite_contacts: int
    ground_contacts: int
    horizon_s: float


def _rotate_z(vector: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z = vector
    return np.array([c * x - s * y, s * x + c * y, z], dtype=float)


def _ground_initial_xyz(lat_deg: float, lon_deg: float, radius_km: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    cos_lat = math.cos(lat)
    return radius_km * np.array(
        [cos_lat * math.cos(lon), cos_lat * math.sin(lon), math.sin(lat)],
        dtype=float,
    )


def ground_position(config: PrototypeConfig, ground_id: int, time_s: float) -> np.ndarray:
    gs = next((item for item in config.ground_stations if item.node_id == ground_id), None)
    if gs is None:
        raise ValueError(f"unknown ground station id {ground_id}")
    initial = _ground_initial_xyz(gs.lat_deg, gs.lon_deg, config.earth_radius_km)
    return _rotate_z(initial, config.earth_rotation_rad_s * float(time_s))


def _satellite_line_of_sight(a: np.ndarray, b: np.ndarray, earth_radius_km: float) -> bool:
    delta = b - a
    length_sq = float(np.dot(delta, delta))
    if length_sq <= 1e-12:
        return True
    t = -float(np.dot(a, delta)) / length_sq
    t = float(np.clip(t, 0.0, 1.0))
    closest = a + t * delta
    # Tangency is treated as blocked, matching the simulator's strict rule.
    return float(np.dot(closest, closest)) > earth_radius_km**2


def _ground_visible(
    sat: np.ndarray,
    ground: np.ndarray,
    min_elevation_deg: float,
) -> tuple[bool, float]:
    delta = sat - ground
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        return True, 0.0
    upward = ground / float(np.linalg.norm(ground))
    sin_elevation = float(np.dot(delta, upward)) / distance
    return sin_elevation >= math.sin(math.radians(min_elevation_deg)), distance


def _window_runs(times: Sequence[float], active: Sequence[bool]) -> list[tuple[float, float, range]]:
    """Stitch sampled booleans into non-zero windows using midpoint boundaries."""
    if len(times) != len(active):
        raise ValueError("times and active must have equal length")
    if len(times) < 2:
        return []

    windows: list[tuple[float, float, range]] = []
    i = 0
    n = len(times)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and active[j + 1]:
            j += 1

        start = float(times[i]) if i == 0 else 0.5 * (float(times[i - 1]) + float(times[i]))
        end = float(times[j]) if j == n - 1 else 0.5 * (float(times[j]) + float(times[j + 1]))
        if end > start:
            windows.append((start, end, range(i, j + 1)))
        i = j + 1
    return windows


def _combined_risk(a: float, b: float) -> float:
    """Independent-risk combination bounded to 0..1."""
    return 1.0 - (1.0 - float(a)) * (1.0 - float(b))


def build_contact_plan(
    snapshots: Sequence[object],
    config: PrototypeConfig,
) -> tuple[ContactPlan, ContactPlanDiagnostics]:
    """Build one directed time-window plan from orbital snapshots.

    Snapshot requirements are intentionally tiny: every object needs `.time`
    and `.positions`, where positions contains exactly the eleven propagated
    spacecraft in fixed node-ID order 0..10.
    """
    if len(snapshots) < 2:
        raise ValueError("at least two snapshots are required to infer contact windows")

    times = [float(s.time) for s in snapshots]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("snapshot times must be strictly increasing")

    positions = [np.asarray(s.positions, dtype=float) for s in snapshots]
    if any(p.shape != (11, 3) for p in positions):
        raise ValueError("every snapshot.positions must have shape (11, 3)")

    contacts: list[Contact] = []
    satellite_contacts = 0
    ground_contacts = 0
    direct_to_ground = 0

    for src in range(NUM_NODES):
        for dst in range(NUM_NODES):
            if src == dst:
                continue
            profile_name = link_profile_name(src, dst)
            if profile_name is None:
                continue
            if profile_name not in config.links:
                raise ValueError(f"config missing link profile {profile_name}")
            profile = config.links[profile_name]

            active: list[bool] = []
            ranges: list[float] = []

            for time_s, sat_positions in zip(times, positions):
                src_role = node_role(src)
                dst_role = node_role(dst)

                if src_role == "GROUND":
                    ok, distance = False, float("inf")
                elif dst_role == "GROUND":
                    sat_pos = sat_positions[src]
                    gs = next(g for g in config.ground_stations if g.node_id == dst)
                    ground_pos = ground_position(config, dst, time_s)
                    ok, distance = _ground_visible(sat_pos, ground_pos, gs.min_elevation_deg)
                else:
                    a = sat_positions[src]
                    b = sat_positions[dst]
                    distance = float(np.linalg.norm(a - b))
                    ok = _satellite_line_of_sight(a, b, config.earth_radius_km)

                if ok and profile.max_range_km is not None and distance > profile.max_range_km:
                    ok = False
                active.append(bool(ok))
                ranges.append(distance)

            for start_s, end_s, sample_range in _window_runs(times, active):
                active_ranges = [ranges[k] for k in sample_range if active[k] and math.isfinite(ranges[k])]
                if not active_ranges:
                    continue
                representative_range = float(sum(active_ranges) / len(active_ranges))
                duration = end_s - start_s
                capacity = max(1, int(duration * profile.data_rate_bps / 8.0))

                weather_risk = profile.weather_risk
                if dst in GROUND_IDS:
                    gs = next(g for g in config.ground_stations if g.node_id == dst)
                    weather_risk = _combined_risk(weather_risk, gs.weather_risk)

                contacts.append(
                    Contact(
                        source_id=src,
                        destination_id=dst,
                        start_s=start_s,
                        end_s=end_s,
                        data_rate_bps=profile.data_rate_bps,
                        range_km=representative_range,
                        propagation_delay_s=representative_range / C_KM_S,
                        residual_capacity_bytes=capacity,
                        reliability=profile.reliability,
                        weather_risk=weather_risk,
                        energy_cost=profile.energy_cost,
                        link_type=profile_name,
                    )
                )
                if dst in GROUND_IDS:
                    ground_contacts += 1
                    if src in SCIENCE_IDS:
                        direct_to_ground += 1
                else:
                    satellite_contacts += 1

    plan = ContactPlan(contacts)
    diagnostics = ContactPlanDiagnostics(
        contacts=len(contacts),
        direct_to_ground_contacts=direct_to_ground,
        satellite_contacts=satellite_contacts,
        ground_contacts=ground_contacts,
        horizon_s=plan.horizon(),
    )
    return plan, diagnostics
