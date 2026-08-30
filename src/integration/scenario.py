"""Build the fixed 14-node physical scenario expected by the RL action space.

The checkpoint interface stays at 14 actions, but the mission layout now has
multiple independent science sources:

    0..2    science spacecraft (3)
    3..8    LEO relays (6)
    9..10   GEO relays (2)
    11..13  ground stations (3, handled by the contact adapter)

Only nodes 0..10 are propagated by ``src.model.network.Network``.
"""

from __future__ import annotations

import math

from .config import GEO_IDS, LEO_IDS, SCIENCE_IDS, PrototypeConfig

EARTH_MU_KM3_S2 = 398600.4418


def circular_angular_velocity(radius_km: float) -> float:
    return math.sqrt(EARTH_MU_KM3_S2 / float(radius_km) ** 3)


def _deg(value: float) -> float:
    return math.radians(float(value))


def build_satellites(config: PrototypeConfig):
    """Return eleven ``Satellite`` objects ordered exactly by node ID."""
    from src.model.nodes import Satellite

    node_cfg = config.raw["nodes"]
    science_cfg = node_cfg["science"]
    leo_cfg = node_cfg["leo"]
    geo_cfg = node_cfg["geo"]

    satellites = [None] * 11

    # Three distinct science spacecraft.  They share the same altitude and
    # inclination class but are deliberately separated in RAAN and phase so
    # bundle source selection changes the physical contact opportunities.
    science_radius = config.earth_radius_km + float(science_cfg["altitude_km"])
    base_inclination = _deg(science_cfg.get("inclination_deg", 97.6))
    base_raan_deg = float(science_cfg.get("raan_deg", 12.0))
    base_phase_deg = float(science_cfg.get("phase_deg", 15.0))
    raan_spacing_deg = float(science_cfg.get("raan_spacing_deg", 120.0))
    phase_spacing_deg = float(science_cfg.get("phase_spacing_deg", 105.0))

    for offset, node_id in enumerate(SCIENCE_IDS):
        satellites[node_id] = Satellite(
            name=f"SCI-{offset}",
            radius=science_radius,
            phase=_deg(base_phase_deg + offset * phase_spacing_deg),
            inclination=base_inclination,
            raan=_deg(base_raan_deg + offset * raan_spacing_deg),
            angular_velocity=circular_angular_velocity(science_radius),
            connection_range=100_000.0,
            storage_capacity=int(science_cfg.get("storage_capacity", 2000)),
            transmit_limit=1,
            link_bandwidth=1,
        )

    leo_radius = config.earth_radius_km + float(leo_cfg["altitude_km"])
    plane_count = int(leo_cfg.get("planes", 3))
    satellites_per_plane = len(LEO_IDS) // plane_count
    if plane_count < 1 or satellites_per_plane * plane_count != len(LEO_IDS):
        raise ValueError("LEO planes must evenly divide the fixed six relays")

    inclination = _deg(leo_cfg.get("inclination_deg", 53.0))
    for offset, node_id in enumerate(LEO_IDS):
        plane = offset // satellites_per_plane
        slot = offset % satellites_per_plane
        raan = 2.0 * math.pi * plane / plane_count
        phase = (
            2.0 * math.pi * slot / satellites_per_plane
            + plane * math.pi / max(1, len(LEO_IDS))
        )
        satellites[node_id] = Satellite(
            name=f"LEO-{node_id}",
            radius=leo_radius,
            phase=phase,
            inclination=inclination,
            raan=raan,
            angular_velocity=circular_angular_velocity(leo_radius),
            connection_range=100_000.0,
            storage_capacity=int(leo_cfg.get("storage_capacity", 2500)),
            transmit_limit=1,
            link_bandwidth=1,
        )

    geo_radius = config.earth_radius_km + float(geo_cfg.get("altitude_km", 35_786.0))
    geo_phases = geo_cfg.get("longitudes_deg", [-30.0, 150.0])
    if len(geo_phases) != len(GEO_IDS):
        raise ValueError("GEO longitudes_deg must contain exactly two values")
    for node_id, phase_deg in zip(GEO_IDS, geo_phases):
        satellites[node_id] = Satellite(
            name=f"GEO-{node_id}",
            radius=geo_radius,
            phase=_deg(phase_deg),
            inclination=0.0,
            raan=0.0,
            angular_velocity=config.earth_rotation_rad_s,
            connection_range=100_000.0,
            storage_capacity=int(geo_cfg.get("storage_capacity", 5000)),
            transmit_limit=1,
            link_bandwidth=1,
        )

    if any(satellite is None for satellite in satellites):
        raise AssertionError("every propagated node ID 0..10 must have a satellite")
    return satellites


def simulate_snapshots(config: PrototypeConfig):
    """Run the existing orbital simulator and return its immutable snapshots."""
    from src.model.network import Network

    satellites = build_satellites(config)
    network = Network(
        satellites,
        earth_radius=config.earth_radius_km,
        # Link availability is recomputed by the adapter with role-specific
        # ranges; Network is used here for propagation and immutable snapshots.
        require_line_of_sight=False,
    )
    snapshots = network.simulate(config.horizon_s, config.frame_count)
    return satellites, snapshots
