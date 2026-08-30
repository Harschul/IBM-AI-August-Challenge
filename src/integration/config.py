"""Configuration loader and validation for the integrated 14-node prototype.

`config/prototype.yaml` is stored as JSON syntax. JSON is valid YAML 1.2, which
lets the repository keep the requested .yaml filename without adding PyYAML as
another demo-time dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


SCIENCE_IDS = tuple(range(0, 3))
LEO_IDS = tuple(range(3, 9))
GEO_IDS = tuple(range(9, 11))
GROUND_IDS = tuple(range(11, 14))
NUM_NODES = 14


@dataclass(frozen=True)
class LinkProfile:
    data_rate_bps: float
    max_range_km: float | None
    reliability: float
    weather_risk: float
    energy_cost: float


@dataclass(frozen=True)
class GroundStation:
    node_id: int
    name: str
    lat_deg: float
    lon_deg: float
    min_elevation_deg: float
    weather_risk: float


@dataclass(frozen=True)
class RuntimeDefaults:
    queue_norm: float
    storage_free_norm: float
    health: float
    battery: float


@dataclass(frozen=True)
class PrototypeConfig:
    raw: Mapping[str, Any]
    seed: int
    horizon_s: float
    sample_step_s: float
    earth_radius_km: float
    earth_rotation_rad_s: float
    ground_stations: tuple[GroundStation, ...]
    links: Mapping[str, LinkProfile]
    runtime_defaults: RuntimeDefaults

    @property
    def frame_count(self) -> int:
        return int(round(self.horizon_s / self.sample_step_s)) + 1


def _need(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing config key: {key}")
    return mapping[key]


def _unit_interval(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in 0..1")
    return value


def load_config(path: str | Path = "config/prototype.yaml") -> PrototypeConfig:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    simulation = _need(payload, "simulation")
    nodes = _need(payload, "nodes")
    links_raw = _need(payload, "links")
    runtime_raw = _need(payload, "runtime_defaults")

    science_cfg = _need(nodes, "science")
    leo_cfg = _need(nodes, "leo")
    geo_cfg = _need(nodes, "geo")
    if int(science_cfg.get("count", len(SCIENCE_IDS))) != len(SCIENCE_IDS):
        raise ValueError(f"science.count must be {len(SCIENCE_IDS)} for the fixed 14-node interface")
    if int(leo_cfg.get("count", len(LEO_IDS))) != len(LEO_IDS):
        raise ValueError(f"leo.count must be {len(LEO_IDS)} for the fixed 14-node interface")
    if int(geo_cfg.get("count", len(GEO_IDS))) != len(GEO_IDS):
        raise ValueError(f"geo.count must be {len(GEO_IDS)} for the fixed 14-node interface")

    ground = tuple(
        GroundStation(
            node_id=int(item["id"]),
            name=str(item["name"]),
            lat_deg=float(item["lat_deg"]),
            lon_deg=float(item["lon_deg"]),
            min_elevation_deg=float(item.get("min_elevation_deg", 0.0)),
            weather_risk=_unit_interval("weather_risk", item.get("weather_risk", 0.0)),
        )
        for item in _need(nodes, "ground_stations")
    )
    if tuple(gs.node_id for gs in ground) != GROUND_IDS:
        raise ValueError(f"ground station IDs must be exactly {GROUND_IDS}")

    links: Dict[str, LinkProfile] = {}
    for name, item in links_raw.items():
        rate = float(item["data_rate_bps"])
        if rate <= 0:
            raise ValueError(f"{name}.data_rate_bps must be positive")
        max_range = item.get("max_range_km")
        max_range = None if max_range is None else float(max_range)
        if max_range is not None and max_range <= 0:
            raise ValueError(f"{name}.max_range_km must be positive when set")
        links[name] = LinkProfile(
            data_rate_bps=rate,
            max_range_km=max_range,
            reliability=_unit_interval(f"{name}.reliability", item.get("reliability", 1.0)),
            weather_risk=_unit_interval(f"{name}.weather_risk", item.get("weather_risk", 0.0)),
            energy_cost=float(item.get("energy_cost", 0.0)),
        )

    runtime = RuntimeDefaults(
        queue_norm=_unit_interval("runtime_defaults.queue_norm", runtime_raw.get("queue_norm", 0.0)),
        storage_free_norm=_unit_interval(
            "runtime_defaults.storage_free_norm", runtime_raw.get("storage_free_norm", 1.0)
        ),
        health=_unit_interval("runtime_defaults.health", runtime_raw.get("health", 1.0)),
        battery=_unit_interval("runtime_defaults.battery", runtime_raw.get("battery", 1.0)),
    )

    horizon = float(simulation["horizon_s"])
    step = float(simulation["sample_step_s"])
    if horizon <= 0 or step <= 0 or step > horizon:
        raise ValueError("simulation horizon/sample step are invalid")

    return PrototypeConfig(
        raw=payload,
        seed=int(payload.get("seed", 42)),
        horizon_s=horizon,
        sample_step_s=step,
        earth_radius_km=float(simulation.get("earth_radius_km", 6371.0)),
        earth_rotation_rad_s=float(simulation.get("earth_rotation_rad_s", 7.2921159e-5)),
        ground_stations=ground,
        links=links,
        runtime_defaults=runtime,
    )


def node_role(node_id: int) -> str:
    if node_id in SCIENCE_IDS:
        return "SCIENCE"
    if node_id in LEO_IDS:
        return "LEO"
    if node_id in GEO_IDS:
        return "GEO"
    if node_id in GROUND_IDS:
        return "GROUND"
    raise ValueError(f"unknown node id {node_id}")


def link_profile_name(source_id: int, destination_id: int) -> str | None:
    """Return the configured directed link class, or None when routing is disallowed."""
    src = node_role(source_id)
    dst = node_role(destination_id)

    if src == "GROUND":
        return None
    if src == "SCIENCE" and dst == "SCIENCE":
        return None
    if dst == "SCIENCE":
        return None

    if dst == "GROUND":
        return f"{src}_GROUND"
    if src == "SCIENCE" and dst in ("LEO", "GEO"):
        return f"SCIENCE_{dst}"
    if src == "LEO" and dst == "LEO":
        return "LEO_LEO"
    if {src, dst} == {"LEO", "GEO"}:
        return "LEO_GEO"
    if src == "GEO" and dst == "GEO":
        return "GEO_GEO"
    return None
