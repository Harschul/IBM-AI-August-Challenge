"""Satellite construction helpers.

This module provides both the original random constellation generator and JSON
save/load helpers so an optimized constellation can be replayed exactly by the
visualization script.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from nodes import Satellite

LOGGER = logging.getLogger(__name__)


def generate_satellites(
    number=50,
    radius=7000,
    radius_fluctuation=100,
    angular_velocity=0.001,
    velocity_fluctuation=0.0001,
    connection_range=5000,
    storage_capacity=100,
    transmit_limit=1,
    link_bandwidth=4096,
    seed=None,
):
    """Generate randomized satellites for the circular-orbit model."""
    number = int(number)
    radius = float(radius)
    radius_fluctuation = float(radius_fluctuation)
    angular_velocity = float(angular_velocity)
    velocity_fluctuation = float(velocity_fluctuation)
    connection_range = float(connection_range)

    if number < 1:
        raise ValueError("number must be at least 1")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if radius_fluctuation < 0:
        raise ValueError("radius_fluctuation cannot be negative")
    if radius - radius_fluctuation <= 0:
        raise ValueError("radius_fluctuation can produce a non-positive radius")
    if not np.isfinite(angular_velocity):
        raise ValueError("angular_velocity must be finite")
    if velocity_fluctuation < 0:
        raise ValueError("velocity_fluctuation cannot be negative")
    if connection_range < 0:
        raise ValueError("connection_range cannot be negative")

    rng = np.random.default_rng(seed)
    satellites = []

    LOGGER.info(
        "Generating %d satellites with seed=%s, base_radius=%.1f km, "
        "omega=%.8f +/- %.8f rad/s, range=%.1f km",
        number,
        seed,
        radius,
        angular_velocity,
        velocity_fluctuation,
        connection_range,
    )

    for i in range(number):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        inclination = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        raan = rng.uniform(0.0, 2.0 * np.pi)
        satellite_radius = radius + rng.uniform(-radius_fluctuation, radius_fluctuation)
        satellite_velocity = angular_velocity + rng.uniform(
            -velocity_fluctuation,
            velocity_fluctuation,
        )

        satellites.append(
            Satellite(
                name=f"Satellite {i}",
                radius=satellite_radius,
                phase=phase,
                inclination=inclination,
                raan=raan,
                angular_velocity=satellite_velocity,
                connection_range=connection_range,
                storage_capacity=storage_capacity,
                transmit_limit=transmit_limit,
                link_bandwidth=link_bandwidth,
            )
        )

    LOGGER.info("Generated %d satellites successfully", len(satellites))
    return satellites


def satellite_to_dict(satellite) -> dict:
    """Serialize the *initial* constellation state of one satellite."""
    initial_phase = (
        satellite.initial_phase()
        if hasattr(satellite, "initial_phase")
        else satellite.phase()
    )
    return {
        "name": satellite.name(),
        "radius": float(satellite.radius()),
        "phase": float(initial_phase),
        "inclination": float(satellite.inclination()),
        "raan": float(satellite.raan()),
        "angular_velocity": float(satellite.angular_velocity()),
        "connection_range": float(satellite.connection_range()),
        "storage_capacity": float(satellite.storage_capacity()),
        "transmit_limit": int(satellite.transmit_limit()),
        "link_bandwidth": int(satellite.link_bandwidth())
    }


def save_satellites(satellites, output) -> Path:
    """Save a constellation as readable JSON for exact later replay."""
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": "satellite-network-constellation-v1",
        "satellites": [satellite_to_dict(sat) for sat in satellites],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def load_satellites(path):
    """Load a constellation previously written by :func:`save_satellites`."""
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        records = payload.get("satellites")
    else:
        records = payload

    if not isinstance(records, list) or not records:
        raise ValueError("Constellation JSON must contain a non-empty satellite list")

    satellites = []
    for record in records:
        satellites.append(
            Satellite(
                name=record["name"],
                radius=record["radius"],
                phase=record["phase"],
                inclination=record["inclination"],
                raan=record["raan"],
                angular_velocity=record["angular_velocity"],
                connection_range=record["connection_range"],
                storage_capacity=record.get("storage_capacity", 100),
                transmit_limit=record.get("transmit_limit", 1),
                link_bandwidth=record.get("link_bandwidth", 4096),
            )
        )

    return satellites
