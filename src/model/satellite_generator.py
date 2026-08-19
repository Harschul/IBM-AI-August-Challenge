"""Random Satellite factory.

Generation belongs here so callers receive fully configured Satellite domain
objects without duplicating setup logic.
"""

from __future__ import annotations

import logging

import numpy as np

from nodes import Satellite

LOGGER = logging.getLogger(__name__)


def generate_satellites(
    number=50,
    radius=7000,
    radius_fluctuation=100,
    angular_velocity=(0.001, 0.0001),
    velocity_fluctuation=(0.0001, 0.00001),
    connection_range=2000,
    seed=None,
):
    number = int(number)
    radius = float(radius)
    radius_fluctuation = float(radius_fluctuation)
    connection_range = float(connection_range)

    if number < 1:
        raise ValueError("number must be at least 1")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if radius_fluctuation < 0:
        raise ValueError("radius_fluctuation cannot be negative")
    if radius - radius_fluctuation <= 0:
        raise ValueError("radius_fluctuation can produce a non-positive radius")
    if connection_range < 0:
        raise ValueError("connection_range cannot be negative")

    angular_velocity = np.asarray(angular_velocity, dtype=float)
    velocity_fluctuation = np.asarray(velocity_fluctuation, dtype=float)
    if angular_velocity.shape != (2,) or velocity_fluctuation.shape != (2,):
        raise ValueError(
            "angular_velocity and velocity_fluctuation must each contain two values"
        )
    if np.any(velocity_fluctuation < 0):
        raise ValueError("velocity_fluctuation values cannot be negative")

    rng = np.random.default_rng(seed)
    satellites = []

    LOGGER.info(
        "Generating %d satellites with seed=%s, base_radius=%.1f km, range=%.1f km",
        number,
        seed,
        radius,
        connection_range,
    )

    for i in range(number):
        longitude = rng.uniform(0.0, 2.0 * np.pi)
        inclination = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        satellite_radius = radius + rng.uniform(
            -radius_fluctuation,
            radius_fluctuation,
        )
        satellite_velocity = angular_velocity + rng.uniform(
            -velocity_fluctuation,
            velocity_fluctuation,
        )

        satellite = Satellite(
            name=f"Satellite {i}",
            radius=satellite_radius,
            longitude=longitude,
            inclination=inclination,
            angular_velocity=satellite_velocity,
            connection_range=connection_range,
        )
        satellites.append(satellite)

        LOGGER.debug(
            "Generated %s: radius=%.3f, lon=%.6f, inc=%.6f, omega=%s",
            satellite.name(),
            satellite.radius(),
            satellite.longitude(),
            satellite.inclination(),
            np.array2string(satellite.angular_velocity(), precision=8),
        )

    LOGGER.info("Generated %d satellites successfully", len(satellites))
    return satellites