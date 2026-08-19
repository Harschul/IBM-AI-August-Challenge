"""Random Satellite factory.

Generation belongs here so callers receive fully configured Satellite domain
objects without duplicating setup logic. Each satellite receives independently
randomized orbital phase, inclination, RAAN, radius, and angular velocity.
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
    angular_velocity=0.001,
    velocity_fluctuation=0.0001,
    connection_range=2000,
    seed=None,
):
    """Generate randomized satellites for the circular-orbit model.

    ``angular_velocity`` is the base orbital phase rate in rad/s and
    ``velocity_fluctuation`` is the +/- random perturbation applied separately
    to every generated satellite.
    """
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
        # Where the satellite starts around its own circular orbit.
        phase = rng.uniform(0.0, 2.0 * np.pi)

        # Tilt of the orbital plane relative to Earth's equatorial plane.
        inclination = rng.uniform(-np.pi / 2.0, np.pi / 2.0)

        # Rotation of the orbital plane around Earth's z-axis. Randomizing RAAN
        # is the key fix for the old shared-x-axis crossing problem.
        raan = rng.uniform(0.0, 2.0 * np.pi)

        satellite_radius = radius + rng.uniform(
            -radius_fluctuation,
            radius_fluctuation,
        )

        # Fresh random draw for every satellite, so each receives its own speed.
        satellite_velocity = angular_velocity + rng.uniform(
            -velocity_fluctuation,
            velocity_fluctuation,
        )

        satellite = Satellite(
            name=f"Satellite {i}",
            radius=satellite_radius,
            phase=phase,
            inclination=inclination,
            raan=raan,
            angular_velocity=satellite_velocity,
            connection_range=connection_range,
        )
        satellites.append(satellite)

        LOGGER.debug(
            "Generated %s: radius=%.3f km, phase=%.6f, inc=%.6f, "
            "raan=%.6f, omega=%.8f rad/s",
            satellite.name(),
            satellite.radius(),
            satellite.phase(),
            satellite.inclination(),
            satellite.raan(),
            satellite.angular_velocity(),
        )

    LOGGER.info("Generated %d satellites successfully", len(satellites))
    return satellites
