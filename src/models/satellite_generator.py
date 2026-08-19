import numpy as np
from nodes import Satellite

def generate_satellites(
    number=50,
    radius=7000,
    radius_fluctuation=100,
    angular_velocity=(0.001, 0.0001),
    velocity_fluctuation=(0.0001, 0.00001),
    connection_range=2000,
    seed=None
):
    rng = np.random.default_rng(seed)

    satellites = []

    angular_velocity = np.array(angular_velocity, dtype=float)
    velocity_fluctuation = np.array(
        velocity_fluctuation,
        dtype=float
    )

    for i in range(number):

        # Random initial angles
        longitude = rng.uniform(0, 2 * np.pi)
        inclination = rng.uniform(-np.pi / 2, np.pi / 2)

        # Radius = base value + fluctuation
        satellite_radius = radius + rng.uniform(
            -radius_fluctuation,
            radius_fluctuation
        )

        # Angular velocity vector = base vector + fluctuation
        satellite_velocity = angular_velocity + rng.uniform(
            -velocity_fluctuation,
            velocity_fluctuation
        )

        satellite = Satellite(
            name = f"Satellite {i}",
            radius=satellite_radius,
            longitude=longitude,
            inclination=inclination,
            angular_velocity=satellite_velocity,
            connection_range=connection_range
        )

        satellites.append(satellite)

    return satellites