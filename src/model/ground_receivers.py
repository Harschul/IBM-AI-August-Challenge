"""Evenly distributed virtual ground receivers and visibility geometry."""

from __future__ import annotations

import numpy as np


def fibonacci_sphere(number_of_points: int = 100, radius: float = 6371.0) -> np.ndarray:
    """Return approximately uniform points on a sphere using a Fibonacci lattice.

    Parameters
    ----------
    number_of_points:
        Number of receiver/sample locations spread across the entire Earth.
    radius:
        Sphere radius in kilometres.
    """
    number_of_points = int(number_of_points)
    radius = float(radius)
    if number_of_points < 1:
        raise ValueError("number_of_points must be at least 1")
    if radius <= 0:
        raise ValueError("radius must be positive")

    i = np.arange(number_of_points, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    # Midpoint sampling avoids placing points exactly at either pole.
    y = 1.0 - 2.0 * (i + 0.5) / number_of_points
    radial = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = golden_angle * i

    x = radial * np.cos(theta)
    z = radial * np.sin(theta)
    return radius * np.column_stack((x, y, z))


def nearest_visible_satellite_distances(
    satellite_positions: np.ndarray,
    ground_points: np.ndarray,
    *,
    earth_radius: float = 6371.0,
    min_elevation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest visible satellite distance and coverage mask per receiver.

    ``satellite_positions`` is shaped ``(N, 3)`` and ``ground_points`` is
    ``(G, 3)``.  A satellite is usable only if it is above the receiver's local
    minimum elevation angle.

    The calculation uses dot products rather than allocating a GxNx3 delta
    tensor.  For a ground point g and satellite s:

        ||s-g||^2 = ||s||^2 + ||g||^2 - 2(s dot g)

    and the elevation test is:

        (s-g) dot (g/R) / ||s-g|| >= sin(min_elevation).
    """
    satellite_positions = np.asarray(satellite_positions, dtype=np.float64)
    ground_points = np.asarray(ground_points, dtype=np.float64)
    earth_radius = float(earth_radius)
    min_elevation_deg = float(min_elevation_deg)

    if satellite_positions.ndim != 2 or satellite_positions.shape[1] != 3:
        raise ValueError("satellite_positions must have shape (N, 3)")
    if ground_points.ndim != 2 or ground_points.shape[1] != 3:
        raise ValueError("ground_points must have shape (G, 3)")
    if earth_radius <= 0:
        raise ValueError("earth_radius must be positive")
    if not (-90.0 <= min_elevation_deg < 90.0):
        raise ValueError("min_elevation_deg must be in [-90, 90)")

    sat_radius_sq = np.sum(satellite_positions * satellite_positions, axis=1)
    ground_radius_sq = np.sum(ground_points * ground_points, axis=1)
    dot = ground_points @ satellite_positions.T  # G x N

    distance_sq = (
        ground_radius_sq[:, None]
        + sat_radius_sq[None, :]
        - 2.0 * dot
    )
    np.maximum(distance_sq, 0.0, out=distance_sq)
    distances = np.sqrt(distance_sq)

    # (s-g) dot g = s dot g - ||g||^2.  Divide by R*distance to get
    # sin(elevation).  We compare without division for better numerical safety.
    upward_dot = dot - ground_radius_sq[:, None]
    sin_min = np.sin(np.deg2rad(min_elevation_deg))
    visible = upward_dot >= (earth_radius * distances * sin_min)

    usable = np.where(visible, distances, np.inf)
    nearest = np.min(usable, axis=1)
    covered = np.isfinite(nearest)
    return nearest, covered
