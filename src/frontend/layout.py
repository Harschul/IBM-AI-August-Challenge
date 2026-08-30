"""Static topology layout used by the synchronized 2D network graph."""

from __future__ import annotations

import math

from src.integration.config import GEO_IDS, GROUND_IDS, LEO_IDS, SCIENCE_IDS


def topology_positions() -> dict[int, tuple[float, float]]:
    """Return a stable, presentation-friendly 2D layout for the fixed 14 nodes."""
    positions: dict[int, tuple[float, float]] = {}

    science_y = (0.95, 0.0, -0.95)
    for node_id, y in zip(SCIENCE_IDS, science_y):
        positions[node_id] = (-2.35, y)

    radius = 1.35
    for idx, node_id in enumerate(LEO_IDS):
        angle = math.radians(150 - idx * 60.0)
        positions[node_id] = (radius * math.cos(angle), radius * math.sin(angle))

    positions[GEO_IDS[0]] = (-0.65, 2.35)
    positions[GEO_IDS[1]] = (0.65, 2.35)

    positions[GROUND_IDS[0]] = (-1.25, -2.1)
    positions[GROUND_IDS[1]] = (0.0, -2.35)
    positions[GROUND_IDS[2]] = (1.25, -2.1)
    return positions
