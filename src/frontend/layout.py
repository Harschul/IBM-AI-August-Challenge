"""Static layered topology layout for the final 14-node network."""

from __future__ import annotations

from src.integration.config import GEO_IDS, GROUND_IDS, LEO_IDS, SCIENCE_IDS


def _spread(ids, y: float, left: float, right: float):
    ids = list(ids)
    if len(ids) == 1:
        return {ids[0]: ((left + right) / 2.0, y)}
    step = (right - left) / max(1, len(ids) - 1)
    return {node_id: (left + i * step, y) for i, node_id in enumerate(ids)}


def topology_positions() -> dict[int, tuple[float, float]]:
    """Return a stable layer-by-layer layout for the fixed final topology.

    The layout is presentation-only.  It deliberately mirrors the architecture
    hierarchy (science -> LEO mesh -> GEO relays -> ground) while all routing
    continues to use the physical ContactPlan.
    """

    positions: dict[int, tuple[float, float]] = {}
    positions.update(_spread(SCIENCE_IDS, 3.0, -1.25, 1.25))
    positions.update(_spread(LEO_IDS, 1.55, -3.05, 3.05))
    positions.update(_spread(GEO_IDS, 0.0, -1.05, 1.05))
    positions.update(_spread(GROUND_IDS, -1.55, -2.2, 2.2))
    return positions
