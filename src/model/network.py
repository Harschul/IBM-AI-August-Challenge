"""Satellite-network simulation engine.

This module owns simulation timesteps, propagation, connectivity, and the full
network history consumed by visualization code.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NetworkSnapshot:
    """Immutable network state for one rendered/simulation timestep."""

    frame: int
    time: float
    positions: np.ndarray
    connection_indices: tuple[tuple[int, int], ...]
    connections: tuple[tuple[str, str], ...]


class Network:
    def __init__(self, satellites, earth_radius=6371.0, require_line_of_sight=True):
        self._satellites = list(satellites)
        self._earth_radius = float(earth_radius)
        self._require_line_of_sight = bool(require_line_of_sight)

        if self._earth_radius <= 0:
            raise ValueError("earth_radius must be positive")
        if not self._satellites:
            raise ValueError("Network requires at least one satellite")

        self._connections: list[tuple[str, str]] = []
        self._connection_indices: list[tuple[int, int]] = []
        self._history: list[NetworkSnapshot] = []

        LOGGER.info(
            "Network initialised with %d satellites (earth_radius=%.1f km, LOS=%s)",
            len(self._satellites),
            self._earth_radius,
            self._require_line_of_sight,
        )

    def satellites(self):
        return tuple(self._satellites)

    def connections(self):
        return tuple(self._connections)

    def connection_indices(self):
        return tuple(self._connection_indices)

    def history(self):
        """Return immutable snapshots for visualization or analysis."""
        return tuple(self._history)

    def earth_radius(self):
        return self._earth_radius

    def reset(self):
        for satellite in self._satellites:
            satellite.reset()

        self._connections.clear()
        self._connection_indices.clear()
        self._history.clear()
        LOGGER.debug("Network reset")

    def update_network(self):
        """Recompute all links from the satellites' current physical states."""
        self._connections.clear()
        self._connection_indices.clear()

        indexed_satellites = enumerate(self._satellites)
        for (i, sat1), (j, sat2) in combinations(indexed_satellites, 2):
            if sat1.can_connect_to(
                sat2,
                earth_radius=self._earth_radius,
                require_line_of_sight=self._require_line_of_sight,
            ):
                self._connection_indices.append((i, j))
                self._connections.append((sat1.name(), sat2.name()))

        LOGGER.debug(
            "Network recomputed at t=%.3f s: %d active links",
            self._satellites[0].time(),
            len(self._connections),
        )
        return self.connections()

    def step(self, delta_time, frame=None):
        """Advance all satellites once, update links, and store a snapshot."""
        for satellite in self._satellites:
            satellite.propagate(delta_time)

        self.update_network()
        if frame is None:
            frame = len(self._history)
        snapshot = self._capture_snapshot(frame=frame)
        self._history.append(snapshot)
        return snapshot

    def simulate(self, simulation_seconds, frame_count):
        """Run the complete network simulation and return frame snapshots.

        The first snapshot is the initial state at t=0. Every later frame is
        produced by propagating the Satellite objects and recomputing Network
        connectivity. The visualization does no orbital/network simulation.
        """
        simulation_seconds = float(simulation_seconds)
        frame_count = int(frame_count)

        if simulation_seconds < 0:
            raise ValueError("simulation_seconds cannot be negative")
        if frame_count < 1:
            raise ValueError("frame_count must be at least 1")

        self.reset()
        self.update_network()
        initial = self._capture_snapshot(frame=0)
        self._history.append(initial)

        if frame_count == 1:
            LOGGER.info("Simulation complete: one frame at t=0")
            return self.history()

        delta_time = simulation_seconds / (frame_count - 1)
        topology_changes = 0
        previous_links = initial.connection_indices

        LOGGER.info(
            "Simulation started: frames=%d, duration=%.3f s, dt=%.6f s",
            frame_count,
            simulation_seconds,
            delta_time,
        )

        for frame in range(1, frame_count):
            snapshot = self.step(delta_time, frame=frame)
            if snapshot.connection_indices != previous_links:
                topology_changes += 1
                LOGGER.info(
                    "Topology changed at frame=%d t=%.3f s: %d active links",
                    frame,
                    snapshot.time,
                    len(snapshot.connection_indices),
                )
                previous_links = snapshot.connection_indices

        LOGGER.info(
            "Simulation complete: %d frames, %d topology changes, final links=%d",
            frame_count,
            topology_changes,
            len(self._connection_indices),
        )
        return self.history()

    def _capture_snapshot(self, frame):
        positions = np.stack([satellite.pos() for satellite in self._satellites])
        positions.setflags(write=False)

        return NetworkSnapshot(
            frame=int(frame),
            time=float(self._satellites[0].time()),
            positions=positions,
            connection_indices=tuple(self._connection_indices),
            connections=tuple(self._connections),
        )