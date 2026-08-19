"""Satellite domain model.

This module owns satellite state, propagation, geometric checks, and per-satellite
state history. Rendering code should only read the state produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SatelliteState:
    """Immutable record of one satellite at one simulation time."""

    time: float
    longitude: float
    inclination: float
    position: np.ndarray


class Satellite:
    def __init__(
        self,
        name,
        radius,
        longitude=0,
        inclination=0,
        angular_velocity=(0, 0),
        connection_range=0,
    ):
        self._name = str(name)
        self._radius = float(radius)
        self._longitude = float(longitude)
        self._inclination = float(inclination)
        self._angular_velocity = np.asarray(angular_velocity, dtype=float)
        self._connection_range = float(connection_range)

        if self._radius <= 0:
            raise ValueError("radius must be positive")
        if self._connection_range < 0:
            raise ValueError("connection_range cannot be negative")
        if self._angular_velocity.shape != (2,):
            raise ValueError("angular_velocity must contain exactly two values")

        self._initial_longitude = self._longitude
        self._initial_inclination = self._inclination
        self._time = 0.0
        self._history: list[SatelliteState] = []
        self._record_state()

        LOGGER.debug(
            "Created %s: radius=%.3f km, lon=%.6f rad, inc=%.6f rad, "
            "omega=(%.8f, %.8f) rad/s, range=%.3f km",
            self._name,
            self._radius,
            self._longitude,
            self._inclination,
            self._angular_velocity[0],
            self._angular_velocity[1],
            self._connection_range,
        )

    def name(self):
        return self._name

    def radius(self):
        return self._radius

    def longitude(self):
        return self._longitude

    def inclination(self):
        return self._inclination

    def angular_velocity(self):
        return self._angular_velocity.copy()

    def connection_range(self):
        return self._connection_range

    def time(self):
        return self._time

    def history(self):
        """Return an immutable view of all recorded states."""
        return tuple(self._history)

    def vertices(self):
        """Backward-compatible position history."""
        return [state.position.copy() for state in self._history]

    def pos(self):
        """Current Cartesian position in kilometres."""
        r = self._radius
        sin_lon = np.sin(self._longitude)

        x = r * np.cos(self._longitude)
        y = r * sin_lon * np.cos(self._inclination)
        z = r * sin_lon * np.sin(self._inclination)
        return np.array([x, y, z], dtype=float)

    def reset(self):
        """Return to the generated initial state and restart the history."""
        self._longitude = self._initial_longitude
        self._inclination = self._initial_inclination
        self._time = 0.0
        self._history.clear()
        self._record_state()
        LOGGER.debug("Reset %s to t=0", self._name)

    def propagate(self, delta_time):
        """Advance the satellite by ``delta_time`` seconds and log the state."""
        delta_time = float(delta_time)
        if delta_time < 0:
            raise ValueError("delta_time cannot be negative")

        self._longitude = (
            self._longitude + self._angular_velocity[0] * delta_time
        ) % (2.0 * np.pi)
        self._inclination += self._angular_velocity[1] * delta_time
        self._time += delta_time
        self._record_state()

        LOGGER.debug(
            "%s propagated to t=%.3f s at position=%s",
            self._name,
            self._time,
            np.array2string(self._history[-1].position, precision=3),
        )
        return self._history[-1]

    def distance_to(self, other):
        return float(np.linalg.norm(self.pos() - other.pos()))

    def has_line_of_sight_to(self, other, earth_radius=6371.0):
        """Return True when the segment to ``other`` does not pass through Earth."""
        earth_radius = float(earth_radius)
        if earth_radius <= 0:
            raise ValueError("earth_radius must be positive")

        a = self.pos()
        b = other.pos()
        delta = b - a
        length_sq = float(np.dot(delta, delta))

        if length_sq <= 1e-12:
            return True

        t = -float(np.dot(a, delta)) / length_sq
        t = float(np.clip(t, 0.0, 1.0))
        closest = a + t * delta
        return float(np.dot(closest, closest)) > earth_radius * earth_radius

    def can_connect_to(self, other, earth_radius=6371.0, require_line_of_sight=True):
        """Evaluate the complete physical link rule.

        A link is symmetric: the distance must be within the smaller endpoint
        range. Optionally, Earth must not intersect the line segment.
        """
        max_distance = min(self.connection_range(), other.connection_range())
        if self.distance_to(other) > max_distance:
            return False

        if require_line_of_sight:
            return self.has_line_of_sight_to(other, earth_radius=earth_radius)

        return True

    def _record_state(self):
        state = SatelliteState(
            time=self._time,
            longitude=self._longitude,
            inclination=self._inclination,
            position=self.pos(),
        )
        # Make the snapshot's NumPy payload read-only so history cannot be mutated
        # accidentally by visualization code.
        state.position.setflags(write=False)
        self._history.append(state)