"""Satellite domain model.

This module owns satellite state, orbital propagation, geometric link checks,
and per-satellite history. Rendering code should only read state produced here.

The orbit model is deliberately simple: each satellite follows a circular orbit
centred on Earth. The orbital plane is defined by inclination and RAAN, while
``phase`` gives the satellite's current position around that circle.
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
    phase: float
    inclination: float
    raan: float
    position: np.ndarray


class Satellite:
    def __init__(
        self,
        name,
        radius,
        phase=0.0,
        inclination=0.0,
        raan=0.0,
        angular_velocity=0.0,
        connection_range=0.0,
        **legacy_kwargs,
    ):
        # Backward compatibility for older callers that still pass
        # ``longitude=...``. In this circular-orbit model longitude is not the
        # right orbital element, so we interpret it as orbital phase.
        if "longitude" in legacy_kwargs:
            if phase != 0.0:
                raise TypeError("Specify either phase or longitude, not both")
            phase = legacy_kwargs.pop("longitude")
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected Satellite arguments: {unexpected}")

        self._name = str(name)
        self._radius = float(radius)
        self._phase = float(phase) % (2.0 * np.pi)
        self._inclination = float(inclination)
        self._raan = float(raan) % (2.0 * np.pi)
        self._angular_velocity = float(angular_velocity)
        self._connection_range = float(connection_range)

        if self._radius <= 0:
            raise ValueError("radius must be positive")
        if self._connection_range < 0:
            raise ValueError("connection_range cannot be negative")
        if not np.isfinite(self._angular_velocity):
            raise ValueError("angular_velocity must be finite")

        self._initial_phase = self._phase
        self._time = 0.0
        self._history: list[SatelliteState] = []
        self._record_state()

        LOGGER.debug(
            "Created %s: radius=%.3f km, phase=%.6f rad, inc=%.6f rad, "
            "raan=%.6f rad, omega=%.8f rad/s, range=%.3f km",
            self._name,
            self._radius,
            self._phase,
            self._inclination,
            self._raan,
            self._angular_velocity,
            self._connection_range,
        )

    def name(self):
        return self._name

    def radius(self):
        return self._radius

    def phase(self):
        return self._phase

    def longitude(self):
        """Backward-compatible alias for phase()."""
        return self._phase

    def inclination(self):
        return self._inclination

    def raan(self):
        return self._raan

    def angular_velocity(self):
        return self._angular_velocity

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

    def orbital_plane_normal(self):
        """Return the unit normal vector of this satellite's orbital plane.

        This is useful for diagnostics and confirms that randomized RAAN values
        produce genuinely differently oriented orbital planes.
        """
        sin_i = np.sin(self._inclination)
        normal = np.array(
            [
                np.sin(self._raan) * sin_i,
                -np.cos(self._raan) * sin_i,
                np.cos(self._inclination),
            ],
            dtype=float,
        )
        return normal

    def pos(self):
        """Current Cartesian position in kilometres.

        Start with a circle in the reference x-y plane, tilt it by inclination,
        then rotate the orbital plane around Earth's z-axis by RAAN. Because
        every satellite can have a different RAAN, all planes no longer share
        the same forced x-axis crossing.
        """
        r = self._radius
        theta = self._phase
        inc = self._inclination
        raan = self._raan

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_inc = np.cos(inc)
        sin_inc = np.sin(inc)
        cos_raan = np.cos(raan)
        sin_raan = np.sin(raan)

        x = r * (
            cos_raan * cos_theta
            - sin_raan * sin_theta * cos_inc
        )
        y = r * (
            sin_raan * cos_theta
            + cos_raan * sin_theta * cos_inc
        )
        z = r * sin_theta * sin_inc

        return np.array([x, y, z], dtype=float)

    def reset(self):
        """Return to the generated initial state and restart the history."""
        self._phase = self._initial_phase
        self._time = 0.0
        self._history.clear()
        self._record_state()
        LOGGER.debug("Reset %s to t=0", self._name)

    def propagate(self, delta_time):
        """Advance the satellite around its circular orbit and log the state.

        Inclination and RAAN define the orbital plane and stay fixed in this
        simple model. Only orbital phase advances with the satellite-specific
        randomized angular velocity.
        """
        delta_time = float(delta_time)
        if delta_time < 0:
            raise ValueError("delta_time cannot be negative")

        self._phase = (
            self._phase + self._angular_velocity * delta_time
        ) % (2.0 * np.pi)
        self._time += delta_time
        self._record_state()

        LOGGER.debug(
            "%s propagated to t=%.3f s: phase=%.6f, position=%s",
            self._name,
            self._time,
            self._phase,
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
        """Evaluate the complete symmetric physical link rule."""
        max_distance = min(self.connection_range(), other.connection_range())
        if self.distance_to(other) > max_distance:
            return False

        if require_line_of_sight:
            return self.has_line_of_sight_to(other, earth_radius=earth_radius)

        return True

    def _record_state(self):
        state = SatelliteState(
            time=self._time,
            phase=self._phase,
            inclination=self._inclination,
            raan=self._raan,
            position=self.pos(),
        )
        # Prevent visualization/analysis code from mutating recorded history.
        state.position.setflags(write=False)
        self._history.append(state)
