"""Fully vectorized optimizer fitness evaluator.

The optimizer bypasses Satellite/Network object creation and evaluates whole
Differential Evolution populations with NumPy.  Geometry is vectorized across:

    C = candidate constellations in a chunk
    F = simulation frames
    N = satellites
    P = unique satellite pairs
    G = virtual Earth receivers

The fitness exactly matches scoring.py and is coverage-first:

    mean_ground  = mean normalized nearest-visible-satellite distance
    worst_ground = worst normalized receiver distance over all frames/receivers
    ground       = blend(mean_ground, worst_ground)
    quality      = (1 + average_link_score - ground) / 2

If worst-frame coverage is incomplete, fitness stays below 0.5 and is driven
primarily by worst-frame coverage. Once worst-frame coverage reaches 100%, the
fitness moves into [0.5, 1] and is ranked by ``quality``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import numpy as np

from fitness import blended_receiver_penalty, coverage_first_fitness, network_quality
from ground_receivers import fibonacci_sphere


_EPS = 1e-12


@dataclass(frozen=True)
class FastScoreDiagnostics:
    score: float
    quality_score: float
    average_link_score: float
    average_receiver_penalty: float
    worst_receiver_penalty: float
    blended_receiver_penalty: float
    average_coverage: float
    worst_coverage: float
    full_coverage: bool
    average_nearest_receiver_distance: float
    worst_nearest_receiver_distance: float
    average_active_links: float
    maximum_active_links: int
    average_link_distance: float
    average_total_link_distance: float


class FastConstellationEvaluator:
    SUPPORTED_PARAMETERS = (
        "phase",
        "inclination",
        "raan",
        "angular_velocity",
    )

    def __init__(
        self,
        *,
        radius,
        phase,
        inclination,
        raan,
        angular_velocity,
        connection_range,
        parameters=("phase", "inclination", "raan"),
        simulation_seconds=20_000.0,
        frame_count=180,
        earth_radius=6371.0,
        require_line_of_sight=True,
        ground_points=None,
        ground_point_count=100,
        ground_distance_scale=5000.0,
        worst_distance_weight=0.5,
        coverage_tolerance=1e-12,
        min_elevation_deg=0.0,
        ground_chunk_size=20,
        candidate_chunk_size=4,
        chunk_workers=4,
    ):
        self.radius = np.ascontiguousarray(radius, dtype=np.float64)
        self.phase = np.ascontiguousarray(phase, dtype=np.float64)
        self.inclination = np.ascontiguousarray(inclination, dtype=np.float64)
        self.raan = np.ascontiguousarray(raan, dtype=np.float64)
        self.angular_velocity = np.ascontiguousarray(angular_velocity, dtype=np.float64)
        self.connection_range = np.ascontiguousarray(connection_range, dtype=np.float64)

        lengths = {
            len(self.radius), len(self.phase), len(self.inclination), len(self.raan),
            len(self.angular_velocity), len(self.connection_range),
        }
        if len(lengths) != 1:
            raise ValueError("All satellite parameter arrays must have equal length")
        self.satellite_count = len(self.radius)
        if self.satellite_count < 1:
            raise ValueError("At least one satellite is required")

        self.parameters = tuple(parameters)
        unknown = set(self.parameters) - set(self.SUPPORTED_PARAMETERS)
        if unknown:
            raise ValueError(f"Unsupported optimized parameters: {sorted(unknown)}")
        if not self.parameters:
            raise ValueError("At least one optimized parameter is required")

        self.simulation_seconds = float(simulation_seconds)
        self.frame_count = int(frame_count)
        self.earth_radius = float(earth_radius)
        self.require_line_of_sight = bool(require_line_of_sight)
        self.ground_distance_scale = float(ground_distance_scale)
        self.worst_distance_weight = float(worst_distance_weight)
        self.coverage_tolerance = float(coverage_tolerance)
        self.min_elevation_deg = float(min_elevation_deg)
        self.ground_chunk_size = max(1, int(ground_chunk_size))
        self.candidate_chunk_size = max(1, int(candidate_chunk_size))
        self.chunk_workers = max(1, int(chunk_workers))

        if self.frame_count < 1:
            raise ValueError("frame_count must be at least 1")
        if self.simulation_seconds < 0:
            raise ValueError("simulation_seconds cannot be negative")
        if self.earth_radius <= 0:
            raise ValueError("earth_radius must be positive")
        if self.ground_distance_scale <= 0:
            raise ValueError("ground_distance_scale must be positive")
        if not 0.0 <= self.worst_distance_weight <= 1.0:
            raise ValueError("worst_distance_weight must be between 0 and 1")
        if self.coverage_tolerance < 0:
            raise ValueError("coverage_tolerance cannot be negative")
        if not (-90.0 <= self.min_elevation_deg < 90.0):
            raise ValueError("min_elevation_deg must be in [-90, 90)")
        if np.any(self.radius <= 0):
            raise ValueError("radius values must be positive")
        if np.any(self.connection_range < 0):
            raise ValueError("connection_range values cannot be negative")

        if ground_points is None:
            ground_points = fibonacci_sphere(ground_point_count, self.earth_radius)
        self.ground_points = np.ascontiguousarray(ground_points, dtype=np.float64)
        if self.ground_points.ndim != 2 or self.ground_points.shape[1] != 3:
            raise ValueError("ground_points must have shape (G, 3)")
        self.ground_point_count = len(self.ground_points)
        self.ground_x = self.ground_points[:, 0]
        self.ground_y = self.ground_points[:, 1]
        self.ground_z = self.ground_points[:, 2]
        self.ground_radius_sq = np.sum(self.ground_points * self.ground_points, axis=1)
        self.sin_min_elevation = np.sin(np.deg2rad(self.min_elevation_deg))

        self.times = np.linspace(0.0, self.simulation_seconds, self.frame_count, dtype=np.float64)
        self.pair_i, self.pair_j = np.triu_indices(self.satellite_count, k=1)
        self.pair_i = self.pair_i.astype(np.intp, copy=False)
        self.pair_j = self.pair_j.astype(np.intp, copy=False)
        self.max_possible_links = len(self.pair_i)

        if self.max_possible_links:
            self.pair_ranges = np.minimum(
                self.connection_range[self.pair_i],
                self.connection_range[self.pair_j],
            )
            self.pair_range_sq = self.pair_ranges * self.pair_ranges
            self.positive_pair_range = self.pair_ranges > 0.0
        else:
            self.pair_ranges = np.empty(0, dtype=np.float64)
            self.pair_range_sq = np.empty(0, dtype=np.float64)
            self.positive_pair_range = np.empty(0, dtype=bool)

        self.earth_radius_sq = self.earth_radius * self.earth_radius
        self.satellite_radius_sq = self.radius * self.radius
        self.variable_count = self.satellite_count * len(self.parameters)

        # Cache terms that remain fixed for the chosen optimization variables.
        self._fixed_phase_advance = None
        if "angular_velocity" not in self.parameters:
            self._fixed_phase_advance = self.times[:, None] * self.angular_velocity[None, :]

        self._fixed_ci = self._fixed_si = None
        if "inclination" not in self.parameters:
            self._fixed_ci = np.cos(self.inclination)[None, None, :]
            self._fixed_si = np.sin(self.inclination)[None, None, :]

        self._fixed_cr = self._fixed_sr = None
        if "raan" not in self.parameters:
            self._fixed_cr = np.cos(self.raan)[None, None, :]
            self._fixed_sr = np.sin(self.raan)[None, None, :]

    @classmethod
    def from_satellites(cls, satellites, **kwargs):
        satellites = tuple(satellites)
        return cls(
            radius=[sat.radius() for sat in satellites],
            phase=[sat.initial_phase() if hasattr(sat, "initial_phase") else sat.phase() for sat in satellites],
            inclination=[sat.inclination() for sat in satellites],
            raan=[sat.raan() for sat in satellites],
            angular_velocity=[sat.angular_velocity() for sat in satellites],
            connection_range=[sat.connection_range() for sat in satellites],
            **kwargs,
        )

    def encode(self) -> np.ndarray:
        return np.column_stack([getattr(self, p) for p in self.parameters]).reshape(-1).astype(np.float64, copy=False)

    def _decode_population(self, population):
        population = np.asarray(population, dtype=np.float64)
        if population.ndim == 1:
            population = population[None, :]
        if population.ndim != 2 or population.shape[1] != self.variable_count:
            raise ValueError(f"Expected candidate shape (C, {self.variable_count}), got {population.shape}")

        c = population.shape[0]
        values = population.reshape(c, self.satellite_count, len(self.parameters))
        decoded = {
            "phase": np.broadcast_to(self.phase, (c, self.satellite_count)),
            "inclination": np.broadcast_to(self.inclination, (c, self.satellite_count)),
            "raan": np.broadcast_to(self.raan, (c, self.satellite_count)),
            "angular_velocity": np.broadcast_to(self.angular_velocity, (c, self.satellite_count)),
        }
        for column, parameter in enumerate(self.parameters):
            decoded[parameter] = values[:, :, column]
        return decoded

    def _positions(self, params):
        phase0 = params["phase"]
        inc = params["inclination"]
        raan = params["raan"]
        omega = params["angular_velocity"]

        if self._fixed_phase_advance is not None:
            theta = phase0[:, None, :] + self._fixed_phase_advance[None, :, :]
        else:
            theta = phase0[:, None, :] + omega[:, None, :] * self.times[None, :, None]

        ct = np.cos(theta)
        st = np.sin(theta)
        if self._fixed_ci is None:
            ci = np.cos(inc)[:, None, :]
            si = np.sin(inc)[:, None, :]
        else:
            ci, si = self._fixed_ci, self._fixed_si
        if self._fixed_cr is None:
            cr = np.cos(raan)[:, None, :]
            sr = np.sin(raan)[:, None, :]
        else:
            cr, sr = self._fixed_cr, self._fixed_sr

        radius = self.radius[None, None, :]
        x = radius * (cr * ct - sr * st * ci)
        y = radius * (sr * ct + cr * st * ci)
        z = radius * (st * si)
        return x, y, z

    def _link_stage(self, x, y, z):
        c = x.shape[0]
        if self.max_possible_links == 0:
            frame_scores = np.zeros((c, self.frame_count), dtype=np.float64)
            active = np.zeros_like(frame_scores, dtype=np.int32)
            total = np.zeros_like(frame_scores)
            distances = None
            valid = None
            return frame_scores, active, total, distances, valid

        ax = x[:, :, self.pair_i]
        ay = y[:, :, self.pair_i]
        az = z[:, :, self.pair_i]
        dx = x[:, :, self.pair_j] - ax
        dy = y[:, :, self.pair_j] - ay
        dz = z[:, :, self.pair_j] - az

        distance_sq = dx * dx + dy * dy + dz * dz
        within_range = distance_sq <= self.pair_range_sq[None, None, :]

        if self.require_line_of_sight:
            safe = np.maximum(distance_sq, _EPS)
            t = -(ax * dx + ay * dy + az * dz) / safe
            np.clip(t, 0.0, 1.0, out=t)
            cx = ax + t * dx
            cy = ay + t * dy
            cz = az + t * dz
            closest_radius_sq = cx * cx + cy * cy + cz * cz
            has_los = (distance_sq <= _EPS) | (closest_radius_sq > self.earth_radius_sq)
            valid = within_range & has_los
        else:
            valid = within_range
        valid &= self.positive_pair_range[None, None, :]

        distances = np.sqrt(distance_sq)
        normalized = np.divide(
            distances,
            self.pair_ranges[None, None, :],
            out=np.zeros_like(distances),
            where=self.positive_pair_range[None, None, :],
        )
        np.clip(normalized, 0.0, 1.0, out=normalized)
        normalized *= valid

        link_scores = normalized.sum(axis=2) / self.max_possible_links
        active_counts = valid.sum(axis=2)
        total_distances = (distances * valid).sum(axis=2)
        return link_scores, active_counts, total_distances, distances, valid

    def _receiver_stage(self, x, y, z, diagnostics=False):
        """Compute receiver metrics in small G-blocks to bound memory.

        The returned arrays are per candidate/per frame:

        - mean receiver penalty,
        - worst receiver penalty,
        - coverage fraction,
        - optional mean nearest covered distance,
        - optional worst nearest distance (``inf`` if that frame is uncovered).
        """
        c = x.shape[0]
        penalty_sum = np.zeros((c, self.frame_count), dtype=np.float64)
        worst_penalty = np.zeros((c, self.frame_count), dtype=np.float64)
        covered_count = np.zeros((c, self.frame_count), dtype=np.int32)

        if diagnostics:
            nearest_sum = np.zeros((c, self.frame_count), dtype=np.float64)
            frame_max_nearest = np.full((c, self.frame_count), -np.inf, dtype=np.float64)

        for start in range(0, self.ground_point_count, self.ground_chunk_size):
            stop = min(start + self.ground_chunk_size, self.ground_point_count)
            gx = self.ground_x[start:stop]
            gy = self.ground_y[start:stop]
            gz = self.ground_z[start:stop]
            gr2 = self.ground_radius_sq[start:stop]

            # C x F x N x B, B <= ground_chunk_size.
            dot = (
                x[..., None] * gx[None, None, None, :]
                + y[..., None] * gy[None, None, None, :]
                + z[..., None] * gz[None, None, None, :]
            )
            distance_sq = (
                self.satellite_radius_sq[None, None, :, None]
                + gr2[None, None, None, :]
                - 2.0 * dot
            )
            np.maximum(distance_sq, 0.0, out=distance_sq)
            distances = np.sqrt(distance_sq)

            upward_dot = dot - gr2[None, None, None, :]
            visible = upward_dot >= (
                self.earth_radius * distances * self.sin_min_elevation
            )
            distances[~visible] = np.inf

            nearest = distances.min(axis=2)  # C x F x B
            covered = np.isfinite(nearest)
            covered_count += covered.sum(axis=2)

            # inf clips to 1, so uncovered receivers automatically get the
            # maximum normalized penalty.
            normalized = np.clip(nearest / self.ground_distance_scale, 0.0, 1.0)
            penalty_sum += normalized.sum(axis=2)
            np.maximum(worst_penalty, normalized.max(axis=2), out=worst_penalty)

            if diagnostics:
                nearest_sum += np.where(covered, nearest, 0.0).sum(axis=2)
                block_max = np.where(covered, nearest, -np.inf).max(axis=2)
                np.maximum(frame_max_nearest, block_max, out=frame_max_nearest)

        receiver_penalty = penalty_sum / self.ground_point_count
        coverage = covered_count / self.ground_point_count

        if not diagnostics:
            return receiver_penalty, worst_penalty, coverage, None, None

        frame_mean_nearest = np.divide(
            nearest_sum,
            covered_count,
            out=np.full_like(nearest_sum, np.nan),
            where=covered_count > 0,
        )
        # If any receiver is uncovered in a frame, the true worst access
        # distance for that frame is unavailable/infinite.
        frame_max_nearest = np.where(
            covered_count == self.ground_point_count,
            frame_max_nearest,
            np.inf,
        )
        return receiver_penalty, worst_penalty, coverage, frame_mean_nearest, frame_max_nearest

    def _evaluate_chunk(self, population, diagnostics=False):
        population = np.asarray(population, dtype=np.float64)
        if population.ndim == 1:
            population = population[None, :]
        c = population.shape[0]

        params = self._decode_population(population)
        x, y, z = self._positions(params)
        link_scores, active_counts, total_distances, _, _ = self._link_stage(x, y, z)
        (
            receiver_penalty,
            frame_worst_receiver_penalty,
            coverage,
            frame_mean_nearest,
            frame_max_nearest,
        ) = self._receiver_stage(x, y, z, diagnostics=diagnostics)

        # Whole-constellation metrics.  Coverage is deliberately based on the
        # WORST sampled frame, not the mean over time.
        average_link_score = link_scores.mean(axis=1)
        average_receiver_penalty = receiver_penalty.mean(axis=1)
        worst_receiver_penalty = frame_worst_receiver_penalty.max(axis=1)
        blended_penalty = blended_receiver_penalty(
            average_receiver_penalty,
            worst_receiver_penalty,
            self.worst_distance_weight,
        )
        quality = network_quality(average_link_score, blended_penalty)
        worst_coverage = coverage.min(axis=1)
        scores = coverage_first_fitness(
            quality,
            worst_coverage,
            self.ground_point_count,
            self.coverage_tolerance,
        )

        if not diagnostics:
            return np.asarray(scores, dtype=np.float64)

        if self.max_possible_links:
            observations = active_counts.sum(axis=1)
            cumulative_distance = total_distances.sum(axis=1)
            mean_link_distance = np.divide(
                cumulative_distance,
                observations,
                out=np.zeros(c, dtype=np.float64),
                where=observations > 0,
            )
            max_links = active_counts.max(axis=1)
            avg_links = active_counts.mean(axis=1)
            avg_total_link_distance = total_distances.mean(axis=1)
        else:
            mean_link_distance = np.zeros(c)
            max_links = np.zeros(c, dtype=int)
            avg_links = np.zeros(c)
            avg_total_link_distance = np.zeros(c)

        rows = []
        for k in range(c):
            valid_means = frame_mean_nearest[k][np.isfinite(frame_mean_nearest[k])]
            full_coverage = bool(
                worst_coverage[k] >= 1.0 - self.coverage_tolerance
            )
            worst_nearest = (
                float(np.max(frame_max_nearest[k]))
                if full_coverage
                else float("inf")
            )
            rows.append(
                FastScoreDiagnostics(
                    score=float(scores[k]),
                    quality_score=float(quality[k]),
                    average_link_score=float(average_link_score[k]),
                    average_receiver_penalty=float(average_receiver_penalty[k]),
                    worst_receiver_penalty=float(worst_receiver_penalty[k]),
                    blended_receiver_penalty=float(blended_penalty[k]),
                    average_coverage=float(coverage[k].mean()),
                    worst_coverage=float(worst_coverage[k]),
                    full_coverage=full_coverage,
                    average_nearest_receiver_distance=(
                        float(valid_means.mean()) if len(valid_means) else float("inf")
                    ),
                    worst_nearest_receiver_distance=worst_nearest,
                    average_active_links=float(avg_links[k]),
                    maximum_active_links=int(max_links[k]),
                    average_link_distance=float(mean_link_distance[k]),
                    average_total_link_distance=float(avg_total_link_distance[k]),
                )
            )
        return np.asarray(scores, dtype=np.float64), rows

    def evaluate_population(self, population) -> np.ndarray:
        population = np.asarray(population, dtype=np.float64)
        single = population.ndim == 1
        if single:
            population = population[None, :]
        if population.ndim != 2:
            raise ValueError("population must be one- or two-dimensional")

        chunks = [
            population[start:min(start + self.candidate_chunk_size, len(population))]
            for start in range(0, len(population), self.candidate_chunk_size)
        ]
        if self.chunk_workers == 1 or len(chunks) <= 1:
            parts = [self._evaluate_chunk(chunk) for chunk in chunks]
        else:
            # NumPy ufuncs release the GIL, so independent chunks can run on CPU
            # cores concurrently.  Ground-distance tensors use extra RAM, hence
            # balanced defaults of chunk-size=4 and workers=4.
            with ThreadPoolExecutor(max_workers=self.chunk_workers) as executor:
                parts = list(executor.map(self._evaluate_chunk, chunks))

        scores = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        return scores[0] if single else scores

    def evaluate_scipy_vectorized(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return -np.asarray(self.evaluate_population(x))
        return -self.evaluate_population(x.T)

    def diagnostics(self, x) -> FastScoreDiagnostics:
        _, rows = self._evaluate_chunk(np.asarray(x)[None, :], diagnostics=True)
        return rows[0]

    def benchmark(self, candidate_count=128, repeats=2, rng_seed=12345):
        candidate_count = max(1, int(candidate_count))
        repeats = max(1, int(repeats))
        rng = np.random.default_rng(rng_seed)
        x0 = self.encode()
        population = np.broadcast_to(x0, (candidate_count, len(x0))).copy()
        population += rng.normal(0.0, 0.05, size=population.shape)
        self.evaluate_population(population[:min(candidate_count, 4)])

        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            self.evaluate_population(population)
            durations.append(time.perf_counter() - start)
        best = min(durations)
        return {
            "candidate_count": candidate_count,
            "seconds": best,
            "candidates_per_second": candidate_count / best,
            "seconds_per_candidate_effective": best / candidate_count,
            "all_seconds": tuple(durations),
        }

    def estimate_de_runtime(self, maxiter, popsize, measured_candidates_per_second):
        population_size = int(popsize) * self.variable_count
        total_candidates = (int(maxiter) + 1) * population_size
        seconds = total_candidates / float(measured_candidates_per_second)
        return {
            "population_size": population_size,
            "total_candidates": total_candidates,
            "seconds": seconds,
        }


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} min"
    hours = minutes / 60.0
    if hours < 48:
        return f"{hours:.2f} h"
    return f"{hours / 24.0:.2f} days"
