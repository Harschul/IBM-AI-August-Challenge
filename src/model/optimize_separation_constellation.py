"""Distance-only maximin optimization of satellite orbital initial conditions.

This optimizer intentionally ignores Earth receivers, network links, routing,
storage, downlink distance, coverage, and every existing fitness term.  Its only
objective is:

    maximize min(distance(satellite_i, satellite_j))

where the minimum is taken across every unique satellite pair and every sampled
simulation frame.  In other words, it maximizes the closest approach that occurs
anywhere in the constellation during the sampled time window.

The resulting constellation is saved in the same JSON format used by
``network_simulation.py``, so it can be rendered directly. Ctrl+C retains and
saves the best candidate from the last completed Differential Evolution batch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time

import numpy as np
from scipy.optimize import differential_evolution

from nodes import Satellite
from satellite_generator import generate_satellites, load_satellites, save_satellites


PARAMETER_BOUNDS = {
    "phase": (0.0, 2.0 * np.pi),
    "inclination": (-np.pi / 2.0, np.pi / 2.0),
    "raan": (0.0, 2.0 * np.pi),
    "angular_velocity": (0.0005, 0.0015),
}

CHECKPOINT_FORMAT = "satellite-network-distance-only-maximin-v1"


def human_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f} min"
    return f"{minutes / 60.0:.2f} h"


def _template(sat):
    return {
        "name": sat.name(),
        "radius": sat.radius(),
        "phase": sat.initial_phase(),
        "inclination": sat.inclination(),
        "raan": sat.raan(),
        "angular_velocity": sat.angular_velocity(),
        "connection_range": sat.connection_range(),
    }


def make_bounds(satellite_count, parameters):
    return [
        PARAMETER_BOUNDS[p]
        for _ in range(int(satellite_count))
        for p in parameters
    ]


def build_satellites(x, templates, parameters):
    values = np.asarray(x, dtype=np.float64).reshape(
        len(templates), len(parameters)
    )
    result = []
    for template, row in zip(templates, values):
        params = dict(template)
        for parameter, value in zip(parameters, row):
            params[parameter] = float(value)
        result.append(Satellite(**params))
    return result


@dataclass(frozen=True)
class SeparationDiagnostics:
    minimum_pair_distance: float
    mean_frame_minimum_distance: float
    mean_pair_distance: float
    worst_frame: int
    worst_time: float
    worst_pair: tuple[int, int]


class SeparationEvaluator:
    """Vectorized evaluator for the distance-only maximin objective."""

    def __init__(
        self,
        *,
        radius,
        phase,
        inclination,
        raan,
        angular_velocity,
        parameters=("phase", "inclination", "raan"),
        simulation_seconds=20_000.0,
        frame_count=180,
        candidate_chunk_size=8,
        chunk_workers=1,
    ):
        self.radius = np.ascontiguousarray(radius, dtype=np.float64)
        self.phase = np.ascontiguousarray(phase, dtype=np.float64)
        self.inclination = np.ascontiguousarray(inclination, dtype=np.float64)
        self.raan = np.ascontiguousarray(raan, dtype=np.float64)
        self.angular_velocity = np.ascontiguousarray(
            angular_velocity, dtype=np.float64
        )

        lengths = {
            len(self.radius),
            len(self.phase),
            len(self.inclination),
            len(self.raan),
            len(self.angular_velocity),
        }
        if len(lengths) != 1:
            raise ValueError("All satellite parameter arrays must have equal length")
        self.satellite_count = len(self.radius)
        if self.satellite_count < 2:
            raise ValueError("Distance-only optimization requires at least 2 satellites")

        self.parameters = tuple(parameters)
        unknown = set(self.parameters) - set(PARAMETER_BOUNDS)
        if unknown:
            raise ValueError(f"Unsupported optimized parameters: {sorted(unknown)}")
        if not self.parameters:
            raise ValueError("At least one optimized parameter is required")

        self.simulation_seconds = float(simulation_seconds)
        self.frame_count = int(frame_count)
        self.candidate_chunk_size = max(1, int(candidate_chunk_size))
        self.chunk_workers = max(1, int(chunk_workers))
        if self.simulation_seconds < 0:
            raise ValueError("simulation_seconds cannot be negative")
        if self.frame_count < 1:
            raise ValueError("frame_count must be at least 1")
        if np.any(self.radius <= 0):
            raise ValueError("radius values must be positive")

        self.times = np.linspace(
            0.0,
            self.simulation_seconds,
            self.frame_count,
            dtype=np.float64,
        )
        self.pair_i, self.pair_j = np.triu_indices(self.satellite_count, k=1)
        self.variable_count = self.satellite_count * len(self.parameters)

    @classmethod
    def from_satellites(cls, satellites, **kwargs):
        satellites = tuple(satellites)
        return cls(
            radius=[sat.radius() for sat in satellites],
            phase=[sat.initial_phase() for sat in satellites],
            inclination=[sat.inclination() for sat in satellites],
            raan=[sat.raan() for sat in satellites],
            angular_velocity=[sat.angular_velocity() for sat in satellites],
            **kwargs,
        )

    def encode(self) -> np.ndarray:
        return np.column_stack(
            [getattr(self, p) for p in self.parameters]
        ).reshape(-1).astype(np.float64, copy=False)

    def _decode_population(self, population):
        population = np.asarray(population, dtype=np.float64)
        if population.ndim == 1:
            population = population[None, :]
        if population.ndim != 2 or population.shape[1] != self.variable_count:
            raise ValueError(
                f"Expected candidate shape (C, {self.variable_count}), "
                f"got {population.shape}"
            )

        c = population.shape[0]
        values = population.reshape(c, self.satellite_count, len(self.parameters))
        decoded = {
            "phase": np.broadcast_to(self.phase, (c, self.satellite_count)),
            "inclination": np.broadcast_to(
                self.inclination, (c, self.satellite_count)
            ),
            "raan": np.broadcast_to(self.raan, (c, self.satellite_count)),
            "angular_velocity": np.broadcast_to(
                self.angular_velocity, (c, self.satellite_count)
            ),
        }
        for column, parameter in enumerate(self.parameters):
            decoded[parameter] = values[:, :, column]
        return decoded

    def _positions(self, params):
        phase0 = params["phase"]
        inc = params["inclination"]
        raan = params["raan"]
        omega = params["angular_velocity"]

        theta = phase0[:, None, :] + omega[:, None, :] * self.times[None, :, None]
        ct = np.cos(theta)
        st = np.sin(theta)
        ci = np.cos(inc)[:, None, :]
        si = np.sin(inc)[:, None, :]
        cr = np.cos(raan)[:, None, :]
        sr = np.sin(raan)[:, None, :]
        radius = self.radius[None, None, :]

        x = radius * (cr * ct - sr * st * ci)
        y = radius * (sr * ct + cr * st * ci)
        z = radius * (st * si)
        return np.stack((x, y, z), axis=-1)

    def _evaluate_chunk(self, population):
        params = self._decode_population(population)
        positions = self._positions(params)
        delta = (
            positions[:, :, self.pair_i, :]
            - positions[:, :, self.pair_j, :]
        )
        distance_sq = np.einsum("...k,...k->...", delta, delta, optimize=True)
        # Monotonic sqrt can be postponed until after the global minimum.
        min_distance_sq = np.min(distance_sq, axis=(1, 2))
        return np.sqrt(np.maximum(min_distance_sq, 0.0))

    def evaluate_population(self, population) -> np.ndarray:
        population = np.asarray(population, dtype=np.float64)
        if population.ndim == 1:
            population = population[None, :]
        if population.ndim != 2 or population.shape[1] != self.variable_count:
            raise ValueError(
                f"Expected candidate shape (C, {self.variable_count}), "
                f"got {population.shape}"
            )

        slices = [
            population[start : start + self.candidate_chunk_size]
            for start in range(0, len(population), self.candidate_chunk_size)
        ]
        if self.chunk_workers == 1 or len(slices) <= 1:
            parts = [self._evaluate_chunk(chunk) for chunk in slices]
        else:
            with ThreadPoolExecutor(max_workers=self.chunk_workers) as pool:
                parts = list(pool.map(self._evaluate_chunk, slices))
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)

    def diagnostics(self, candidate) -> SeparationDiagnostics:
        params = self._decode_population(np.asarray(candidate, dtype=np.float64))
        positions = self._positions(params)[0]
        delta = positions[:, self.pair_i, :] - positions[:, self.pair_j, :]
        distances = np.linalg.norm(delta, axis=-1)  # F x P
        frame_min = np.min(distances, axis=1)
        flat = int(np.argmin(distances))
        frame, pair_index = np.unravel_index(flat, distances.shape)
        return SeparationDiagnostics(
            minimum_pair_distance=float(distances[frame, pair_index]),
            mean_frame_minimum_distance=float(np.mean(frame_min)),
            mean_pair_distance=float(np.mean(distances)),
            worst_frame=int(frame),
            worst_time=float(self.times[frame]),
            worst_pair=(
                int(self.pair_i[pair_index]),
                int(self.pair_j[pair_index]),
            ),
        )

    def benchmark(self, candidate_count=128, repeats=2, seed=12345):
        candidate_count = max(1, int(candidate_count))
        repeats = max(1, int(repeats))
        rng = np.random.default_rng(seed)
        bounds = np.asarray(
            make_bounds(self.satellite_count, self.parameters), dtype=np.float64
        )
        low, high = bounds[:, 0], bounds[:, 1]
        population = rng.uniform(low, high, size=(candidate_count, self.variable_count))

        self.evaluate_population(population[: min(candidate_count, 4)])
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            self.evaluate_population(population)
            timings.append(time.perf_counter() - started)
        seconds = min(timings)
        return {
            "candidate_count": candidate_count,
            "seconds": seconds,
            "candidates_per_second": candidate_count / seconds if seconds else np.inf,
        }


def checkpoint_payload(x, score, parameters, evaluations, elapsed):
    return {
        "format": CHECKPOINT_FORMAT,
        "objective": "maximize minimum pairwise distance over all sampled frames",
        "score_km": float(score),
        "parameters": list(parameters),
        "evaluations": int(evaluations),
        "elapsed_seconds": float(elapsed),
        "x": np.asarray(x, dtype=float).tolist(),
    }


class VectorizedSeparationObjective:
    def __init__(self, evaluator, parameters, checkpoint_path=None):
        self.evaluator = evaluator
        self.parameters = tuple(parameters)
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path
            else None
        )
        self.batch_calls = 0
        self.evaluations = 0
        self.best_score = -np.inf
        self.best_x = None
        self.started = time.perf_counter()

    def save_checkpoint(self):
        if self.checkpoint_path is None or self.best_x is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = checkpoint_payload(
            self.best_x,
            self.best_score,
            self.parameters,
            self.evaluations,
            time.perf_counter() - self.started,
        )
        temp = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(self.checkpoint_path)

    def __call__(self, x):
        # SciPy vectorized=True normally supplies shape (variables, population).
        x = np.asarray(x, dtype=np.float64)
        population = x[None, :] if x.ndim == 1 else x.T
        scores = np.atleast_1d(
            self.evaluator.evaluate_population(population)
        ).astype(np.float64)

        self.batch_calls += 1
        self.evaluations += len(population)
        idx = int(np.argmax(scores))
        candidate_best = float(scores[idx])
        improved = candidate_best > self.best_score
        if improved:
            self.best_score = candidate_best
            self.best_x = np.array(population[idx], copy=True)
            self.save_checkpoint()

        elapsed = time.perf_counter() - self.started
        rate = self.evaluations / elapsed if elapsed else 0.0
        print(
            f"batch={self.batch_calls:3d} | evaluated={self.evaluations:6d} | "
            f"best min separation={self.best_score:10.3f} km | "
            f"{rate:7.1f} candidates/s"
            + (" NEW BEST" if improved else "")
        )
        return -scores


def load_checkpoint(path):
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported distance-only checkpoint format")
    return payload


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--satellites", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--baseline-constellation",
        type=Path,
        help=(
            "Optimize this saved constellation instead of regenerating the "
            "seeded baseline. Use the same file as the coverage-first optimizer "
            "for a controlled comparison later."
        ),
    )
    p.add_argument("--optimizer-seed", type=int, default=4321)
    p.add_argument("--simulation-seconds", type=float, default=20_000.0)
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--maxiter", type=int, default=10)
    p.add_argument("--popsize", type=int, default=5)
    p.add_argument(
        "--parameters",
        nargs="+",
        choices=tuple(PARAMETER_BOUNDS),
        default=["phase", "inclination", "raan"],
    )
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--chunk-workers", type=int, default=1)
    p.add_argument("--benchmark-candidates", type=int, default=128)
    p.add_argument("--benchmark-repeats", type=int, default=2)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("distance_optimizer_checkpoint.json"),
    )
    p.add_argument("--resume-checkpoint", type=Path)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("distance_optimized_constellation.json"),
    )
    return p.parse_args()


def print_diagnostics(label, d: SeparationDiagnostics):
    a, b = d.worst_pair
    print(f"\n{label}")
    print(f"minimum pair distance       : {d.minimum_pair_distance:.3f} km")
    print(f"mean frame minimum distance : {d.mean_frame_minimum_distance:.3f} km")
    print(f"mean all-pair distance      : {d.mean_pair_distance:.3f} km")
    print(f"closest-approach frame      : {d.worst_frame}")
    print(f"closest-approach time       : {d.worst_time:.3f} s")
    print(f"closest satellite pair      : {a}, {b}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.baseline_constellation:
        baseline = load_satellites(args.baseline_constellation)
        print(f"Loaded baseline constellation: {args.baseline_constellation.resolve()}")
    else:
        baseline = generate_satellites(number=args.satellites, seed=args.seed)

    if len(baseline) < 2:
        raise ValueError("Distance-only optimization requires at least 2 satellites")

    templates = [_template(s) for s in baseline]
    bounds = make_bounds(len(baseline), args.parameters)
    evaluator = SeparationEvaluator.from_satellites(
        baseline,
        parameters=args.parameters,
        simulation_seconds=args.simulation_seconds,
        frame_count=args.frames,
        candidate_chunk_size=args.chunk_size,
        chunk_workers=args.chunk_workers,
    )

    baseline_x = evaluator.encode()
    x0 = baseline_x.copy()
    if args.resume_checkpoint:
        cp = load_checkpoint(args.resume_checkpoint)
        if tuple(cp["parameters"]) != tuple(args.parameters):
            raise ValueError("Checkpoint optimized parameters do not match this run")
        x0 = np.asarray(cp["x"], dtype=np.float64)
        if x0.shape != baseline_x.shape:
            raise ValueError("Checkpoint variable count does not match this run")
        rescored = evaluator.diagnostics(x0)
        print(
            "Resuming checkpoint vector: "
            f"stored min separation={cp['score_km']:.3f} km; "
            f"current={rescored.minimum_pair_distance:.3f} km"
        )

    baseline_diag = evaluator.diagnostics(baseline_x)
    print_diagnostics("BASELINE DISTANCE GEOMETRY", baseline_diag)

    population_size = args.popsize * len(bounds)
    total_candidates = (args.maxiter + 1) * population_size
    print("\nDISTANCE-ONLY OPTIMIZER CONFIGURATION")
    print(f"satellites                  : {len(baseline)}")
    print(f"optimized variables         : {len(bounds)} ({', '.join(args.parameters)})")
    print(f"sampled frames / candidate  : {args.frames}")
    print(f"simulation window           : {args.simulation_seconds:.1f} s")
    print("objective                   : maximize global minimum pair distance")
    print("Earth receiver terms        : NOT USED")
    print("network/link terms          : NOT USED")
    print("routing/storage terms       : NOT USED")
    print(f"DE population               : {population_size}")
    print(f"max candidates              : ~{total_candidates:,}")
    print(f"candidate chunk size        : {args.chunk_size}")
    print(f"chunk workers               : {args.chunk_workers}")

    if args.benchmark_candidates > 0:
        b = evaluator.benchmark(
            args.benchmark_candidates,
            args.benchmark_repeats,
            args.optimizer_seed + 991,
        )
        estimated_seconds = total_candidates / b["candidates_per_second"]
        print("\nLOCAL SPEED BENCHMARK")
        print(
            f"{b['candidate_count']} candidates in {b['seconds']:.3f}s -> "
            f"{b['candidates_per_second']:.1f} candidates/s"
        )
        print(
            "Approximate raw evaluator time: "
            f"{human_duration(estimated_seconds)}"
        )

    objective = VectorizedSeparationObjective(
        evaluator, args.parameters, args.checkpoint
    )
    started = time.perf_counter()
    result = None
    interrupted = False

    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=args.maxiter,
            popsize=args.popsize,
            polish=False,
            seed=args.optimizer_seed,
            disp=True,
            updating="deferred",
            workers=1,
            vectorized=True,
            x0=x0,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\nCtrl+C received: retaining the best completed vectorized batch.")

    elapsed = time.perf_counter() - started
    if result is not None:
        best_x = np.asarray(result.x, dtype=np.float64)
    elif objective.best_x is not None:
        best_x = objective.best_x
    else:
        raise RuntimeError("Stopped before a candidate batch completed")

    best_diag = evaluator.diagnostics(best_x)
    best_satellites = build_satellites(best_x, templates, args.parameters)
    saved = save_satellites(best_satellites, args.output)

    print_diagnostics("DISTANCE-ONLY OPTIMIZED RESULT", best_diag)
    print(
        "minimum-separation improvement: "
        f"{best_diag.minimum_pair_distance-baseline_diag.minimum_pair_distance:+.3f} km"
    )
    print(f"elapsed                     : {human_duration(elapsed)}")
    print(f"evaluated candidates        : {objective.evaluations:,}")
    print(f"saved constellation         : {saved}")
    if interrupted:
        print("status                      : interrupted; best-so-far saved")

    print("\nRender the distance-only result:")
    print(
        "  python network_simulation.py "
        f'--constellation "{saved}" '
        "--output VIDEO_B_distance_only.mp4 "
        '--video-label "B - DISTANCE-ONLY MAXIMIN SEPARATION"'
    )


if __name__ == "__main__":
    main()
