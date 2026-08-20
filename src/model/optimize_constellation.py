"""Coverage-first vectorized optimization of satellite initial conditions.

The optimizer changes the selected orbital parameters (phase, inclination and
RAAN by default) and uses SciPy Differential Evolution to maximize a strict
coverage-first fitness.

Whole-constellation scoring:

    mean_ground  = mean normalized nearest-visible-satellite receiver penalty
    worst_ground = worst normalized receiver penalty over all frames/receivers
    ground       = weighted blend(mean_ground, worst_ground)
    quality      = (1 + average_link_score - ground) / 2

    if worst_frame_coverage < 100%:
        fitness < 0.5 and is ranked primarily by worst-frame coverage
    else:
        fitness = 0.5 + 0.5 * quality

This makes full sampled Earth coverage a feasibility requirement rather than a
quantity that can be traded away for longer inter-satellite links. Ctrl+C keeps
and saves the best candidate from the last completed vectorized batch.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
from scipy.optimize import differential_evolution

from fast_scoring import FastConstellationEvaluator, human_duration
from network import Network
from nodes import Satellite
from satellite_generator import generate_satellites, load_satellites, save_satellites
from scoring import print_score_summary, score_network, write_score_csv


PARAMETER_BOUNDS = {
    "phase": (0.0, 2.0 * np.pi),
    "inclination": (-np.pi / 2.0, np.pi / 2.0),
    "raan": (0.0, 2.0 * np.pi),
    "angular_velocity": (0.0005, 0.0015),
}

CHECKPOINT_FORMAT = "satellite-network-coverage-first-optimizer-v3"
OLD_CHECKPOINT_FORMAT = "satellite-network-ground-optimizer-checkpoint-v2"


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
    x = np.asarray(x, dtype=np.float64)
    values = x.reshape(len(templates), len(parameters))
    result = []
    for template, row in zip(templates, values):
        params = dict(template)
        for parameter, value in zip(parameters, row):
            params[parameter] = float(value)
        result.append(Satellite(**params))
    return result


def checkpoint_payload(x, score, parameters, evaluations, elapsed):
    return {
        "format": CHECKPOINT_FORMAT,
        "score": float(score),
        "parameters": list(parameters),
        "evaluations": int(evaluations),
        "elapsed_seconds": float(elapsed),
        "x": np.asarray(x, dtype=float).tolist(),
    }


class VectorizedObjective:
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
        temp = self.checkpoint_path.with_suffix(
            self.checkpoint_path.suffix + ".tmp"
        )
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(self.checkpoint_path)

    def __call__(self, x):
        # With scipy vectorized=True, x is normally shaped (variables, population).
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
            f"best={self.best_score:.6f} | {rate:7.1f} candidates/s"
            + (" NEW BEST" if improved else "")
        )
        return -scores


def load_checkpoint(path):
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    fmt = payload.get("format")
    if fmt not in {CHECKPOINT_FORMAT, OLD_CHECKPOINT_FORMAT}:
        raise ValueError("Unsupported checkpoint format")
    if fmt == OLD_CHECKPOINT_FORMAT:
        print(
            "WARNING: loading a checkpoint made with the old soft-coverage "
            "objective. Its stored score is not comparable; the candidate "
            "vector will be rescored using the new coverage-first objective."
        )
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
            "seeded baseline. Recommended for exact before/after comparisons."
        ),
    )
    p.add_argument("--optimizer-seed", type=int, default=1234)
    p.add_argument("--simulation-seconds", type=float, default=20_000.0)
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--earth-radius", type=float, default=6371.0)

    p.add_argument("--ground-points", type=int, default=100)
    p.add_argument("--ground-distance-scale", type=float, default=5000.0)
    p.add_argument(
        "--worst-distance-weight",
        type=float,
        default=0.5,
        help=(
            "Weight assigned to the worst receiver-distance penalty; the "
            "remaining weight goes to the mean penalty (default: 0.5)."
        ),
    )
    p.add_argument(
        "--coverage-tolerance",
        type=float,
        default=1e-12,
        help="Numerical tolerance used when deciding whether coverage is 100%%.",
    )
    p.add_argument("--min-elevation-deg", type=float, default=0.0)

    p.add_argument("--maxiter", type=int, default=10)
    p.add_argument("--popsize", type=int, default=5)
    p.add_argument(
        "--parameters",
        nargs="+",
        choices=tuple(PARAMETER_BOUNDS),
        default=["phase", "inclination", "raan"],
    )
    p.add_argument("--ground-chunk-size", type=int, default=20)
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--chunk-workers", type=int, default=4)
    p.add_argument("--benchmark-candidates", type=int, default=128)
    p.add_argument("--benchmark-repeats", type=int, default=2)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("optimizer_checkpoint.json"),
    )
    p.add_argument("--resume-checkpoint", type=Path)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("optimized_constellation.json"),
    )
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("optimized_constellation_metrics.csv"),
    )
    p.add_argument("--skip-reference-validation", action="store_true")
    return p.parse_args()


def _distance_text(value):
    return f"{value:.2f} km" if np.isfinite(value) else "uncovered"


def print_fast(label, d):
    print(f"\n{label}")
    print(f"fitness                     : {d.score:.6f} / 1.000000")
    print(f"coverage status             : {'FULL' if d.full_coverage else 'INCOMPLETE'}")
    print(f"worst-frame Earth coverage  : {100*d.worst_coverage:.2f}%")
    print(f"average Earth coverage      : {100*d.average_coverage:.2f}%")
    print(f"quality score               : {d.quality_score:.6f}")
    print(f"average link score          : {d.average_link_score:.6f}")
    print(f"mean receiver penalty       : {d.average_receiver_penalty:.6f}")
    print(f"worst receiver penalty      : {d.worst_receiver_penalty:.6f}")
    print(f"blended receiver penalty    : {d.blended_receiver_penalty:.6f}")
    print(f"avg nearest receiver->sat   : {_distance_text(d.average_nearest_receiver_distance)}")
    print(f"worst sampled receiver dist : {_distance_text(d.worst_nearest_receiver_distance)}")
    print(f"average active links        : {d.average_active_links:.2f}")
    print(f"average valid-link distance : {d.average_link_distance:.2f} km")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)

    if not 0.0 <= args.worst_distance_weight <= 1.0:
        raise ValueError("worst-distance-weight must be between 0 and 1")
    if args.coverage_tolerance < 0:
        raise ValueError("coverage-tolerance cannot be negative")

    if args.baseline_constellation:
        baseline = load_satellites(args.baseline_constellation)
        print(
            "Loaded baseline constellation: "
            f"{args.baseline_constellation.resolve()}"
        )
    else:
        baseline = generate_satellites(number=args.satellites, seed=args.seed)

    templates = [_template(s) for s in baseline]
    bounds = make_bounds(len(baseline), args.parameters)

    evaluator = FastConstellationEvaluator.from_satellites(
        baseline,
        parameters=args.parameters,
        simulation_seconds=args.simulation_seconds,
        frame_count=args.frames,
        earth_radius=args.earth_radius,
        require_line_of_sight=True,
        ground_point_count=args.ground_points,
        ground_distance_scale=args.ground_distance_scale,
        worst_distance_weight=args.worst_distance_weight,
        coverage_tolerance=args.coverage_tolerance,
        min_elevation_deg=args.min_elevation_deg,
        ground_chunk_size=args.ground_chunk_size,
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
            f"Resuming checkpoint vector: previous stored score={cp['score']:.6f}; "
            f"current coverage-first score={rescored.score:.6f}"
        )

    baseline_diag = evaluator.diagnostics(baseline_x)
    print_fast("BASELINE (FAST EVALUATOR)", baseline_diag)

    population_size = args.popsize * len(bounds)
    total_candidates = (args.maxiter + 1) * population_size
    print("\nOPTIMIZER CONFIGURATION")
    print(f"satellites                 : {len(baseline)}")
    print(f"optimized variables        : {len(bounds)} ({', '.join(args.parameters)})")
    print(f"frames / candidate         : {args.frames}")
    print(f"virtual ground points      : {args.ground_points}")
    print(f"receiver distance scale    : {args.ground_distance_scale:.1f} km")
    print(f"worst-distance weight      : {args.worst_distance_weight:.2f}")
    print(f"minimum elevation          : {args.min_elevation_deg:.1f} deg")
    print("coverage objective         : worst-frame coverage first; 100% required")
    print(f"DE population              : {population_size}")
    print(f"max candidates             : ~{total_candidates:,}")
    print(f"ground block size          : {args.ground_chunk_size}")
    print(f"NumPy candidate chunk size : {args.chunk_size}")
    print(f"chunk workers              : {args.chunk_workers}")

    if args.benchmark_candidates > 0:
        b = evaluator.benchmark(
            args.benchmark_candidates,
            args.benchmark_repeats,
            args.optimizer_seed + 991,
        )
        est = evaluator.estimate_de_runtime(
            args.maxiter,
            args.popsize,
            b["candidates_per_second"],
        )
        print("\nLOCAL SPEED BENCHMARK")
        print(
            f"{b['candidate_count']} candidates in {b['seconds']:.3f}s -> "
            f"{b['candidates_per_second']:.1f} candidates/s"
        )
        print(
            "Estimated optimizer wall time: "
            f"{human_duration(est['seconds']*1.05)} - "
            f"{human_duration(est['seconds']*1.35)}"
        )

    objective = VectorizedObjective(evaluator, args.parameters, args.checkpoint)
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

    print_fast("OPTIMIZED RESULT (FAST EVALUATOR)", best_diag)
    print(f"elapsed                     : {human_duration(elapsed)}")
    print(f"evaluated candidates        : {objective.evaluations:,}")
    print(f"fitness improvement         : {best_diag.score-baseline_diag.score:+.6f}")
    print(f"saved constellation         : {saved}")
    if interrupted:
        print("status                      : interrupted; best-so-far saved")

    if not args.skip_reference_validation:
        network = Network(
            best_satellites,
            earth_radius=args.earth_radius,
            require_line_of_sight=True,
        )
        snapshots = network.simulate(args.simulation_seconds, args.frames)
        reference = score_network(
            snapshots,
            best_satellites,
            ground_point_count=args.ground_points,
            earth_radius=args.earth_radius,
            ground_distance_scale=args.ground_distance_scale,
            worst_distance_weight=args.worst_distance_weight,
            coverage_tolerance=args.coverage_tolerance,
            min_elevation_deg=args.min_elevation_deg,
        )
        print("\nREFERENCE-SIMULATOR VALIDATION")
        print_score_summary(reference, len(best_satellites))
        print(f"fast score      : {best_diag.score:.12f}")
        print(f"reference score : {reference.score:.12f}")
        print(f"absolute error  : {abs(best_diag.score-reference.score):.3e}")
        print(
            "Saved optimized metrics: "
            f"{write_score_csv(reference, args.metrics_csv)}"
        )

    print("\nRender the result with matching defaults:")
    print(f'  python network_simulation.py --constellation "{saved}"')


if __name__ == "__main__":
    main()
