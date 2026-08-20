"""Optimize satellite initial conditions against the normalized network score.

The optimizer reuses exactly the same scoring function as network_simulation.py.
By default it optimizes phase, inclination and RAAN while keeping each random
baseline satellite's radius, angular velocity and connection range fixed.

Differential evolution is used because the valid-link / line-of-sight objective
is discontinuous and therefore not well suited to gradient-based minimizers.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from network import Network
from nodes import Satellite
from satellite_generator import generate_satellites, save_satellites
from scoring import print_score_summary, score_network, write_score_csv


PARAMETER_BOUNDS = {
    "phase": (0.0, 2.0 * np.pi),
    "inclination": (-np.pi / 2.0, np.pi / 2.0),
    "raan": (0.0, 2.0 * np.pi),
}


def _template_from_satellite(satellite) -> dict:
    return {
        "name": satellite.name(),
        "radius": satellite.radius(),
        "phase": satellite.initial_phase(),
        "inclination": satellite.inclination(),
        "raan": satellite.raan(),
        "angular_velocity": satellite.angular_velocity(),
        "connection_range": satellite.connection_range(),
    }


def encode_satellites(satellites, parameters) -> np.ndarray:
    """Flatten selected initial-condition fields into SciPy's x vector."""
    values = []
    for satellite in satellites:
        template = _template_from_satellite(satellite)
        for parameter in parameters:
            values.append(template[parameter])
    return np.asarray(values, dtype=float)


def make_bounds(satellite_count: int, parameters) -> list[tuple[float, float]]:
    bounds = []
    for _ in range(satellite_count):
        for parameter in parameters:
            bounds.append(PARAMETER_BOUNDS[parameter])
    return bounds


def build_satellites(x, templates, parameters):
    """Decode SciPy's x vector into fresh Satellite objects at t=0."""
    x = np.asarray(x, dtype=float)
    width = len(parameters)
    expected = len(templates) * width
    if x.size != expected:
        raise ValueError(f"Expected {expected} variables, got {x.size}")

    values = x.reshape(len(templates), width)
    satellites = []

    for template, row in zip(templates, values):
        params = dict(template)
        for parameter, value in zip(parameters, row):
            params[parameter] = float(value)
        satellites.append(Satellite(**params))

    return satellites


class ConstellationObjective:
    """Callable objective that SciPy minimizes (negative fitness)."""

    def __init__(
        self,
        templates,
        parameters,
        simulation_seconds,
        frame_count,
        earth_radius,
        report_every=25,
    ):
        self.templates = templates
        self.parameters = tuple(parameters)
        self.simulation_seconds = float(simulation_seconds)
        self.frame_count = int(frame_count)
        self.earth_radius = float(earth_radius)
        self.report_every = max(1, int(report_every))
        self.evaluations = 0
        self.best_score = -np.inf

    def __call__(self, x):
        satellites = build_satellites(x, self.templates, self.parameters)
        network = Network(
            satellites,
            earth_radius=self.earth_radius,
            require_line_of_sight=True,
        )
        snapshots = network.simulate(self.simulation_seconds, self.frame_count)
        results = score_network(snapshots, satellites)

        self.evaluations += 1
        improved = results.score > self.best_score
        if improved:
            self.best_score = results.score

        if improved or self.evaluations % self.report_every == 0:
            marker = " BEST" if improved else ""
            print(
                f"eval={self.evaluations:6d} | score={results.score:.6f} | "
                f"avg_links={results.average_active_links:7.2f} | "
                f"avg_distance={results.average_link_distance:8.1f} km{marker}"
            )

        # scipy.optimize minimizes, so negate the normalized fitness.
        return -results.score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--satellites", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer-seed", type=int, default=1234)
    parser.add_argument("--simulation-seconds", type=float, default=20_000.0)
    parser.add_argument(
        "--frames",
        type=int,
        default=180,
        help="Cheap scoring frames used during optimization (default: 180).",
    )
    parser.add_argument("--earth-radius", type=float, default=6371.0)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=tuple(PARAMETER_BOUNDS),
        default=["phase", "inclination", "raan"],
        help="Initial conditions to optimize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("optimized_constellation.json"),
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("optimized_constellation_metrics.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        # Optimization evaluates many simulations; WARNING keeps output readable.
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    baseline = generate_satellites(number=args.satellites, seed=args.seed)
    templates = [_template_from_satellite(sat) for sat in baseline]
    x0 = encode_satellites(baseline, args.parameters)
    bounds = make_bounds(len(baseline), args.parameters)

    # Establish the random baseline using exactly the same reduced simulation used
    # during optimization, so the comparison is apples-to-apples.
    baseline_network = Network(
        baseline,
        earth_radius=args.earth_radius,
        require_line_of_sight=True,
    )
    baseline_snapshots = baseline_network.simulate(args.simulation_seconds, args.frames)
    baseline_results = score_network(baseline_snapshots, baseline)

    print("\nRANDOM BASELINE")
    print_score_summary(baseline_results, len(baseline))
    print(
        f"Optimizing {len(bounds)} variables: {', '.join(args.parameters)}\n"
        f"DE maxiter={args.maxiter}, popsize={args.popsize}, frames={args.frames}\n"
    )

    objective = ConstellationObjective(
        templates=templates,
        parameters=args.parameters,
        simulation_seconds=args.simulation_seconds,
        frame_count=args.frames,
        earth_radius=args.earth_radius,
    )

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        polish=False,
        seed=args.optimizer_seed,
        disp=True,
        updating="immediate",
        workers=1,
        x0=x0,
    )

    best_satellites = build_satellites(result.x, templates, args.parameters)
    best_network = Network(
        best_satellites,
        earth_radius=args.earth_radius,
        require_line_of_sight=True,
    )
    best_snapshots = best_network.simulate(args.simulation_seconds, args.frames)
    best_results = score_network(best_snapshots, best_satellites)

    print("\nOPTIMIZED CONSTELLATION")
    print_score_summary(best_results, len(best_satellites))
    print(f"Random score    : {baseline_results.score:.6f}")
    print(f"Optimized score : {best_results.score:.6f}")
    print(f"Improvement     : {best_results.score - baseline_results.score:+.6f}")

    constellation_path = save_satellites(best_satellites, args.output)
    metrics_path = write_score_csv(best_results, args.metrics_csv)
    print(f"Saved optimized constellation: {constellation_path}")
    print(f"Saved optimized metrics      : {metrics_path}")
    print("\nRender it with:")
    print(
        f"  python network_simulation.py --constellation \"{constellation_path}\""
    )


if __name__ == "__main__":
    main()
