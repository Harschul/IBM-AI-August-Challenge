"""Verify fast/reference scoring agreement and benchmark optimizer throughput."""

from __future__ import annotations

import argparse
import numpy as np

from fast_scoring import FastConstellationEvaluator, human_duration
from network import Network
from satellite_generator import generate_satellites
from scoring import score_network


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--satellites", type=int, default=30)
    p.add_argument("--frames", type=int, default=180)
    p.add_argument("--simulation-seconds", type=float, default=20_000.0)
    p.add_argument("--ground-points", type=int, default=100)
    p.add_argument("--ground-distance-scale", type=float, default=5000.0)
    p.add_argument("--worst-distance-weight", type=float, default=0.5)
    p.add_argument("--coverage-tolerance", type=float, default=1e-12)
    p.add_argument("--min-elevation-deg", type=float, default=0.0)
    p.add_argument("--maxiter", type=int, default=10)
    p.add_argument("--popsize", type=int, default=5)
    p.add_argument("--ground-chunk-size", type=int, default=20)
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--chunk-workers", type=int, default=4)
    p.add_argument("--candidates", type=int, default=128)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _dist(value):
    return f"{value:.3f} km" if np.isfinite(value) else "uncovered"


def main():
    a = parse_args()
    sats = generate_satellites(a.satellites, seed=a.seed)
    evaluator = FastConstellationEvaluator.from_satellites(
        sats,
        parameters=("phase", "inclination", "raan"),
        simulation_seconds=a.simulation_seconds,
        frame_count=a.frames,
        ground_point_count=a.ground_points,
        ground_distance_scale=a.ground_distance_scale,
        worst_distance_weight=a.worst_distance_weight,
        coverage_tolerance=a.coverage_tolerance,
        min_elevation_deg=a.min_elevation_deg,
        ground_chunk_size=a.ground_chunk_size,
        candidate_chunk_size=a.chunk_size,
        chunk_workers=a.chunk_workers,
    )

    fast = evaluator.diagnostics(evaluator.encode())
    network = Network(sats, earth_radius=6371.0, require_line_of_sight=True)
    reference = score_network(
        network.simulate(a.simulation_seconds, a.frames),
        sats,
        ground_point_count=a.ground_points,
        ground_distance_scale=a.ground_distance_scale,
        worst_distance_weight=a.worst_distance_weight,
        coverage_tolerance=a.coverage_tolerance,
        min_elevation_deg=a.min_elevation_deg,
    )

    print("CORRECTNESS CHECK")
    print(f"fast fitness                 : {fast.score:.12f}")
    print(f"reference fitness            : {reference.score:.12f}")
    print(f"absolute error               : {abs(fast.score-reference.score):.3e}")
    print(f"fast quality                 : {fast.quality_score:.12f}")
    print(f"reference quality            : {reference.quality_score:.12f}")
    print(f"fast worst coverage          : {100*fast.worst_coverage:.4f}%")
    print(f"reference worst coverage     : {100*reference.worst_coverage:.4f}%")
    print(f"fast worst receiver penalty  : {fast.worst_receiver_penalty:.12f}")
    print(f"reference worst recv penalty : {reference.worst_receiver_penalty:.12f}")
    print(f"fast worst receiver distance : {_dist(fast.worst_nearest_receiver_distance)}")
    print(f"ref worst receiver distance  : {_dist(reference.worst_nearest_receiver_distance)}")

    b = evaluator.benchmark(a.candidates, a.repeats, 12345)
    est = evaluator.estimate_de_runtime(
        a.maxiter,
        a.popsize,
        b["candidates_per_second"],
    )
    print("\nSPEED")
    print(f"candidate throughput : {b['candidates_per_second']:.1f} candidates/s")
    print(f"DE population        : {est['population_size']}")
    print(f"max candidates       : ~{est['total_candidates']:,}")
    print(f"raw evaluator time   : {human_duration(est['seconds'])}")
    print(
        "practical estimate   : "
        f"{human_duration(est['seconds']*1.05)} - "
        f"{human_duration(est['seconds']*1.35)}"
    )


if __name__ == "__main__":
    main()
