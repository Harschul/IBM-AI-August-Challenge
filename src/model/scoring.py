"""Reference scoring for satellite links and Earth receiver access.

This module is the readable/reference implementation used for final validation,
CSV diagnostics and rendering.  The vectorized optimizer in ``fast_scoring.py``
uses the same shared fitness functions from ``fitness.py``.

The final whole-constellation objective applies three important rules:

1. **Worst-frame coverage matters.** A brief coverage hole cannot be hidden by
   excellent coverage at other times.
2. **Worst receiver distance matters as well as the mean.** The normalized mean
   and worst receiver penalties are blended (50/50 by default).
3. **Full sampled coverage is effectively mandatory.** Incomplete candidates
   are confined below 0.5 fitness and ranked primarily by worst-frame coverage.
   A candidate with 100% worst-frame coverage enters the [0.5, 1.0] range and is
   then ranked by link-minus-receiver quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np

from fitness import blended_receiver_penalty, coverage_first_fitness, network_quality
from ground_receivers import fibonacci_sphere, nearest_visible_satellite_distances


@dataclass(frozen=True)
class SnapshotScore:
    frame: int
    time: float
    active_links: int
    max_possible_links: int
    link_fraction: float
    total_link_distance: float
    mean_link_distance: float
    normalized_distance_sum: float
    link_score: float
    receiver_count: int
    covered_receivers: int
    coverage_fraction: float
    mean_nearest_receiver_distance: float
    max_nearest_receiver_distance: float
    receiver_penalty: float
    worst_receiver_penalty: float
    blended_receiver_penalty: float
    base_score: float
    score: float
    running_score: float


@dataclass(frozen=True)
class NetworkScore:
    score: float
    quality_score: float
    frame_scores: tuple[SnapshotScore, ...]
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
    cumulative_link_distance: float
    total_link_observations: int
    max_possible_links: int
    receiver_count: int
    ground_distance_scale: float
    worst_distance_weight: float
    coverage_tolerance: float
    min_elevation_deg: float


def _connection_ranges(satellites) -> np.ndarray:
    return np.fromiter(
        (sat.connection_range() for sat in satellites),
        dtype=np.float64,
        count=len(satellites),
    )


def _link_metrics(snapshot, satellites, connection_ranges):
    n = len(satellites)
    max_possible_links = n * (n - 1) // 2
    active_links = len(snapshot.connection_indices)

    if max_possible_links == 0 or active_links == 0:
        return 0.0, active_links, 0.0, 0.0, 0.0, max_possible_links

    active = np.asarray(snapshot.connection_indices, dtype=np.intp)
    i_idx = active[:, 0]
    j_idx = active[:, 1]
    delta = snapshot.positions[i_idx] - snapshot.positions[j_idx]
    distances = np.linalg.norm(delta, axis=1)
    pair_ranges = np.minimum(connection_ranges[i_idx], connection_ranges[j_idx])

    normalized = np.divide(
        distances,
        pair_ranges,
        out=np.zeros_like(distances),
        where=pair_ranges > 0.0,
    )
    np.clip(normalized, 0.0, 1.0, out=normalized)

    normalized_sum = float(np.sum(normalized))
    total_distance = float(np.sum(distances))
    mean_distance = float(np.mean(distances))
    link_score = normalized_sum / max_possible_links
    return (
        float(np.clip(link_score, 0.0, 1.0)),
        active_links,
        total_distance,
        mean_distance,
        normalized_sum,
        max_possible_links,
    )


def score_network(
    snapshots,
    satellites,
    *,
    ground_points=None,
    ground_point_count: int = 100,
    earth_radius: float = 6371.0,
    ground_distance_scale: float = 5000.0,
    worst_distance_weight: float = 0.5,
    coverage_tolerance: float = 1e-12,
    min_elevation_deg: float = 0.0,
) -> NetworkScore:
    """Score an entire reference simulation using coverage-first fitness.

    ``ground_distance_scale`` maps receiver distance to a normalized penalty:
    ``distance / ground_distance_scale``, clipped to [0, 1].  An uncovered
    receiver receives a penalty of exactly 1.

    ``worst_distance_weight`` controls the blend between the mean receiver
    penalty and the single worst sampled receiver penalty.  The default 0.5 is
    an equal mean/worst blend.
    """
    snapshots = tuple(snapshots)
    satellites = tuple(satellites)
    if not snapshots:
        raise ValueError("snapshots cannot be empty")
    if not satellites:
        raise ValueError("satellites cannot be empty")
    if ground_distance_scale <= 0:
        raise ValueError("ground_distance_scale must be positive")
    if not 0.0 <= float(worst_distance_weight) <= 1.0:
        raise ValueError("worst_distance_weight must be between 0 and 1")
    if float(coverage_tolerance) < 0:
        raise ValueError("coverage_tolerance cannot be negative")

    if ground_points is None:
        ground_points = fibonacci_sphere(ground_point_count, radius=earth_radius)
    ground_points = np.asarray(ground_points, dtype=np.float64)
    if ground_points.ndim != 2 or ground_points.shape[1] != 3:
        raise ValueError("ground_points must have shape (G, 3)")

    receiver_count = len(ground_points)
    connection_ranges = _connection_ranges(satellites)
    max_possible_links = len(satellites) * (len(satellites) - 1) // 2

    rows = []
    running_total = 0.0
    total_link_observations = 0
    cumulative_link_distance = 0.0
    maximum_active_links = 0

    for k, snapshot in enumerate(snapshots, start=1):
        (
            link_score,
            active_links,
            total_link_distance,
            mean_link_distance,
            normalized_distance_sum,
            _,
        ) = _link_metrics(snapshot, satellites, connection_ranges)

        nearest, covered = nearest_visible_satellite_distances(
            snapshot.positions,
            ground_points,
            earth_radius=earth_radius,
            min_elevation_deg=min_elevation_deg,
        )
        covered_count = int(np.count_nonzero(covered))
        coverage_fraction = covered_count / receiver_count

        # Uncovered receivers stay at penalty=1.  Covered distances are scaled
        # and clipped into [0, 1].
        normalized_receiver = np.ones(receiver_count, dtype=np.float64)
        if covered_count:
            normalized_receiver[covered] = np.clip(
                nearest[covered] / ground_distance_scale,
                0.0,
                1.0,
            )
            mean_nearest = float(np.mean(nearest[covered]))
        else:
            mean_nearest = float("inf")

        # If even one receiver is uncovered, the true worst access distance is
        # unbounded/unavailable rather than merely the worst of the covered set.
        if covered_count == receiver_count:
            max_nearest = float(np.max(nearest))
        else:
            max_nearest = float("inf")

        receiver_penalty = float(np.mean(normalized_receiver))
        worst_receiver_penalty = float(np.max(normalized_receiver))
        blended_penalty = float(
            blended_receiver_penalty(
                receiver_penalty,
                worst_receiver_penalty,
                worst_distance_weight,
            )
        )
        base_score = float(network_quality(link_score, blended_penalty))
        frame_score = float(
            coverage_first_fitness(
                base_score,
                coverage_fraction,
                receiver_count,
                coverage_tolerance,
            )
        )

        running_total += frame_score
        running_score = running_total / k
        total_link_observations += active_links
        cumulative_link_distance += total_link_distance
        maximum_active_links = max(maximum_active_links, active_links)

        rows.append(
            SnapshotScore(
                frame=int(snapshot.frame),
                time=float(snapshot.time),
                active_links=int(active_links),
                max_possible_links=int(max_possible_links),
                link_fraction=(active_links / max_possible_links if max_possible_links else 0.0),
                total_link_distance=float(total_link_distance),
                mean_link_distance=float(mean_link_distance),
                normalized_distance_sum=float(normalized_distance_sum),
                link_score=float(link_score),
                receiver_count=int(receiver_count),
                covered_receivers=covered_count,
                coverage_fraction=float(coverage_fraction),
                mean_nearest_receiver_distance=mean_nearest,
                max_nearest_receiver_distance=max_nearest,
                receiver_penalty=receiver_penalty,
                worst_receiver_penalty=worst_receiver_penalty,
                blended_receiver_penalty=blended_penalty,
                base_score=base_score,
                score=frame_score,
                running_score=float(running_score),
            )
        )

    link_scores = np.array([r.link_score for r in rows], dtype=np.float64)
    receiver_penalties = np.array([r.receiver_penalty for r in rows], dtype=np.float64)
    worst_receiver_penalties = np.array([r.worst_receiver_penalty for r in rows], dtype=np.float64)
    coverage = np.array([r.coverage_fraction for r in rows], dtype=np.float64)
    active_links = np.array([r.active_links for r in rows], dtype=np.float64)
    total_distances = np.array([r.total_link_distance for r in rows], dtype=np.float64)

    average_link_score = float(np.mean(link_scores))
    average_receiver_penalty = float(np.mean(receiver_penalties))
    worst_receiver_penalty = float(np.max(worst_receiver_penalties))
    blended_penalty = float(
        blended_receiver_penalty(
            average_receiver_penalty,
            worst_receiver_penalty,
            worst_distance_weight,
        )
    )
    quality_score = float(network_quality(average_link_score, blended_penalty))
    worst_coverage = float(np.min(coverage))
    full_coverage = bool(worst_coverage >= 1.0 - float(coverage_tolerance))
    final_score = float(
        coverage_first_fitness(
            quality_score,
            worst_coverage,
            receiver_count,
            coverage_tolerance,
        )
    )

    finite_nearest = [
        r.mean_nearest_receiver_distance
        for r in rows
        if np.isfinite(r.mean_nearest_receiver_distance)
    ]
    # Any uncovered point at any frame makes the worst receiver distance infinite.
    if full_coverage:
        worst_nearest = max(r.max_nearest_receiver_distance for r in rows)
    else:
        worst_nearest = float("inf")

    return NetworkScore(
        score=final_score,
        quality_score=quality_score,
        frame_scores=tuple(rows),
        average_link_score=average_link_score,
        average_receiver_penalty=average_receiver_penalty,
        worst_receiver_penalty=worst_receiver_penalty,
        blended_receiver_penalty=blended_penalty,
        average_coverage=float(np.mean(coverage)),
        worst_coverage=worst_coverage,
        full_coverage=full_coverage,
        average_nearest_receiver_distance=(
            float(np.mean(finite_nearest)) if finite_nearest else float("inf")
        ),
        worst_nearest_receiver_distance=float(worst_nearest),
        average_active_links=float(np.mean(active_links)),
        maximum_active_links=int(np.max(active_links)),
        average_link_distance=(
            cumulative_link_distance / total_link_observations
            if total_link_observations
            else 0.0
        ),
        average_total_link_distance=float(np.mean(total_distances)),
        cumulative_link_distance=float(cumulative_link_distance),
        total_link_observations=int(total_link_observations),
        max_possible_links=int(max_possible_links),
        receiver_count=int(receiver_count),
        ground_distance_scale=float(ground_distance_scale),
        worst_distance_weight=float(worst_distance_weight),
        coverage_tolerance=float(coverage_tolerance),
        min_elevation_deg=float(min_elevation_deg),
    )


def _distance_text(value: float) -> str:
    return f"{value:.2f} km" if np.isfinite(value) else "uncovered"


def print_score_summary(results: NetworkScore, satellite_count: int) -> None:
    print()
    print("=" * 76)
    print(" COVERAGE-FIRST CONSTELLATION SCORE")
    print("=" * 76)
    print(f"Satellites                         : {satellite_count}")
    print(f"Virtual Earth receivers            : {results.receiver_count}")
    print(f"Simulation snapshots               : {len(results.frame_scores)}")
    print(f"Maximum possible sat links         : {results.max_possible_links}")
    print("-" * 76)
    print(f"FINAL FITNESS                      : {results.score:.6f} / 1.000000")
    print(f"Coverage status                    : {'FULL' if results.full_coverage else 'INCOMPLETE'}")
    print(f"Worst-frame Earth coverage         : {100*results.worst_coverage:.2f}%")
    print(f"Average Earth coverage             : {100*results.average_coverage:.2f}%")
    print("-" * 76)
    print(f"Quality score (after coverage)     : {results.quality_score:.6f}")
    print(f"Average satellite link score       : {results.average_link_score:.6f}")
    print(f"Mean receiver penalty              : {results.average_receiver_penalty:.6f}")
    print(f"Worst receiver penalty             : {results.worst_receiver_penalty:.6f}")
    print(f"Blended receiver penalty           : {results.blended_receiver_penalty:.6f}")
    print(f"Mean/worst receiver weight         : {1-results.worst_distance_weight:.2f} / {results.worst_distance_weight:.2f}")
    print(f"Avg nearest visible satellite      : {_distance_text(results.average_nearest_receiver_distance)}")
    print(f"Worst sampled receiver distance    : {_distance_text(results.worst_nearest_receiver_distance)}")
    print("-" * 76)
    print(f"Average active links               : {results.average_active_links:.2f}")
    print(f"Maximum active links               : {results.maximum_active_links}")
    print(f"Average valid-link distance        : {results.average_link_distance:.2f} km")
    print(f"Avg total sat-link dist / frame    : {results.average_total_link_distance:.2f} km")
    print(f"Receiver distance scale            : {results.ground_distance_scale:.1f} km")
    print(f"Minimum elevation                  : {results.min_elevation_deg:.1f} deg")
    print("=" * 76)
    print()


def print_frame_samples(results: NetworkScore, every_frames: int = 60) -> None:
    every_frames = max(1, int(every_frames))
    print(
        f"{'Frame':>7} {'Sim time':>11} {'Links':>7} {'Link':>8} "
        f"{'Cover':>8} {'Near km':>10} {'WorstP':>8} {'Frame':>8}"
    )
    print("-" * 84)

    def print_row(row):
        near = row.mean_nearest_receiver_distance
        near_text = f"{near:.0f}" if np.isfinite(near) else "inf"
        print(
            f"{row.frame:7d} {row.time:9.1f}s {row.active_links:7d} "
            f"{row.link_score:8.4f} {100*row.coverage_fraction:7.1f}% "
            f"{near_text:>10} {row.worst_receiver_penalty:8.4f} {row.score:8.5f}"
        )

    for row in results.frame_scores[::every_frames]:
        print_row(row)
    if results.frame_scores[-1].frame % every_frames != 0:
        print_row(results.frame_scores[-1])
    print()


def write_score_csv(results: NetworkScore, output) -> Path:
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [field.name for field in SnapshotScore.__dataclass_fields__.values()]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results.frame_scores:
            writer.writerow({name: getattr(row, name) for name in fields})
    return output
