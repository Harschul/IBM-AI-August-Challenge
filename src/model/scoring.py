"""Normalized scoring and diagnostics for a satellite network simulation.

The score deliberately rewards both goals at once:

* more simultaneously valid links;
* longer valid links.

For every possible satellite pair, a disconnected pair contributes 0.  A
connected pair contributes ``distance / pair_connection_range``.  Therefore a
single pair is always worth between 0 and 1, and the score for a snapshot is
the mean contribution over *all possible pairs*.  The final network score is
the mean snapshot score over the complete simulation, so it is also in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SnapshotScore:
    """Scoring diagnostics for one immutable NetworkSnapshot."""

    frame: int
    time: float
    active_links: int
    max_possible_links: int
    link_fraction: float
    total_link_distance: float
    mean_link_distance: float
    normalized_distance_sum: float
    score: float
    running_score: float


@dataclass(frozen=True)
class NetworkScore:
    """Complete normalized score plus useful simulation diagnostics."""

    score: float
    frame_scores: tuple[SnapshotScore, ...]
    average_active_links: float
    maximum_active_links: int
    average_link_distance: float
    average_total_link_distance: float
    cumulative_link_distance: float
    total_link_observations: int
    max_possible_links: int


def _pair_connection_range(satellites, i: int, j: int) -> float:
    """Return the same symmetric range limit used by Satellite.can_connect_to()."""
    return min(
        float(satellites[i].connection_range()),
        float(satellites[j].connection_range()),
    )


def score_snapshot(snapshot, satellites) -> tuple[float, int, float, float, float]:
    """Score one snapshot.

    Returns
    -------
    score:
        Normalized score in [0, 1].
    active_links:
        Number of currently valid links.
    total_link_distance:
        Sum of physical distances of all currently valid links, in km.
    mean_link_distance:
        Mean physical distance of currently valid links, in km.
    normalized_distance_sum:
        Sum of each active link's ``distance / pair_connection_range``.

    Notes
    -----
    A disconnected pair contributes zero automatically because it does not
    appear in ``snapshot.connection_indices``.
    """
    satellite_count = len(satellites)
    max_possible_links = satellite_count * (satellite_count - 1) // 2

    if max_possible_links == 0:
        return 0.0, 0, 0.0, 0.0, 0.0

    active_links = len(snapshot.connection_indices)
    if active_links == 0:
        return 0.0, 0, 0.0, 0.0, 0.0

    active = np.asarray(snapshot.connection_indices, dtype=np.int32)
    i_idx = active[:, 0]
    j_idx = active[:, 1]

    deltas = snapshot.positions[i_idx] - snapshot.positions[j_idx]
    distances = np.linalg.norm(deltas, axis=1)

    pair_ranges = np.asarray(
        [_pair_connection_range(satellites, int(i), int(j)) for i, j in active],
        dtype=float,
    )

    # A zero-range pair can only connect at zero distance.  Define its normalized
    # contribution as zero rather than dividing by zero.
    normalized = np.zeros_like(distances, dtype=float)
    valid_range = pair_ranges > 0.0
    normalized[valid_range] = distances[valid_range] / pair_ranges[valid_range]

    # Connectivity already enforces distance <= range; clip only to protect the
    # advertised [0, 1] score from tiny floating-point overshoots.
    normalized = np.clip(normalized, 0.0, 1.0)

    normalized_distance_sum = float(np.sum(normalized))
    total_link_distance = float(np.sum(distances))
    mean_link_distance = float(np.mean(distances))
    score = normalized_distance_sum / max_possible_links

    return (
        float(np.clip(score, 0.0, 1.0)),
        active_links,
        total_link_distance,
        mean_link_distance,
        normalized_distance_sum,
    )


def score_network(snapshots, satellites) -> NetworkScore:
    """Score an entire simulation and return per-frame diagnostics.

    The final score is the arithmetic mean of every snapshot score.  It remains
    between 0 and 1 and is suitable for direct use as a fitness value.
    """
    snapshots = tuple(snapshots)
    satellites = tuple(satellites)

    if not snapshots:
        raise ValueError("snapshots cannot be empty")
    if not satellites:
        raise ValueError("satellites cannot be empty")

    max_possible_links = len(satellites) * (len(satellites) - 1) // 2

    frame_scores: list[SnapshotScore] = []
    score_sum = 0.0
    total_link_observations = 0
    cumulative_link_distance = 0.0
    maximum_active_links = 0

    for count, snapshot in enumerate(snapshots, start=1):
        (
            frame_score,
            active_links,
            total_link_distance,
            mean_link_distance,
            normalized_distance_sum,
        ) = score_snapshot(snapshot, satellites)

        score_sum += frame_score
        running_score = score_sum / count

        total_link_observations += active_links
        cumulative_link_distance += total_link_distance
        maximum_active_links = max(maximum_active_links, active_links)

        link_fraction = (
            active_links / max_possible_links if max_possible_links else 0.0
        )

        frame_scores.append(
            SnapshotScore(
                frame=int(snapshot.frame),
                time=float(snapshot.time),
                active_links=int(active_links),
                max_possible_links=int(max_possible_links),
                link_fraction=float(link_fraction),
                total_link_distance=float(total_link_distance),
                mean_link_distance=float(mean_link_distance),
                normalized_distance_sum=float(normalized_distance_sum),
                score=float(frame_score),
                running_score=float(running_score),
            )
        )

    final_score = score_sum / len(frame_scores)
    average_active_links = total_link_observations / len(frame_scores)
    average_total_link_distance = cumulative_link_distance / len(frame_scores)

    # This is weighted by actual link observations rather than by frames, so
    # frames with no links do not artificially drag the physical link length down.
    average_link_distance = (
        cumulative_link_distance / total_link_observations
        if total_link_observations
        else 0.0
    )

    return NetworkScore(
        score=float(np.clip(final_score, 0.0, 1.0)),
        frame_scores=tuple(frame_scores),
        average_active_links=float(average_active_links),
        maximum_active_links=int(maximum_active_links),
        average_link_distance=float(average_link_distance),
        average_total_link_distance=float(average_total_link_distance),
        cumulative_link_distance=float(cumulative_link_distance),
        total_link_observations=int(total_link_observations),
        max_possible_links=int(max_possible_links),
    )


def print_score_summary(results: NetworkScore, satellite_count: int) -> None:
    """Print a compact, human-readable summary of a constellation run."""
    print()
    print("=" * 64)
    print(" CONSTELLATION SCORE")
    print("=" * 64)
    print(f"Satellites                  : {satellite_count}")
    print(f"Maximum possible links      : {results.max_possible_links}")
    print(f"Simulation snapshots        : {len(results.frame_scores)}")
    print("-" * 64)
    print(f"Final normalized score      : {results.score:.6f} / 1.000000")
    print(f"Average active links        : {results.average_active_links:.2f}")
    print(f"Maximum active links        : {results.maximum_active_links}")
    print(f"Average valid-link distance : {results.average_link_distance:.2f} km")
    print(f"Avg total distance / frame  : {results.average_total_link_distance:.2f} km")
    print(f"Cumulative sampled distance : {results.cumulative_link_distance:.2f} km")
    print("=" * 64)
    print()


def print_frame_samples(results: NetworkScore, every_frames: int = 60) -> None:
    """Print sampled snapshot metrics without flooding the console."""
    every_frames = max(1, int(every_frames))

    print(
        f"{'Frame':>7} {'Sim time':>12} {'Links':>8} "
        f"{'Mean dist':>12} {'Total dist':>13} {'Score':>9} {'Running':>9}"
    )
    print("-" * 84)

    for index in range(0, len(results.frame_scores), every_frames):
        row = results.frame_scores[index]
        print(
            f"{row.frame:7d} {row.time:10.1f}s {row.active_links:8d} "
            f"{row.mean_link_distance:10.1f}km "
            f"{row.total_link_distance:11.1f}km "
            f"{row.score:9.4f} {row.running_score:9.4f}"
        )

    # Always show the last frame if it was not already printed.
    if results.frame_scores and (len(results.frame_scores) - 1) % every_frames:
        row = results.frame_scores[-1]
        print(
            f"{row.frame:7d} {row.time:10.1f}s {row.active_links:8d} "
            f"{row.mean_link_distance:10.1f}km "
            f"{row.total_link_distance:11.1f}km "
            f"{row.score:9.4f} {row.running_score:9.4f}"
        )
    print()


def write_score_csv(results: NetworkScore, output) -> Path:
    """Write one diagnostics row per simulation snapshot."""
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "simulation_time_s",
                "active_links",
                "max_possible_links",
                "link_fraction",
                "mean_link_distance_km",
                "total_link_distance_km",
                "normalized_distance_sum",
                "snapshot_score",
                "running_score",
            ]
        )

        for row in results.frame_scores:
            writer.writerow(
                [
                    row.frame,
                    f"{row.time:.9f}",
                    row.active_links,
                    row.max_possible_links,
                    f"{row.link_fraction:.9f}",
                    f"{row.mean_link_distance:.9f}",
                    f"{row.total_link_distance:.9f}",
                    f"{row.normalized_distance_sum:.9f}",
                    f"{row.score:.9f}",
                    f"{row.running_score:.9f}",
                ]
            )

    return output
