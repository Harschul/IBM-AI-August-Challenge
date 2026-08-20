"""Shared normalized fitness functions for reference and vectorized scoring.

The optimizer follows a coverage-first, two-stage objective:

1. While *any sampled frame* has incomplete Earth coverage, the score is driven
   primarily by the worst-frame coverage fraction.  A tiny quality tie-break is
   allowed, but it is too small to let a lower-coverage candidate outrank a
   candidate that covers even one additional receiver in its worst frame.

2. Once worst-frame coverage is complete, the candidate enters the upper half
   of the [0, 1] fitness range and is ranked by network quality.

Network quality keeps the original "link distance minus receiver distance"
idea, but receiver distance combines both the mean and the worst sampled
receiver penalty so a single poorly served location cannot disappear inside a
strong average.
"""

from __future__ import annotations

import numpy as np


def blended_receiver_penalty(mean_penalty, worst_penalty, worst_distance_weight=0.5):
    """Blend mean and worst receiver-distance penalties in [0, 1]."""
    weight = float(worst_distance_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("worst_distance_weight must be between 0 and 1")
    return (1.0 - weight) * np.asarray(mean_penalty) + weight * np.asarray(worst_penalty)


def network_quality(link_score, receiver_penalty):
    """Return normalized link-minus-ground quality in [0, 1].

    ``link_score`` is larger-is-better. ``receiver_penalty`` is
    smaller-is-better. Both are expected in [0, 1].
    """
    value = (1.0 + np.asarray(link_score) - np.asarray(receiver_penalty)) / 2.0
    return np.clip(value, 0.0, 1.0)


def coverage_first_fitness(
    quality,
    worst_coverage,
    receiver_count: int,
    coverage_tolerance: float = 1e-12,
):
    """Map coverage + quality to a strict coverage-first fitness in [0, 1].

    Incomplete candidates live below 0.5::

        0.5 * worst_coverage + tiny_quality_tiebreak

    Full-coverage candidates live at or above 0.5::

        0.5 + 0.5 * quality

    The tie-break is ``0.25 / receiver_count * quality``.  Because one receiver
    changes coverage by ``1 / receiver_count``, one additional receiver in the
    worst frame is always worth more than the entire quality tie-break.
    """
    receiver_count = int(receiver_count)
    if receiver_count < 1:
        raise ValueError("receiver_count must be at least 1")
    coverage_tolerance = float(coverage_tolerance)
    if coverage_tolerance < 0:
        raise ValueError("coverage_tolerance cannot be negative")

    q = np.clip(np.asarray(quality, dtype=np.float64), 0.0, 1.0)
    c = np.clip(np.asarray(worst_coverage, dtype=np.float64), 0.0, 1.0)

    full = c >= (1.0 - coverage_tolerance)
    tie_break_weight = 0.25 / receiver_count
    incomplete_score = 0.5 * c + tie_break_weight * q
    complete_score = 0.5 + 0.5 * q
    result = np.where(full, complete_score, incomplete_score)
    return np.clip(result, 0.0, 1.0)
