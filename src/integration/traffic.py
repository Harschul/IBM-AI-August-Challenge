"""Deterministic science-bundle generation for fair policy comparison."""

from __future__ import annotations

import random

from src.models.bundle import DataBundle

from .config import SCIENCE_IDS, PrototypeConfig


def generate_bundles(
    config: PrototypeConfig,
    count: int = 100,
    seed: int | None = None,
) -> list[DataBundle]:
    """Generate bundles and randomly distribute their origins across SCIENCE_IDS.

    A fixed seed still gives identical source selection and traffic to every
    policy, so baseline/RL comparisons remain paired and reproducible.
    """
    if count < 1:
        raise ValueError("count must be positive")

    traffic = config.raw.get("traffic", {})
    rng = random.Random(config.seed + 100 if seed is None else seed)
    creation_horizon = float(traffic.get("creation_horizon_s", config.horizon_s * 0.55))
    urgent_fraction = float(traffic.get("urgent_fraction", 0.30))
    min_size_mb = float(traffic.get("min_size_mb", 10.0))
    max_size_mb = float(traffic.get("max_size_mb", 120.0))

    bundles: list[DataBundle] = []
    for index in range(count):
        source_id = int(rng.choice(SCIENCE_IDS))
        created = rng.uniform(0.0, creation_horizon)
        urgent = rng.random() < urgent_fraction
        if urgent:
            priority = rng.uniform(0.75, 1.0)
            ttl = rng.uniform(180.0, 480.0)
            data_type = "TRANSIENT"
        else:
            priority = rng.uniform(0.1, 0.7)
            ttl = rng.uniform(600.0, max(601.0, config.horizon_s - created))
            data_type = rng.choice(["STAR_FIELD", "CALIBRATION", "HOUSEKEEPING"])

        size_bytes = int(rng.uniform(min_size_mb, max_size_mb) * 1_000_000)
        bundles.append(
            DataBundle(
                bundle_id=f"B{index:04d}",
                source_id=source_id,
                size_bytes=size_bytes,
                created_s=created,
                science_priority=priority,
                deadline_s=min(config.horizon_s, created + ttl),
                data_type=data_type,
            )
        )

    bundles.sort(key=lambda b: (b.created_s, b.bundle_id))
    return bundles
