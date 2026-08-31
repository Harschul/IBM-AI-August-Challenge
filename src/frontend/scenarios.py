"""Plain-English summaries of the fixed held-out benchmark traffic runs.

The underlying experiment still uses the exact locked traffic and stochastic
seed pairs. These labels are presentation-only summaries computed from the real
500 generated bundles for each held-out run.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from src.experiment.spec import FinalExperimentSpec
from src.integration.config import PrototypeConfig, SCIENCE_IDS
from src.integration.traffic import generate_bundles


@dataclass(frozen=True)
class ScenarioProfile:
    offset: int
    label: str
    description: str
    traffic_seed: int
    stochastic_seed: int
    urgent_count: int
    total_gb: float
    mean_payload_mb: float
    mean_deadline_window_s: float
    source_counts: tuple[int, ...]


def build_scenario_profiles(spec: FinalExperimentSpec, config: PrototypeConfig) -> tuple[ScenarioProfile, ...]:
    profiles: list[ScenarioProfile] = []
    total = spec.benchmark.bundles_per_seed
    for offset in range(spec.benchmark.num_seeds):
        traffic_seed, stochastic_seed = spec.benchmark.seeds(offset)
        bundles = generate_bundles(config, count=total, seed=traffic_seed)
        urgent_count = sum(bundle.data_type == "TRANSIENT" for bundle in bundles)
        total_gb = sum(bundle.size_bytes for bundle in bundles) / 1e9
        mean_payload_mb = fmean(bundle.size_bytes / 1e6 for bundle in bundles)
        mean_deadline_window_s = fmean((bundle.deadline_s or config.horizon_s) - bundle.created_s for bundle in bundles)
        source_counts = tuple(sum(bundle.source_id == source_id for bundle in bundles) for source_id in SCIENCE_IDS)
        urgent_pct = 100.0 * urgent_count / max(1, total)
        label = (
            f"Run {offset + 1:02d} · {total} packets · {urgent_pct:.0f}% urgent · "
            f"{mean_payload_mb:.0f} MB avg payload · {mean_deadline_window_s:.0f}s avg deadline window"
        )
        origin_parts = [
            f"Research satellite {idx + 1}: {count} packets ({100.0 * count / max(1, total):.0f}%)"
            for idx, count in enumerate(source_counts)
        ]
        description = "Packet origins — " + " · ".join(origin_parts)
        profiles.append(
            ScenarioProfile(
                offset=offset,
                label=label,
                description=description,
                traffic_seed=traffic_seed,
                stochastic_seed=stochastic_seed,
                urgent_count=urgent_count,
                total_gb=total_gb,
                mean_payload_mb=mean_payload_mb,
                mean_deadline_window_s=mean_deadline_window_s,
                source_counts=source_counts,
            )
        )
    return tuple(profiles)
