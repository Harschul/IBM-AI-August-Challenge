"""Final paired Temporal-vs-pure-PPO benchmark using the locked experiment spec."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Sequence

from src.experiment.runner import ExperimentResult, aggregate, build_world, bundles_for_seed, load_final_policy, run_algorithm
from src.experiment.spec import FinalExperimentSpec, sha256_file
from src.integration.config import SCIENCE_IDS
from src.integration.stochastic_transfer import StochasticTransferSettings

METRICS = [
    "delivery_ratio",
    "deadline_success",
    "priority_weighted_timely",
    "mean_latency_s",
    "mean_hops",
    "mean_attempts",
    "transfer_failure_rate",
    "mean_failures",
    "mean_wasted_mb",
    "fallback_rate",
]


def ci95(values: list[float]) -> tuple[float, float, float]:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return float("nan"), float("nan"), float("nan")
    mean = fmean(finite)
    if len(finite) == 1:
        return mean, mean, mean
    half = 1.96 * stdev(finite) / math.sqrt(len(finite))
    return mean, mean - half, mean + half


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_final_benchmark(
    spec: FinalExperimentSpec,
    *,
    output_dir: Path | None = None,
    num_seeds: int | None = None,
    bundles_per_seed: int | None = None,
    include_operational_fallback: bool = False,
) -> dict[str, object]:
    spec.verify_config()
    model_metadata = spec.verify_model(require_exists=True)
    world = build_world(spec)
    policy = load_final_policy(spec)
    settings = StochasticTransferSettings.from_config(world.config)

    seed_count = spec.benchmark.num_seeds if num_seeds is None else int(num_seeds)
    if seed_count < 1 or seed_count > spec.benchmark.num_seeds:
        raise ValueError(f"num_seeds must be 1..{spec.benchmark.num_seeds}")
    bundle_count = spec.benchmark.bundles_per_seed if bundles_per_seed is None else int(bundles_per_seed)
    if bundle_count < 1:
        raise ValueError("bundles_per_seed must be positive")

    algorithms = ["temporal", "rl_pure"]
    if include_operational_fallback:
        algorithms.append("rl_with_temporal_fallback")

    out = output_dir or spec.benchmark.output_dir
    out.mkdir(parents=True, exist_ok=True)

    per_seed: dict[str, list[dict[str, float]]] = defaultdict(list)
    source_metrics: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    bundle_rows: list[dict[str, object]] = []
    attempt_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []

    print("Final stochastic physical benchmark")
    print("-----------------------------------")
    print(f"experiment          {spec.name}")
    print(f"config SHA          {sha256_file(spec.scenario_config)}")
    print(f"model               {spec.model}")
    print(f"contacts            {world.diagnostics.contacts}")
    print(f"science IDs         {SCIENCE_IDS}")
    print(f"stochastic enabled  {settings.enabled}")
    print("primary comparison  temporal vs pure PPO (zero fallback)")

    for offset in range(seed_count):
        traffic_seed, stochastic_seed = spec.benchmark.seeds(offset)
        # Keep the exact same generator/seed pair for all algorithms.
        from src.integration.traffic import generate_bundles
        bundles = generate_bundles(world.config, count=bundle_count, seed=traffic_seed)

        result_sets = {
            algorithm: run_algorithm(
                algorithm=algorithm,
                plan=world.plan,
                config=world.config,
                bundles=bundles,
                policy=policy if algorithm.startswith("rl") else None,
                stochastic_seed=stochastic_seed,
                max_hops=spec.benchmark.max_hops,
                max_attempts=spec.benchmark.max_attempts,
            )
            for algorithm in algorithms
        }

        for algorithm, rows in result_sets.items():
            metrics = aggregate(rows)
            per_seed[algorithm].append(metrics)
            seed_rows.append({
                "seed_offset": offset,
                "traffic_seed": traffic_seed,
                "stochastic_seed": stochastic_seed,
                "algorithm": algorithm,
                **metrics,
            })
            for source_id in SCIENCE_IDS:
                source_metrics[algorithm][source_id].append(
                    aggregate([row for row in rows if row.source_id == source_id])
                )
            for row in rows:
                bundle_rows.append({
                    "seed_offset": offset,
                    "traffic_seed": traffic_seed,
                    "stochastic_seed": stochastic_seed,
                    "algorithm": algorithm,
                    "bundle_id": row.bundle_id,
                    "source_id": row.source_id,
                    "science_priority": row.science_priority,
                    "delivered": row.delivered,
                    "on_time": row.on_time,
                    "arrival_s": row.arrival_s,
                    "latency_s": row.latency_s,
                    "hops": row.hops,
                    "attempts": row.attempts,
                    "transfer_failures": row.transfer_failures,
                    "retries": row.retries,
                    "wasted_capacity_bytes": row.wasted_capacity_bytes,
                    "fallbacks": row.fallbacks,
                    "actual_algorithms": "|".join(row.actual_algorithms),
                    "path": "-".join(str(node) for node in row.path),
                    "reason": row.reason,
                })
                for attempt in row.attempt_trace:
                    attempt_rows.append({
                        "seed_offset": offset,
                        "traffic_seed": traffic_seed,
                        "stochastic_seed": stochastic_seed,
                        "algorithm": algorithm,
                        "bundle_id": row.bundle_id,
                        "source_id": row.source_id,
                        "science_priority": row.science_priority,
                        "attempt_index": attempt.attempt_index,
                        "holder_id": attempt.holder_id,
                        "destination_id": attempt.destination_id,
                        "requested_algorithm": attempt.requested_algorithm,
                        "actual_algorithm": attempt.actual_algorithm,
                        "fallback_used": attempt.fallback_used,
                        "fallback_reason": attempt.fallback_reason,
                        "contact_start_s": attempt.contact_start_s,
                        "contact_end_s": attempt.contact_end_s,
                        "failure_probability": attempt.failure_probability,
                        "success_draw": attempt.success_draw,
                        "success": attempt.success,
                        "transfer_progress": attempt.transfer_progress,
                        "capacity_bytes_consumed": attempt.capacity_bytes_consumed,
                        "depart_s": attempt.depart_s,
                        "event_time_s": attempt.event_time_s,
                        "arrival_s": attempt.arrival_s,
                    })

        print(
            f"seed {offset:02d}: temporal timely={per_seed['temporal'][-1]['deadline_success']:.3f} "
            f"rl={per_seed['rl_pure'][-1]['deadline_success']:.3f}"
        )

    algorithms_summary: dict[str, object] = {}
    for algorithm in algorithms:
        algorithms_summary[algorithm] = {
            metric: dict(zip(("mean", "ci95_low", "ci95_high"), ci95([row[metric] for row in per_seed[algorithm]])))
            for metric in METRICS
        }

    per_source_summary: dict[str, object] = {}
    for algorithm in algorithms:
        per_source_summary[algorithm] = {}
        for source_id in SCIENCE_IDS:
            per_source_summary[algorithm][str(source_id)] = {
                metric: dict(zip(("mean", "ci95_low", "ci95_high"), ci95([row[metric] for row in source_metrics[algorithm][source_id]])))
                for metric in METRICS
            }

    paired_delta = {
        metric: dict(
            zip(
                ("mean", "ci95_low", "ci95_high"),
                ci95([
                    rl_row[metric] - temporal_row[metric]
                    for temporal_row, rl_row in zip(per_seed["temporal"], per_seed["rl_pure"])
                ]),
            )
        )
        for metric in METRICS
    }

    summary: dict[str, object] = {
        "experiment": spec.name,
        "scenario_config": str(spec.scenario_config.relative_to(spec.scenario_config.parents[1])),
        "config_sha256": spec.scenario_config_sha256,
        "model": str(spec.model),
        "model_metadata": model_metadata,
        "num_seeds": seed_count,
        "bundles_per_seed": bundle_count,
        "traffic_seed_base": spec.benchmark.traffic_seed_base,
        "stochastic_seed_base": spec.benchmark.stochastic_seed_base,
        "stochastic_settings": settings.__dict__,
        "algorithms": algorithms_summary,
        "per_source": per_source_summary,
        "paired_delta_rl_minus_temporal": paired_delta,
    }

    _write_csv(out / "seed_metrics.csv", seed_rows)
    _write_csv(out / "bundle_results.csv", bundle_rows)
    _write_csv(out / "attempt_log.csv", attempt_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote final benchmark artifacts to {out}")
    return summary
