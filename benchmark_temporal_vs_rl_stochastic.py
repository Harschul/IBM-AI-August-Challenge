#!/usr/bin/env python3
"""Paired stochastic benchmark: Temporal vs pure PPO on identical traffic/draws.

The benchmark uses common random numbers. Temporal and PPO each receive an
independent capacity ledger, but their TransferOracle instances share the same
stochastic seed. Therefore matching attempts on the same bundle/contact/ordinal
see exactly the same failure draw.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Sequence

from src.integration.capacity import CapacityLedger
from src.integration.config import GROUND_IDS, SCIENCE_IDS, PrototypeConfig, load_config
from src.integration.contact_plan import build_contact_plan
from src.integration.rl_bridge import MaskablePPOPolicy, best_feasible_contact, build_observation
from src.integration.scenario import simulate_snapshots
from src.integration.simulation import clone_bundle
from src.integration.stochastic_transfer import StochasticTransferSettings, TransferOracle
from src.integration.traffic import generate_bundles
from src.models.bundle import DataBundle
from src.models.contact import ContactPlan
from src.routing.temporal_baseline import earliest_arrival


@dataclass(frozen=True)
class AttemptRecord:
    bundle_id: str
    attempt_index: int
    holder_id: int
    destination_id: int
    actual_algorithm: str
    contact_start_s: float
    contact_end_s: float
    failure_probability: float
    success_draw: float
    success: bool
    transfer_progress: float
    capacity_bytes_consumed: int
    depart_s: float
    event_time_s: float
    arrival_s: float | None


@dataclass(frozen=True)
class StochasticResult:
    bundle_id: str
    source_id: int
    policy: str
    requested_algorithm: str
    actual_algorithms: tuple[str, ...]
    delivered: bool
    on_time: bool
    arrival_s: float | None
    latency_s: float | None
    hops: int
    attempts: int
    transfer_failures: int
    retries: int
    wasted_capacity_bytes: int
    path: tuple[int, ...]
    fallbacks: int
    reason: str
    science_priority: float
    attempt_trace: tuple[AttemptRecord, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_config(model_path: Path, config_path: Path, allow_mismatch: bool) -> None:
    metadata_path = model_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        print(f"WARNING: metadata file not found: {metadata_path}")
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = metadata.get("config_sha256")
    actual = sha256_file(config_path)
    if expected and expected != actual:
        message = (
            "Model/config mismatch: model was trained with config SHA "
            f"{expected}, current config SHA is {actual}."
        )
        if allow_mismatch:
            print("WARNING:", message)
        else:
            raise SystemExit(message + " Retrain or pass --allow-config-mismatch intentionally.")


def _temporal_next_hop(
    ledger: CapacityLedger,
    bundle: DataBundle,
    now_s: float,
) -> int | None:
    plan = ledger.planning_plan()
    route = earliest_arrival(
        plan,
        bundle.current_holder,
        GROUND_IDS,
        bundle.remaining_bytes,
        start_s=now_s,
        deadline_s=bundle.deadline_s,
    )
    if route is None:
        route = earliest_arrival(
            plan,
            bundle.current_holder,
            GROUND_IDS,
            bundle.remaining_bytes,
            start_s=now_s,
            deadline_s=None,
        )
    return None if route is None or not route.hops else int(route.next_hop())


def _rl_next_hop(
    ledger: CapacityLedger,
    config: PrototypeConfig,
    bundle: DataBundle,
    now_s: float,
    policy: MaskablePPOPolicy,
) -> tuple[int | None, str | None]:
    obs, mask = build_observation(ledger.planning_plan(), bundle, now_s, config)
    if not mask.any():
        return None, "no_feasible_action"
    try:
        action = int(policy.choose(obs, mask))
    except Exception as exc:
        return None, f"policy_error:{type(exc).__name__}"
    if action < 0 or action >= len(mask) or not mask[action]:
        return None, "invalid_action"
    return action, None


def run_algorithm(
    *,
    algorithm: str,
    plan: ContactPlan,
    config: PrototypeConfig,
    bundles: Sequence[DataBundle],
    policy: MaskablePPOPolicy | None,
    stochastic_seed: int,
    max_hops: int = 8,
    max_attempts: int = 24,
) -> list[StochasticResult]:
    if algorithm not in {"temporal", "rl_pure", "rl_with_temporal_fallback"}:
        raise ValueError(f"unknown algorithm {algorithm}")
    if algorithm.startswith("rl") and policy is None:
        raise ValueError("RL algorithm requires a policy")

    ledger = CapacityLedger(plan)
    settings = StochasticTransferSettings.from_config(config)
    oracle = TransferOracle(stochastic_seed, config, settings)
    results: list[StochasticResult] = []

    for original in sorted(bundles, key=lambda b: (b.created_s, b.bundle_id)):
        bundle = clone_bundle(original)
        now_s = bundle.created_s
        path = [bundle.source_id]
        actual_algorithms: list[str] = []
        attempt_trace: list[AttemptRecord] = []
        attempts = 0
        failures = 0
        fallbacks = 0
        wasted = 0
        successful_hops = 0
        reason = "no_route"

        while attempts < max_attempts and successful_hops < max_hops:
            if bundle.current_holder in GROUND_IDS:
                reason = "delivered"
                break

            actual_algorithm = algorithm
            fallback_reason = None
            if algorithm == "temporal":
                action = _temporal_next_hop(ledger, bundle, now_s)
                actual_algorithm = "temporal"
                if action is None:
                    reason = "no_route"
                    break
            else:
                action, rl_error = _rl_next_hop(
                    ledger,
                    config,
                    bundle,
                    now_s,
                    policy,  # type: ignore[arg-type]
                )
                actual_algorithm = "rl"
                if action is None and algorithm == "rl_pure":
                    reason = rl_error or "no_feasible_action"
                    break
                if action is None:
                    fallbacks += 1
                    fallback_reason = rl_error or "no_feasible_action"
                    action = _temporal_next_hop(ledger, bundle, now_s)
                    actual_algorithm = "temporal"
                    if action is None:
                        reason = f"fallback_failed:{fallback_reason}"
                        break

            contact = best_feasible_contact(
                ledger.planning_plan(),
                bundle.current_holder,
                int(action),
                now_s,
                bundle.remaining_bytes,
            )
            if contact is None:
                if algorithm == "rl_with_temporal_fallback" and actual_algorithm == "rl":
                    fallbacks += 1
                    fallback_reason = "contact_drift"
                    action = _temporal_next_hop(ledger, bundle, now_s)
                    actual_algorithm = "temporal"
                    if action is not None:
                        contact = best_feasible_contact(
                            ledger.planning_plan(),
                            bundle.current_holder,
                            int(action),
                            now_s,
                            bundle.remaining_bytes,
                        )
                if contact is None:
                    reason = "contact_drift"
                    break

            outcome = oracle.attempt(
                bundle_id=bundle.bundle_id,
                contact=contact,
                size_bytes=bundle.remaining_bytes,
                now_s=now_s,
            )
            ledger.reserve_contact(contact, outcome.capacity_bytes_consumed)
            attempts += 1
            actual_algorithms.append(actual_algorithm)
            attempt_trace.append(
                AttemptRecord(
                    bundle_id=bundle.bundle_id,
                    attempt_index=attempts,
                    holder_id=bundle.current_holder,
                    destination_id=int(action),
                    actual_algorithm=actual_algorithm,
                    contact_start_s=float(contact.start_s),
                    contact_end_s=float(contact.end_s),
                    failure_probability=outcome.failure_probability,
                    success_draw=outcome.success_draw,
                    success=outcome.success,
                    transfer_progress=outcome.transfer_progress,
                    capacity_bytes_consumed=outcome.capacity_bytes_consumed,
                    depart_s=outcome.depart_s,
                    event_time_s=outcome.event_time_s,
                    arrival_s=outcome.arrival_s,
                )
            )

            if not outcome.success:
                failures += 1
                wasted += outcome.wasted_capacity_bytes
                now_s = outcome.event_time_s
                reason = "transmission_failed"

                if bundle.deadline_s is not None and now_s > bundle.deadline_s:
                    reason = "missed_deadline_after_failure"
                    break
                # Replanning on the next iteration is intentional. Capacity
                # already spent on this failed attempt stays consumed.
                continue

            now_s = float(outcome.arrival_s)
            bundle.current_holder = int(action)
            bundle.route_history.append(int(action))
            path.append(int(action))
            successful_hops += 1

            if bundle.current_holder in GROUND_IDS:
                reason = "delivered"
                break
            if bundle.deadline_s is not None and now_s > bundle.deadline_s:
                reason = "missed_deadline"
                break
        else:
            reason = "max_attempts" if attempts >= max_attempts else "max_hops"

        delivered = bundle.current_holder in GROUND_IDS
        on_time = delivered and (bundle.deadline_s is None or now_s <= bundle.deadline_s)
        if delivered:
            reason = "delivered_on_time" if on_time else "delivered_late"

        results.append(
            StochasticResult(
                bundle_id=bundle.bundle_id,
                source_id=bundle.source_id,
                policy=algorithm,
                requested_algorithm=algorithm,
                actual_algorithms=tuple(actual_algorithms),
                delivered=delivered,
                on_time=on_time,
                arrival_s=now_s if delivered else None,
                latency_s=(now_s - bundle.created_s) if delivered else None,
                hops=successful_hops,
                attempts=attempts,
                transfer_failures=failures,
                retries=failures,
                wasted_capacity_bytes=wasted,
                path=tuple(path),
                fallbacks=fallbacks,
                reason=reason,
                science_priority=bundle.science_priority,
                attempt_trace=tuple(attempt_trace),
            )
        )

    return results


def aggregate(rows: Sequence[StochasticResult]) -> dict[str, float]:
    if not rows:
        return {key: float("nan") for key in [
            "delivery_ratio", "deadline_success", "priority_weighted_timely",
            "mean_latency_s", "mean_hops", "mean_attempts",
            "transfer_failure_rate", "mean_failures", "mean_wasted_mb", "fallback_rate",
        ]}

    delivered = [row for row in rows if row.delivered]
    total_priority = sum(row.science_priority for row in rows)
    timely_priority = sum(row.science_priority for row in rows if row.on_time)
    attempts = sum(row.attempts for row in rows)
    failures = sum(row.transfer_failures for row in rows)
    executed_hops = sum(row.hops for row in rows)
    fallbacks = sum(row.fallbacks for row in rows)
    return {
        "delivery_ratio": len(delivered) / len(rows),
        "deadline_success": sum(row.on_time for row in rows) / len(rows),
        "priority_weighted_timely": timely_priority / total_priority if total_priority else 0.0,
        "mean_latency_s": fmean(row.latency_s for row in delivered) if delivered else float("nan"),
        "mean_hops": fmean(row.hops for row in rows),
        "mean_attempts": fmean(row.attempts for row in rows),
        "transfer_failure_rate": failures / attempts if attempts else 0.0,
        "mean_failures": fmean(row.transfer_failures for row in rows),
        "mean_wasted_mb": fmean(row.wasted_capacity_bytes / 1_000_000 for row in rows),
        "fallback_rate": fallbacks / executed_hops if executed_hops else 0.0,
    }


def ci95(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = fmean(values)
    if len(values) == 1:
        return mean, mean, mean
    half = 1.96 * stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/prototype.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=900_000_000, help="traffic seed base")
    parser.add_argument("--stochastic-seed-base", type=int, default=700_000_000)
    parser.add_argument("--bundles", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=24)
    parser.add_argument("--out", default="benchmark_results/stochastic_temporal_vs_rl")
    parser.add_argument("--include-operational-fallback", action="store_true")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    model_path = Path(args.model)
    verify_frozen_config(model_path, config_path, args.allow_config_mismatch)

    config = load_config(config_path)
    settings = StochasticTransferSettings.from_config(config)
    _, snapshots = simulate_snapshots(config)
    plan, diagnostics = build_contact_plan(snapshots, config)
    policy = MaskablePPOPolicy(model_path)

    algorithms = ["temporal", "rl_pure"]
    if args.include_operational_fallback:
        algorithms.append("rl_with_temporal_fallback")

    per_seed: dict[str, list[dict[str, float]]] = defaultdict(list)
    source_metrics: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    bundle_rows: list[dict[str, object]] = []
    attempt_rows: list[dict[str, object]] = []

    print("Stochastic physical benchmark")
    print("-----------------------------")
    print(f"config SHA          {sha256_file(config_path)}")
    print(f"contacts            {diagnostics.contacts}")
    print(f"science IDs         {SCIENCE_IDS}")
    print(f"stochastic enabled  {settings.enabled}")
    print(f"base p_fail         {settings.base_failure_probability:.3f}")
    print("common randomness   same transfer seed per paired algorithm")
    print("primary comparison  temporal vs pure PPO (no fallback)")

    for offset in range(args.num_seeds):
        traffic_seed = args.seed_base + offset
        stochastic_seed = args.stochastic_seed_base + offset
        bundles = generate_bundles(config, count=args.bundles, seed=traffic_seed)

        result_sets = {
            algorithm: run_algorithm(
                algorithm=algorithm,
                plan=plan,
                config=config,
                bundles=bundles,
                policy=policy if algorithm.startswith("rl") else None,
                stochastic_seed=stochastic_seed,
                max_attempts=args.max_attempts,
            )
            for algorithm in algorithms
        }

        for algorithm, rows in result_sets.items():
            metrics = aggregate(rows)
            per_seed[algorithm].append(metrics)
            for source_id in SCIENCE_IDS:
                source_metrics[algorithm][source_id].append(
                    aggregate([row for row in rows if row.source_id == source_id])
                )
            for row in rows:
                bundle_rows.append(
                    {
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
                    }
                )
                for attempt in row.attempt_trace:
                    attempt_rows.append(
                        {
                            "traffic_seed": traffic_seed,
                            "stochastic_seed": stochastic_seed,
                            "algorithm": algorithm,
                            "bundle_id": row.bundle_id,
                            "source_id": row.source_id,
                            "science_priority": row.science_priority,
                            "attempt_index": attempt.attempt_index,
                            "holder_id": attempt.holder_id,
                            "destination_id": attempt.destination_id,
                            "actual_algorithm": attempt.actual_algorithm,
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
                        }
                    )

        print(
            f"traffic={traffic_seed} stochastic={stochastic_seed}: "
            f"temporal timely={per_seed['temporal'][-1]['deadline_success']:.3f} "
            f"rl={per_seed['rl_pure'][-1]['deadline_success']:.3f}"
        )

    metrics_to_report = [
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
    summary: dict[str, object] = {
        "config_sha256": sha256_file(config_path),
        "model": str(model_path),
        "num_seeds": args.num_seeds,
        "bundles_per_seed": args.bundles,
        "traffic_seed_base": args.seed_base,
        "stochastic_seed_base": args.stochastic_seed_base,
        "stochastic_settings": settings.__dict__,
        "algorithms": {},
        "per_source": {},
        "paired_delta_rl_minus_temporal": {},
    }

    for algorithm in algorithms:
        summary["algorithms"][algorithm] = {}
        for metric in metrics_to_report:
            values = [row[metric] for row in per_seed[algorithm] if not math.isnan(row[metric])]
            mean, low, high = ci95(values)
            summary["algorithms"][algorithm][metric] = {
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
            }

        summary["per_source"][algorithm] = {}
        for source_id in SCIENCE_IDS:
            summary["per_source"][algorithm][str(source_id)] = {}
            for metric in metrics_to_report:
                values = [
                    row[metric]
                    for row in source_metrics[algorithm][source_id]
                    if not math.isnan(row[metric])
                ]
                mean, low, high = ci95(values)
                summary["per_source"][algorithm][str(source_id)][metric] = {
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }

    for metric in metrics_to_report:
        deltas = []
        for temporal, rl in zip(per_seed["temporal"], per_seed["rl_pure"]):
            if not math.isnan(temporal[metric]) and not math.isnan(rl[metric]):
                deltas.append(rl[metric] - temporal[metric])
        mean, low, high = ci95(deltas)
        summary["paired_delta_rl_minus_temporal"][metric] = {
            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_csv = out_dir / "bundle_results.csv"
    fieldnames = list(bundle_rows[0].keys()) if bundle_rows else []
    with bundle_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bundle_rows)

    attempt_csv = out_dir / "attempt_log.csv"
    attempt_fields = list(attempt_rows[0].keys()) if attempt_rows else []
    with attempt_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=attempt_fields)
        writer.writeheader()
        writer.writerows(attempt_rows)

    seed_csv = out_dir / "seed_metrics.csv"
    with seed_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["algorithm", "traffic_seed", "stochastic_seed", *metrics_to_report],
        )
        writer.writeheader()
        for algorithm in algorithms:
            for offset, metrics in enumerate(per_seed[algorithm]):
                writer.writerow(
                    {
                        "algorithm": algorithm,
                        "traffic_seed": args.seed_base + offset,
                        "stochastic_seed": args.stochastic_seed_base + offset,
                        **metrics,
                    }
                )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFinal paired stochastic summary")
    print("-------------------------------")
    for algorithm in algorithms:
        metrics = summary["algorithms"][algorithm]
        print(
            f"{algorithm:28s} "
            f"delivery={metrics['delivery_ratio']['mean']:.3f} "
            f"timely={metrics['deadline_success']['mean']:.3f} "
            f"priority={metrics['priority_weighted_timely']['mean']:.3f} "
            f"failure_rate={metrics['transfer_failure_rate']['mean']:.3f}"
        )
    print(f"\nWrote {bundle_csv}")
    print(f"Wrote {attempt_csv}")
    print(f"Wrote {seed_csv}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
