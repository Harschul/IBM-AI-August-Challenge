#!/usr/bin/env python3
"""CLI replay for one seed from the locked final stochastic experiment.

Defaults to pure PPO using the newly retrained checkpoint. Use --algorithm
temporal for the paired baseline or --algorithm rl_with_temporal_fallback only
for an operational safety-mode demonstration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.experiment.runner import aggregate, build_world, load_final_policy, run_algorithm
from src.experiment.spec import load_final_spec, repo_path
from src.integration.traffic import generate_bundles


def main() -> None:
    spec = load_final_spec()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["rl_pure", "temporal", "rl_with_temporal_fallback"], default="rl_pure")
    parser.add_argument("--seed-offset", type=int, default=spec.demo.default_seed_offset)
    parser.add_argument("--out", default="artifacts/final_experiment/demo/cli")
    args = parser.parse_args()

    world = build_world(spec)
    traffic_seed, stochastic_seed = spec.benchmark.seeds(args.seed_offset)
    bundles = generate_bundles(world.config, count=spec.benchmark.bundles_per_seed, seed=traffic_seed)
    policy = load_final_policy(spec) if args.algorithm.startswith("rl") else None
    rows = run_algorithm(
        algorithm=args.algorithm,
        plan=world.plan,
        config=world.config,
        bundles=bundles,
        policy=policy,
        stochastic_seed=stochastic_seed,
        max_hops=spec.benchmark.max_hops,
        max_attempts=spec.benchmark.max_attempts,
    )
    metrics = aggregate(rows)
    out = repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": spec.name,
        "algorithm": args.algorithm,
        "reported_experiment": args.algorithm in spec.demo.reported_modes,
        "seed_offset": args.seed_offset,
        "traffic_seed": traffic_seed,
        "stochastic_seed": stochastic_seed,
        "bundles": spec.benchmark.bundles_per_seed,
        "model": str(spec.model) if policy is not None else None,
        "config_sha256": spec.scenario_config_sha256,
        "metrics": metrics,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "bundle_results.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["bundle_id", "source_id", "science_priority", "delivered", "on_time", "arrival_s", "latency_s", "hops", "attempts", "transfer_failures", "wasted_capacity_bytes", "fallbacks", "path", "reason"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
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
                "wasted_capacity_bytes": row.wasted_capacity_bytes,
                "fallbacks": row.fallbacks,
                "path": "-".join(map(str, row.path)),
                "reason": row.reason,
            })
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
