#!/usr/bin/env python3
"""Run orbital physics -> temporal contacts -> baseline/RL on identical traffic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

from src.integration.config import load_config
from src.integration.contact_plan import build_contact_plan
from src.integration.rl_bridge import MaskablePPOPolicy
from src.integration.scenario import simulate_snapshots
from src.integration.simulation import IntegratedSimulator, aggregate_results
from src.integration.traffic import generate_bundles


def _clean_number(value):
    return None if isinstance(value, float) and math.isnan(value) else value


def write_results(out_dir: Path, rows, summary: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "integrated_bundle_results.csv"
    json_path = out_dir / "integrated_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "bundle_id", "policy", "delivered", "on_time", "arrival_s",
                "latency_s", "hops", "path", "fallbacks", "reason",
                "science_priority",
            ],
        )
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["path"] = "-".join(str(n) for n in row.path)
            writer.writerow(payload)

    json_path.write_text(
        json.dumps(
            {
                key: {k: _clean_number(v) for k, v in metrics.items()}
                for key, metrics in summary.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return csv_path, json_path


def print_summary(name: str, metrics: dict):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"delivery ratio             {metrics['delivery_ratio']:.3f}")
    print(f"deadline success           {metrics['deadline_success']:.3f}")
    print(f"priority-weighted timely   {metrics['priority_weighted_timely']:.3f}")
    if not math.isnan(metrics["mean_latency_s"]):
        print(f"mean latency               {metrics['mean_latency_s']:.1f} s")
    print(f"mean hops                  {metrics['mean_hops']:.2f}")
    if name.startswith("RL"):
        print(f"fallbacks / executed hops  {metrics['fallback_rate']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/prototype.yaml")
    parser.add_argument("--bundles", type=int, default=100)
    parser.add_argument("--traffic-seed", type=int, default=20260830)
    parser.add_argument(
        "--model",
        default="RL/rl_env_v0/models/rl_agent_seed_42.zip",
        help="MaskablePPO checkpoint. Use --no-rl to skip model loading.",
    )
    parser.add_argument("--no-rl", action="store_true")
    parser.add_argument("--out", default="integrated_results")
    args = parser.parse_args()

    config = load_config(args.config)
    satellites, snapshots = simulate_snapshots(config)
    plan, diagnostics = build_contact_plan(snapshots, config)
    bundles = generate_bundles(config, count=args.bundles, seed=args.traffic_seed)

    print("Physical contact plan")
    print("---------------------")
    print(f"contacts                  {diagnostics.contacts}")
    print(f"satellite contacts        {diagnostics.satellite_contacts}")
    print(f"ground contacts           {diagnostics.ground_contacts}")
    print(f"direct science->ground    {diagnostics.direct_to_ground_contacts}")
    print(f"horizon                   {diagnostics.horizon_s:.1f} s")

    simulator = IntegratedSimulator(plan, config)
    baseline_rows = simulator.run_baseline(bundles)
    summary = {"baseline": aggregate_results(baseline_rows)}
    all_rows = list(baseline_rows)
    print_summary("Baseline", summary["baseline"])

    if not args.no_rl:
        policy = MaskablePPOPolicy(args.model)
        rl_rows = simulator.run_rl_with_fallback(bundles, policy)
        summary["rl_fallback"] = aggregate_results(rl_rows)
        all_rows.extend(rl_rows)
        print_summary("RL + deterministic fallback", summary["rl_fallback"])

    csv_path, json_path = write_results(Path(args.out), all_rows, summary)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
