#!/usr/bin/env python3
"""Run the locked final Temporal-vs-PPO stochastic benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment.benchmark import run_final_benchmark
from src.experiment.spec import load_final_spec, repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="config/final_experiment.json")
    parser.add_argument("--num-seeds", type=int)
    parser.add_argument("--bundles", type=int)
    parser.add_argument("--out")
    parser.add_argument("--include-operational-fallback", action="store_true")
    args = parser.parse_args()
    spec = load_final_spec(args.spec)
    run_final_benchmark(
        spec,
        output_dir=repo_path(args.out) if args.out else None,
        num_seeds=args.num_seeds,
        bundles_per_seed=args.bundles,
        include_operational_fallback=args.include_operational_fallback,
    )


if __name__ == "__main__":
    main()
