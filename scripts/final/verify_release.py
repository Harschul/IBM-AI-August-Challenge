#!/usr/bin/env python3
"""Verify that demo, benchmark and checkpoint all refer to one experiment."""

from __future__ import annotations

import json

from src.experiment.spec import load_final_spec
from src.experiment.runner import build_world


def main() -> None:
    spec = load_final_spec()
    print(f"experiment : {spec.name}")
    print(f"config     : {spec.scenario_config}")
    print(f"config SHA : {spec.verify_config()}")
    print(f"model      : {spec.model}")
    metadata = spec.verify_model(require_exists=True)
    print(f"model cfg  : {metadata.get('config_sha256')}")
    world = build_world(spec)
    print(f"contacts   : {world.diagnostics.contacts}")
    print(f"seeds      : {spec.benchmark.num_seeds}")
    print(f"bundles    : {spec.benchmark.bundles_per_seed} per seed")
    print("algorithms : temporal vs rl_pure")
    summary = spec.benchmark.output_dir / "summary.json"
    if summary.exists():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if payload.get("config_sha256") != spec.scenario_config_sha256:
            raise SystemExit("benchmark summary config SHA does not match final experiment")
        if str(payload.get("model")) != str(spec.model):
            raise SystemExit("benchmark summary model path does not match final experiment")
        print(f"benchmark  : verified {summary}")
    else:
        print("benchmark  : not generated yet")
    print("\nPASS: final release paths are coherent.")


if __name__ == "__main__":
    main()
