#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python run_final_benchmark.py --num-seeds 2 --bundles 50 --out artifacts/final_experiment/benchmark_smoke
