#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python benchmark_temporal_vs_rl_stochastic.py \
  --model RL/rl_env_v0/models/physical_multisource_stochastic_smoke.zip \
  --num-seeds 3 \
  --seed-base 910000000 \
  --stochastic-seed-base 710000000 \
  --bundles 100 \
  --out benchmark_results/stochastic_smoke
