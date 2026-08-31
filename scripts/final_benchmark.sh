#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python benchmark_temporal_vs_rl_stochastic.py \
  --model RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip \
  --num-seeds 20 \
  --seed-base 900000000 \
  --stochastic-seed-base 700000000 \
  --bundles 500 \
  --out benchmark_results/final_stochastic
