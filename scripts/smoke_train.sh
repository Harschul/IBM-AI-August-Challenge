#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python train_physical_ppo_stochastic.py \
  --timesteps 100000 \
  --n-envs 2 \
  --bundles-per-episode 24 \
  --max-attempts 24 \
  --seed 42 \
  --out RL/rl_env_v0/models/physical_multisource_stochastic_smoke.zip
