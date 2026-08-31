# Apply the stochastic transfer bundle

Copy this bundle into the root of the repository **after** the v2 multisource
integration/frontend code and physical PPO retraining code.

```bash
cp -a /path/to/ppo_stochastic_transfer_bundle/. \
  /home/harshul/IBM-AI-August-Challenge/
```

This intentionally replaces:

```text
config/prototype.yaml
src/integration/physical_rl_env.py
```

and adds:

```text
src/integration/stochastic_transfer.py
train_physical_ppo_stochastic.py
benchmark_temporal_vs_rl_stochastic.py
requirements-stochastic-training.txt
tests/test_stochastic_transfer.py
tests/test_stochastic_benchmark.py
```

The changed `prototype.yaml` means its SHA-256 changes. That is intentional:
PPO should be retrained after stochastic transfer semantics are enabled.

## Install

```bash
cd /home/harshul/IBM-AI-August-Challenge
source .venv-ppo/bin/activate   # if you created the earlier training venv
python -m pip install -r requirements-stochastic-training.txt
```

## Tests

```bash
python -m pytest \
  tests/test_integration.py \
  tests/test_stochastic_transfer.py \
  tests/test_stochastic_benchmark.py \
  -q
```

## Smoke training

```bash
python train_physical_ppo_stochastic.py \
  --timesteps 100000 \
  --n-envs 2 \
  --bundles-per-episode 24 \
  --out RL/rl_env_v0/models/physical_multisource_stochastic_smoke.zip
```

## Final training

```bash
python train_physical_ppo_stochastic.py \
  --timesteps 2000000 \
  --n-envs 4 \
  --bundles-per-episode 32 \
  --seed 42 \
  --out RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip
```

## Final paired benchmark

```bash
python benchmark_temporal_vs_rl_stochastic.py \
  --model RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip \
  --num-seeds 20 \
  --seed-base 900000000 \
  --stochastic-seed-base 700000000 \
  --bundles 500 \
  --out benchmark_results/final_stochastic
```
