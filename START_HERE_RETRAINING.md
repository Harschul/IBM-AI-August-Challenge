# Start here: retrain MaskablePPO on the frozen physical stochastic environment

This bundle is for the exact 14-node scenario:

- science spacecraft: `0, 1, 2`
- LEO relays: `3, 4, 5, 6, 7, 8`
- GEO relays: `9, 10`
- ground stations: `11, 12, 13`
- action space: 14 next-hop IDs
- observation: 158 floats

The physical contact plan is generated from `config/prototype.yaml`. Science
bundles are generated from the three science spacecraft. Transfer outcomes are
stochastic and seeded from weather risk, node health, and link reliability.
Failed transfers waste time and contact capacity and leave the full bundle at
the sender, so PPO must retry or reroute.

## 0. Prerequisite

The v2 integration/multi-source code must already exist in the repository. In
particular, the repo must have:

- `src/integration/config.py` with `SCIENCE_IDS`
- `src/integration/scenario.py`
- `src/integration/contact_plan.py`
- `src/integration/rl_bridge.py`
- `src/integration/traffic.py`
- `requirements-integration.txt`

## 1. Copy this bundle into the repo root

```bash
cp -a /path/to/ppo_retraining_tutorial_bundle/. /path/to/IBM-AI-August-Challenge/
cd /path/to/IBM-AI-August-Challenge
```

The bundle intentionally replaces `config/prototype.yaml` with the stochastic
version and adds the physical stochastic PPO environment/training scripts.

## 2. Create the isolated PPO environment

Recommended one-command setup:

```bash
bash scripts/setup_training_env.sh
```

Equivalent manual commands:

```bash
python3.11 -m venv .venv-ppo
source .venv-ppo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-stochastic-training.txt
python check_training_setup.py
```

Do not install the root `requirements.txt` just to train PPO. The training
requirements are deliberately scoped to the integration + RL stack.

## 3. Confirm the environment is frozen correctly

Run:

```bash
python check_training_setup.py
```

Expected key lines:

```text
SCIENCE_IDS : (0, 1, 2)
LEO_IDS     : (3, 4, 5, 6, 7, 8)
GEO_IDS     : (9, 10)
GROUND_IDS  : (11, 12, 13)
NUM_NODES   : 14
OBS_LEN     : 158
obs shape   : (158,)
mask shape  : (14,)
stochastic  : True
PASS: environment is ready ...
```

If those IDs or shapes differ, do not train. Fix the code/config mismatch first.

## 4. Run tests before spending training time

```bash
python -m pytest \
  tests/test_integration.py \
  tests/test_stochastic_transfer.py \
  tests/test_stochastic_benchmark.py \
  -q
```

If `tests/test_integration.py` is not present in your branch, run the two
stochastic tests plus any existing integration tests in the repo.

## 5. Run a 100k-timestep smoke training

```bash
bash scripts/smoke_train.sh
```

Equivalent command:

```bash
python train_physical_ppo_stochastic.py \
  --timesteps 100000 \
  --n-envs 2 \
  --bundles-per-episode 24 \
  --max-attempts 24 \
  --seed 42 \
  --out RL/rl_env_v0/models/physical_multisource_stochastic_smoke.zip
```

This is not the final model. It verifies that MaskablePPO can reset, collect
rollouts, handle action masks, experience stochastic failures, and save a model.

Expected outputs:

```text
RL/rl_env_v0/models/physical_multisource_stochastic_smoke.zip
RL/rl_env_v0/models/physical_multisource_stochastic_smoke.metadata.json
```

The metadata stores the SHA-256 of the config used for training. The benchmark
checks this so you do not accidentally benchmark a model on a different
constellation/config.

## 6. Run the smoke Temporal-vs-RL benchmark

```bash
bash scripts/smoke_benchmark.sh
```

This uses held-out traffic and stochastic seeds. The primary comparison is
**Temporal vs pure PPO**. A failed PPO decision is not silently rescued by the
Temporal router.

Expected directory:

```text
benchmark_results/stochastic_smoke/
```

Inspect:

- `bundle_results.csv` for bundle-level outcomes
- `attempt_log.csv` for each stochastic transmission attempt
- `seed_metrics.csv` for paired per-seed metrics
- `summary.json` for aggregate results

## 7. Watch training metrics with TensorBoard

In another terminal, with the same virtual environment active:

```bash
tensorboard --logdir RL/rl_env_v0/logs/physical_multisource_stochastic
```

Look for stable/improving episode reward rather than judging the model from one
short rollout. Failure rate may remain nonzero because failures are physical
stochastic events, not necessarily policy mistakes.

## 8. Train the final model

Once the smoke pipeline works:

```bash
bash scripts/final_train.sh
```

Equivalent command:

```bash
python train_physical_ppo_stochastic.py \
  --timesteps 2000000 \
  --n-envs 4 \
  --bundles-per-episode 32 \
  --max-attempts 24 \
  --seed 42 \
  --out RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip
```

Two million timesteps is a starting point, not a guaranteed convergence point.
If learning is still clearly improving near 2M, continue to 3M-5M.

## 9. Prefer multiple training seeds for final reporting

For a stronger result, train three independent PPO models, for example seeds
41, 42, and 43. Do not choose the best one on your final benchmark seeds. Either
report all three or select a model on a separate validation seed set.

Example:

```bash
python train_physical_ppo_stochastic.py --timesteps 2000000 --n-envs 4 --seed 41 --out RL/rl_env_v0/models/physical_stochastic_seed41.zip
python train_physical_ppo_stochastic.py --timesteps 2000000 --n-envs 4 --seed 42 --out RL/rl_env_v0/models/physical_stochastic_seed42.zip
python train_physical_ppo_stochastic.py --timesteps 2000000 --n-envs 4 --seed 43 --out RL/rl_env_v0/models/physical_stochastic_seed43.zip
```

## 10. Run the final paired benchmark

```bash
bash scripts/final_benchmark.sh
```

Equivalent command:

```bash
python benchmark_temporal_vs_rl_stochastic.py \
  --model RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip \
  --num-seeds 20 \
  --seed-base 900000000 \
  --stochastic-seed-base 700000000 \
  --bundles 500 \
  --out benchmark_results/final_stochastic
```

That evaluates 20 held-out paired scenarios, each with 500 science bundles.
Temporal and PPO see the same physical constellation, traffic distribution,
science priorities, deadlines, source spacecraft, and deterministic keyed
stochastic draws when they make matching transfer attempts.

## 11. What to report

Use the benchmark outputs to report at least:

- delivery ratio
- deadline success rate
- priority-weighted timely delivery
- mean latency
- mean successful hops
- transmission failure count/rate
- retry count
- wasted capacity from failed transfers
- results by science source (`SCI-0`, `SCI-1`, `SCI-2`)
- paired Temporal-vs-RL differences across held-out seeds

The headline algorithm comparison should be **Temporal vs pure PPO**. If you
also evaluate `RL + Temporal fallback`, label it as a third operational mode,
not as pure RL.

## 12. How the reward treats scientific importance

On a successful ground delivery, the environment gives a base delivery reward
and an additional on-time bonus proportional to `science_priority`. Failed
transfers also have a priority-dependent penalty. This allows PPO to learn that
risky routing is more costly for high-value science without changing the
physical failure probability itself.

## 13. How stochastic failures work

For a selected contact, the environment computes a failure probability from:

```text
base failure probability
+ weather weight × weather risk
+ health weight × (1 - health)
+ reliability weight × (1 - link reliability)
```

A seeded draw determines whether the transfer succeeds. On failure:

- the bundle stays completely at the sender
- time advances to the sampled failure point
- a fraction of contact capacity is consumed/wasted
- PPO receives the failure penalty
- the next action can retry or choose a different route

This is an atomic-bundle model; it does not yet implement fragment-level partial
delivery or erasure coding.
