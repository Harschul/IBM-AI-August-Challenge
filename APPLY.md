# Apply the full integration + frontend v2 bundle

This archive is laid out relative to the repository root.

## Copy it over the cloned repo

```bash
cp -a /path/to/ibm_full_bundle_v2/. /home/harshul/IBM-AI-August-Challenge/
cd /home/harshul/IBM-AI-August-Challenge
```

## Lightweight install (frontend + physical temporal routing)

```bash
python -m pip install -r requirements-frontend.txt
```

## Optional true RL execution

```bash
python -m pip install -r requirements-integration-rl.txt
```

Without the optional RL environment/checkpoint dependencies, the UI still runs
and explicitly records `requested=rl`, `actual=temporal`, `fallback=true`.

## Test

```bash
python -m pytest \
  tests/test_temporal_router.py \
  tests/test_integration.py \
  tests/test_frontend_bundle.py \
  -q
```

## Run the frontend

```bash
python run_frontend.py
```

## Run the CLI experiment

Temporal baseline only:

```bash
python run_integrated_demo.py --no-rl --bundles 100
```

With the trained checkpoint (after the optional RL install):

```bash
python run_integrated_demo.py \
  --model RL/rl_env_v0/models/rl_agent_seed_42.zip \
  --bundles 100
```

## Fixed 14-node map in v2

```text
SCIENCE_IDS = (0, 1, 2)
LEO_IDS     = (3, 4, 5, 6, 7, 8)
GEO_IDS     = (9, 10)
GROUND_IDS  = (11, 12, 13)
```

The action space remains 14 entries and the observation stays 158 floats, so the
existing checkpoint is shape-compatible. It was trained on the previous
single-source distribution, so retraining on the v2 physical plans is still
recommended before using multi-source results as the final RL headline.
