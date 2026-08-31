# Final release

This bundle is intentionally trimmed to the runnable final experiment while retaining the scientific runtime, canonical 2M PPO checkpoint, tests, and final benchmark evidence.

## Run

```bash
python -m pip install -r requirements.txt
python scripts/final/verify_release.py
python run_frontend.py
```

## Locked experiment

- science spacecraft: 0, 1, 2
- LEO relays: 3–8
- GEO relays: 9–10
- ground receivers: 11–13
- PPO actions: 14
- observation features: 158
- stochastic seeded transfer failures
- reported comparison: Temporal vs pure PPO, zero fallback

Canonical checkpoint:

`RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip`

Canonical benchmark evidence:

`artifacts/final_experiment/benchmark/`
