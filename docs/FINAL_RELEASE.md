# Final release: physical stochastic Temporal vs PPO

This is the canonical entry point for the integrated experiment and demo.

Everything that is reported or visualized is locked by `config/final_experiment.json`.
That file points to one physical scenario config and one retrained PPO checkpoint.

## Commands

```bash
python -m pip install -r requirements-frontend-rl.txt
python check_training_setup.py
python run_frontend.py
python run_final_benchmark.py
```

## Frozen experiment

- 3 science spacecraft: node IDs 0, 1, 2
- 6 LEO relays: node IDs 3–8
- 2 GEO relays: node IDs 9–10
- 3 ground stations: node IDs 11–13
- 14 PPO actions
- 158 observation features
- stochastic transfer failures with seeded common random numbers
- shared capacity semantics within each algorithm run
- primary comparison: Temporal vs **pure PPO**, zero fallback

## Canonical PPO checkpoint

`RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip`

The final demo refuses to silently substitute the older single-source checkpoint,
the smoke checkpoint, or Temporal routing when "Reported PPO" is selected.
