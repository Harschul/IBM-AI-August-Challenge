# Optional PPO retraining

The final checkpoint is already included; retraining is not required to run the project.

If the physical/stochastic experiment is deliberately changed:

```bash
python -m pip install -r requirements-training.txt
python train_physical_ppo_stochastic.py --timesteps 2000000 --n-envs 4 \
  --bundles-per-episode 32 --seed 42 \
  --out RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip
```

The trainer writes companion metadata containing the physical config SHA and node/action/observation contract. The final demo refuses a mismatched checkpoint.
