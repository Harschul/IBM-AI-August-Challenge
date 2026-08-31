# PPO training

The final checkpoint path is defined in `config/final_experiment.json` and defaults to:

`RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip`

Install:

```bash
python -m pip install -r requirements-training.txt
```

Verify:

```bash
python check_training_setup.py
```

Train/retrain:

```bash
python train_physical_ppo_stochastic.py
```

The trainer writes the model and a companion metadata file. The metadata stores the
physical config SHA and frozen node/action/observation contract. The final demo will
refuse a mismatched checkpoint instead of silently using it.

TensorBoard:

```bash
tensorboard --logdir RL/rl_env_v0/logs/physical_multisource_stochastic
```
