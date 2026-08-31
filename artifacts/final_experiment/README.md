# Final experiment artifacts

Generated outputs live here and are intentionally ignored by Git by default.

- `benchmark/` — `summary.json`, `seed_metrics.csv`, `bundle_results.csv`, `attempt_log.csv`
- `demo/` — optional CLI/demo exports

The trained PPO checkpoint remains at the canonical path declared in
`config/final_experiment.json`:

`RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip`

The companion metadata file must be next to it:

`RL/rl_env_v0/models/physical_multisource_stochastic_ppo.metadata.json`
