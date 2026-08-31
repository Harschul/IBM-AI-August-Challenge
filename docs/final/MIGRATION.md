# Migration from earlier bundles

Earlier bundle documentation referenced the old checkpoint
`RL/rl_env_v0/models/rl_agent_seed_42.zip`. That checkpoint is no longer the final
demo default.

The canonical final model is:

`RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip`

The earlier deterministic frontend replay has also been replaced for reported modes.
Reported visualization now calls the same stochastic experiment runner as the benchmark.

Run `bash scripts/final/cleanup_legacy_docs.sh` after copying this release if you want
the previous bundle-specific documentation moved into `docs/archive/legacy_bundle_docs/`.
