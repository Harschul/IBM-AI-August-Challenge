# Experiment definition

`config/final_experiment.json` is the single source of truth.

It locks:

- physical scenario: `config/prototype.yaml`
- expected scenario SHA-256: `cf9f11e8ac81dc066fad151d751b9207201c6e5354dbd21a26440da192ce3004`
- final PPO checkpoint: `RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip`
- Temporal-vs-PPO benchmark seeds
- stochastic seeds
- 500 bundles per benchmark seed
- 20 held-out benchmark seeds
- max hops and max attempts
- final artifact paths

The demo and benchmark both call `src.experiment.runner.run_algorithm`.
There is no separate simplified routing simulator for reported visualizations.

## Reported algorithms

`temporal`
: deterministic temporal earliest-arrival router facing stochastic transfer outcomes.

`rl_pure`
: the newly retrained MaskablePPO checkpoint. No Temporal fallback is allowed.

`rl_with_temporal_fallback`
: operational safety mode only. It is not used for the headline comparison.

The UI's interactive mid-run switch also uses the same physical world and stochastic
transfer engine, but it is explicitly marked **not reported**.
