# Final benchmark

Run:

```bash
python run_final_benchmark.py
```

Defaults come entirely from `config/final_experiment.json`:

- 20 seed pairs
- traffic seeds `900000000 .. 900000019`
- stochastic seeds `700000000 .. 700000019`
- 500 bundles per seed
- Temporal vs pure PPO
- no fallback in PPO result

Outputs:

```text
artifacts/final_experiment/benchmark/
├── summary.json
├── seed_metrics.csv
├── bundle_results.csv
└── attempt_log.csv
```

The benchmark uses common random numbers keyed by bundle/contact/attempt. If Temporal
and PPO make the same physical transmission attempt, they receive the same stochastic draw.

The UI can then read `summary.json` in its **Reported benchmark** tab, while rendering
any one of the exact 20 seed pairs using the same runner.
