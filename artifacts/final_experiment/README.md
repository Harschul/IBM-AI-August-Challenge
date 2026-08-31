# Final experiment artifacts

`benchmark/` contains the committed evidence for the reported 20-seed × 500-bundle comparison.

- `summary.json` — aggregate metrics and 95% confidence intervals
- `seed_metrics.csv` — paired per-seed metrics
- `bundle_results.csv` — bundle-level outcomes for Temporal and pure PPO

The very large per-attempt transmission log is intentionally not shipped in the final bundle; it can be regenerated with `python run_final_benchmark.py`.
