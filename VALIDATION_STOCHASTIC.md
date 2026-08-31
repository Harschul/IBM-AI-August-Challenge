# Validation

The bundle was validated in the local API-compatible repository harness.

- Python syntax compilation passed for the stochastic layer, physical PPO env,
  stochastic trainer, stochastic benchmark and tests.
- Stochastic unit tests: **4 passed**.
- Benchmark reproducibility test: **1 passed**.
- Combined integration + stochastic suite: **13 passed** (`tests/test_integration.py`, `tests/test_stochastic_transfer.py`, `tests/test_stochastic_benchmark.py`).
- A physical Temporal smoke run with 12 bundles produced real stochastic
  behavior: 28 attempted transmissions, one sampled transfer failure, capacity
  waste, replanning, and eventual delivery of all 12 bundles in that seed.
- Repeated `TransferOracle` instances with the same seed/contact/bundle/ordinal
  returned byte-for-byte identical outcomes.

Full MaskablePPO training was not executed in this assembly environment because
Gymnasium, Stable-Baselines3 and sb3-contrib are not installed here. Run the
100k-timestep smoke training command before the final multi-million-step run.
