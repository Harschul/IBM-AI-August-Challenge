# Final visual demo

Install and run:

```bash
python -m pip install -r requirements.txt
python scripts/final/verify_release.py
python run_frontend.py
```

The main page is a white, single-view orbital simulation. The network topology and detailed tables are optional expanders so they do not crowd the core visualization.

Modes:

- **Reported PPO** — pure retrained PPO using the locked final experiment.
- **Reported Temporal** — deterministic temporal earliest-arrival baseline on the same seed family.
- **Interactive Temporal ↔ RL** — same physical/stochastic engine, but explicitly not part of the reported benchmark.

A failed stochastic transmission remains visible in the event/attempt history; the reported modes use the same runner as the benchmark.
