# Final visual demo

Install:

```bash
python -m pip install -r requirements-frontend-rl.txt
```

Run:

```bash
python run_frontend.py
```

The default mode is **Reported PPO**. It:

1. loads `config/final_experiment.json`;
2. verifies the locked physical config hash;
3. verifies the final PPO metadata was trained on that config;
4. loads `physical_multisource_stochastic_ppo.zip`;
5. generates the exact benchmark traffic/stochastic seed pair for the selected seed offset;
6. simulates all 500 bundles with the same capacity and stochastic transfer logic used by the benchmark;
7. renders those attempts in the orbital and topology views.

A failed transmission is shown as a failed in-flight packet and is also written to
the event log with its `p_fail`, seeded random draw, failure progress and consumed capacity.

## Modes

- **Reported PPO** — pure PPO, exact final experiment.
- **Reported Temporal** — Temporal baseline, exact same seed family.
- **Interactive Temporal ↔ RL** — same world and stochastic layer, operator can switch mid-run; clearly marked not part of the reported benchmark.
