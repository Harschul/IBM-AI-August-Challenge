# Multi-Orbit Scientific Data Relay Network

Final IBM AI August Challenge project: a physical orbital routing simulation that generates science data across three research spacecraft, routes it through LEO/GEO relays, and delivers it to one of three operational ground receivers.

The final release has one locked experiment definition and one execution engine shared by the benchmark and visual demo. The reported comparison is **Temporal earliest-arrival routing vs pure MaskablePPO** on identical physical scenarios with seeded stochastic transfer failures.

## What the demo shows

- 3 research satellites generating science bundles
- 6 LEO relays and 2 GEO relays
- 3 real ground receivers (not the old Earth-coverage sample points)
- physically derived temporal contact windows
- capacity-aware, contact-window-valid transfers
- stochastic failures from configured weather/health/reliability assumptions
- animated packet paths on the orbital simulation
- switchable **Reported PPO**, **Reported Temporal**, and clearly marked interactive switch mode

## Quick start

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/final/verify_release.py
python run_frontend.py
```

The final PPO checkpoint is already included at:

```text
RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip
```

No retraining is required to run the demo.

## Final benchmark

The locked benchmark uses:

- 20 held-out paired seed scenarios
- 500 science bundles per seed
- identical traffic and stochastic seed families for both algorithms
- 3 science sources, 6 LEO relays, 2 GEO relays, 3 ground receivers
- pure PPO with **no Temporal fallback** in the headline RL result

Committed aggregate results:

| Metric | Temporal | Retrained PPO |
|---|---:|---:|
| Delivery ratio | **75.28%** | 62.16% |
| On-time delivery | **71.08%** | 59.47% |
| Priority-weighted timely delivery | **67.21%** | 57.08% |
| Mean latency of successful deliveries | 216.3 s | **181.3 s** |
| Mean hops | **2.23** | 3.39 |
| Transfer failure rate | 4.39% | **3.40%** |

The result is intentionally reported as measured by the simulation: PPO is faster on successful deliveries and selects lower-risk links, but the deterministic Temporal router achieves higher overall and on-time delivery.

The committed evidence is in:

```text
artifacts/final_experiment/benchmark/
├── summary.json
├── seed_metrics.csv
└── bundle_results.csv
```

Run the benchmark again with:

```bash
python run_final_benchmark.py
```

That command also regenerates the detailed per-attempt log.

## Experiment identity

`config/final_experiment.json` is the single source of truth for the reported experiment. It locks the physical scenario config, expected config SHA-256, final PPO checkpoint, seed family, bundle count, and algorithm modes.

Both the benchmark and reported frontend replay call the same execution engine:

```text
src/experiment/runner.py
```

The frontend does not use a separate simplified routing simulator for reported results.

## Repository layout

```text
config/                     locked scenario + experiment definition
src/model/                  orbital simulator/research utilities
src/models/                 ContactPlan and DataBundle data models
src/routing/                Temporal earliest-arrival router
src/integration/            physical contacts, capacity, stochastic transfers, PPO bridge
src/experiment/             canonical final runner + benchmark
src/frontend/               Streamlit/Plotly visualization
RL/rl_env_v0/models/        canonical final PPO checkpoint + metadata
artifacts/final_experiment/ committed final benchmark evidence
tests/                      regression/integration tests
docs/final/                 concise experiment/demo/training notes
```

## Validation

Without installing PPO/Streamlit, the lightweight test suite is:

```bash
python -m pip install -r requirements-ci.txt
python -m pytest -q
python scripts/final/verify_release.py
```

The contact feasibility invariant is enforced during routing/execution:

```text
departure_time + transmission_time <= contact.end_s
```

## Optional retraining

Retraining is not required to run the final project. If you intentionally change the physical/stochastic experiment and need a new policy:

```bash
python -m pip install -r requirements-training.txt
python train_physical_ppo_stochastic.py --timesteps 2000000 --n-envs 4 \
  --bundles-per-episode 32 --seed 42 \
  --out RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip
```

Changing the physical config changes its hash; the final demo will refuse to present a checkpoint whose metadata does not match the locked config.

## Scientific scope

Orbital geometry, contact windows, traffic and stochastic failures are **simulation outputs/assumptions**, not measurements of an operational satellite network. The three operational ground receivers are distinct from any Earth-surface coverage sampling used by earlier constellation experiments.
