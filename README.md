# IBM-AI-August-Challenge

Multi-Orbit Scientific Data Relay Network

An AI-orchestrated relay network that routes high-value science data across
direct-to-ground, GEO and LEO paths before its value expires. This repository
holds the simulator, the routing baseline, the RL environment and the
evaluation harness.

---

## Requirements

**Python 3.11 or newer** for the full repository. The root `requirements.txt`
pins `contourpy==1.3.3`, `numpy==2.5.2` and `matplotlib==3.11.1`, none of which
publish wheels for older versions — on Python 3.9 or 3.10 the install fails at
the first package with a `Requires-Python` error.

The RL environment alone runs on **Python 3.9+** via
`RL/rl_env_v0/requirements.txt`, which uses lower bounds rather than pins. If
the demo machine is on an older Python, use that path.

## Install

Everything (needs Python 3.11+):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

RL environment and evaluation only (works on Python 3.9+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r RL/rl_env_v0/requirements.txt
```

No component makes network calls, so everything below runs offline once
installed.

## Running things

### Routing baseline

The temporal earliest-arrival router, its tests and a walkthrough. Pure standard
library — no third-party packages needed beyond `pytest` for the tests.

```bash
python3 demo_route.py               # worked examples with route timelines
python3 -m pytest tests/ -q         # 11 tests with hand-calculated answers
```

### RL environment

Run these from `RL/rl_env_v0/`, which is where `src/rl/` and the model
checkpoints sit:

```bash
cd RL/rl_env_v0

python3 run_demo.py                        # action-masking sanity check
python3 src/rl/eval_all.py                 # agents vs masked-random
python3 src/rl/eval_with_baseline.py       # agents vs the temporal router
python3 src/rl/architecture_comparison.py  # GEO-only vs LEO mesh vs hybrid
python3 src/rl/route_trace.py              # side-by-side route traces
```

`eval_with_baseline.py` and `architecture_comparison.py` write to
`RL/rl_env_v0/results/` — CSV per bundle, plus JSON carrying the git commit and
checkpoint hashes that produced each run.

## Repository layout

| path | what it is |
|---|---|
| `src/model/` | orbital simulation, constellation optimisation, packet routing |
| `src/models/` | `Contact`, `ContactPlan`, `DataBundle` |
| `src/routing/` | temporal earliest-arrival / CGR-style router and route traces |
| `RL/rl_env_v0/` | Gymnasium routing environment, training, evaluation |
| `config/` | scenario configuration |
| `tests/` | routing baseline tests |

## Prototype assumptions

Numbers in this repository are **simulation assumptions**, not measurements of
real systems, unless a source is named:

- The RL environment runs on a **mock contact graph**, not real orbital
  geometry. Contact windows, data rates, congestion and weather are randomly
  generated.
- **Direct-to-ground is not modelled.** `mock_graph._allowed_pairs()` generates
  no science-satellite-to-ground link, so that architecture cannot currently be
  evaluated.
- Link rates, ranges and priorities are chosen so the routing problem is
  realistic, not because they match a specific spacecraft.
- `config/prototype.yaml` is currently empty; the frozen physical parameters it
  is meant to hold have not landed yet.
