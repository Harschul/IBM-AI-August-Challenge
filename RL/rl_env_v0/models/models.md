# Model Card: RL Agent for Multi-Orbit Data Relay Network

**Version:** 1.2 (Post-PR #8 Environment Correction)
**Date:** 28 August 2026
**Author:** Sudeepa (RL & Evaluation Lead)

---

## 1. Environment Specification (Updated post-PR #8)

| Component | Specification |
| :--- | :--- |
| **Observation space** | 158-dimensional vector (Box, -1 to 1) |
| | 4 bundle features: `[priority, size_norm, deadline_remaining_norm, age_norm]` |
| | 14 candidate nodes × 11 features each: `[valid, link_rate_norm, contact_remaining_norm, prop_delay_norm, queue_norm, storage_free_norm, health, battery, weather_risk, estimated_arrival_norm, reliability]` |
| **Action space** | Discrete(14) — fixed node IDs |
| | `0: Science satellite`, `1-8: LEO relays`, `9-10: GEO relays`, `11-13: Ground stations` |
| **Action masking** | `env.action_masks()` returns binary mask requiring `depart + tx_time <= contact.end_s` |
| **Horizon** | 1800 seconds (30 minutes) per episode |
| **Max hops** | 8 per bundle |
| **Ablation modes** | `ablation=None` (full), `"no_priority"`, `"no_weather_health"` — mask what the agent *observes* while keeping true reward/dynamics unchanged |

### Reward Function (from `RoutingEnv.step()`)

| Event | Reward |
| :--- | :--- |
| Delivery to ground | `+100` |
| On-time delivery (≤ deadline) | `+100 × science_priority` |
| Late delivery (> deadline) | `-25` |
| Missed deadline (expired in transit) | `-25` |
| Invalid action (unreachable node) | `-15` |
| No feasible onward contact | `-25` (terminates episode safely without impossible choices) |
| Per-hop cost | `-2` |
| Congestion penalty | `-5 × queue_norm` |
| Latency cost | `-0.05 × t` (sim time in seconds) |
| Max hops exceeded (8) | `-25` |
| **Failed-transmission risk** | `-15 × p_fail`, where `p_fail = clip(0.5·weather_risk + 0.3·(1-health) + 0.2·(1-reliability), 0, 0.9)` |
| **Energy penalty** | `-2 × (1 - battery)` of the relay being routed through |

---

## 2. Training Configuration

| Parameter | Value |
| :--- | :--- |
| **Algorithm** | MaskablePPO (`sb3-contrib`) |
| **Policy** | MaskableActorCriticPolicy |
| **Learning rate** | `3e-4` (baseline) / `1e-3` (higher exploration) |
| **n_steps** | `2048` |
| **batch_size** | `64` |
| **n_epochs** | `10` |
| **Gamma** | `0.99` |
| **Entropy coefficient** | `0.01` (baseline) / `0.02` (higher exploration) |
| **Clip range** | `0.2` |
| **Total timesteps** | `200,000` per seed |
| **Training seeds** | `42`, `7`, `123` |
| **Environment horizon** | `1800s` |

---

## 3. Evaluation Results & Physical Environment Impact

### Impact of PR #8 (Contact Window Feasibility Fix)
In PR #8, the environment's contact-window feasibility bug was corrected (`_action_mask()` and `step()` now strictly enforce `depart + tx_time <= contact.end_s`). 

When testing the pre-fix trained checkpoints on the fixed physical environment:
- **Pre-fix RL Agents**: Score drops from ~**0.74** (92% delivery under unconstrained rules) down to ~**0.50** under physically valid windows. The policy had learned to exploit simulator overruns that are physically impossible in orbit.
- **Physical Baseline Router (`baseline_physical`)**: Unaffected by simulator exploits, holding steady at **0.735 – 0.744** (92.8% delivery ratio, 82.4% on-time).
- **Claim Update**: We have removed any claims that old RL checkpoints match the physical router until the models are retrained directly on the corrected physics environment.

| Model / Policy | Environment Rules | Priority-Weighted Timely | Delivery Ratio | On-Time Rate | Mean Latency | Mean Reward |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Masked Random** | Fixed Physics | 32.9% (±2.8%) | 57.3% (±7.6%) | 43.0% (±2.0%) | 671s (±49s) | 21.0 (±4.9) |
| **Baseline Physical (Earliest-Arrival)** | **Fixed Physics** | **80.5% (±6.8%)** | **96.0% (±1.0%)** | **88.0% (±5.0%)** | **271s (±22s)** | **102.3 (±2.8)** |
| Baseline (Permissive Env Rules) | Permissive | 69.6% (±6.2%) | 86.7% (±5.5%) | 75.3% (±5.0%) | 447s (±66s) | 78.7 (±3.9) |
| **RL Agent (3-seed mean on Fixed Env)** | **Fixed Physics** | **54.2% (±5.7%)** | **80.9% (±4.4%)** | **63.7% (±5.2%)** | **586s (±63s)** | **59.8 (±4.9)** |
| ↳ *RL Seed 42* | Fixed Physics | 55.3% (±5.1%) | 80.7% (±3.1%) | 64.7% (±4.6%) | 586s (±69s) | 60.5 (±4.9) |
| ↳ *RL Seed 7* | Fixed Physics | 53.5% (±6.4%) | 80.3% (±5.0%) | 62.7% (±5.5%) | 585s (±62s) | 58.9 (±5.1) |
| ↳ *RL Seed 123* | Fixed Physics | 53.8% (±5.6%) | 81.7% (±5.1%) | 63.7% (±5.5%) | 586s (±59s) | 60.1 (±4.6) |
| RL Agent (Pre-fix Weights on Permissive Env) | Permissive | 73.9% (±9.6%) | 91.6% | 84.2% | 298s | 94.4 |

---

## 4. Ablation Study

**Setup:** same environment and reward; only the agent's *observation* is restricted. Two variants tested: `no_priority` (science_priority zeroed) and `no_weather_health` (weather_risk and health zeroed), each trained on the same 3 seeds and evaluated on the same held-out range as the baseline.

| Variant | Delivered (Permissive) | On-time | Mean Reward |
| :--- | :--- | :--- | :--- |
| Full observation | 92% (91–93%) | 81% (81–82%) | 94.3 |
| No priority | 92% (91–93%) | 81% (81–83%) | 93.9 |
| No weather/health | 92% (92–93%) | 81% (80–83%) | 93.6 |

**Finding:** on the initial testbed, the gap between variants (0.4–0.7 on mean reward) is smaller than the seed-to-seed spread within the full-observation baseline itself (2.7). A planned retraining batch on the corrected environment will evaluate whether tight contact constraints magnify the impact of weather and health awareness.

---

## 5. Hyperparameter Sweep

**Setup:** 3 candidate configs, each trained on the same 3 seeds, evaluated on the same held-out range.

| Config | Key params (vs. baseline) | Delivered | On-time | Mean Reward |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | lr=3e-4, ent_coef=0.01 | 92.3% | 81.4% | 94.3 |
| **Higher exploration (selected)** | lr=1e-3, ent_coef=0.02 | **93.8%** | **81.9%** | **95.9** |
| Conservative | lr=1e-4, batch_size=32, n_epochs=5 | 90.2% | 78.9% | 88.7 |

**Decision:** `higher_exploration` (`lr=1e-3`, `ent_coef=0.02`) is selected as the primary training configuration for retraining under the corrected physics.

---

## 6. Fallback Behavior

Per section 5.6 of the team plan, the AI agent is deployed alongside the temporal baseline:
- `rl_fallback_unsafe`: If an RL policy proposes a hop that cannot complete within its contact window, control immediately defaults to the earliest-arrival baseline.
- `rl_fallback_invalid`: Catches mask-violating or out-of-distribution selections.

---

## 7. Status of Known Issues

- **Contact-window feasibility bug:** **RESOLVED in PR #8**. `_action_mask()` and `step()` now check transfer completion before `contact.end_s`.
- **Numpy 2.x / 1.x compatibility:** **RESOLVED** in `eval_with_baseline.py` and `eval_all.py` using `_numpy2_compat()` and `custom_objects` space overrides.

---

## 8. Limitations & Next Steps

1. **Retraining on Corrected Physics**: Retrain 3-seed MaskablePPO agents (`train_multiseed.py`) against the fixed `RoutingEnv` with `depart + tx_time <= contact.end_s`.
2. **Orbital Geometry Integration**: Bridge RL contact evaluation with the Three.js / orbital mechanics contact generator rather than pure mock graphs.
3. **Stress Testing**: Evaluate trained agents under simulated ground station outages and atmospheric rain fade.

---

## 9. Model Artifacts

| File | Description |
| :--- | :--- |
| `models/rl_agent_seed_42.zip`, `_7.zip`, `_123.zip` | Baseline full-observation agents |
| `models/ablations/rl_agent_full_seed_{42,7,123}.zip` | Ablation baseline re-runs |
| `models/ablations/rl_agent_no_priority_seed_{42,7,123}.zip` | No-priority ablation |
| `models/ablations/rl_agent_no_weather_health_seed_{42,7,123}.zip` | No-weather/health ablation |
| `results/ablation_results.csv`, `results/hyperparam_sweep_results.csv` | Raw per-seed metrics |
| `results/final_summary_table.md`, `results/ablation_comparison.png`, `results/hyperparam_comparison.png` | Pitch-deck artifacts |

---

## 10. Next Steps

1. Fix the contact-window feasibility bug (Section 7) and re-verify overrun rate hits 0%.
2. Replace `MockContactPlan` with Serafin's real orbital contact generator.
3. Implement the fallback-to-baseline handoff described in Section 6.
4. Retrain final agent under `higher_exploration` config once the contact-window fix lands.
5. Compare against Jiwoo's temporal baseline (CGR-like) for final evaluation.
6. Add weather, failure modes, and burst traffic stress tests.
7. Integrate with team dashboard for live visualization.

---

## 11. Reproduction Instructions

```bash
pip install -r requirements.txt

# Train 3 baseline agents
python -m src.rl.train_multiseed

# Evaluate all models against masked-random baseline
python -m src.rl.eval_all

# Run the ablation study
python -m src.rl.ablations --timesteps 200000

# Run the hyperparameter sweep
python -m src.rl.hyperparam_sweep --timesteps 200000

# Generate final plots/tables (run after the two above)
python -m src.rl.final_plots
```