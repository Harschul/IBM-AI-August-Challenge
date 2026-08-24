# Model Card: RL Agent for Multi-Orbit Data Relay Network

**Version:** 1.1
**Date:** 25 August 2026
**Author:** Sudeepa (RL & Evaluation Lead)

---

## 1. Environment Specification (Frozen as of 20 Aug 2026)

| Component | Specification |
| :--- | :--- |
| **Observation space** | 158-dimensional vector (Box, -1 to 1) |
| | 4 bundle features: `[priority, size_norm, deadline_remaining_norm, age_norm]` |
| | 14 candidate nodes × 11 features each: `[valid, link_rate_norm, contact_remaining_norm, prop_delay_norm, queue_norm, storage_free_norm, health, battery, weather_risk, estimated_arrival_norm, reliability]` |
| **Action space** | Discrete(14) — fixed node IDs |
| | `0: Science satellite`, `1-8: LEO relays`, `9-10: GEO relays`, `11-13: Ground stations` |
| **Action masking** | `env.action_masks()` returns binary mask of currently valid next-hops |
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
| Per-hop cost | `-2` |
| Congestion penalty | `-5 × queue_norm` |
| Latency cost | `-0.05 × t` (sim time in seconds) |
| Max hops exceeded (8) | `-25` |
| **Failed-transmission risk** | `-15 × p_fail`, where `p_fail = clip(0.5·weather_risk + 0.3·(1-health) + 0.2·(1-reliability), 0, 0.9)` — expected cost, not a stochastic hard failure |
| **Energy penalty** | `-2 × (1 - battery)` of the relay being routed through |

Added 21–25 Aug: the failed-transmission and energy terms were not present in v1.0 of this card; weather/health/battery were observable but had no effect on reward or dynamics until this update.

---

## 2. Training Configuration

| Parameter | Value |
| :--- | :--- |
| **Algorithm** | MaskablePPO (`sb3-contrib`) |
| **Policy** | MaskableActorCriticPolicy |
| **Learning rate** | `3e-4` |
| **n_steps** | `2048` |
| **batch_size** | `64` |
| **n_epochs** | `10` |
| **Gamma** | `0.99` |
| **Entropy coefficient** | `0.01` |
| **Clip range** | `0.2` |
| **Total timesteps** | `200,000` per seed |
| **Training seeds** | `42`, `7`, `123` |
| **Environment horizon** | `1800s` |

---

## 3. Baseline Evaluation Results

**Methodology:** 300 episodes per policy, held-out seeds `999`–`1298` (not used in training), same seeds across all policies. Metrics: delivery rate, on-time delivery rate, invalid action rate, mean reward.

| Model | Delivered | On-time | Invalid | Mean Reward |
| :--- | :--- | :--- | :--- | :--- |
| Masked Random (baseline) | 59% | 48% | 0% | 24.2 |
| RL Agent — full obs, default hyperparams (3-seed mean) | 92.3% (91–93%) | 81.4% (81–82%) | 0% | 94.3 |

The RL agent consistently and significantly outperforms masked-random across all 3 seeds, with a tight spread (±1pp) — stable, reproducible training.

---

## 4. Ablation Study

**Setup:** same environment and reward; only the agent's *observation* is restricted. Two variants tested: `no_priority` (science_priority zeroed) and `no_weather_health` (weather_risk and health zeroed), each trained on the same 3 seeds and evaluated on the same held-out range as the baseline.

| Variant | Delivered | On-time | Mean Reward |
| :--- | :--- | :--- | :--- |
| Full observation | 92% (91–93%) | 81% (81–82%) | 94.3 |
| No priority | 92% (91–93%) | 81% (81–83%) | 93.9 |
| No weather/health | 92% (92–93%) | 81% (80–83%) | 93.6 |

**Finding:** no statistically meaningful effect. The gap between variants (0.4–0.7 on mean reward) is smaller than the seed-to-seed spread within the full-observation baseline itself (2.7). This does not confirm the expected hypothesis that removing these signals would hurt performance. A plausible explanation: the failed-transmission and energy reward terms are small relative to the dominant `+100`/`+100×priority` delivery bonus, and the masked features may be redundant with other still-visible observation values (e.g. `queue_norm`, `reliability`, `storage_free_norm`). With more time, a larger seed count per variant would be needed to properly test significance.

---

## 5. Hyperparameter Sweep

**Setup:** 3 candidate configs, each trained on the same 3 seeds, evaluated on the same held-out range.

| Config | Key params (vs. baseline) | Delivered | On-time | Mean Reward |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | lr=3e-4, ent_coef=0.01 | 92.3% | 81.4% | 94.3 |
| **Higher exploration (selected)** | lr=1e-3, ent_coef=0.02 | **93.8%** | **81.9%** | **95.9** |
| Conservative | lr=1e-4, batch_size=32, n_epochs=5 | 90.2% | 78.9% | 88.7 |

**Finding:** unlike the ablation, this is a real effect — the spread across configs (7.2 on mean reward) exceeds the seed-to-seed noise seen within any one config (~2.7). `conservative` underperforms consistently across all 3 of its seeds. `higher_exploration` has the best mean on every metric and wins on 2 of 3 seeds individually.

**Decision:** freezing `higher_exploration` (`lr=1e-3`, `ent_coef=0.02`, all else unchanged) as the final config going forward.

---

## 6. Fallback Behavior

Per section 5.6 of the team plan, the AI agent is not the sole path to delivery: if the RL policy selects an invalid action, exceeds a decision-time budget, or encounters an out-of-distribution state, control should hand off to the deterministic baseline router rather than let routing fail outright.

**Current status:** invalid actions are masked to 0% by `action_masks()` in all evaluation runs to date, so the RL agent has not been observed to require this handoff during testing. An explicit fallback-to-baseline handoff (e.g. a wrapper that detects a masked/degenerate state and calls a deterministic router instead of the policy) is **not yet implemented** — this is a known gap, not a tested-and-passing safety feature, and should not be presented as such in the demo.

---

## 7. Known Issues

- **Contact-window feasibility bug (open, unfixed as of this version):** `_action_mask()` and `step()` do not verify that a bundle's transfer can complete before `contact.end_s` closes — only that the contact exists. A teammate's repro (GitHub issue #4) measured ~61% of hops overrunning their contact window. All results in Sections 3–5 of this card were generated **before** this is fixed, so absolute delivered/on-time numbers may shift once corrected; relative comparisons (agent vs. baseline, variant vs. variant) are expected to remain informative since all runs share the same bug equally. Fix is scoped but not yet applied.

---

## 8. Limitations

- **Mocked graph only:** trained and evaluated on `MockContactPlan` (randomly generated contacts), not real orbital geometry or the final physics-based contact generator.
- **Fixed episode horizon:** `horizon_s=1800` (30 minutes), max 20 steps per episode.
- **Failed-transmission risk is a deterministic expected cost, not a stochastic hard-failure event** — no bundle is ever actually dropped due to weather/health in the current environment.
- **No stress testing:** not yet evaluated on longer horizons, higher traffic loads, or degraded-network scenarios.
- **Ablation result is inconclusive** given only 3 seeds per variant (see Section 4).
- **Contact-window bug is open** (see Section 7) — absolute performance numbers are provisional.

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