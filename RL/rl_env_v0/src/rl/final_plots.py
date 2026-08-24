#!/usr/bin/env python3
"""Generate final comparison charts and summary table for the pitch deck.

Reads ablation and hyperparameter sweep results, computes masked-random baseline
inline, and produces:
    - results/final_comparison.png: grouped bar charts (ablation + hyperparam)
    - results/final_summary_table.md: markdown table for model card

Usage:
    python -m src.rl.final_plots

This script does NOT train any models — it only reads existing CSVs and
computes the baseline inline.
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Optional pandas import with clear error message
try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "pandas is required for final_plots.py. "
        "Run: pip install pandas>=2.0.0"
    )

from src.rl.env import RoutingEnv


# ============================================================
# BASELINE COMPUTATION (inline, no model needed)
# ============================================================

def compute_masked_random_baseline(n_episodes=300, start_seed=999):
    """Compute masked-random baseline on held-out seeds."""
    env = RoutingEnv(horizon_s=1800.0)
    rng = random.Random(start_seed)
    events = []
    rewards = []

    for _ in range(n_episodes):
        obs, info = env.reset(seed=rng.randint(0, 10**9))
        total_reward = 0.0
        for _ in range(20):
            mask = info["action_mask"]
            valid_ids = [i for i, v in enumerate(mask) if v]
            action = rng.choice(valid_ids) if valid_ids else rng.randrange(14)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        events.append(info.get("event", "unknown"))
        rewards.append(total_reward)

    n = len(events)
    delivered = sum(1 for e in events if e.startswith("delivered")) / n
    on_time = sum(1 for e in events if e == "delivered_on_time") / n
    invalid = sum(1 for e in events if e == "invalid_action") / n
    avg_reward = np.mean(rewards)

    return {
        "label": "Masked Random",
        "delivered": delivered,
        "on_time": on_time,
        "invalid": invalid,
        "avg_reward": avg_reward,
        "delivered_range": (0.0, 0.0),  # no range for single run
        "on_time_range": (0.0, 0.0),
    }


# ============================================================
# LOAD DATA WITH GRACEFUL FALLBACK
# ============================================================

def load_ablation_results():
    """Load ablation results, return aggregated stats per variant."""
    csv_path = "results/ablation_results.csv"
    if not os.path.exists(csv_path):
        print(f"⚠️  Warning: {csv_path} not found — skipping ablation plot")
        return None

    df = pd.read_csv(csv_path)

    # Determine which reward column exists
    reward_col = "mean_reward" if "mean_reward" in df.columns else "avg_reward"
    if reward_col not in df.columns:
        print(f"⚠️  Warning: No reward column found in {csv_path} — skipping reward in ablation summary")
        reward_col = None

    # Aggregate per variant across seeds
    summary = []
    for variant in df["variant"].unique():
        subset = df[df["variant"] == variant]
        row = {
            "label": variant,
            "delivered": subset["delivered"].mean(),
            "on_time": subset["on_time"].mean(),
            "invalid": subset["invalid"].mean(),
            "delivered_range": (subset["delivered"].min(), subset["delivered"].max()),
            "on_time_range": (subset["on_time"].min(), subset["on_time"].max()),
        }
        if reward_col:
            row["avg_reward"] = subset[reward_col].mean()
        else:
            row["avg_reward"] = np.nan
        summary.append(row)

    # Rename variants for display
    label_map = {
        "full": "Full RL",
        "no_priority": "No Priority",
        "no_weather_health": "No Weather/Health",
    }
    for s in summary:
        s["label_display"] = label_map.get(s["label"], s["label"])

    return summary


def load_hyperparam_results():
    """Load hyperparameter sweep results, return aggregated stats per config."""
    csv_path = "results/hyperparam_sweep_results.csv"
    if not os.path.exists(csv_path):
        print(f"⚠️  Warning: {csv_path} not found — skipping hyperparam plot")
        return None

    df = pd.read_csv(csv_path)

    # Ensure avg_reward exists
    if "avg_reward" not in df.columns:
        print(f"⚠️  Warning: No avg_reward column in {csv_path} — skipping reward in hyperparam summary")
        reward_col = None
    else:
        reward_col = "avg_reward"

    # Aggregate per config across seeds
    summary = []
    for config in df["config"].unique():
        subset = df[df["config"] == config]
        row = {
            "label": config,
            "delivered": subset["delivered"].mean(),
            "on_time": subset["on_time"].mean(),
            "invalid": subset["invalid"].mean(),
            "delivered_range": (subset["delivered"].min(), subset["delivered"].max()),
            "on_time_range": (subset["on_time"].min(), subset["on_time"].max()),
        }
        if reward_col:
            row["avg_reward"] = subset[reward_col].mean()
        else:
            row["avg_reward"] = np.nan
        summary.append(row)

    # Rename for display
    label_map = {
        "baseline": "Baseline (lr=3e-4)",
        "higher_exploration": "Higher Exploration",
        "conservative": "Conservative",
    }
    for s in summary:
        s["label_display"] = label_map.get(s["label"], s["label"])

    return summary


# ============================================================
# PLOTTING
# ============================================================

def create_ablation_chart(ablation_data, baseline_data):
    """Create grouped bar chart for ablation comparison."""
    if ablation_data is None:
        return None

    # Order: Baseline, Full, No Priority, No Weather
    labels = ["Masked Random", "Full RL", "No Priority", "No Weather/Health"]

    # Find matching data
    full = next((d for d in ablation_data if d["label"] == "full"), None)
    no_priority = next((d for d in ablation_data if d["label"] == "no_priority"), None)
    no_weather = next((d for d in ablation_data if d["label"] == "no_weather_health"), None)

    data = [
        baseline_data,
        full,
        no_priority,
        no_weather,
    ]

    delivered_vals = []
    delivered_err_lower = []
    delivered_err_upper = []
    on_time_vals = []
    on_time_err_lower = []
    on_time_err_upper = []

    for d in data:
        if d is None:
            delivered_vals.append(0)
            delivered_err_lower.append(0)
            delivered_err_upper.append(0)
            on_time_vals.append(0)
            on_time_err_lower.append(0)
            on_time_err_upper.append(0)
        else:
            delivered_vals.append(d["delivered"])
            # Clamp to avoid negative error values
            delivered_err_lower.append(max(0, d["delivered"] - d["delivered_range"][0]))
            delivered_err_upper.append(max(0, d["delivered_range"][1] - d["delivered"]))
            on_time_vals.append(d["on_time"])
            on_time_err_lower.append(max(0, d["on_time"] - d["on_time_range"][0]))
            on_time_err_upper.append(max(0, d["on_time_range"][1] - d["on_time"]))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to (2, n) format for matplotlib's errorbar
    delivered_err = [delivered_err_lower, delivered_err_upper]
    on_time_err = [on_time_err_lower, on_time_err_upper]

    bars1 = ax.bar(x - width/2, delivered_vals, width, yerr=delivered_err,
                   label="Delivered", color="#2e86de", capsize=4, error_kw={'capsize': 4})
    bars2 = ax.bar(x + width/2, on_time_vals, width, yerr=on_time_err,
                   label="On-time", color="#e67e22", capsize=4, error_kw={'capsize': 4})

    ax.set_ylabel("Rate")
    ax.set_title("Ablation Study: RL Agent Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Add value labels on top of bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        if height > 0.01:
            ax.annotate(f"{height:.0%}",
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8)

    plt.tight_layout()
    return fig

def create_hyperparam_chart(hyperparam_data):
    """Create grouped bar chart for hyperparameter comparison."""
    if hyperparam_data is None:
        return None

    # Sort by delivered (descending)
    sorted_data = sorted(hyperparam_data, key=lambda x: x["delivered"], reverse=True)

    labels = [d["label_display"] for d in sorted_data]
    delivered_vals = [d["delivered"] for d in sorted_data]
    delivered_err_lower = [max(0, d["delivered"] - d["delivered_range"][0]) for d in sorted_data]
    delivered_err_upper = [max(0, d["delivered_range"][1] - d["delivered"]) for d in sorted_data]
    on_time_vals = [d["on_time"] for d in sorted_data]
    on_time_err_lower = [max(0, d["on_time"] - d["on_time_range"][0]) for d in sorted_data]
    on_time_err_upper = [max(0, d["on_time_range"][1] - d["on_time"]) for d in sorted_data]

    delivered_err = [delivered_err_lower, delivered_err_upper]
    on_time_err = [on_time_err_lower, on_time_err_upper]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, delivered_vals, width, yerr=delivered_err,
                   label="Delivered", color="#2e86de", capsize=4, error_kw={'capsize': 4})
    bars2 = ax.bar(x + width/2, on_time_vals, width, yerr=on_time_err,
                   label="On-time", color="#e67e22", capsize=4, error_kw={'capsize': 4})

    ax.set_ylabel("Rate")
    ax.set_title("Hyperparameter Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Add value labels on top of bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        if height > 0.01:
            ax.annotate(f"{height:.0%}",
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8)

    plt.tight_layout()
    return fig


# ============================================================
# SUMMARY TABLE (Markdown)
# ============================================================

def generate_summary_table(baseline_data, ablation_data, hyperparam_data):
    """Generate markdown summary table combining all results."""
    rows = []

    # Baseline
    rows.append({
        "Model": "Masked Random (baseline)",
        "Delivered": f"{baseline_data['delivered']:.0%}",
        "On-time": f"{baseline_data['on_time']:.0%}",
        "Invalid": f"{baseline_data['invalid']:.0%}",
        "Avg Reward": f"{baseline_data['avg_reward']:.1f}",
    })

    # Ablation variants
    if ablation_data:
        label_map = {
            "full": "RL Agent (Full)",
            "no_priority": "RL Agent (No Priority)",
            "no_weather_health": "RL Agent (No Weather/Health)",
        }
        for d in sorted(ablation_data, key=lambda x: x["delivered"], reverse=True):
            rows.append({
                "Model": label_map.get(d["label"], d["label"]),
                "Delivered": f"{d['delivered']:.0%} ({d['delivered_range'][0]:.0%}-{d['delivered_range'][1]:.0%})",
                "On-time": f"{d['on_time']:.0%} ({d['on_time_range'][0]:.0%}-{d['on_time_range'][1]:.0%})",
                "Invalid": f"{d['invalid']:.0%}",
                "Avg Reward": f"{d.get('avg_reward', 0.0):.1f}",
            })

    # Hyperparam variants
    if hyperparam_data:
        label_map = {
            "baseline": "RL Agent (Baseline HP)",
            "higher_exploration": "RL Agent (Higher Exploration)",
            "conservative": "RL Agent (Conservative)",
        }
        for d in sorted(hyperparam_data, key=lambda x: x["delivered"], reverse=True):
            rows.append({
                "Model": label_map.get(d["label"], d["label"]),
                "Delivered": f"{d['delivered']:.0%} ({d['delivered_range'][0]:.0%}-{d['delivered_range'][1]:.0%})",
                "On-time": f"{d['on_time']:.0%} ({d['on_time_range'][0]:.0%}-{d['on_time_range'][1]:.0%})",
                "Invalid": f"{d['invalid']:.0%}",
                "Avg Reward": f"{d.get('avg_reward', 0.0):.1f}",
            })

    # Build markdown table
    lines = []
    lines.append("| Model | Delivered | On-time | Invalid | Avg Reward |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for row in rows:
        lines.append(f"| {row['Model']} | {row['Delivered']} | {row['On-time']} | {row['Invalid']} | {row['Avg Reward']} |")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("FINAL PLOTS: Generating Pitch-Deck Artifacts")
    print("=" * 60)

    # 1. Compute baseline
    print("\n📊 Computing masked-random baseline...")
    baseline = compute_masked_random_baseline(n_episodes=300, start_seed=999)
    print(f"   Delivered: {baseline['delivered']:.0%}  On-time: {baseline['on_time']:.0%}")

    # 2. Load ablation results
    print("\n📊 Loading ablation results...")
    ablation_data = load_ablation_results()
    if ablation_data:
        for d in ablation_data:
            print(f"   {d['label']}: {d['delivered']:.0%} ({d['delivered_range'][0]:.0%}-{d['delivered_range'][1]:.0%})")

    # 3. Load hyperparam results
    print("\n📊 Loading hyperparameter sweep results...")
    hyperparam_data = load_hyperparam_results()
    if hyperparam_data:
        for d in hyperparam_data:
            print(f"   {d['label']}: {d['delivered']:.0%} ({d['delivered_range'][0]:.0%}-{d['delivered_range'][1]:.0%})")

    # 4. Generate plots
    print("\n📈 Generating plots...")
    os.makedirs("results", exist_ok=True)

    # Ablation chart
    fig1 = create_ablation_chart(ablation_data, baseline)
    if fig1:
        fig1.savefig("results/ablation_comparison.png", dpi=150)
        print("   ✅ Saved results/ablation_comparison.png")
        plt.close(fig1)

    # Hyperparam chart
    fig2 = create_hyperparam_chart(hyperparam_data)
    if fig2:
        fig2.savefig("results/hyperparam_comparison.png", dpi=150)
        print("   ✅ Saved results/hyperparam_comparison.png")
        plt.close(fig2)

    # 5. Generate summary table
    print("\n📋 Generating summary table...")
    table = generate_summary_table(baseline, ablation_data, hyperparam_data)
    with open("results/final_summary_table.md", "w") as f:
        f.write("# Final Summary Table\n\n")
        f.write(table)
        f.write("\n\n*Generated by `src/rl/final_plots.py`*\n")
    print("   ✅ Saved results/final_summary_table.md")

    # Print table to console
    print("\n" + "=" * 60)
    print("FINAL SUMMARY TABLE")
    print("=" * 60)
    print(table)
    print("=" * 60)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()