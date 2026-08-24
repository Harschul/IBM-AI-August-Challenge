#!/usr/bin/env python3
"""Hyperparameter sweep over 3 candidate configs.

Trains each config across 3 seeds, evaluates on held-out seeds (999..1298),
and reports which config performs best.

Usage:
    python -m src.rl.hyperparam_sweep --timesteps 200000
"""

import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rl.env import RoutingEnv


# ============================================================
# HYPERPARAMETER CONFIGS
# ============================================================

CONFIGS = {
    "baseline": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "gamma": 0.99,
    },
    "higher_exploration": {
        "learning_rate": 1e-3,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.02,
        "clip_range": 0.2,
        "gamma": 0.99,
    },
    "conservative": {
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 32,
        "n_epochs": 5,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "gamma": 0.99,
    },
}


# ============================================================
# EVALUATION HELPERS (mirroring eval_all.py)
# ============================================================

def run_eval_episode(env, model, rng):
    obs, info = env.reset(seed=rng.randint(0, 10**9))
    total_reward = 0.0
    for _ in range(20):
        mask = info["action_mask"]
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return total_reward, info.get("event", "unknown")
    return total_reward, "no_terminal"


def evaluate_model(model, n_episodes=300, start_seed=999):
    """Evaluate a trained model on held-out seeds."""
    env = RoutingEnv(horizon_s=1800.0)
    rng = random.Random(start_seed)
    events = []
    rewards = []

    for _ in range(n_episodes):
        r, ev = run_eval_episode(env, model, rng)
        rewards.append(r)
        events.append(ev)

    n = len(events)
    delivered = sum(1 for e in events if e.startswith("delivered")) / n
    on_time = sum(1 for e in events if e == "delivered_on_time") / n
    invalid = sum(1 for e in events if e == "invalid_action") / n
    avg_reward = np.mean(rewards)

    return {
        "delivered": delivered,
        "on_time": on_time,
        "invalid": invalid,
        "avg_reward": avg_reward,
    }


# ============================================================
# TRAINING HELPERS
# ============================================================

def train_agent(config_name, config_kwargs, seed, total_timesteps):
    """Train one agent with a given config and seed."""
    env = RoutingEnv(horizon_s=1800.0, seed=seed)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=0,
        seed=seed,
        **config_kwargs,
    )

    model.learn(total_timesteps=total_timesteps)
    return model


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Total timesteps per training run (default: 200000)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 7, 123],
        help="Training seeds to use (default: 42 7 123)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=300,
        help="Number of evaluation episodes per model (default: 300)",
    )
    parser.add_argument(
        "--eval-start-seed",
        type=int,
        default=999,
        help="Starting seed for held-out evaluation (default: 999)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("HYPERPARAMETER SWEEP")
    print("=" * 70)
    print(f"Timesteps per run: {args.timesteps}")
    print(f"Training seeds: {args.seeds}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Evaluation start seed: {args.eval_start_seed}")
    print()
    print("Configs:")
    for name in CONFIGS:
        print(f"  - {name}: {CONFIGS[name]}")
    print("=" * 70)

    results = []

    for config_name, config_kwargs in CONFIGS.items():
        print(f"\n📋 Training config: {config_name}")

        for seed in args.seeds:
            print(f"  🔹 Seed {seed}...", end=" ", flush=True)

            # Train
            model = train_agent(config_name, config_kwargs, seed, args.timesteps)

            # Evaluate
            stats = evaluate_model(model, args.eval_episodes, args.eval_start_seed)

            # Store result
            results.append({
                "config": config_name,
                "seed": seed,
                "delivered": stats["delivered"],
                "on_time": stats["on_time"],
                "invalid": stats["invalid"],
                "avg_reward": stats["avg_reward"],
            })

            print(f"done. delivered={stats['delivered']:.0%}  on_time={stats['on_time']:.0%}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    csv_path = "results/hyperparam_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved results to {csv_path}")

    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    print("\n" + "=" * 70)
    print("📊 SUMMARY: Per-Config Averages (across 3 seeds)")
    print("=" * 70)

    summary = df.groupby("config").agg({
        "delivered": ["mean", "min", "max"],
        "on_time": ["mean", "min", "max"],
        "avg_reward": ["mean", "min", "max"],
        "invalid": ["mean"],
    }).round(4)

    print(summary.to_string())
    print()

    # Determine best config by avg_reward
    best_idx = df.groupby("config")["avg_reward"].mean().idxmax()
    best_reward = df.groupby("config")["avg_reward"].mean().max()

    print(f"🏆 Best config (highest avg_reward): {best_idx} ({best_reward:.1f})")

    # Sanity check: all configs should have 0% invalid actions
    max_invalid = df.groupby("config")["invalid"].max()
    for name, val in max_invalid.items():
        if val > 0.01:
            print(f"⚠️  WARNING: {name} has non-zero invalid actions ({val:.1%}) — this config is broken!")

    print("=" * 70)
    print(f"Recommended config for final model: {best_idx}")
    print("=" * 70)


if __name__ == "__main__":
    main()