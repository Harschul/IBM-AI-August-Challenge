#!/usr/bin/env python3
"""Evaluate all trained models against masked-random baseline."""

import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from src.rl.env import RoutingEnv
from sb3_contrib import MaskablePPO

def run_episode(env, policy, rng, model=None):
    obs, info = env.reset(seed=rng.randint(0, 10**9))
    total_reward = 0.0
    for _ in range(20):
        mask = info["action_mask"]
        if policy == "masked_random":
            valid_ids = [i for i, v in enumerate(mask) if v]
            action = rng.choice(valid_ids) if valid_ids else rng.randrange(14)
        elif policy == "rl_agent":
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        else:
            raise ValueError("Unknown policy")
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return total_reward, info.get("event", "unknown")
    return total_reward, "no_terminal"

def run_batch(policy, n_episodes=300, start_seed=999, model=None):
    env = RoutingEnv(horizon_s=1800.0)
    rng = random.Random(start_seed)  # Same held-out seeds for all models
    events = []
    rewards = []
    for _ in range(n_episodes):
        r, ev = run_episode(env, policy, rng, model)
        rewards.append(r)
        events.append(ev)
    return rewards, events

def summarize(rewards, events, label):
    """Print summary statistics for a batch of episodes."""
    n = len(events)
    if n == 0:
        print(f"{label:>20}: no episodes")
        return

    delivered = sum(1 for e in events if e.startswith("delivered"))
    on_time = sum(1 for e in events if e == "delivered_on_time")
    invalid = sum(1 for e in events if e == "invalid_action")
    avg_reward = np.mean(rewards) if rewards else 0.0

    print(f"{label:>20}: delivered={delivered/n:.0%}  on_time={on_time/n:.0%}  "
          f"invalid={invalid/n:.0%}  avg_reward={avg_reward:.1f}")
    return {
        "delivered": delivered / n,
        "on_time": on_time / n,
        "invalid": invalid / n,
        "avg_reward": avg_reward,
    }

if __name__ == "__main__":
    print("=" * 70)
    print("Running 300 episodes for masked-random (held-out seeds 999..1298)")
    print("=" * 70)

    random_r, random_e = run_batch("masked_random", 300, start_seed=999)
    baseline_stats = summarize(random_r, random_e, "Masked Random")
    print()

    print("=" * 70)
    print("Evaluating 3 trained agents on same held-out seeds")
    print("=" * 70)

    model_seeds = [42, 7, 123]
    results = {}

    for seed in model_seeds:
        model_path = f"models/rl_agent_seed_{seed}.zip"
        try:
            model = MaskablePPO.load(model_path)
            rewards, events = run_batch("rl_agent", 300, start_seed=999, model=model)
            stats = summarize(rewards, events, f"RL (seed {seed})")
            results[seed] = stats
        except FileNotFoundError:
            print(f"❌ Model not found: {model_path}")

    if results:
        d_vals = [r["delivered"] for r in results.values()]
        o_vals = [r["on_time"] for r in results.values()]
        i_vals = [r["invalid"] for r in results.values()]
        r_vals = [r["avg_reward"] for r in results.values()]

        d_mean, d_min, d_max = np.mean(d_vals), np.min(d_vals), np.max(d_vals)
        o_mean, o_min, o_max = np.mean(o_vals), np.min(o_vals), np.max(o_vals)
        r_mean = np.mean(r_vals)

        print("\n" + "=" * 70)
        print("📊 SUMMARY: 3-Seed Evaluation")
        print("=" * 70)
        print(f"Delivered:     {d_mean:.0%}  (range: {d_min:.0%} - {d_max:.0%})")
        print(f"On-time:       {o_mean:.0%}  (range: {o_min:.0%} - {o_max:.0%})")
        print(f"Avg Reward:    {r_mean:.1f}")
        print(f"Baseline (masked random): delivered={baseline_stats['delivered']:.0%}  "
              f"on_time={baseline_stats['on_time']:.0%}  "
              f"avg_reward={baseline_stats['avg_reward']:.1f}")
        print("=" * 70)