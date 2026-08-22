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

def summarize(events, label):
    n = len(events)
    delivered = sum(1 for e in events if e.startswith("delivered"))
    on_time = sum(1 for e in events if e == "delivered_on_time")
    invalid = sum(1 for e in events if e == "invalid_action")
    avg_reward = np.mean([r for r, _ in zip(rewards, events)]) if 'rewards' in dir() else 0
    print(f"{label:>20}: delivered={delivered/n:.0%}  on_time={on_time/n:.0%}  invalid={invalid/n:.0%}")

if __name__ == "__main__":
    # First, get the masked-random baseline
    print("=" * 60)
    print("Running 300 episodes for masked-random (held-out seeds 999..1298)")
    print("=" * 60)
    random_r, random_e = run_batch("masked_random", 300, start_seed=999)
    n = len(random_e)
    r_delivered = sum(1 for e in random_e if e.startswith("delivered")) / n
    r_ontime = sum(1 for e in random_e if e == "delivered_on_time") / n
    r_invalid = sum(1 for e in random_e if e == "invalid_action") / n
    print(f"\n{'Masked Random':>20}: delivered={r_delivered:.0%}  on_time={r_ontime:.0%}  invalid={r_invalid:.0%}\n")

    # Now evaluate all 3 trained models
    model_seeds = [42, 7, 123]
    results = {}

    print("=" * 60)
    print("Evaluating 3 trained agents on same held-out seeds")
    print("=" * 60)

    for seed in model_seeds:
        model_path = f"models/rl_agent_seed_{seed}.zip"
        try:
            model = MaskablePPO.load(model_path)
            rewards, events = run_batch("rl_agent", 300, start_seed=999, model=model)
            n = len(events)
            delivered = sum(1 for e in events if e.startswith("delivered")) / n
            on_time = sum(1 for e in events if e == "delivered_on_time") / n
            invalid = sum(1 for e in events if e == "invalid_action") / n
            results[seed] = {"delivered": delivered, "on_time": on_time, "invalid": invalid}
            print(f"Seed {seed:>3}: delivered={delivered:.0%}  on_time={on_time:.0%}  invalid={invalid:.0%}")
        except FileNotFoundError:
            print(f"❌ Model not found: {model_path}")

    # Compute mean and spread
    if results:
        d_vals = [r["delivered"] for r in results.values()]
        o_vals = [r["on_time"] for r in results.values()]
        i_vals = [r["invalid"] for r in results.values()]

        d_mean, d_min, d_max = np.mean(d_vals), np.min(d_vals), np.max(d_vals)
        o_mean, o_min, o_max = np.mean(o_vals), np.min(o_vals), np.max(o_vals)

        print("\n" + "=" * 60)
        print("📊 SUMMARY: 3-Seed Evaluation")
        print("=" * 60)
        print(f"Delivered:  {d_mean:.0%}  (range: {d_min:.0%} - {d_max:.0%})")
        print(f"On-time:    {o_mean:.0%}  (range: {o_min:.0%} - {o_max:.0%})")
        print(f"Masked Random baseline: delivered={r_delivered:.0%}  on_time={r_ontime:.0%}")
        print("=" * 60)