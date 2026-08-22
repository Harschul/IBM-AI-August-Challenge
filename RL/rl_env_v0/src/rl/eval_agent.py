#!/usr/bin/env python3
"""Evaluate trained RL agent against masked-random baseline."""

import random
import numpy as np
from src.rl.env import RoutingEnv
from sb3_contrib import MaskablePPO

def run_episode(env, policy, rng):
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

def run_batch(policy, n_episodes=300, start_seed=999):
    env = RoutingEnv(horizon_s=1800.0)
    rng = random.Random(start_seed)  # use a different seed range from training
    events = []
    rewards = []
    for ep in range(n_episodes):
        r, ev = run_episode(env, policy, rng)
        rewards.append(r)
        events.append(ev)
    return rewards, events

def summarize(events, label):
    n = len(events)
    delivered = sum(1 for e in events if e.startswith("delivered"))
    on_time = sum(1 for e in events if e == "delivered_on_time")
    invalid = sum(1 for e in events if e == "invalid_action")
    print(f"{label:>15}: delivered={delivered/n:.0%}  on_time={on_time/n:.0%}  invalid_action={invalid/n:.0%}")

if __name__ == "__main__":
    # Load the trained model
    model = MaskablePPO.load("models/rl_agent_masked.zip")

    print("Running 300 episodes for each policy (seeds 999..1298)...\n")
    random_r, random_e = run_batch("masked_random", 300, start_seed=999)
    rl_r, rl_e = run_batch("rl_agent", 300, start_seed=999)

    summarize(random_e, "Masked Random")
    summarize(rl_e, "RL Agent")
    print("\n✅ Done. Compare the percentages above.")