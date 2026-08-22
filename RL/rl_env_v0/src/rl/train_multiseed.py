#!/usr/bin/env python3
"""Train MaskablePPO agents with multiple seeds for reproducibility."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from src.rl.env import RoutingEnv

def train(seed=42, total_timesteps=200_000):
    print(f"\n🚀 Training with seed={seed}...")
    env = RoutingEnv(horizon_s=1800.0, seed=seed)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=0,  # Reduce output for multiple runs
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.2,
        seed=seed,  # Set the seed for reproducibility
    )

    model.learn(total_timesteps=total_timesteps)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/rl_agent_seed_{seed}.zip"
    model.save(model_path)
    print(f"✅ Saved to {model_path}")

    # Quick test episode
    obs, info = env.reset()
    total_reward = 0
    for _ in range(20):
        mask = info["action_mask"]
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"🎯 Test reward: {total_reward:.1f}")
    return model_path

if __name__ == "__main__":
    # Train on 3 different seeds
    seeds = [42, 7, 123]
    print("=" * 50)
    print(f"Training {len(seeds)} agents on seeds: {seeds}")
    print("=" * 50)

    for s in seeds:
        train(seed=s, total_timesteps=200_000)

    print("\n" + "=" * 50)
    print("✅ All agents trained! Models saved to models/")
    print("   - models/rl_agent_seed_42.zip")
    print("   - models/rl_agent_seed_7.zip")
    print("   - models/rl_agent_seed_123.zip")
    print("=" * 50)