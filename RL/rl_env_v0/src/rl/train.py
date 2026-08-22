#!/usr/bin/env python3
"""Train a MaskablePPO agent on the frozen RoutingEnv.

Usage:
    python -m src.rl.train

This will train for 200,000 timesteps and save the model to
models/rl_agent_masked.zip. Increase total_timesteps for better performance.
"""

import os
import sys

# Add project root to path if running from src/rl/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from src.rl.env import RoutingEnv  # Your frozen environment


def main():
    print("🚀 Initializing RoutingEnv...")
    env = RoutingEnv(horizon_s=1800.0, seed=42)

    print("🧠 Creating MaskablePPO agent...")
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,                     # Print training progress
        learning_rate=3e-4,
        n_steps=2048,                  # Steps per rollout
        batch_size=64,
        n_epochs=10,                   # Epochs per rollout
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.2,
        tensorboard_log="./logs/",     # Optional: view with tensorboard
    )

    print("⚡ Training for 200,000 timesteps...")
    model.learn(total_timesteps=200_000)

    os.makedirs("models", exist_ok=True)
    model.save("models/rl_agent_masked.zip")
    print("✅ Model saved to models/rl_agent_masked.zip")

    # Quick test: run one episode to show it works
    obs, info = env.reset()
    total_reward = 0
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"🎯 Test episode reward: {total_reward:.1f}")
    print("✅ Training complete!")


if __name__ == "__main__":
    main()