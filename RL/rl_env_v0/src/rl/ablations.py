#!/usr/bin/env python3
"""Ablation study for section 11.4 ("perform at least two ablations").

Trains three variants of the same MaskablePPO setup that differ ONLY in
what the agent is allowed to observe:

    full             -- baseline, full observation (matches train_multiseed.py)
    no_priority      -- science_priority is zeroed out in the observation
    no_weather_health -- weather_risk and health are zeroed out in the observation

In every variant the environment's true reward and dynamics are unchanged
(see RoutingEnv's `ablation` docstring in env.py) -- only the agent's view
is restricted. Each variant is trained on `seeds` and evaluated on a
disjoint held-out seed range, same convention as eval_all.py, so ablation
results are directly comparable to the seed-42/7/123 full-observation runs
you already have.

Usage:
    python -m src.rl.ablations                 # default: seeds 42/7/123, 200k steps each
    python -m src.rl.ablations --timesteps 20000  # quick smoke test

Writes results/ablation_results.csv (one row per variant x seed) and prints
a summary table comparing delivered / on_time / invalid rates across
variants, so you can report "does performance change as expected" directly.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from src.rl.env import RoutingEnv

VARIANTS = ["full", "no_priority", "no_weather_health"]
SEEDS = [42, 7, 123]
HELD_OUT_START_SEED = 999
HELD_OUT_EPISODES = 300
MAX_STEPS_PER_EPISODE = 20


def _ablation_kwarg(variant: str):
    return None if variant == "full" else variant


def train_variant(variant: str, seed: int, total_timesteps: int):
    ablation = _ablation_kwarg(variant)
    env = RoutingEnv(horizon_s=1800.0, seed=seed, ablation=ablation)
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.2,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)

    os.makedirs("models/ablations", exist_ok=True)
    path = f"models/ablations/rl_agent_{variant}_seed_{seed}.zip"
    model.save(path)
    return model, path


def run_episode(env, model, rng):
    obs, info = env.reset(seed=rng.randint(0, 10**9))
    total_reward = 0.0
    for _ in range(MAX_STEPS_PER_EPISODE):
        mask = info["action_mask"]
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return total_reward, info.get("event", "unknown")
    return total_reward, "no_terminal"


def evaluate_variant(variant: str, model):
    # NOTE: eval env always uses the FULL true dynamics/reward, same convention
    # as env.py's ablation docstring -- we're only restricting what the model
    # was trained to observe, not changing what "delivered on time" means.
    ablation = _ablation_kwarg(variant)
    env = RoutingEnv(horizon_s=1800.0, ablation=ablation)
    rng = __import__("random").Random(HELD_OUT_START_SEED)
    rewards, events = [], []
    for _ in range(HELD_OUT_EPISODES):
        r, ev = run_episode(env, model, rng)
        rewards.append(r)
        events.append(ev)
    n = len(events)
    return {
        "variant": variant,
        "delivered": sum(1 for e in events if e.startswith("delivered")) / n,
        "on_time": sum(1 for e in events if e == "delivered_on_time") / n,
        "invalid": sum(1 for e in events if e == "invalid_action") / n,
        "mean_reward": float(np.mean(rewards)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    rows = []

    for variant in VARIANTS:
        print("=" * 60)
        print(f"Variant: {variant}")
        print("=" * 60)
        for seed in args.seeds:
            model, path = train_variant(variant, seed, args.timesteps)
            metrics = evaluate_variant(variant, model)
            metrics["seed"] = seed
            rows.append(metrics)
            print(
                f"  seed={seed:>4}  delivered={metrics['delivered']:.0%}  "
                f"on_time={metrics['on_time']:.0%}  invalid={metrics['invalid']:.0%}  "
                f"mean_reward={metrics['mean_reward']:.1f}"
            )

    csv_path = "results/ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "seed", "delivered", "on_time", "invalid", "mean_reward"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    print("\n" + "=" * 60)
    print("SUMMARY (mean over seeds, held-out seeds 999.. )")
    print("=" * 60)
    for variant in VARIANTS:
        vrows = [r for r in rows if r["variant"] == variant]
        d = np.mean([r["delivered"] for r in vrows])
        o = np.mean([r["on_time"] for r in vrows])
        i = np.mean([r["invalid"] for r in vrows])
        mr = np.mean([r["mean_reward"] for r in vrows])
        print(f"{variant:>18}: delivered={d:.0%}  on_time={o:.0%}  invalid={i:.0%}  mean_reward={mr:.1f}")
    print("\nExpectation to report against: no_priority and no_weather_health should")
    print("both under-perform 'full' on mean_reward and/or on_time, since the agent")
    print("can no longer see signals that the reward now genuinely depends on.")


if __name__ == "__main__":
    main()