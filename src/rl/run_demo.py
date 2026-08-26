"""Today's deliverable: prove the RL environment interface works.

Runs two sanity policies over many episodes on the mocked dynamic graph:

  - unmasked random  : picks any of the 14 node IDs uniformly at random
                        (this is the "random valid next hop" sanity floor
                        FROM THE OUTSIDE, i.e. it doesn't even know the mask)
  - masked random     : picks uniformly among only the currently-valid
                        (masked) next hops

This is not the RL agent yet -- it is the acceptance test for the frozen
API: fixed 158-d observation, Discrete(14) action, working action_masks().
Once this is green, Jiwoo's baseline and the actual MaskablePPO agent both
plug into the exact same env unmodified.
"""

import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.rl.env import RoutingEnv, OBS_LEN


def run_episode(env: RoutingEnv, masked: bool, rng: random.Random):
    obs, info = env.reset(seed=rng.randint(0, 10**9))
    assert obs.shape == (OBS_LEN,), f"observation shape drifted: {obs.shape}"
    total_reward = 0.0
    for _ in range(20):
        mask = info["action_mask"]
        valid_ids = [i for i, v in enumerate(mask) if v]
        if masked:
            action = rng.choice(valid_ids) if valid_ids else rng.randrange(14)
        else:
            action = rng.randrange(14)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return total_reward, info.get("event", "unknown")
    return total_reward, "no_terminal"


def run_batch(n_episodes: int, masked: bool, seed: int):
    env = RoutingEnv(horizon_s=1800.0)
    rng = random.Random(seed)
    events = []
    rewards = []
    for _ in range(n_episodes):
        r, event = run_episode(env, masked, rng)
        rewards.append(r)
        events.append(event)
    return rewards, events


def summarize(events, label):
    n = len(events)
    delivered = sum(1 for e in events if e.startswith("delivered"))
    on_time = sum(1 for e in events if e == "delivered_on_time")
    invalid = sum(1 for e in events if e == "invalid_action")
    print(f"{label:>15}: delivered={delivered/n:.0%}  on_time={on_time/n:.0%}  invalid_action={invalid/n:.0%}  n={n}")
    return delivered / n, on_time / n, invalid / n


if __name__ == "__main__":
    N = 300
    print("Fixed observation length:", OBS_LEN, "| Fixed action space: Discrete(14)\n")

    unmasked_r, unmasked_e = run_batch(N, masked=False, seed=1)
    masked_r, masked_e = run_batch(N, masked=True, seed=1)

    u_delivered, u_ontime, u_invalid = summarize(unmasked_e, "unmasked random")
    m_delivered, m_ontime, m_invalid = summarize(masked_e, "masked random")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    labels = ["Delivered", "On-time"]
    unmasked_vals = [u_delivered, u_ontime]
    masked_vals = [m_delivered, m_ontime]
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, unmasked_vals, width, label="Unmasked random")
    axes[0].bar(x + width / 2, masked_vals, width, label="Masked random")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of episodes")
    axes[0].set_title("Delivery outcomes (N=300 mocked episodes)")
    axes[0].legend()

    axes[1].bar(["Unmasked random", "Masked random"], [u_invalid, m_invalid], color=["#d9534f", "#5cb85c"])
    axes[1].set_ylabel("Invalid-action rate")
    axes[1].set_title("Action-mask sanity check")

    fig.suptitle("RoutingEnv API check -- 20 Aug interface freeze", fontsize=11)
    fig.tight_layout()
    fig.savefig("rl_env_sanity_check.png", dpi=150)
    print("\nSaved chart to /mnt/user-data/outputs/rl_env_sanity_check.png")
