#!/usr/bin/env python3
"""RL agents vs the temporal earliest-arrival baseline (PDF sections 10.2, 10.4).

`eval_all.py` compares the trained agents against masked-random. Section 10.2
calls masked-random the *sanity floor* and names the temporal earliest-arrival /
CGR router the *primary* non-AI benchmark. This script adds that benchmark, so
the headline claim is "the agent beats a real router" rather than "the agent
beats random".

Every policy sees identical scenarios from identical seeds (section 10.5), so
any difference is attributable to the policy.

TWO BASELINES, ON PURPOSE
-------------------------
`rl_env_v0` currently lets a transfer run past the end of its contact window
(issue #4, ~61% of hops) and applies no propagation delay. That leaves two
different questions, and they deserve different rows:

  baseline_env_rules   plans under the environment's ACTUAL dynamics -- window
                       overrun allowed, no propagation delay. This is the fair
                       head-to-head: the agent and the router are playing the
                       same game, exploiting the same permissiveness.

  baseline_physical    plans under REAL physics -- a transfer must fit inside
                       its contact window. This is what a correct router does,
                       and the gap between the two rows is a direct measure of
                       how much the open bug is worth.

Reporting only the first would hide the bug's effect; reporting only the second
would penalise the agent for an environment property it did not choose.

Run from `RL/rl_env_v0/`:

    python3 src/rl/eval_with_baseline.py
"""

import argparse
import heapq
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np  # noqa: E402
from gymnasium import spaces  # noqa: E402

from src.rl.env import OBS_LEN, RoutingEnv  # noqa: E402
from src.rl.mock_graph import GROUND_IDS, NUM_NODES  # noqa: E402

def _numpy2_compat():
    """Let numpy-1.26 load checkpoints pickled under numpy 2.x.

    The saved models reference `numpy._core.*`, which is numpy 2's internal
    layout; numpy 1.26 calls the same modules `numpy.core.*`. numpy 2 needs
    Python 3.10+, and this machine is on 3.9.6, so alias the old names to the
    new ones rather than chase an interpreter upgrade. Arrays themselves are
    format-compatible; only the module PATH moved.
    """
    import numpy.core

    aliases = ["", ".numeric", ".multiarray", ".umath", ".numerictypes",
               ".overrides", "._multiarray_umath"]
    for suffix in aliases:
        old = "numpy.core" + suffix
        new = "numpy._core" + suffix
        if new in sys.modules:
            continue
        try:
            sys.modules[new] = __import__(old, fromlist=["_"])
        except Exception:
            pass


URGENT_PRIORITY = 0.8
HELD_OUT_START = 999          # same held-out seeds eval_all.py uses
DEFAULT_EPISODES = 300
MAX_STEPS = 20


# --------------------------------------------------------------------------
# Temporal earliest-arrival search
#
# Same algorithm as `src/routing/temporal_baseline.py` (merged in #3),
# reimplemented here against rl_env_v0's Contact type. The repo has two
# different packages both named `src` (repo root, and RL/rl_env_v0), so they
# cannot be imported into the same process -- hence the duplication rather than
# an import. Keep the two in sync if either changes.
# --------------------------------------------------------------------------

def earliest_arrival(contacts, source_id, targets, size_bytes, start_s,
                     require_fit):
    """Earliest time `size_bytes` can reach any node in `targets`.

    Returns (next_hop_id, arrival_s), or None if no route exists.

    `require_fit=True` enforces the section 9.1 condition that the transfer
    complete before the contact closes. `require_fit=False` mirrors the
    environment's current behaviour, where it need not.
    """
    targets = set(targets)
    if source_id in targets:
        return None

    by_source = {}
    for c in contacts:
        by_source.setdefault(c.source_id, []).append(c)

    best = {source_id: start_s}
    counter = 0
    heap = [(start_s, counter, source_id, None)]

    while heap:
        arrival_s, _, node, first_hop = heapq.heappop(heap)
        if arrival_s > best.get(node, float("inf")):
            continue
        if node in targets:
            return first_hop, arrival_s

        for contact in by_source.get(node, ()):
            if contact.end_s < arrival_s:
                continue
            depart = max(arrival_s, contact.start_s)
            if depart >= contact.end_s:
                continue

            tx_s = size_bytes * 8 / contact.data_rate_bps
            if require_fit and depart + tx_s > contact.end_s:
                continue

            candidate = depart + tx_s          # env adds no propagation delay
            nxt = contact.destination_id
            if candidate < best.get(nxt, float("inf")):
                best[nxt] = candidate
                counter += 1
                hop = first_hop if first_hop is not None else nxt
                heapq.heappush(heap, (candidate, counter, nxt, hop))

    return None


def baseline_action(env, mask, require_fit):
    """Next hop the temporal router would take, restricted to masked-legal moves."""
    legal = [i for i, v in enumerate(mask) if v]
    if not legal:
        return None

    result = earliest_arrival(
        env.contact_plan.contacts,
        env.bundle.current_holder,
        GROUND_IDS,
        env.bundle.remaining_bytes,
        env.t,
        require_fit,
    )
    if result is not None and result[0] in legal:
        return result[0]

    # No route the router will commit to. Still move the data -- late science
    # beats lost science -- by taking the legal hop that lands soonest.
    def arrival_of(dest):
        best = float("inf")
        for c in env.contact_plan.contacts:
            if c.source_id != env.bundle.current_holder or c.destination_id != dest:
                continue
            if c.end_s < env.t:
                continue
            depart = max(env.t, c.start_s)
            tx_s = env.bundle.remaining_bytes * 8 / c.data_rate_bps
            best = min(best, depart + tx_s)
        return best

    return min(legal, key=arrival_of)


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------

def run_episode(env, policy, rng, model=None):
    obs, info = env.reset(seed=rng.randint(0, 10**9))
    priority = env.bundle.science_priority
    deadline = env.bundle.deadline_s
    total_reward = 0.0

    for _ in range(MAX_STEPS):
        mask = info["action_mask"]

        if policy == "masked_random":
            legal = [i for i, v in enumerate(mask) if v]
            action = rng.choice(legal) if legal else rng.randrange(NUM_NODES)
        elif policy == "rl_agent":
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            action = int(action)
        elif policy in ("baseline_env_rules", "baseline_physical"):
            action = baseline_action(env, mask, policy == "baseline_physical")
            if action is None:
                break
        else:
            raise ValueError("unknown policy %r" % policy)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            event = info.get("event", "unknown")
            return {
                "event": event,
                "reward": total_reward,
                "priority": priority,
                "latency_s": env.t,
                "delivered": event.startswith("delivered"),
                "on_time": event == "delivered_on_time",
                "deadline_s": deadline,
            }

    return {"event": "no_terminal", "reward": total_reward, "priority": priority,
            "latency_s": env.t, "delivered": False, "on_time": False,
            "deadline_s": deadline}


def run_batch(policy, episodes, model=None, seed=HELD_OUT_START):
    env = RoutingEnv(horizon_s=1800.0)
    rng = random.Random(seed)          # identical scenarios for every policy
    return [run_episode(env, policy, rng, model) for _ in range(episodes)]


# --------------------------------------------------------------------------
# Metrics (PDF section 10.4)
# --------------------------------------------------------------------------

def _percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[k]


def aggregate(rows):
    n = len(rows)
    total_priority = sum(r["priority"] for r in rows)
    on_time_priority = sum(r["priority"] for r in rows if r["on_time"])
    delivered = [r for r in rows if r["delivered"]]
    urgent = [r for r in delivered if r["priority"] >= URGENT_PRIORITY]

    return {
        "priority_weighted_timely": on_time_priority / total_priority if total_priority else 0.0,
        "urgent_p95_latency_s": _percentile([r["latency_s"] for r in urgent], 0.95),
        "deadline_success": sum(1 for r in rows if r["on_time"]) / n,
        "delivery_ratio": len(delivered) / n,
        "mean_latency_s": statistics.fmean([r["latency_s"] for r in delivered]) if delivered else float("nan"),
        "mean_reward": statistics.fmean([r["reward"] for r in rows]),
        "episodes": n,
    }


ROWS = [
    ("priority_weighted_timely", "Priority-weighted timely delivery", "{:.3f}", "PRIMARY"),
    ("urgent_p95_latency_s", "Urgent p95 latency (s)", "{:.1f}", "PRIMARY"),
    ("deadline_success", "Deadline success rate", "{:.3f}", "PRIMARY"),
    ("delivery_ratio", "Delivery ratio", "{:.3f}", "core"),
    ("mean_latency_s", "Mean latency (s)", "{:.1f}", "core"),
    ("mean_reward", "Mean episode reward", "{:.1f}", "diag"),
]


def render_table(results):
    names = list(results)
    label_w = max(len(r[1]) for r in ROWS) + 2
    col = max(20, max(len(n) for n in names) + 2)

    out = ["=" * (label_w + col * len(names) + 10)]
    out.append("metric".ljust(label_w) + "".join(n.rjust(col) for n in names) + "     priority")
    out.append("-" * (label_w + col * len(names) + 10))
    for key, label, fmt, tag in ROWS:
        line = label.ljust(label_w)
        for name in names:
            line += fmt.format(results[name][key]).rjust(col)
        out.append(line + "     " + tag)
    out.append("=" * (label_w + col * len(names) + 10))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--models", default="models")
    args = parser.parse_args()

    _numpy2_compat()
    from sb3_contrib import MaskablePPO

    results = {}

    print("Running %d held-out episodes per policy (seeds from %d)...\n"
          % (args.episodes, HELD_OUT_START))

    for policy in ("masked_random", "baseline_env_rules", "baseline_physical"):
        print("  %s..." % policy, flush=True)
        results[policy] = aggregate(run_batch(policy, args.episodes))

    for seed in (7, 42, 123):
        path = os.path.join(args.models, "rl_agent_seed_%d.zip" % seed)
        if not os.path.exists(path):
            print("  (missing %s -- skipped)" % path)
            continue
        print("  rl_agent_seed_%d..." % seed, flush=True)
        # The pickled Space objects carry a numpy Generator, which will not
        # cross the numpy 2 -> 1.26 boundary. Supply the spaces directly --
        # they are fixed by the frozen interface anyway (158 / Discrete(14)) --
        # so sb3 never has to unpickle them.
        model = MaskablePPO.load(path, device="cpu", custom_objects={
            "observation_space": spaces.Box(low=-1.0, high=1.0,
                                            shape=(OBS_LEN,), dtype=np.float32),
            "action_space": spaces.Discrete(NUM_NODES),
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        })
        results["rl_seed_%d" % seed] = aggregate(
            run_batch("rl_agent", args.episodes, model=model))

    print("\n" + render_table(results))
    print("""
baseline_env_rules  plans under the environment's current dynamics (contact-window
                    overrun permitted, issue #4). Fair head-to-head with the agent.
baseline_physical   plans under real physics (transfer must fit in its window).
                    The gap between the two rows measures what the open bug is worth.
""")


if __name__ == "__main__":
    main()
