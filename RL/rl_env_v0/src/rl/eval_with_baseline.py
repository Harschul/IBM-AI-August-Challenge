#!/usr/bin/env python3
"""RL agents vs the temporal earliest-arrival baseline (PDF sections 10.2, 10.4, 10.5).

`eval_all.py` compares the trained agents against masked-random. Section 10.2
calls masked-random the *sanity floor* and names the temporal earliest-arrival /
CGR router the *primary* non-AI benchmark. This script adds that benchmark, the
RL+fallback policy section 10.2 recommends for the demo, and the statistical
discipline section 10.5 requires.

All four policies from section 10.2 are covered:

    masked_random        sanity floor
    baseline_*           primary non-AI benchmark
    rl_seed_*            the mandatory AI contribution
    rl_fallback_*        recommended deployed/demo policy

TWO BASELINES, ON PURPOSE
-------------------------
`rl_env_v0` currently lets a transfer run past the end of its contact window
(issue #4, ~61% of hops) and applies no propagation delay:

  baseline_env_rules   plans under the environment's ACTUAL dynamics -- window
                       overrun permitted. The fair head-to-head: agent and
                       router play the same game.
  baseline_physical    plans under REAL physics -- the transfer must fit inside
                       its contact window. The gap between the rows measures
                       what the open bug is worth.

TWO FALLBACKS, ALSO ON PURPOSE
------------------------------
Section 9.2: "If invalid/unsafe/unavailable, call temporal baseline rather than
drop the bundle." That leaves "unsafe" open to interpretation, so both readings
are measured:

  rl_fallback_invalid  literal: substitute only when the agent picks a move the
                       mask forbids. Masking makes that nearly impossible, so
                       this should score the same as the bare agent -- reported
                       anyway, because "the safety net never fires" is a result,
                       not an omission.
  rl_fallback_unsafe   treats physically impossible as unsafe: if the agent's
                       chosen hop cannot complete inside its contact window, the
                       baseline's choice is substituted. Fires often while issue
                       #4 is open.

STATISTICS (section 10.5)
-------------------------
Runs REPLICATES independent groups of episodes and reports mean +/- standard
deviation across groups rather than one number from one run. Every policy sees
identical scenarios within each replicate.

Results are written to results/ as CSV (one row per bundle, per appendix B5) and
JSON (aggregates plus git commit, checkpoint SHA-256s and configuration), so any
number in the pitch can be traced to the exact code and weights behind it.

Run from `RL/rl_env_v0/`:

    python3 src/rl/eval_with_baseline.py
    python3 src/rl/eval_with_baseline.py --replicates 20 --episodes 25
"""

import argparse
import csv
import hashlib
import heapq
import json
import os
import random
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np  # noqa: E402
from gymnasium import spaces  # noqa: E402

from src.rl.env import OBS_LEN, RoutingEnv  # noqa: E402
from src.rl.mock_graph import GROUND_IDS, NUM_NODES  # noqa: E402

URGENT_PRIORITY = 0.8
HELD_OUT_START = 999          # same held-out seeds eval_all.py uses
REPLICATE_STRIDE = 10_000     # keeps replicate seed ranges disjoint
MAX_STEPS = 20
MODEL_SEEDS = (7, 42, 123)


def _numpy2_compat():
    """Let numpy-1.26 load checkpoints pickled under numpy 2.x.

    The saved models reference `numpy._core.*`, numpy 2's internal layout;
    numpy 1.26 calls the same modules `numpy.core.*`. numpy 2 needs Python
    3.10+, so alias rather than chase an interpreter upgrade.
    """
    import numpy.core  # noqa: F401

    for suffix in ("", ".numeric", ".multiarray", ".umath", ".numerictypes",
                   ".overrides", "._multiarray_umath"):
        new = "numpy._core" + suffix
        if new in sys.modules:
            continue
        try:
            sys.modules[new] = __import__("numpy.core" + suffix, fromlist=["_"])
        except Exception:
            pass


# --------------------------------------------------------------------------
# Temporal earliest-arrival search
#
# Same algorithm as `src/routing/temporal_baseline.py` (merged in #3),
# reimplemented against rl_env_v0's Contact type. The repo has two different
# packages both named `src`, so they cannot be imported into one process.
# Keep the two in sync if either changes.
# --------------------------------------------------------------------------

def earliest_arrival(contacts, source_id, targets, size_bytes, start_s,
                     require_fit):
    """Earliest time `size_bytes` can reach any node in `targets`.

    Returns (first_hop_id, arrival_s) or None. `require_fit=True` enforces the
    section 9.1 condition that the transfer complete before the contact closes;
    False mirrors the environment's current permissiveness.
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
                heapq.heappush(
                    heap,
                    (candidate, counter, nxt,
                     first_hop if first_hop is not None else nxt))

    return None


def hop_is_feasible(env, dest_id):
    """Can the bundle actually complete a transfer to dest_id?

    This is the check `env.step()` is missing (issue #4). Used by
    rl_fallback_unsafe to veto physically impossible agent choices.
    """
    for c in env.contact_plan.contacts:
        if c.source_id != env.bundle.current_holder or c.destination_id != dest_id:
            continue
        if c.end_s < env.t:
            continue
        depart = max(env.t, c.start_s)
        if depart >= c.end_s:
            continue
        if depart + env.bundle.remaining_bytes * 8 / c.data_rate_bps <= c.end_s:
            return True
    return False


def baseline_action(env, mask, require_fit):
    """Next hop the temporal router would take, restricted to masked-legal moves."""
    legal = [i for i, v in enumerate(mask) if v]
    if not legal:
        return None

    result = earliest_arrival(
        env.contact_plan.contacts, env.bundle.current_holder, GROUND_IDS,
        env.bundle.remaining_bytes, env.t, require_fit)
    if result is not None and result[0] in legal:
        return result[0]

    # No route the router will commit to. Still move the data -- late science
    # beats lost science -- via the legal hop that lands soonest.
    def arrival_of(dest):
        best = float("inf")
        for c in env.contact_plan.contacts:
            if c.source_id != env.bundle.current_holder or c.destination_id != dest:
                continue
            if c.end_s < env.t:
                continue
            depart = max(env.t, c.start_s)
            best = min(best, depart + env.bundle.remaining_bytes * 8 / c.data_rate_bps)
        return best

    return min(legal, key=arrival_of)


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------

def choose_action(policy, env, obs, mask, rng, model, stats):
    """One next-hop decision. `stats` accumulates fallback counters."""
    legal = [i for i, v in enumerate(mask) if v]

    if policy == "masked_random":
        return rng.choice(legal) if legal else rng.randrange(NUM_NODES)

    if policy.startswith("baseline"):
        return baseline_action(env, mask, policy == "baseline_physical")

    action, _ = model.predict(obs, deterministic=True, action_masks=mask)
    action = int(action)
    stats["decisions"] += 1

    if policy == "rl_agent":
        return action

    if policy == "rl_fallback_invalid":
        if mask[action]:
            return action
        stats["fallbacks"] += 1
        return baseline_action(env, mask, False)

    if policy == "rl_fallback_unsafe":
        if mask[action] and hop_is_feasible(env, action):
            return action
        stats["fallbacks"] += 1
        # Physically-correct baseline: the point of vetoing is to be safe.
        return baseline_action(env, mask, True)

    raise ValueError("unknown policy %r" % policy)


def run_episode(env, policy, rng, model, stats, record_path=False):
    obs, info = env.reset(seed=rng.randint(0, 10**9))
    priority = env.bundle.science_priority
    total_reward = 0.0
    hops = 0
    path = [env.bundle.current_holder]

    for _ in range(MAX_STEPS):
        mask = info["action_mask"]
        action = choose_action(policy, env, obs, mask, rng, model, stats)
        if action is None:
            break

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        hops += 1
        path.append(int(action))

        if terminated or truncated:
            event = info.get("event", "unknown")
            return {
                "event": event, "reward": total_reward, "priority": priority,
                "latency_s": env.t, "hops": hops,
                "delivered": event.startswith("delivered"),
                "on_time": event == "delivered_on_time",
                "path": "-".join(str(p) for p in path),
            }

    return {"event": "no_terminal", "reward": total_reward, "priority": priority,
            "latency_s": env.t, "hops": hops, "delivered": False,
            "on_time": False, "path": "-".join(str(p) for p in path)}


def run_batch(policy, episodes, base_seed, model=None):
    env = RoutingEnv(horizon_s=1800.0)
    rng = random.Random(base_seed)     # identical scenarios for every policy
    stats = {"decisions": 0, "fallbacks": 0}
    rows = [run_episode(env, policy, rng, model, stats) for _ in range(episodes)]
    return rows, stats


# --------------------------------------------------------------------------
# Metrics (section 10.4)
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
        "mean_hops": statistics.fmean([r["hops"] for r in rows]),
        "mean_reward": statistics.fmean([r["reward"] for r in rows]),
    }


def mean_std(values):
    clean = [v for v in values if v == v]      # drop NaN
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.fmean(clean), statistics.stdev(clean)


METRICS = [
    ("priority_weighted_timely", "Priority-weighted timely delivery", "{:.3f}", "PRIMARY"),
    ("urgent_p95_latency_s", "Urgent p95 latency (s)", "{:.0f}", "PRIMARY"),
    ("deadline_success", "Deadline success rate", "{:.3f}", "PRIMARY"),
    ("delivery_ratio", "Delivery ratio", "{:.3f}", "core"),
    ("mean_latency_s", "Mean latency (s)", "{:.0f}", "core"),
    ("mean_hops", "Mean hops", "{:.2f}", "core"),
    ("mean_reward", "Mean episode reward", "{:.1f}", "diag"),
]


def render_table(summary):
    names = list(summary)
    label_w = max(len(m[1]) for m in METRICS) + 2
    col = max(20, max(len(n) for n in names) + 2)
    width = label_w + col * len(names) + 10

    out = ["=" * width]
    out.append("metric".ljust(label_w) + "".join(n.rjust(col) for n in names)
               + "     priority")
    out.append("-" * width)
    for key, label, fmt, tag in METRICS:
        line = label.ljust(label_w)
        for name in names:
            mean, std = summary[name][key]
            line += (fmt.format(mean) + " ±" + fmt.format(std)).rjust(col)
        out.append(line + "     " + tag)
    out.append("=" * width)
    return "\n".join(out)


# --------------------------------------------------------------------------
# Provenance (section 10.5)
# --------------------------------------------------------------------------

def sha256_of(path, chars=16):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:chars]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=20,
                        help="independent seed groups (section 10.5 wants >= 20)")
    parser.add_argument("--episodes", type=int, default=25,
                        help="episodes per replicate")
    parser.add_argument("--models", default="models")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    _numpy2_compat()
    from sb3_contrib import MaskablePPO

    space_overrides = {
        # The pickled Spaces carry a numpy Generator that will not cross the
        # numpy 2 -> 1.26 boundary. The interface is frozen anyway, so supply
        # them directly rather than unpickle.
        "observation_space": spaces.Box(low=-1.0, high=1.0,
                                        shape=(OBS_LEN,), dtype=np.float32),
        "action_space": spaces.Discrete(NUM_NODES),
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    models, checkpoints = {}, {}
    for seed in MODEL_SEEDS:
        path = os.path.join(args.models, "rl_agent_seed_%d.zip" % seed)
        if os.path.exists(path):
            models[seed] = MaskablePPO.load(path, device="cpu",
                                            custom_objects=space_overrides)
            checkpoints[os.path.basename(path)] = sha256_of(path)
        else:
            print("  (missing %s -- skipped)" % path)

    policies = ["masked_random", "baseline_env_rules", "baseline_physical"]
    policies += ["rl_seed_%d" % s for s in models]
    primary = next(iter(models), None)
    if primary is not None:
        # Fallback variants use the first checkpoint, so the only difference
        # from that agent's own row is the fallback itself.
        policies += ["rl_fallback_invalid", "rl_fallback_unsafe"]

    print("Policies: %d   replicates: %d x %d episodes = %d per policy\n"
          % (len(policies), args.replicates, args.episodes,
             args.replicates * args.episodes))

    per_replicate = {p: [] for p in policies}
    all_rows = []
    fallback_stats = {}

    for r in range(args.replicates):
        base_seed = HELD_OUT_START + r * REPLICATE_STRIDE
        print("  replicate %2d/%d (seed %d)" % (r + 1, args.replicates, base_seed),
              flush=True)

        for policy in policies:
            if policy.startswith("rl_seed_"):
                model = models[int(policy.rsplit("_", 1)[1])]
                kind = "rl_agent"
            elif policy.startswith("rl_fallback"):
                model = models[primary]
                kind = policy
            else:
                model, kind = None, policy

            rows, stats = run_batch(kind, args.episodes, base_seed, model)
            per_replicate[policy].append(aggregate(rows))

            if stats["decisions"]:
                acc = fallback_stats.setdefault(
                    policy, {"decisions": 0, "fallbacks": 0})
                acc["decisions"] += stats["decisions"]
                acc["fallbacks"] += stats["fallbacks"]

            for i, row in enumerate(rows):
                all_rows.append({"policy": policy, "replicate": r,
                                 "episode": i, **row})

    summary = {
        policy: {key: mean_std([rep[key] for rep in reps])
                 for key, _, _, _ in METRICS}
        for policy, reps in per_replicate.items()
    }

    print("\n" + render_table(summary))
    print("\nmean ± standard deviation across %d independent replicates "
          "(section 10.5)" % args.replicates)

    fb = {p: a for p, a in fallback_stats.items() if p.startswith("rl_fallback")}
    if fb:
        print("\nFallback activity:")
        for policy, acc in fb.items():
            print("  %-22s %d/%d decisions overridden (%.1f%%)"
                  % (policy, acc["fallbacks"], acc["decisions"],
                     100.0 * acc["fallbacks"] / max(1, acc["decisions"])))

    # --- freeze the results (section 10.5) ---------------------------------
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "baseline_vs_rl_episodes.csv")
    json_path = os.path.join(args.out, "baseline_vs_rl_summary.json")

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    with open(json_path, "w") as fh:
        json.dump({
            "provenance": {
                "git_commit": git_commit(),
                "checkpoints_sha256": checkpoints,
                "replicates": args.replicates,
                "episodes_per_replicate": args.episodes,
                "held_out_seed_start": HELD_OUT_START,
                "replicate_stride": REPLICATE_STRIDE,
                "horizon_s": 1800.0,
                # Detected, not asserted. This started life as a hardcoded
                # string and silently went stale the moment #8 merged -- it
                # then claimed the bug was open on runs where it was fixed.
                # A wrong label on correct numbers is worse than no label.
                "contact_window_fix_present": hasattr(
                    RoutingEnv, "_feasible_contact_to"),
                "note": ("contact-window fix (issue #4) IS present"
                         if hasattr(RoutingEnv, "_feasible_contact_to")
                         else "contact-window bug (issue #4) OPEN at time of run"),
            },
            "summary": {
                policy: {key: {"mean": m, "std": s}
                         for key, (m, s) in metrics.items()}
                for policy, metrics in summary.items()
            },
            "fallback_activity": fallback_stats,
        }, fh, indent=2)

    print("\nWrote %s (%d rows)\n      %s" % (csv_path, len(all_rows), json_path))


if __name__ == "__main__":
    main()
