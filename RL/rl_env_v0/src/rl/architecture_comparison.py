#!/usr/bin/env python3
"""Architecture comparison A / B / C / D (PDF section 10.1).

The section 2 acceptance table lists this as a minimum criterion:

    Current architecture benchmark -- "Direct-to-ground and two-GEO-relay
    scenarios run using the same traffic and seeds."
    LEO/hybrid architecture -- "Eight LEO relays are selectable alongside GEO
    and ground paths."

Everything else measured so far is hybrid-only, so the project's central claim --
that a hybrid network beats direct-only and GEO-only -- has had no evidence
behind it. This produces that evidence.

    A  direct only   science satellite -> ground station
    B  GEO only      science satellite -> one of 2 GEO relays -> ground
    C  LEO mesh      science satellite -> one or more of 8 LEO relays -> ground
    D  hybrid        direct, LEO and GEO all available; the router selects

An architecture is expressed as the set of nodes a bundle may traverse. The
restriction is applied in two places at once, and it must be both:

    the action mask   so no policy can route through an excluded node
    the contact list  so the router PLANS within the architecture rather than
                      planning a hybrid route and then having it vetoed

Applying only the first would measure "hybrid router with its hands tied", which
is a different and much less interesting question.

Run from `RL/rl_env_v0/`:

    python3 src/rl/architecture_comparison.py
    python3 src/rl/architecture_comparison.py --replicates 20 --episodes 25
"""

import argparse
import json
import os
import random
import itertools
import math
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rl.env import RoutingEnv  # noqa: E402
from src.rl.mock_graph import (  # noqa: E402
    GEO_IDS, GROUND_IDS, LEO_IDS, NUM_NODES, SCIENCE_ID,
)
from src.rl.eval_with_baseline import (  # noqa: E402
    HELD_OUT_START, METRICS, MAX_STEPS, REPLICATE_STRIDE,
    aggregate, earliest_arrival, git_commit, mean_std, render_table,
)

# Node sets a bundle may traverse. The science satellite is always included --
# it is where the data starts, not a routing choice.
ARCHITECTURES = {
    "A_direct_only": set(GROUND_IDS),
    "B_geo_only": set(GEO_IDS) | set(GROUND_IDS),
    "C_leo_mesh": set(LEO_IDS) | set(GROUND_IDS),
    "D_hybrid": set(range(NUM_NODES)),
}


def topology_supports(env, allowed):
    """Does the contact plan contain any first hop this architecture could use?

    `mock_graph._allowed_pairs()` generates SCI->LEO, SCI->GEO, LEO->LEO,
    LEO->GEO, LEO->ground and GEO->ground. It does NOT generate SCI->ground, so
    architecture A (direct only) has no link to route over and scores zero for a
    reason that has nothing to do with direct-to-ground being a poor
    architecture. Reporting that as a result would be misleading, so detect it
    and say so instead.
    """
    return any(c.source_id == SCIENCE_ID and c.destination_id in allowed
               for c in env.contact_plan.contacts)


def allowed_contacts(env, allowed):
    """Contacts that stay inside the architecture."""
    permitted = allowed | {SCIENCE_ID}
    return [c for c in env.contact_plan.contacts
            if c.source_id in permitted and c.destination_id in permitted]


def restricted_mask(mask, allowed):
    return [bool(v) and i in allowed for i, v in enumerate(mask)]


def architecture_action(env, mask, allowed):
    """Temporal earliest-arrival, planning only within this architecture."""
    legal = [i for i, v in enumerate(restricted_mask(mask, allowed)) if v]
    if not legal:
        return None

    contacts = allowed_contacts(env, allowed)
    result = earliest_arrival(
        contacts, env.bundle.current_holder, GROUND_IDS,
        env.bundle.remaining_bytes, env.t, require_fit=True)
    if result is not None and result[0] in legal:
        return result[0]

    def arrival_of(dest):
        best = float("inf")
        for c in contacts:
            if c.source_id != env.bundle.current_holder or c.destination_id != dest:
                continue
            if c.end_s < env.t:
                continue
            depart = max(env.t, c.start_s)
            best = min(best, depart + env.bundle.remaining_bytes * 8 / c.data_rate_bps)
        return best

    return min(legal, key=arrival_of)


def run_episode(env, allowed, rng):
    env.reset(seed=rng.randint(0, 10**9))
    priority = env.bundle.science_priority
    total_reward = 0.0
    hops = 0

    for _ in range(MAX_STEPS):
        action = architecture_action(env, env._action_mask(), allowed)
        if action is None:
            # No path exists inside this architecture. That is a real outcome,
            # not an error -- it is precisely what direct-only looks like
            # between ground passes.
            break

        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        hops += 1
        if terminated or truncated:
            event = info.get("event", "unknown")
            return {"event": event, "reward": total_reward, "priority": priority,
                    "latency_s": env.t, "hops": hops,
                    "delivered": event.startswith("delivered"),
                    "on_time": event == "delivered_on_time"}

    return {"event": "no_path", "reward": total_reward, "priority": priority,
            "latency_s": env.t, "hops": hops, "delivered": False,
            "on_time": False}


def run_batch(allowed, episodes, base_seed):
    env = RoutingEnv(horizon_s=1800.0)
    rng = random.Random(base_seed)   # identical traffic across architectures
    return [run_episode(env, allowed, rng) for _ in range(episodes)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    # Which architectures the environment can actually express?
    probe = RoutingEnv(horizon_s=1800.0)
    probe.reset(seed=HELD_OUT_START)
    evaluable, unsupported = {}, {}
    for name, allowed in ARCHITECTURES.items():
        (evaluable if topology_supports(probe, allowed) else unsupported)[name] = allowed

    if unsupported:
        print("NOT EVALUABLE in this environment -- the contact plan contains no")
        print("first hop these architectures could use:\n")
        for name in unsupported:
            print("  %s" % name)
        print("""
mock_graph._allowed_pairs() generates SCI->LEO, SCI->GEO, LEO->LEO, LEO->GEO,
LEO->ground and GEO->ground. There is no SCI->ground pair, so direct-to-ground
is not modelled at all. These architectures would score zero for a reason that
says nothing about the architecture, so they are excluded rather than reported.

Section 2's acceptance criterion "direct-to-ground and two-GEO-relay scenarios
run using the same traffic and seeds" cannot be fully met until mock_graph adds
the SCI->ground pair.
""")

    print("Architectures: %d evaluable   replicates: %d x %d episodes\n"
          % (len(evaluable), args.replicates, args.episodes))

    per_replicate = {name: [] for name in evaluable}
    no_path = {name: 0 for name in evaluable}
    totals = {name: 0 for name in evaluable}

    for r in range(args.replicates):
        base_seed = HELD_OUT_START + r * REPLICATE_STRIDE
        print("  replicate %2d/%d" % (r + 1, args.replicates), flush=True)
        for name, allowed in evaluable.items():
            rows = run_batch(allowed, args.episodes, base_seed)
            per_replicate[name].append(aggregate(rows))
            no_path[name] += sum(1 for row in rows if row["event"] == "no_path")
            totals[name] += len(rows)

    summary = {
        name: {key: mean_std([rep[key] for rep in reps])
               for key, _, _, _ in METRICS}
        for name, reps in per_replicate.items()
    }

    print("\n" + render_table(summary))
    print("\nmean ± standard deviation across %d replicates, identical traffic "
          "and seeds\nacross architectures (sections 10.1, 10.5). Router is the "
          "temporal earliest-arrival\nbaseline planning within each architecture."
          % args.replicates)

    # --- paired comparisons ------------------------------------------------
    # The replicate standard deviations overlap heavily, but replicates are
    # PAIRED -- every architecture saw the same scenarios -- so the per-replicate
    # difference is far less noisy than either column alone. Comparing the two
    # means without pairing would wrongly call these indistinguishable.
    key = "priority_weighted_timely"
    names = list(evaluable)
    if len(names) > 1:
        crit = 2.093 if args.replicates == 20 else 2.1
        print("\nPaired per-replicate differences in %s" % key)
        print("(same scenarios; |t| > ~%.2f is significant at 95%% for n=%d)\n"
              % (crit, args.replicates))
        for a, b in itertools.combinations(names, 2):
            diffs = [x[key] - y[key]
                     for x, y in zip(per_replicate[a], per_replicate[b])]
            mean = statistics.fmean(diffs)
            se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
            t = mean / se if se else float("inf")
            print("  %-14s vs %-14s  %+.3f ± %.3f   t=%+5.2f   %s"
                  % (a, b, mean, se, t,
                     "SIGNIFICANT" if abs(t) > crit else "not distinguishable"))

    print("\nEpisodes with no path available inside the architecture:")
    for name in evaluable:
        print("  %-16s %4d/%d  (%.0f%%)"
              % (name, no_path[name], totals[name],
                 100.0 * no_path[name] / totals[name]))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "architecture_comparison.json")
    with open(path, "w") as fh:
        json.dump({
            "provenance": {
                "git_commit": git_commit(),
                "replicates": args.replicates,
                "episodes_per_replicate": args.episodes,
                "held_out_seed_start": HELD_OUT_START,
                "horizon_s": 1800.0,
                "policy": "temporal earliest-arrival, require_fit=True",
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
            "architectures": {
                name: sorted(nodes) for name, nodes in evaluable.items()
            },
            "not_evaluable": {
                name: "no SCI->allowed contact exists in mock_graph"
                for name in unsupported
            },
            "summary": {
                name: {key: {"mean": m, "std": s} for key, (m, s) in metrics.items()}
                for name, metrics in summary.items()
            },
            "per_replicate": {
                name: [rep[key] for rep in reps]
                for name, reps in per_replicate.items()
                for key in ["priority_weighted_timely"]
            },
            "no_path_episodes": {
                name: {"count": no_path[name], "total": totals[name]}
                for name in evaluable
            },
        }, fh, indent=2)
    print("\nWrote %s" % path)


if __name__ == "__main__":
    main()
