#!/usr/bin/env python3
"""Route trace logs for the demo (PDF sections 11.3, 2 - "competition demo").

The acceptance table in section 2 asks for one thing to be visible on screen:

    "A visible route changes when congestion, weather or a relay failure
     makes another path preferable."

A metrics table cannot show that. This can: it runs two policies over the SAME
scenario, prints the path each one took with real timings, finds the first
decision where they disagreed, and shows the options that were on the table at
that moment -- so an audience can see not just that the routes differ, but why.

Pure ASCII on purpose. No plotting library, works over SSH, pastes into Slack,
and cannot break the demo machine.

Run from `RL/rl_env_v0/`:

    python3 src/rl/route_trace.py                 # find and show divergences
    python3 src/rl/route_trace.py --seed 1234     # one specific scenario
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np  # noqa: E402
from gymnasium import spaces  # noqa: E402

from src.rl.env import OBS_LEN, RoutingEnv  # noqa: E402
from src.rl.mock_graph import GROUND_IDS, NUM_NODES  # noqa: E402
from src.rl.eval_with_baseline import (  # noqa: E402
    MODEL_SEEDS, _numpy2_compat, baseline_action, hop_is_feasible,
)

NAMES = {0: "SCI", 9: "GEO1", 10: "GEO2", 11: "GNDA", 12: "GNDB", 13: "GNDC"}
for _i in range(1, 9):
    NAMES[_i] = "LEO%d" % _i

MAX_STEPS = 20


def name(node_id):
    return NAMES.get(node_id, "N%d" % node_id)


def _describe(env, contact):
    size = env.bundle.remaining_bytes
    depart = max(env.t, contact.start_s)
    tx_s = size * 8 / contact.data_rate_bps
    return {
        "wait": depart - env.t, "tx": tx_s, "arrive": depart + tx_s,
        "fits": depart + tx_s <= contact.end_s,
        "rate_mbps": contact.data_rate_bps / 1e6,
    }


def candidate_table(env, mask):
    """What every legal next hop offers, AS THE ENVIRONMENT WILL EXECUTE IT.

    Two different contacts matter here and it is important not to conflate them:

      actual   the contact `env._best_contact_to()` will really use. It sorts by
               earliest DEPARTURE, so an early slow window beats a later fast
               one -- observed picking 1.14 Mbps over 48.95 Mbps, arriving
               4208 s instead of 251 s.
      best     the contact with the earliest ARRIVAL, which is what any sane
               planner assumes it will get.

    The demo must show `actual`, or the printed arrival times will not match the
    outcomes underneath them. `best` is carried alongside so the gap is visible.
    """
    rows = []
    for dest in range(NUM_NODES):
        if not mask[dest]:
            continue

        reachable = [
            c for c in env.contact_plan.contacts
            if c.source_id == env.bundle.current_holder
            and c.destination_id == dest and c.end_s >= env.t
            and max(env.t, c.start_s) < c.end_s
        ]
        if not reachable:
            continue

        # Use the contact step() will really execute. Before #8 that was
        # _best_contact_to (earliest DEPARTURE, which could pick a slow window);
        # since #8 it is _feasible_contact_to (earliest ARRIVAL among contacts
        # the transfer can actually complete in). Reading the old helper here
        # made the trace report arrivals the environment would never produce.
        actual = (env._feasible_contact_to(dest)
                  if hasattr(env, "_feasible_contact_to")
                  else env._best_contact_to(dest))
        if actual is None:
            continue
        # Compare only against contacts the transfer can actually COMPLETE in.
        # Comparing against every reachable contact flagged "a faster option
        # existed" when that option was a window too short to carry the bundle
        # -- not an option at all. Post-#8 this correctly never fires.
        usable = [c for c in reachable if _describe(env, c)["fits"]]
        best = min(usable or reachable, key=lambda c: _describe(env, c)["arrive"])

        row = {"dest": dest}
        row.update(_describe(env, actual))
        row["best_arrive"] = _describe(env, best)["arrive"]
        row["mispicked"] = row["best_arrive"] < row["arrive"] - 1.0
        rows.append(row)

    return sorted(rows, key=lambda r: r["arrive"])


def trace_episode(seed, policy, model=None):
    """Run one scenario under one policy, recording every decision."""
    env = RoutingEnv(horizon_s=1800.0)
    obs, info = env.reset(seed=seed)

    record = {
        "priority": env.bundle.science_priority,
        "size_mb": env.bundle.size_bytes / 1e6,
        "deadline_s": env.bundle.deadline_s,
        "hops": [],
        "event": "no_terminal",
    }

    for _ in range(MAX_STEPS):
        mask = info["action_mask"]
        t_before = env.t
        options = candidate_table(env, mask)

        if policy == "baseline":
            action = baseline_action(env, mask, require_fit=True)
        else:
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            action = int(action)
        if action is None:
            break

        chosen = next((o for o in options if o["dest"] == action), None)
        holder = env.bundle.current_holder
        obs, _, terminated, truncated, info = env.step(action)

        record["hops"].append({
            "from": holder, "to": action, "t_before": t_before,
            "t_after": env.t, "options": options,
            "wait": chosen["wait"] if chosen else 0.0,
            "tx": chosen["tx"] if chosen else 0.0,
            "fits": chosen["fits"] if chosen else True,
        })

        if terminated or truncated:
            record["event"] = info.get("event", "unknown")
            break

    record["latency_s"] = env.t
    return record


def render_path(record):
    path = [record["hops"][0]["from"]] if record["hops"] else []
    path += [h["to"] for h in record["hops"]]
    return " -> ".join(name(p) for p in path)


def render_comparison(seed, base, rl, rl_label):
    out = []
    out.append("=" * 78)
    out.append("SCENARIO seed=%d" % seed)
    out.append("  bundle: priority %.2f   %.0f MB   deadline t=%.0f s"
               % (base["priority"], base["size_mb"], base["deadline_s"]))
    out.append("=" * 78)
    out.append("")

    for label, rec in (("temporal baseline", base), (rl_label, rl)):
        flag = "on time" if rec["event"] == "delivered_on_time" else rec["event"]
        out.append("  %-18s %-38s %s" % (label, render_path(rec), flag))
        out.append("  %-18s %s" % ("", "delivered t=%.0f s, %d hop(s)"
                                   % (rec["latency_s"], len(rec["hops"]))))
        out.append("")

    # --- where they first disagreed ------------------------------------
    diverge = None
    for i in range(min(len(base["hops"]), len(rl["hops"]))):
        if base["hops"][i]["to"] != rl["hops"][i]["to"]:
            diverge = i
            break

    if diverge is None:
        out.append("  Both policies chose the same route.")
        out.append("=" * 78)
        return "\n".join(out)

    hop_b, hop_r = base["hops"][diverge], rl["hops"][diverge]
    out.append("-" * 78)
    out.append("  DIVERGENCE at hop %d, holding at %s, t=%.0f s"
               % (diverge + 1, name(hop_b["from"]), hop_b["t_before"]))
    out.append("-" * 78)
    out.append("")
    out.append("    next hop | wait (s) | transmit (s) | arrives | fits | chosen by")
    out.append("    " + "-" * 72)

    mispicks = 0
    for opt in hop_b["options"]:
        who = []
        if opt["dest"] == hop_b["to"]:
            who.append("baseline")
        if opt["dest"] == hop_r["to"]:
            who.append(rl_label)
        note = ""
        if opt["mispicked"]:
            mispicks += 1
            note = "   [env picked a slow window; %.0f s was available]" % opt["best_arrive"]
        out.append("    %-8s | %8.1f | %12.1f | %7.0f | %-4s | %s%s"
                   % (name(opt["dest"]), opt["wait"], opt["tx"], opt["arrive"],
                      "yes" if opt["fits"] else "NO", ", ".join(who), note))

    out.append("")
    if mispicks:
        out.append("    %d option(s) show the contact-selection bug: _best_contact_to()"
                   % mispicks)
        out.append("    sorts by earliest DEPARTURE, not earliest arrival.")
        out.append("")
    if not hop_r["fits"]:
        out.append("    The agent's choice does not fit inside its contact window")
        out.append("    (issue #4 -- the environment permits it anyway).")
    else:
        gap = hop_r["t_after"] - hop_b["t_after"]
        out.append("    Both hops are physically valid. The baseline's lands %.0f s %s."
                   % (abs(gap), "earlier" if gap > 0 else "later"))
    out.append("=" * 78)
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="trace one specific scenario")
    parser.add_argument("--show", type=int, default=3,
                        help="how many diverging scenarios to show")
    parser.add_argument("--search", type=int, default=200,
                        help="how many seeds to search for divergences")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()

    _numpy2_compat()
    from sb3_contrib import MaskablePPO

    model_path = None
    for seed in MODEL_SEEDS:
        candidate = os.path.join(args.models, "rl_agent_seed_%d.zip" % seed)
        if os.path.exists(candidate):
            model_path, model_seed = candidate, seed
            break
    if model_path is None:
        sys.exit("no checkpoint found in %s" % args.models)

    model = MaskablePPO.load(model_path, device="cpu", custom_objects={
        "observation_space": spaces.Box(low=-1.0, high=1.0,
                                        shape=(OBS_LEN,), dtype=np.float32),
        "action_space": spaces.Discrete(NUM_NODES),
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    })
    rl_label = "rl_seed_%d" % model_seed

    if args.seed is not None:
        base = trace_episode(args.seed, "baseline")
        rl = trace_episode(args.seed, "rl", model)
        print(render_comparison(args.seed, base, rl, rl_label))
        return

    # Search for scenarios where the two policies actually route differently
    # AND the outcome differs -- those are the ones worth putting on screen.
    print("Searching %d scenarios for route divergences...\n" % args.search)
    rng = random.Random(999)
    shown = 0

    for _ in range(args.search):
        seed = rng.randint(0, 10**9)
        base = trace_episode(seed, "baseline")
        rl = trace_episode(seed, "rl", model)

        paths_differ = render_path(base) != render_path(rl)
        outcome_differs = base["event"] != rl["event"]
        if not (paths_differ and outcome_differs):
            continue

        print(render_comparison(seed, base, rl, rl_label))
        print()
        shown += 1
        if shown >= args.show:
            break

    if shown == 0:
        print("No scenario in %d had both a different route and a different "
              "outcome." % args.search)
    else:
        print("Reproduce any of these with:  python3 src/rl/route_trace.py --seed <seed>")


if __name__ == "__main__":
    main()
