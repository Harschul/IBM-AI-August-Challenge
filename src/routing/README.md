# Temporal routing baseline

*The non-AI benchmark: given a schedule of communication windows that open and
close, which path gets this bundle to the ground soonest?*

**Scope.** This document covers `src/models/` and `src/routing/` only — one
workstream (Jinwoo, PDF §11.3) inside the five-person project. Orbital physics,
the simulation clock, the RL agent, the dashboard and the experiment runner are
owned by other people and live elsewhere in this repo. For the full project, see
the plan PDF.

August Challenge 2026 · prototype deadline 31 August 2026

---

## 1. The problem, explained simply

A science satellite in low orbit takes pictures. It cannot send them to Earth
whenever it wants, because a ground station has to be **below it and in view**.

ESA gives a representative Earth-observation case: roughly **10 minutes of
downlink in a ~100 minute orbit**. (That is a motivating example, not a law of
physics — the real number depends on altitude, ground-station location and
antenna geometry.)

So the satellite spends most of its time collecting data it cannot send yet.

Two things make this worse:

- **Cameras got better faster than radios did.** Modern sensors generate data
  faster than those short windows can drain it. The onboard storage fills up.
- **Not all data is equal.** A routine calibration frame and a once-a-year
  astronomical transient take up the same bandwidth. If the routine frame goes
  first, the rare event might arrive after anyone can act on it.

### The bus-stop analogy

You are on a remote island with parcels to send.

- **Boats only stop at scheduled times.** You cannot leave whenever you like.
- **Some parcels are urgent** — medicine that expires in 3 minutes.
- **Some are not** — a birthday card that can wait 6 hours.
- **Boats have limited space**, and some are already half full.
- **Some routes are longer but leave sooner**; some are direct but leave late.
- **Weather can cancel a departure** without warning.

The question this project answers: **which boat should this particular parcel
take, right now?**

---

## 2. The network we simulate

14 "islands", all moving:

| Node | Count | Role |
|---|---|---|
| Science satellite | 1 | Takes the pictures. The source of all data. |
| LEO relays | 8 | Low, fast, close together. Short hops, but they move constantly. |
| GEO relays | 2 | Very high up. Can see a huge area for a long time, but far away, so slower links. |
| Ground stations | 3 | The destination. In different countries, so different weather. |

A **bundle** is one package of science data. It carries:

```
size            900 MB
priority        0.96      (0 = boring, 1 = once-in-a-decade discovery)
deadline        180 s     (after this it is scientifically worthless)
type            TRANSIENT
```

A **contact** is a link that exists *only during a time window*:

```
LEO-3  ->  LEO-5     from t=120s  to t=400s     10 Mbit/s
```

> **This is the single most important idea in the project.**
> We never say "LEO-3 is connected to LEO-5". The satellites are moving, so the
> link exists between t=120 and t=400 and then it is gone.

---

## 3. Why this is not just Dijkstra

Ordinary shortest-path routing assumes roads are always open.

Here, roads open and close on a schedule — **and waiting is a legal move.**

If your data arrives at a relay at t=5, and the only link onward opens at t=60,
that is **not** a dead end. You store the data for 55 seconds and then send it.
This is called **store-and-forward**, and it is standard practice in space
networking (CCSDS Schedule-Aware Bundle Routing).

A router that only looks at *"what can I see right now?"* would report **"no
route exists"** — and would be wrong.

---

## 4. The toy example, with real numbers

This actually runs. `python demo_route.py` from the repo root

A 90 MB urgent transient, deadline 600 seconds:

```
 SCI-0 -> LEO-3   depart    0.0s   tx 72.0s   arrive   72.0s
 LEO-3 -> LEO-5   depart  120.0s   tx 72.0s   arrive  192.0s   (waited 48s for window)
 LEO-5 -> GND-B   depart  300.0s   tx 72.0s   arrive  372.0s   (waited 108s for window)

 route   : SCI-0 -> LEO-3 -> LEO-5 -> GND-B
 arrival : 372.0s   (deadline 600s)  -> MADE IT
```

The router **deliberately waited twice** — 48 seconds, then 108 seconds — for
links that did not exist yet.

The obvious alternative was the big GEO relay, which is visible the whole time.
On the same bundle it arrives at **720.2 seconds** — the link is slower because
GEO is so far away, so the data misses its deadline by two minutes.

**That is the whole project in one example:** the smart path is not the obvious
one, and picking it correctly is worth real science.

---

## 5. Where the AI comes in

The example above was solved by a **deterministic** router — it searches every
possible future path and picks the earliest arrival. That is the *baseline*.

The baseline has limits. It works from a plan, and reality drifts: a relay's
queue fills up, clouds roll over a ground station, a satellite fails mid-route.
Recomputing the perfect answer for every bundle, every few seconds, gets
expensive and still assumes the plan is accurate.

So we also train a **reinforcement learning agent**. At each decision point it
sees the bundle (priority, deadline, size) and the network (which links are
open, how congested, how healthy, what the weather is) and picks **one next
hop** out of the 14 node IDs. Illegal choices are masked out so it cannot pick
an impossible link.

It is rewarded for delivering **high-priority data before its deadline**, and
penalised for lateness, wasted hops, congestion and failures.

The competition result is the **comparison**: under what conditions does the AI
beat the deterministic baseline? Both run on identical traffic and identical
random seeds, so the comparison is fair.

> **Safety rule:** the AI is never the only path to delivery. If it picks
> something invalid, or takes too long, or hits a situation it was not trained
> for, the deterministic router takes over.

---

## 6. What this module contains

```
src/models/contact.py             Contact (a link with a time window) + ContactPlan
src/models/bundle.py              DataBundle (science data with priority + deadline)
src/routing/temporal_baseline.py  the earliest-arrival router  <- the core algorithm
src/routing/trace.py              ASCII timeline renderer for demo route traces
tests/test_temporal_router.py     11 tests, every answer worked out by hand
demo_route.py                     the example from section 4
```

Pure standard library — `heapq`, `dataclasses`, `typing`. No third-party runtime
dependency, so this adds no install risk to `main`. `pytest` is needed for the
tests only.

### Run it

```bash
python -m pytest tests/ -q     # 11 tests
python demo_route.py           # the route trace
```

### The algorithm in eight lines

Every time the router considers using a link:

```python
depart = max(when_i_arrived_here, contact.start_s)   # 1. wait if not open yet
if depart >= contact.end_s:        skip              # 2. window already over
tx = (size_bytes * 8) / data_rate_bps                #    bytes -> bits!
if depart + tx > contact.end_s:    skip              # 3. must FINISH in time
if residual_capacity < size_bytes: skip              # 4. room for this bundle?
arrival = depart + tx + propagation_delay
```

Those four checks are the entire difference between this and textbook Dijkstra.

### Not built yet, in this module

- **Capacity reservation.** `residual_capacity_bytes` is checked but never
  decremented, so two bundles routed independently can each be told they fit the
  same contact. Needs the simulator to reserve on commit (PDF §11.3, 23 Aug).
- **Queue / congestion delay** at intermediate nodes.
- **Reroute on contact failure** — currently a failed contact means recomputing
  from scratch.
- **Gymnasium environment and action mask** — co-owned with Sudeepa.

---

## 7. Scope discipline

This is a **simulation**. We are building and evaluating the *intelligence
layer* for an emerging multi-provider relay ecosystem.

We are **not** building satellites, and we do **not** own or have access to
NASA, ESA or commercial relay assets. Real systems (TDRSS, EDRS, HydRON, PExT,
Telesat Lightspeed) are cited as evidence that the architecture direction is
credible — not as infrastructure we can command.

All link rates, orbit geometry, weather thresholds and reward weights are
labelled **simulation assumptions** unless sourced to a specific real system.
