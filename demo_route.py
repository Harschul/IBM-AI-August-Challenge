"""
Watch the temporal router work.

    ./.venv/bin/python demo_route.py

Three scenarios on the same 14-node network. The point of each is written above it.
"""

from src.models.bundle import DataBundle
from src.models.contact import Contact, ContactPlan
from src.routing.temporal_baseline import earliest_arrival
from src.routing.trace import render_summary, render_timeline

NAMES = {0: "SCI", 3: "LEO3", 5: "LEO5", 9: "GEO1", 11: "GNDA", 12: "GNDB"}
LEO, GEO = 10_000_000, 2_000_000


def show(title, why, plan, bundle, t_end):
    route = earliest_arrival(plan, bundle.source_id, [11, 12, 13],
                             bundle.size_bytes, deadline_s=bundle.deadline_s)
    print("\n" + "=" * 80)
    print(title)
    print(why)
    print(f"\nbundle {bundle.bundle_id}: {bundle.size_bytes/1e6:.0f} MB  "
          f"priority {bundle.science_priority}  deadline {bundle.deadline_s}s\n")
    print(render_timeline(plan, route, NAMES, bundle.size_bytes, t_end=t_end))
    print()
    print(render_summary(route, bundle, NAMES))


def urgent(bid="OBS-004812"):
    return DataBundle(bid, source_id=0, size_bytes=90_000_000,
                      science_priority=0.96, deadline_s=600, data_type="TRANSIENT")


# 1 --------------------------------------------------------------------------
show(
    "SCENARIO 1  -  HYBRID NETWORK, ROUTER PICKS THE LEO MESH",
    "GEO1 is visible the whole time, but its link is slow. The router takes three\n"
    "LEO hops instead, WAITING twice for windows that have not opened yet.",
    ContactPlan([
        Contact(0, 9,  start_s=0,   end_s=900, data_rate_bps=GEO, propagation_delay_s=0.12),
        Contact(9, 11, start_s=0,   end_s=900, data_rate_bps=GEO, propagation_delay_s=0.12),
        Contact(0, 3,  start_s=0,   end_s=180, data_rate_bps=LEO, propagation_delay_s=0.004),
        Contact(3, 5,  start_s=120, end_s=400, data_rate_bps=LEO, propagation_delay_s=0.006),
        Contact(5, 12, start_s=300, end_s=600, data_rate_bps=LEO, propagation_delay_s=0.003),
    ]),
    urgent(), t_end=900,
)

# 2 --------------------------------------------------------------------------
show(
    "SCENARIO 2  -  SAME BUNDLE, GEO ONLY  (the architecture we are benchmarking)",
    "Take the LEO relays away. GEO is always visible, so a naive 'is there a link?'\n"
    "check says yes - but the slow rate means the data lands 120s too late.",
    ContactPlan([
        Contact(0, 9,  start_s=0, end_s=900, data_rate_bps=GEO, propagation_delay_s=0.12),
        Contact(9, 11, start_s=0, end_s=900, data_rate_bps=GEO, propagation_delay_s=0.12),
    ]),
    urgent("OBS-004812"), t_end=900,
)

# 3 --------------------------------------------------------------------------
show(
    "SCENARIO 3  -  LEO5 FAILS MID-ROUTE",
    "The relay that scenario 1 depended on is gone. The only remaining LEO link is a\n"
    "dead end, and GEO is too slow. The router reports NO ROUTE rather than inventing\n"
    "one - an honest negative is a valid answer the simulator needs.",
    ContactPlan([
        Contact(0, 9,  start_s=0,   end_s=900, data_rate_bps=GEO, propagation_delay_s=0.12),
        Contact(9, 11, start_s=0,   end_s=900, data_rate_bps=GEO, propagation_delay_s=0.12),
        Contact(0, 3,  start_s=0,   end_s=180, data_rate_bps=LEO, propagation_delay_s=0.004),
    ]),
    urgent("OBS-004813"), t_end=900,
)

print("\n" + "=" * 80)
print("Legend:  |----|  contact window open    #  transmitting    .  waiting in storage")
print("=" * 80)
