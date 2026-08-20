"""
Toy contact-plan tests with KNOWN optimal arrival times.

Written before any orbital physics exists, on purpose (PDF section 11.3). If the
router is wrong here - where you can check the answer with a pen - you will never
find the bug once real satellite geometry is generating thousands of contacts.

Node id convention (the fixed 14):
    0       science satellite
    1-8     LEO relays
    9-10    GEO relays
    11-13   ground stations

Numbers are chosen so the arithmetic is exact:
    data_rate = 1_000_000 bits/s
    bundle    = 125_000 bytes = 1_000_000 bits
    => transmission time = exactly 1.0 s
"""

import pytest

from src.models.contact import Contact, ContactPlan
from src.routing.temporal_baseline import earliest_arrival

RATE = 1_000_000          # bits per second
SIZE = 125_000            # bytes -> 1.0 s of transmission
GROUND = [11, 12, 13]


def test_transmission_time_converts_bytes_to_bits():
    """The *8 conversion. Off-by-8x errors here poison every other result."""
    c = Contact(0, 11, start_s=0, end_s=100, data_rate_bps=RATE)
    assert c.transmission_time_s(SIZE) == pytest.approx(1.0)


def test_direct_pass():
    """One contact, already open.
    HAND CALC: depart=0, tx=1.0, prop=0.5  ->  arrival = 1.5"""
    plan = ContactPlan([
        Contact(0, 11, start_s=0, end_s=100, data_rate_bps=RATE, propagation_delay_s=0.5),
    ])
    route = earliest_arrival(plan, 0, GROUND, SIZE)

    assert route is not None
    assert route.arrival_s == pytest.approx(1.5)
    assert route.path_ids == [0, 11]
    assert route.next_hop() == 11


def test_must_wait_at_relay_before_forwarding():
    """THE important test. The bundle reaches relay 1 at t=1.1 but the only link
    to ground does not open until t=60. A snapshot router says 'no path'.
    A temporal router stores the bundle and departs at 60.

    HAND CALC:
      hop 1: depart max(0, 0)  = 0    tx 1.0  prop 0.1  -> at node 1 at 1.1
      hop 2: depart max(1.1,60)= 60   tx 1.0  prop 0.2  -> at node 11 at 61.2
    """
    plan = ContactPlan([
        Contact(0, 1, start_s=0,  end_s=10,  data_rate_bps=RATE, propagation_delay_s=0.1),
        Contact(1, 11, start_s=60, end_s=120, data_rate_bps=RATE, propagation_delay_s=0.2),
    ])
    route = earliest_arrival(plan, 0, GROUND, SIZE)

    assert route is not None
    assert route.arrival_s == pytest.approx(61.2)
    assert route.path_ids == [0, 1, 11]


def test_contact_too_short_to_finish_transfer():
    """A 0.5 s window cannot carry a 1.0 s transfer, even though it is open now.
    The router must reject it and take the slower two-hop path.

    HAND CALC (via relay 2): 0 -> 2 arrives 1.0, 2 -> 12 arrives 2.0"""
    plan = ContactPlan([
        Contact(0, 11, start_s=0, end_s=0.5, data_rate_bps=RATE),   # too short
        Contact(0, 2,  start_s=0, end_s=100, data_rate_bps=RATE),
        Contact(2, 12, start_s=0, end_s=100, data_rate_bps=RATE),
    ])
    route = earliest_arrival(plan, 0, GROUND, SIZE)

    assert route is not None
    assert route.path_ids == [0, 2, 12]
    assert route.arrival_s == pytest.approx(2.0)


def test_earliest_arrival_is_not_fewest_hops():
    """A direct link exists but does not open until t=500. Two hops through a
    LEO relay land the data at t=2.0. Earliest arrival must win over hop count."""
    plan = ContactPlan([
        Contact(0, 11, start_s=500, end_s=600, data_rate_bps=RATE),  # 1 hop, arrives 501.0
        Contact(0, 3,  start_s=0,   end_s=100, data_rate_bps=RATE),
        Contact(3, 11, start_s=0,   end_s=100, data_rate_bps=RATE),  # 2 hops, arrives 2.0
    ])
    route = earliest_arrival(plan, 0, GROUND, SIZE)

    assert route.arrival_s == pytest.approx(2.0)
    assert len(route.hops) == 2


def test_insufficient_residual_capacity_is_rejected():
    """The window is long enough, but other traffic already reserved the capacity."""
    plan = ContactPlan([
        Contact(0, 11, start_s=0, end_s=100, data_rate_bps=RATE,
                residual_capacity_bytes=SIZE - 1),                  # 1 byte short
        Contact(0, 4,  start_s=0, end_s=100, data_rate_bps=RATE),
        Contact(4, 12, start_s=0, end_s=100, data_rate_bps=RATE),
    ])
    route = earliest_arrival(plan, 0, GROUND, SIZE)

    assert route.path_ids == [0, 4, 12]


def test_no_route_returns_none():
    """No path to ground is a legitimate answer, not a crash."""
    plan = ContactPlan([
        Contact(0, 5, start_s=0, end_s=100, data_rate_bps=RATE),    # dead end
    ])
    assert earliest_arrival(plan, 0, GROUND, SIZE) is None


def test_deadline_prunes_unreachable_route():
    """An urgent transient with a 30 s deadline cannot use a path arriving at 61.2."""
    plan = ContactPlan([
        Contact(0, 1,  start_s=0,  end_s=10,  data_rate_bps=RATE, propagation_delay_s=0.1),
        Contact(1, 11, start_s=60, end_s=120, data_rate_bps=RATE, propagation_delay_s=0.2),
    ])
    assert earliest_arrival(plan, 0, GROUND, SIZE, deadline_s=30.0) is None
    # ...but the same plan succeeds for routine data with a generous deadline.
    assert earliest_arrival(plan, 0, GROUND, SIZE, deadline_s=3600.0) is not None


def test_late_start_time_misses_an_early_window():
    """Starting at t=50 means the 0-10 s window is gone. Same plan, different answer,
    because time is part of the state."""
    plan = ContactPlan([
        Contact(0, 1,  start_s=0,  end_s=10,  data_rate_bps=RATE),
        Contact(0, 11, start_s=80, end_s=200, data_rate_bps=RATE),
    ])
    route = earliest_arrival(plan, 0, GROUND, SIZE, start_s=50.0)

    assert route.path_ids == [0, 11]
    assert route.arrival_s == pytest.approx(81.0)


def test_contact_plan_indexes_by_source():
    plan = ContactPlan([
        Contact(0, 1, start_s=50, end_s=60, data_rate_bps=RATE),
        Contact(0, 2, start_s=10, end_s=20, data_rate_bps=RATE),
        Contact(1, 11, start_s=0, end_s=5,  data_rate_bps=RATE),
    ])
    departing = plan.from_node(0)
    assert len(departing) == 2
    assert [c.start_s for c in departing] == [10, 50]   # sorted earliest first
    assert plan.from_node(99) == []
    assert plan.horizon() == 60


def test_invalid_contact_rejected():
    with pytest.raises(ValueError):
        Contact(0, 1, start_s=100, end_s=100, data_rate_bps=RATE)
