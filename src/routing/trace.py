"""
Route trace rendering (PDF section 11.3 - "route trace logs used in the final demo").

A contact plan is hard to reason about as a list of numbers. Drawn as a timeline
it becomes obvious: you can SEE that the link you wanted has already closed, or
that the bundle sat waiting for 108 seconds.

Pure ASCII on purpose - no plotting library, works over SSH, pastes into Slack,
and never breaks the demo machine.
"""

from typing import List, Optional

from src.models.contact import Contact, ContactPlan
from src.routing.temporal_baseline import Route


def _col(t: float, t0: float, t1: float, width: int) -> int:
    if t1 <= t0:
        return 0
    return max(0, min(width - 1, round((t - t0) / (t1 - t0) * (width - 1))))


def render_timeline(
    plan: ContactPlan,
    route: Optional[Route] = None,
    names: Optional[dict] = None,
    size_bytes: int = 0,
    width: int = 58,
    t_end: Optional[float] = None,
) -> str:
    """
    Draw every contact as a horizontal window, and overlay the chosen route.

        |---------|   a contact window: the link is usable in here
        #########     the bundle is actually transmitting
        .......       the bundle is sitting in storage, waiting
    """
    names = names or {}
    label = lambda i: names.get(i, f"N{i}")

    t0 = 0.0
    t1 = t_end if t_end is not None else plan.horizon()
    lines: List[str] = []

    # --- time axis -------------------------------------------------------
    axis = ["-"] * width
    ticks = []
    for k in range(5):
        t = t0 + (t1 - t0) * k / 4
        c = _col(t, t0, t1, width)
        axis[c] = "+"
        ticks.append((c, f"{t:.0f}s"))
    tick_line = [" "] * width
    for c, s in ticks:
        start = min(c, width - len(s))
        for j, ch in enumerate(s):
            tick_line[start + j] = ch

    hdr = " " * 22
    lines.append(hdr + "".join(tick_line))
    lines.append(hdr + "".join(axis))

    # Which contacts were used, and when the transfer actually happened.
    used = {}
    if route:
        t = 0.0
        for c in route.hops:
            depart = max(t, c.start_s)
            tx = c.transmission_time_s(size_bytes) if size_bytes else 0.0
            arrive = depart + tx + c.propagation_delay_s
            used[id(c)] = (t, depart, arrive)   # (available_at, depart, arrive)
            t = arrive

    # --- one row per contact ---------------------------------------------
    for c in plan.contacts:
        row = [" "] * width
        a, b = _col(c.start_s, t0, t1, width), _col(c.end_s, t0, t1, width)
        for x in range(a, b + 1):
            row[x] = "-"
        row[a], row[b] = "|", "|"

        marker = "   "
        if id(c) in used:
            avail, depart, arrive = used[id(c)]
            # the wait: bundle was ready but the window had not opened
            for x in range(_col(avail, t0, t1, width), _col(depart, t0, t1, width)):
                if row[x] == " ":
                    row[x] = "."
            # the transfer itself
            for x in range(_col(depart, t0, t1, width), _col(arrive, t0, t1, width) + 1):
                row[x] = "#"
            marker = "-> "

        rate = f"{c.data_rate_bps/1e6:.0f}Mb"
        name = f"{marker}{label(c.source_id)}>{label(c.destination_id)} {rate}"
        lines.append(f"{name:<22}" + "".join(row))

    return "\n".join(lines)


def render_summary(route: Optional[Route], bundle, names: Optional[dict] = None) -> str:
    names = names or {}
    label = lambda i: names.get(i, f"N{i}")

    if route is None:
        return ("  RESULT   : NO FEASIBLE ROUTE\n"
                f"  bundle {bundle.bundle_id} cannot reach any ground station before "
                f"its deadline ({bundle.deadline_s}s).")

    on_time = bundle.deadline_s is None or route.arrival_s <= bundle.deadline_s
    verdict = "DELIVERED ON TIME" if on_time else "MISSED DEADLINE"
    slack = "" if bundle.deadline_s is None else f"  ({bundle.deadline_s - route.arrival_s:+.1f}s vs deadline)"

    return (f"  route    : {' -> '.join(label(n) for n in route.path_ids)}\n"
            f"  arrival  : {route.arrival_s:.1f}s{slack}\n"
            f"  hops     : {len(route.hops)}\n"
            f"  next hop : {label(route.next_hop())}\n"
            f"  RESULT   : {verdict}")
