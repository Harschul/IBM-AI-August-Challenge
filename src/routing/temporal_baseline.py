"""
Temporal earliest-arrival router  (PDF section 9.1)

This is the non-AI benchmark the RL agent must beat. If this is weak, the whole
comparison is worthless, so it must be a real contact-graph router - not Dijkstra
on a static map.

Difference from ordinary shortest-path:
  ordinary: an edge is always available; cost is a fixed number.
  here:     an edge exists only during [start_s, end_s], and WAITING IS LEGAL.

That second point is the whole trick. Being at a node at t=5 when the useful
link opens at t=60 is not a dead end - you store the bundle and depart at 60.
Routing on "what can I see right now" would wrongly report no path.

Rooted in CCSDS Schedule-Aware Bundle Routing [16].
"""

import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

from src.models.contact import Contact, ContactPlan


@dataclass
class Route:
    """The answer: which contacts to take, and when the data lands."""
    hops: List[Contact]
    arrival_s: float

    @property
    def path_ids(self) -> List[int]:
        """Node ids visited, e.g. [0, 3, 11]."""
        if not self.hops:
            return []
        return [self.hops[0].source_id] + [c.destination_id for c in self.hops]

    def next_hop(self) -> Optional[int]:
        """What the simulator actually needs: the single next node id."""
        return self.hops[0].destination_id if self.hops else None


def earliest_arrival(
    plan: ContactPlan,
    source_id: int,
    destinations: Iterable[int],
    size_bytes: int,
    start_s: float = 0.0,
    deadline_s: Optional[float] = None,
) -> Optional[Route]:
    """
    Find the route that gets size_bytes from source_id to ANY destination soonest.

    Returns None if no feasible route exists (that is a valid answer - it means
    the bundle must wait, be dropped, or the deadline is unreachable).

    This is Dijkstra where the "distance" is arrival TIME rather than cost.
    Because arrival time only ever increases as we take more hops, the first
    time we pop a destination off the queue we have the earliest arrival. That
    is the same argument that makes Dijkstra correct.
    """
    targets: Set[int] = set(destinations)
    if source_id in targets:
        return Route(hops=[], arrival_s=start_s)

    # Earliest time we know we can BE at each node holding the whole bundle.
    best: Dict[int, float] = {source_id: start_s}

    # Priority queue ordered by arrival time. The counter is a tie-breaker so
    # Python never tries to compare two Route objects (it cannot).
    counter = 0
    heap: List[tuple] = [(start_s, counter, source_id, [])]

    while heap:
        arrival_s, _, node, hops = heapq.heappop(heap)

        # Stale entry: we already found a better way to this node.
        if arrival_s > best.get(node, float("inf")):
            continue

        if node in targets:
            return Route(hops=hops, arrival_s=arrival_s)

        for contact in plan.from_node(node):
            # --- the four feasibility checks ---------------------------------

            # 1. WAIT if the window has not opened yet. This is the store-and-
            #    forward behaviour; it is what makes this a temporal router.
            depart_s = max(arrival_s, contact.start_s)

            # 2. The window may already be over.
            if depart_s >= contact.end_s:
                continue

            # 3. The transfer must FINISH before the window closes. A 10-second
            #    window is useless for an 18-second transfer.
            tx_s = contact.transmission_time_s(size_bytes)
            if depart_s + tx_s > contact.end_s:
                continue

            # 4. The contact must have enough capacity left after other traffic.
            if contact.residual_capacity_bytes < size_bytes:
                continue

            # -----------------------------------------------------------------
            candidate_s = depart_s + tx_s + contact.propagation_delay_s

            if deadline_s is not None and candidate_s > deadline_s:
                continue  # this path cannot make the deadline; prune it

            nxt = contact.destination_id
            if candidate_s < best.get(nxt, float("inf")):
                best[nxt] = candidate_s
                counter += 1
                heapq.heappush(heap, (candidate_s, counter, nxt, hops + [contact]))

    return None
