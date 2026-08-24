"""
Contact = a communication opportunity that exists only during a time window.

This is THE idea of the whole project (PDF section 7.3). Do not store
"LEO-2 is connected to LEO-5" as a permanent boolean. Satellites move, so a
link exists from start_s to end_s and then it is gone.

All times are seconds since the start of the simulation.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Contact:
    source_id: int
    destination_id: int
    start_s: float
    end_s: float
    data_rate_bps: float                      # BITS per second
    range_km: float = 0.0
    propagation_delay_s: float = 0.0          # range / speed of light
    residual_capacity_bytes: int = 10**15     # how much can still be sent in this window
    reliability: float = 1.0
    weather_risk: float = 0.0
    energy_cost: float = 0.0
    link_type: str = "RF_ISL"

    def __post_init__(self):
        if self.end_s <= self.start_s:
            raise ValueError(f"contact must end after it starts: {self}")
        if self.data_rate_bps <= 0:
            raise ValueError("data_rate_bps must be positive")

    def transmission_time_s(self, size_bytes: int) -> float:
        """How long to push size_bytes through this link.
        Note the *8: size is in BYTES, rate is in BITS per second.
        Getting this wrong is the single most common bug in this file."""
        return (size_bytes * 8) / self.data_rate_bps


class ContactPlan:
    """
    The list of all predicted contacts, indexed for fast lookup.

    Your router asks the same question thousands of times: "given that I am at
    node 3, what links might I use?" A plain list means scanning everything
    every time. So we index by source node once, up front.
    """

    def __init__(self, contacts: List[Contact]):
        self.contacts = list(contacts)
        self._by_source: Dict[int, List[Contact]] = {}
        for c in self.contacts:
            self._by_source.setdefault(c.source_id, []).append(c)
        # Sorted by start time so we can stop scanning early later if needed.
        for lst in self._by_source.values():
            lst.sort(key=lambda c: c.start_s)

    def from_node(self, node_id: int) -> List[Contact]:
        """All contacts departing this node, earliest first."""
        return self._by_source.get(node_id, [])

    def horizon(self) -> float:
        """Last moment anything is possible. Useful as a search cutoff."""
        return max((c.end_s for c in self.contacts), default=0.0)

    def __len__(self) -> int:
        return len(self.contacts)
