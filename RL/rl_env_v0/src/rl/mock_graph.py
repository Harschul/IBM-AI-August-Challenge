"""Mocked dynamic graph for early RL development.

This deliberately does NOT depend on Harshul's orbital/physics code. Per the
proposal (section 11.4 / 11.6), Sudeepa's RL track starts on a random/mock
dynamic graph with the *same fixed node-ID interface* the real environment
will use, so training code is never blocked on orbital integration. When
Serafin's real contact-plan generator lands, only `mock_graph.py` gets
swapped out — `env.py` and `features.py` do not change.

Fixed 14-node action space (frozen, must match Appendix A of the plan doc):
    0      -> SCIENCE satellite (bundle source)
    1..8   -> LEO relays (8)
    9..10  -> GEO relays (2)
    11..13 -> Ground stations (3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random

NUM_NODES = 14
SCIENCE_ID = 0
LEO_IDS = list(range(1, 9))
GEO_IDS = list(range(9, 11))
GROUND_IDS = list(range(11, 14))


def node_type(node_id: int) -> str:
    if node_id == SCIENCE_ID:
        return "SCIENCE"
    if node_id in LEO_IDS:
        return "LEO_RELAY"
    if node_id in GEO_IDS:
        return "GEO_RELAY"
    return "GROUND"


@dataclass
class Contact:
    source_id: int
    destination_id: int
    start_s: float
    end_s: float
    data_rate_bps: float
    range_km: float
    queue_norm: float
    storage_free_norm: float
    health: float
    battery: float
    weather_risk: float
    reliability: float

    def is_open(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s


@dataclass
class Bundle:
    science_priority: float
    size_bytes: float
    deadline_s: float
    created_s: float = 0.0
    current_holder: int = SCIENCE_ID
    remaining_bytes: float = field(init=False)

    def __post_init__(self):
        self.remaining_bytes = self.size_bytes


class MockContactPlan:
    """Generates a random-but-plausible time-varying contact plan.

    Ground stations are never directly reachable from other ground stations;
    LEO/GEO relays can reach each other and ground; science can only reach
    LEO/GEO (never ground directly, to force at least one hop -- swap this
    later for the real geometry-driven version).
    """

    def __init__(self, horizon_s: float = 1800.0, seed: int | None = None):
        self.horizon_s = horizon_s
        self.rng = random.Random(seed)
        self.contacts: list[Contact] = self._generate()

    def _allowed_pairs(self):
        pairs = []
        for a in [SCIENCE_ID, *LEO_IDS]:
            for b in LEO_IDS + GEO_IDS:
                if a != b:
                    pairs.append((a, b))
        for a in LEO_IDS + GEO_IDS:
            for b in GROUND_IDS:
                pairs.append((a, b))
        return pairs

    def _generate(self):
        contacts = []
        for src, dst in self._allowed_pairs():
            t = 0.0
            while t < self.horizon_s:
                gap = self.rng.uniform(0, 120)
                t += gap
                dur = self.rng.uniform(30, 240)
                if t + dur > self.horizon_s:
                    break
                contacts.append(
                    Contact(
                        source_id=src,
                        destination_id=dst,
                        start_s=t,
                        end_s=t + dur,
                        data_rate_bps=self.rng.uniform(1e6, 5e7),
                        range_km=self.rng.uniform(500, 40000),
                        queue_norm=self.rng.uniform(0.0, 0.9),
                        storage_free_norm=self.rng.uniform(0.1, 1.0),
                        health=self.rng.uniform(0.8, 1.0),
                        battery=self.rng.uniform(0.5, 1.0),
                        weather_risk=self.rng.uniform(0.0, 0.6),
                        reliability=self.rng.uniform(0.7, 1.0),
                    )
                )
                t += dur
        return contacts

    def open_contacts_from(self, node_id: int, t: float) -> list[Contact]:
        return [c for c in self.contacts if c.source_id == node_id and c.is_open(t)]

    def sample_bundle(self, t: float, rng: random.Random) -> Bundle:
        urgent = rng.random() < 0.3
        priority = rng.uniform(0.7, 1.0) if urgent else rng.uniform(0.1, 0.6)
        deadline = rng.uniform(120, 400) if urgent else rng.uniform(600, self.horizon_s)
        return Bundle(
            science_priority=priority,
            size_bytes=rng.uniform(50e6, 900e6),
            deadline_s=deadline,
            created_s=t,
        )
