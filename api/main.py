"""
Thin FastAPI wrapper around the tested temporal earliest-arrival router
(src/routing/temporal_baseline.py). No new routing logic lives here — this
module only translates HTTP requests into the existing Contact/ContactPlan/
DataBundle dataclasses, calls earliest_arrival(), and translates the Route
back into a shape a browser can render as a timeline (the visual equivalent
of src/routing/trace.py's ASCII output).

Run:
    ./.venv/bin/uvicorn api.main:app --reload --port 8000
"""

from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.bundle import DataBundle
from src.models.contact import Contact, ContactPlan
from src.routing.temporal_baseline import earliest_arrival

app = FastAPI(title="Deorbit Intelligence — Temporal Router API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NODE_ROLE = {
    0: "SCIENCE_SAT",
    **{i: "LEO_RELAY" for i in range(1, 9)},
    9: "GEO_RELAY", 10: "GEO_RELAY",
    11: "GROUND", 12: "GROUND", 13: "GROUND",
}

NODE_NAME = {
    0: "SCI",
    1: "LEO1", 2: "LEO2", 3: "LEO3", 4: "LEO4",
    5: "LEO5", 6: "LEO6", 7: "LEO7", 8: "LEO8",
    9: "GEO1", 10: "GEO2",
    11: "GNDA", 12: "GNDB", 13: "GNDC",
}


class ContactIn(BaseModel):
    source_id: int
    destination_id: int
    start_s: float
    end_s: float
    data_rate_bps: float = Field(gt=0)
    propagation_delay_s: float = 0.0
    residual_capacity_bytes: int = 10**15


class BundleIn(BaseModel):
    bundle_id: str = "BUNDLE-1"
    source_id: int
    size_bytes: int = Field(gt=0)
    science_priority: float = Field(default=0.5, ge=0.0, le=1.0)
    deadline_s: Optional[float] = None
    data_type: str = "STAR_FIELD"


class RouteRequest(BaseModel):
    contacts: list[ContactIn]
    bundle: BundleIn
    destinations: list[int] = [11, 12, 13]


class HopOut(BaseModel):
    source_id: int
    destination_id: int
    start_s: float
    end_s: float
    data_rate_bps: float
    propagation_delay_s: float
    available_at_s: float   # when the bundle arrived at source_id, ready to send
    depart_s: float         # when it actually left (after waiting for the window)
    arrive_s: float         # when it lands at destination_id
    waited_s: float


class RouteResponse(BaseModel):
    feasible: bool
    verdict: Literal["DELIVERED_ON_TIME", "MISSED_DEADLINE", "NO_FEASIBLE_ROUTE"]
    path_ids: list[int]
    path_names: list[str]
    hops: list[HopOut]
    arrival_s: Optional[float] = None
    slack_s: Optional[float] = None
    node_names: dict[int, str]
    node_roles: dict[int, str]


def _to_domain(req: RouteRequest) -> tuple[ContactPlan, DataBundle]:
    contacts = [
        Contact(
            source_id=c.source_id,
            destination_id=c.destination_id,
            start_s=c.start_s,
            end_s=c.end_s,
            data_rate_bps=c.data_rate_bps,
            propagation_delay_s=c.propagation_delay_s,
            residual_capacity_bytes=c.residual_capacity_bytes,
        )
        for c in req.contacts
    ]
    bundle = DataBundle(
        bundle_id=req.bundle.bundle_id,
        source_id=req.bundle.source_id,
        size_bytes=req.bundle.size_bytes,
        science_priority=req.bundle.science_priority,
        deadline_s=req.bundle.deadline_s,
        data_type=req.bundle.data_type,
    )
    return ContactPlan(contacts), bundle


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    plan, bundle = _to_domain(req)
    result = earliest_arrival(
        plan, bundle.source_id, req.destinations,
        bundle.size_bytes, deadline_s=bundle.deadline_s,
    )

    if result is None:
        return RouteResponse(
            feasible=False,
            verdict="NO_FEASIBLE_ROUTE",
            path_ids=[],
            path_names=[],
            hops=[],
            node_names=NODE_NAME,
            node_roles=NODE_ROLE,
        )

    hops: list[HopOut] = []
    t = 0.0
    for c in result.hops:
        depart = max(t, c.start_s)
        tx = c.transmission_time_s(bundle.size_bytes)
        arrive = depart + tx + c.propagation_delay_s
        hops.append(HopOut(
            source_id=c.source_id, destination_id=c.destination_id,
            start_s=c.start_s, end_s=c.end_s,
            data_rate_bps=c.data_rate_bps, propagation_delay_s=c.propagation_delay_s,
            available_at_s=t, depart_s=depart, arrive_s=arrive,
            waited_s=depart - t,
        ))
        t = arrive

    on_time = bundle.deadline_s is None or result.arrival_s <= bundle.deadline_s
    slack = None if bundle.deadline_s is None else bundle.deadline_s - result.arrival_s

    return RouteResponse(
        feasible=True,
        verdict="DELIVERED_ON_TIME" if on_time else "MISSED_DEADLINE",
        path_ids=result.path_ids,
        path_names=[NODE_NAME.get(n, f"N{n}") for n in result.path_ids],
        hops=hops,
        arrival_s=result.arrival_s,
        slack_s=slack,
        node_names=NODE_NAME,
        node_roles=NODE_ROLE,
    )


# --- Built-in scenarios, mirroring demo_route.py exactly --------------------

LEO, GEO = 10_000_000, 2_000_000

SCENARIOS = {
    "hybrid-leo-mesh": {
        "title": "Hybrid network — router picks the LEO mesh",
        "why": (
            "GEO1 is visible the whole time, but its link is slow. The router takes "
            "three LEO hops instead, waiting twice for windows that have not opened yet."
        ),
        "t_end": 900,
        "contacts": [
            dict(source_id=0, destination_id=9, start_s=0, end_s=900, data_rate_bps=GEO, propagation_delay_s=0.12),
            dict(source_id=9, destination_id=11, start_s=0, end_s=900, data_rate_bps=GEO, propagation_delay_s=0.12),
            dict(source_id=0, destination_id=3, start_s=0, end_s=180, data_rate_bps=LEO, propagation_delay_s=0.004),
            dict(source_id=3, destination_id=5, start_s=120, end_s=400, data_rate_bps=LEO, propagation_delay_s=0.006),
            dict(source_id=5, destination_id=12, start_s=300, end_s=600, data_rate_bps=LEO, propagation_delay_s=0.003),
        ],
        "bundle": dict(bundle_id="OBS-004812", source_id=0, size_bytes=90_000_000,
                        science_priority=0.96, deadline_s=600, data_type="TRANSIENT"),
    },
    "geo-only": {
        "title": "Same bundle, GEO only — the architecture being benchmarked",
        "why": (
            "Take the LEO relays away. GEO is always visible, so a naive link check says "
            "yes — but the slow rate means the data lands 120s too late."
        ),
        "t_end": 900,
        "contacts": [
            dict(source_id=0, destination_id=9, start_s=0, end_s=900, data_rate_bps=GEO, propagation_delay_s=0.12),
            dict(source_id=9, destination_id=11, start_s=0, end_s=900, data_rate_bps=GEO, propagation_delay_s=0.12),
        ],
        "bundle": dict(bundle_id="OBS-004812", source_id=0, size_bytes=90_000_000,
                        science_priority=0.96, deadline_s=600, data_type="TRANSIENT"),
    },
    "leo5-fails": {
        "title": "LEO5 fails mid-route",
        "why": (
            "The relay scenario 1 depended on is gone. The only remaining LEO link is a "
            "dead end, and GEO is too slow. The router reports NO ROUTE rather than inventing one."
        ),
        "t_end": 900,
        "contacts": [
            dict(source_id=0, destination_id=9, start_s=0, end_s=900, data_rate_bps=GEO, propagation_delay_s=0.12),
            dict(source_id=9, destination_id=11, start_s=0, end_s=900, data_rate_bps=GEO, propagation_delay_s=0.12),
            dict(source_id=0, destination_id=3, start_s=0, end_s=180, data_rate_bps=LEO, propagation_delay_s=0.004),
        ],
        "bundle": dict(bundle_id="OBS-004813", source_id=0, size_bytes=90_000_000,
                        science_priority=0.96, deadline_s=600, data_type="TRANSIENT"),
    },
}


@app.get("/scenarios")
def list_scenarios():
    return SCENARIOS


@app.get("/health")
def health():
    return {"status": "ok"}
