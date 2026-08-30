from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.integration.capacity import CapacityLedger
from src.integration.config import (
    GROUND_IDS,
    LEO_IDS,
    SCIENCE_IDS,
    GroundStation,
    load_config,
)
from src.integration.contact_plan import build_contact_plan
from src.integration.rl_bridge import OBS_LEN, build_observation
from src.integration.scenario import build_satellites
from src.integration.simulation import IntegratedSimulator
from src.integration.traffic import generate_bundles
from src.models.bundle import DataBundle
from src.models.contact import Contact, ContactPlan

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "config" / "prototype.yaml")


def test_config_keeps_14_actions_with_multiple_science_sources():
    cfg = config()
    assert SCIENCE_IDS == (0, 1, 2)
    assert LEO_IDS == (3, 4, 5, 6, 7, 8)
    assert GROUND_IDS == (11, 12, 13)
    assert cfg.raw["nodes"]["science"]["count"] == 3
    assert cfg.raw["nodes"]["leo"]["count"] == 6
    assert cfg.frame_count == 361


def test_physical_scenario_builds_three_distinct_science_spacecraft():
    satellites = build_satellites(config())
    assert len(satellites) == 11
    science = [satellites[node_id] for node_id in SCIENCE_IDS]
    assert [sat.name() for sat in science] == ["SCI-0", "SCI-1", "SCI-2"]
    assert len({tuple(np.round(sat.pos(), 3)) for sat in science}) == 3


def test_physical_adapter_creates_direct_contacts_from_science_sources():
    cfg = config()
    cfg = replace(
        cfg,
        earth_rotation_rad_s=0.0,
        ground_stations=(
            GroundStation(11, "A", 0.0, 0.0, 0.0, 0.0),
            GroundStation(12, "B", 0.0, 90.0, 0.0, 0.0),
            GroundStation(13, "C", 0.0, 180.0, 0.0, 0.0),
        ),
    )

    positions = np.zeros((11, 3), dtype=float)
    # Put all science sources over the first ground station and spread the
    # remaining propagated nodes slightly so coincident geometry is avoided.
    for i in range(11):
        positions[i] = np.array([7000.0, 20.0 * i, 10.0 * i])

    snapshots = [
        SimpleNamespace(time=0.0, positions=positions.copy()),
        SimpleNamespace(time=5.0, positions=positions.copy()),
        SimpleNamespace(time=10.0, positions=positions.copy()),
    ]
    plan, diagnostics = build_contact_plan(snapshots, cfg)

    direct_sources = {
        c.source_id
        for c in plan.contacts
        if c.source_id in SCIENCE_IDS and c.destination_id == 11
    }
    assert direct_sources == set(SCIENCE_IDS)
    assert diagnostics.direct_to_ground_contacts >= len(SCIENCE_IDS)


def test_generated_bundles_are_randomly_distributed_across_science_sources():
    bundles = generate_bundles(config(), count=90, seed=12345)
    source_ids = {bundle.source_id for bundle in bundles}
    assert source_ids == set(SCIENCE_IDS)
    assert all(bundle.source_id in SCIENCE_IDS for bundle in bundles)


def test_capacity_ledger_prevents_double_booking():
    c = Contact(
        0, 11, 0.0, 10.0, data_rate_bps=8_000.0,
        residual_capacity_bytes=1_000,
    )
    ledger = CapacityLedger(ContactPlan([c]))
    ledger.reserve_contact(c, 700)
    assert ledger.remaining(c) == 300
    with pytest.raises(ValueError):
        ledger.reserve_contact(c, 301)


def test_rl_bridge_preserves_frozen_158_feature_interface_for_any_science_source():
    cfg = config()
    source_id = SCIENCE_IDS[-1]
    plan = ContactPlan([
        Contact(source_id, 3, 0.0, 100.0, 10_000_000.0, range_km=1000.0),
        Contact(source_id, 11, 10.0, 100.0, 10_000_000.0, range_km=1200.0),
    ])
    bundle = DataBundle("x", source_id, 1_000_000, deadline_s=500.0)
    obs, mask = build_observation(plan, bundle, 0.0, cfg)
    assert obs.shape == (OBS_LEN,) == (158,)
    assert mask.shape == (14,)
    assert mask[3] == 1
    assert mask[11] == 1
    assert int(mask.sum()) == 2


def test_integrated_baseline_routes_from_nonzero_science_source_and_reserves_capacity():
    cfg = config()
    size = 1_000_000
    source_id = SCIENCE_IDS[1]
    plan = ContactPlan([
        Contact(source_id, 3, 0.0, 20.0, 8_000_000.0, residual_capacity_bytes=size),
        Contact(3, 11, 0.0, 20.0, 8_000_000.0, residual_capacity_bytes=size),
    ])
    bundles = [
        DataBundle("a", source_id, size, created_s=0.0, deadline_s=20.0),
        DataBundle("b", source_id, size, created_s=0.1, deadline_s=20.0),
    ]
    results = IntegratedSimulator(plan, cfg).run_baseline(bundles)
    assert results[0].delivered is True
    assert results[0].path == (source_id, 3, 11)
    assert results[1].delivered is False


class DirectGroundPolicy:
    def choose(self, observation, mask):
        if mask[11]:
            return 11
        return int(np.flatnonzero(mask)[0])


def test_rl_policy_executes_on_canonical_plan_with_fallback_contract():
    cfg = config()
    source_id = SCIENCE_IDS[2]
    plan = ContactPlan([
        Contact(source_id, 11, 0.0, 100.0, 80_000_000.0, range_km=1000.0),
    ])
    bundle = DataBundle("rl", source_id, 5_000_000, deadline_s=100.0)
    result = IntegratedSimulator(plan, cfg).run_rl_with_fallback(
        [bundle], DirectGroundPolicy()
    )[0]
    assert result.delivered is True
    assert result.on_time is True
    assert result.path == (source_id, 11)
    assert result.fallbacks == 0
