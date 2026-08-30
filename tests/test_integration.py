from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.integration.capacity import CapacityLedger
from src.integration.config import GroundStation, load_config
from src.integration.contact_plan import build_contact_plan
from src.integration.rl_bridge import OBS_LEN, action_mask, build_observation
from src.integration.simulation import IntegratedSimulator
from src.models.bundle import DataBundle
from src.models.contact import Contact, ContactPlan

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "config" / "prototype.yaml")


def test_config_freezes_expected_14_node_ground_ids():
    cfg = config()
    assert [g.node_id for g in cfg.ground_stations] == [11, 12, 13]
    assert cfg.frame_count == 361


def test_physical_adapter_creates_direct_to_ground_contact():
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
    positions[:] = np.array([7000.0, 0.0, 0.0])
    # Spread relay points enough that coincident geometry does not matter to this test.
    for i in range(1, 11):
        positions[i] = np.array([7000.0, 20.0 * i, 10.0 * i])

    snapshots = [
        SimpleNamespace(time=0.0, positions=positions.copy()),
        SimpleNamespace(time=5.0, positions=positions.copy()),
        SimpleNamespace(time=10.0, positions=positions.copy()),
    ]
    plan, diagnostics = build_contact_plan(snapshots, cfg)

    direct = [c for c in plan.contacts if c.source_id == 0 and c.destination_id == 11]
    assert direct, "science-to-ground must be a first-class contact"
    assert diagnostics.direct_to_ground_contacts >= 1
    assert direct[0].propagation_delay_s > 0
    assert direct[0].residual_capacity_bytes > 0


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


def test_rl_bridge_preserves_frozen_158_feature_interface():
    cfg = config()
    plan = ContactPlan([
        Contact(0, 1, 0.0, 100.0, 10_000_000.0, range_km=1000.0),
        Contact(0, 11, 10.0, 100.0, 10_000_000.0, range_km=1200.0),
    ])
    bundle = DataBundle("x", 0, 1_000_000, deadline_s=500.0)
    obs, mask = build_observation(plan, bundle, 0.0, cfg)
    assert obs.shape == (OBS_LEN,) == (158,)
    assert mask.shape == (14,)
    assert mask[1] == 1
    assert mask[11] == 1
    assert int(mask.sum()) == 2


def test_integrated_baseline_routes_and_reserves_capacity():
    cfg = config()
    size = 1_000_000
    # Each contact only has room for one bundle, so the second bundle cannot be
    # independently told it owns the exact same path.
    plan = ContactPlan([
        Contact(0, 1, 0.0, 20.0, 8_000_000.0, residual_capacity_bytes=size),
        Contact(1, 11, 0.0, 20.0, 8_000_000.0, residual_capacity_bytes=size),
    ])
    bundles = [
        DataBundle("a", 0, size, created_s=0.0, deadline_s=20.0),
        DataBundle("b", 0, size, created_s=0.1, deadline_s=20.0),
    ]
    results = IntegratedSimulator(plan, cfg).run_baseline(bundles)
    assert results[0].delivered is True
    assert results[0].path == (0, 1, 11)
    assert results[1].delivered is False


class DirectGroundPolicy:
    def choose(self, observation, mask):
        if mask[11]:
            return 11
        return int(np.flatnonzero(mask)[0])


def test_rl_policy_executes_on_canonical_plan_with_fallback_contract():
    cfg = config()
    plan = ContactPlan([
        Contact(0, 11, 0.0, 100.0, 80_000_000.0, range_km=1000.0),
    ])
    bundle = DataBundle("rl", 0, 5_000_000, deadline_s=100.0)
    result = IntegratedSimulator(plan, cfg).run_rl_with_fallback(
        [bundle], DirectGroundPolicy()
    )[0]
    assert result.delivered is True
    assert result.on_time is True
    assert result.path == (0, 11)
    assert result.fallbacks == 0
