from __future__ import annotations

import plotly.graph_objects as go

from src.frontend.figures import build_orbital_figure, build_topology_figure
from src.frontend.layout import topology_positions
from src.frontend.replay import _choose_action, build_replay
from src.integration.capacity import CapacityLedger
from src.integration.config import SCIENCE_IDS, load_config
from src.models.bundle import DataBundle
from src.models.contact import Contact, ContactPlan


def test_topology_layout_covers_fixed_nodes_and_multiple_science_sources():
    layout = topology_positions()
    assert len(layout) == 14
    assert all(node_id in layout for node_id in SCIENCE_IDS)
    assert len({layout[node_id] for node_id in SCIENCE_IDS}) == len(SCIENCE_IDS)
    assert 13 in layout


def test_requested_rl_with_missing_model_is_explicit_temporal_execution():
    cfg = load_config("config/prototype.yaml")
    source_id = SCIENCE_IDS[1]
    plan = ContactPlan([
        Contact(source_id, 11, 0.0, 100.0, 80_000_000.0, range_km=1000.0),
    ])
    bundle = DataBundle("fallback", source_id, 5_000_000, deadline_s=100.0)
    action, actual, reason = _choose_action(
        "rl",
        None,
        CapacityLedger(plan),
        bundle,
        0.0,
        cfg,
    )
    assert action == 11
    assert actual == "temporal"
    assert reason == "rl_model_unavailable"


def test_replay_labels_requested_and_actual_algorithms_separately():
    replay = build_replay(
        bundles=8,
        traffic_seed=123,
        before_policy="rl",
        after_policy="rl",
        switch_time_s=None,
        model_path="RL/rl_env_v0/models/does_not_exist.zip",
        allow_missing_model=True,
    )
    assert replay.model_loaded is False
    assert replay.requested_algorithm_at(100.0) == "rl"

    hops = [hop for bundle in replay.bundle_runs for hop in bundle.hops]
    assert hops, "the replay should include at least one routing decision"
    assert all(hop.requested_algorithm == "rl" for hop in hops)
    assert all(hop.actual_algorithm == "temporal" for hop in hops)
    assert all(hop.fallback_used for hop in hops)


def test_requested_algorithm_switch_is_visible_independently_of_execution():
    replay = build_replay(
        bundles=4,
        traffic_seed=456,
        before_policy="temporal",
        after_policy="rl",
        switch_time_s=600.0,
        model_path="RL/rl_env_v0/models/does_not_exist.zip",
        allow_missing_model=True,
    )
    assert replay.requested_algorithm_at(100.0) == "temporal"
    assert replay.requested_algorithm_at(1200.0) == "rl"


def test_figures_render_with_multiple_science_nodes():
    replay = build_replay(
        bundles=4,
        traffic_seed=321,
        before_policy="temporal",
        after_policy="temporal",
        switch_time_s=None,
        model_path="RL/rl_env_v0/models/does_not_exist.zip",
        allow_missing_model=True,
    )
    t = replay.times[min(10, len(replay.times) - 1)]
    orbital = build_orbital_figure(replay, t)
    topology = build_topology_figure(replay, t)
    assert isinstance(orbital, go.Figure)
    assert isinstance(topology, go.Figure)
    assert len(orbital.data) > 0
    assert len(topology.data) > 0
