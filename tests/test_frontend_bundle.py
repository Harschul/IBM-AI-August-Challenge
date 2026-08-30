from __future__ import annotations

import plotly.graph_objects as go

from src.frontend.figures import build_orbital_figure, build_topology_figure
from src.frontend.layout import topology_positions
from src.frontend.replay import build_replay


def test_topology_layout_covers_fixed_nodes():
    layout = topology_positions()
    assert len(layout) == 14
    assert 0 in layout
    assert 13 in layout


def test_replay_builds_without_rl_checkpoint():
    replay = build_replay(
        bundles=4,
        traffic_seed=123,
        before_policy="temporal",
        after_policy="rl",
        switch_time_s=600.0,
        model_path="RL/rl_env_v0/models/does_not_exist.zip",
        allow_missing_model=True,
    )
    assert len(replay.bundle_runs) == 4
    assert replay.switch_time_s == 600.0
    assert replay.current_policy_label(100.0) == "temporal"
    assert replay.current_policy_label(1200.0) == "rl"


def test_figures_render():
    replay = build_replay(
        bundles=3,
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
