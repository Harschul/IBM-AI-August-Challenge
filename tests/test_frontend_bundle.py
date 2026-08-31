from __future__ import annotations

import plotly.graph_objects as go

from src.experiment.runner import run_algorithm
from src.frontend.figures import build_orbital_figure, build_topology_figure
from src.frontend.layout import topology_positions
from src.frontend.replay import build_replay
from src.integration.config import SCIENCE_IDS, load_config
from src.models.bundle import DataBundle
from src.models.contact import Contact, ContactPlan


def test_topology_layout_covers_fixed_nodes_and_multiple_science_sources():
    layout = topology_positions()
    assert len(layout) == 14
    assert all(node_id in layout for node_id in SCIENCE_IDS)
    assert len({layout[node_id] for node_id in SCIENCE_IDS}) == len(SCIENCE_IDS)
    assert 13 in layout


class BrokenPolicy:
    def choose(self, observation, mask):
        raise RuntimeError("intentional")


def test_fallback_records_requested_rl_but_actual_temporal():
    cfg = load_config("config/prototype.yaml")
    plan = ContactPlan([
        Contact(0, 11, 0.0, 100.0, 80_000_000.0, range_km=1000.0, residual_capacity_bytes=20_000_000),
    ])
    bundle = DataBundle("fallback", 0, 5_000_000, deadline_s=100.0)
    row = run_algorithm(
        algorithm="rl_with_temporal_fallback",
        plan=plan,
        config=cfg,
        bundles=[bundle],
        policy=BrokenPolicy(),
        stochastic_seed=777,
    )[0]
    assert row.attempt_trace
    attempt = row.attempt_trace[0]
    assert attempt.requested_algorithm == "rl"
    assert attempt.actual_algorithm == "temporal"
    assert attempt.fallback_used is True


def test_reported_temporal_replay_uses_locked_benchmark_seed_family():
    replay = build_replay(mode="reported_temporal", seed_offset=0)
    assert replay.reported_experiment is True
    assert replay.traffic_seed == replay.spec.benchmark.traffic_seed_base
    assert replay.stochastic_seed == replay.spec.benchmark.stochastic_seed_base
    assert len(replay.bundle_runs) == replay.spec.benchmark.bundles_per_seed
    assert replay.requested_algorithm_at(100.0) == "temporal"


def test_figures_render_from_reported_temporal_experiment():
    replay = build_replay(mode="reported_temporal", seed_offset=0)
    t = replay.times[min(10, len(replay.times) - 1)]
    orbital = build_orbital_figure(replay, t)
    topology = build_topology_figure(replay, t)
    assert isinstance(orbital, go.Figure)
    assert isinstance(topology, go.Figure)
    assert len(orbital.data) > 0
    assert len(topology.data) > 0
