from __future__ import annotations

from benchmark_temporal_vs_rl_stochastic import run_algorithm
from src.integration.config import load_config
from src.models.bundle import DataBundle
from src.models.contact import Contact, ContactPlan


def test_temporal_benchmark_is_reproducible_under_common_random_seed():
    config = load_config("config/prototype.yaml")
    plan = ContactPlan(
        [
            Contact(
                source_id=0,
                destination_id=11,
                start_s=0.0,
                end_s=1000.0,
                data_rate_bps=8_000_000.0,
                range_km=1000.0,
                propagation_delay_s=0.01,
                residual_capacity_bytes=100_000_000,
                reliability=0.9,
                weather_risk=0.2,
                energy_cost=0.1,
                link_type="SCIENCE_GROUND",
            )
        ]
    )
    bundles = [
        DataBundle(
            bundle_id="B0",
            source_id=0,
            size_bytes=1_000_000,
            created_s=0.0,
            science_priority=0.95,
            deadline_s=900.0,
            data_type="TRANSIENT",
        )
    ]

    first = run_algorithm(
        algorithm="temporal",
        plan=plan,
        config=config,
        bundles=bundles,
        policy=None,
        stochastic_seed=777,
    )
    second = run_algorithm(
        algorithm="temporal",
        plan=plan,
        config=config,
        bundles=bundles,
        policy=None,
        stochastic_seed=777,
    )
    assert first == second
