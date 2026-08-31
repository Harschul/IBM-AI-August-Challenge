from __future__ import annotations

from src.integration.config import load_config
from src.integration.stochastic_transfer import (
    StochasticTransferSettings,
    TransferOracle,
    failure_probability,
    sample_transfer,
)
from src.models.contact import Contact


def contact(*, reliability=1.0, weather=0.0):
    return Contact(
        source_id=0,
        destination_id=3,
        start_s=10.0,
        end_s=110.0,
        data_rate_bps=8_000_000.0,
        range_km=1000.0,
        propagation_delay_s=0.01,
        residual_capacity_bytes=100_000_000,
        reliability=reliability,
        weather_risk=weather,
        energy_cost=0.1,
        link_type="SCIENCE_LEO",
    )


def test_common_random_numbers_are_reproducible():
    cfg = load_config("config/prototype.yaml")
    c = contact(reliability=0.9, weather=0.1)
    a = TransferOracle(12345, cfg)
    b = TransferOracle(12345, cfg)

    first_a = a.attempt(bundle_id="B1", contact=c, size_bytes=1_000_000, now_s=0.0)
    first_b = b.attempt(bundle_id="B1", contact=c, size_bytes=1_000_000, now_s=0.0)
    second_a = a.attempt(bundle_id="B1", contact=c, size_bytes=1_000_000, now_s=0.0)
    second_b = b.attempt(bundle_id="B1", contact=c, size_bytes=1_000_000, now_s=0.0)

    assert first_a == first_b
    assert second_a == second_b


def test_forced_failure_consumes_partial_capacity_and_no_delivery():
    cfg = load_config("config/prototype.yaml")
    c = contact()
    settings = StochasticTransferSettings(
        enabled=True,
        base_failure_probability=1.0,
        weather_weight=0.0,
        health_weight=0.0,
        reliability_weight=0.0,
        max_failure_probability=1.0,
        min_failure_progress=0.5,
        max_failure_progress=0.5,
    )
    outcome = sample_transfer(
        stochastic_seed=1,
        bundle_id="B2",
        contact=c,
        attempt_ordinal=1,
        size_bytes=2_000_000,
        now_s=0.0,
        config=cfg,
        settings=settings,
    )
    assert not outcome.success
    assert outcome.capacity_bytes_consumed == 1_000_000
    assert outcome.bundle_bytes_delivered == 0
    assert outcome.arrival_s is None
    # Wait to contact start (10s), then half of a 2s full transmission.
    assert abs(outcome.event_time_s - 11.0) < 1e-9


def test_forced_success_consumes_full_capacity_and_arrives():
    cfg = load_config("config/prototype.yaml")
    c = contact()
    settings = StochasticTransferSettings(
        enabled=True,
        base_failure_probability=0.0,
        weather_weight=0.0,
        health_weight=0.0,
        reliability_weight=0.0,
        max_failure_probability=1.0,
    )
    outcome = sample_transfer(
        stochastic_seed=2,
        bundle_id="B3",
        contact=c,
        attempt_ordinal=1,
        size_bytes=2_000_000,
        now_s=0.0,
        config=cfg,
        settings=settings,
    )
    assert outcome.success
    assert outcome.capacity_bytes_consumed == 2_000_000
    assert outcome.bundle_bytes_delivered == 2_000_000
    assert abs(outcome.arrival_s - 12.01) < 1e-9


def test_bad_weather_and_low_reliability_raise_failure_probability():
    cfg = load_config("config/prototype.yaml")
    good = contact(reliability=0.999, weather=0.0)
    bad = contact(reliability=0.8, weather=0.4)
    assert failure_probability(bad, cfg) > failure_probability(good, cfg)
