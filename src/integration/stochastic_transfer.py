"""Deterministic-seeded stochastic transfer outcomes for physical contacts.

The routing algorithms still choose *which* contact to attempt. This layer then
samples whether that physical transmission succeeds. Failed attempts consume
part of the contact's capacity and wall-clock time, while the bundle remains at
its current holder and must be retried or rerouted.

For fair Temporal-vs-RL benchmarking, randomness is keyed by
(seed, bundle, contact, attempt ordinal). Two algorithms that attempt the same
bundle on the same contact for the same ordinal therefore see the same random
draw, even if their other routing decisions differ. This is a common-random-
numbers design that reduces benchmark noise without forcing both algorithms to
follow the same route.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping

from src.integration.config import PrototypeConfig
from src.models.contact import Contact


@dataclass(frozen=True)
class StochasticTransferSettings:
    enabled: bool = True
    base_failure_probability: float = 0.01
    weather_weight: float = 0.50
    health_weight: float = 0.30
    reliability_weight: float = 0.20
    max_failure_probability: float = 0.90
    min_failure_progress: float = 0.25
    max_failure_progress: float = 0.95
    risk_shaping_weight: float = 2.0
    failure_penalty_base: float = 8.0
    failure_penalty_priority: float = 12.0

    @classmethod
    def from_config(cls, config: PrototypeConfig) -> "StochasticTransferSettings":
        raw: Mapping[str, object] = config.raw.get("stochastic_transfer", {})
        settings = cls(
            enabled=bool(raw.get("enabled", True)),
            base_failure_probability=float(raw.get("base_failure_probability", 0.01)),
            weather_weight=float(raw.get("weather_weight", 0.50)),
            health_weight=float(raw.get("health_weight", 0.30)),
            reliability_weight=float(raw.get("reliability_weight", 0.20)),
            max_failure_probability=float(raw.get("max_failure_probability", 0.90)),
            min_failure_progress=float(raw.get("min_failure_progress", 0.25)),
            max_failure_progress=float(raw.get("max_failure_progress", 0.95)),
            risk_shaping_weight=float(raw.get("risk_shaping_weight", 2.0)),
            failure_penalty_base=float(raw.get("failure_penalty_base", 8.0)),
            failure_penalty_priority=float(raw.get("failure_penalty_priority", 12.0)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not 0.0 <= self.base_failure_probability <= 1.0:
            raise ValueError("base_failure_probability must be in 0..1")
        if min(self.weather_weight, self.health_weight, self.reliability_weight) < 0.0:
            raise ValueError("failure probability weights must be non-negative")
        if not 0.0 <= self.max_failure_probability <= 1.0:
            raise ValueError("max_failure_probability must be in 0..1")
        if not 0.0 < self.min_failure_progress <= self.max_failure_progress <= 1.0:
            raise ValueError("failure progress range must satisfy 0 < min <= max <= 1")
        if self.risk_shaping_weight < 0.0:
            raise ValueError("risk_shaping_weight must be non-negative")
        if min(self.failure_penalty_base, self.failure_penalty_priority) < 0.0:
            raise ValueError("failure penalties must be non-negative")


@dataclass(frozen=True)
class TransferOutcome:
    success: bool
    failure_probability: float
    success_draw: float
    depart_s: float
    event_time_s: float
    arrival_s: float | None
    transfer_progress: float
    capacity_bytes_consumed: int
    bundle_bytes_delivered: int

    @property
    def wasted_capacity_bytes(self) -> int:
        return 0 if self.success else self.capacity_bytes_consumed


def _contact_identity(contact: Contact) -> str:
    return "|".join(
        [
            str(int(contact.source_id)),
            str(int(contact.destination_id)),
            f"{float(contact.start_s):.9f}",
            f"{float(contact.end_s):.9f}",
            str(contact.link_type),
        ]
    )


def _uniform01(*parts: object) -> float:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # 53 bits map cleanly into a Python float mantissa.
    integer = int.from_bytes(digest[:8], "big") >> 11
    return integer / float(1 << 53)


def failure_probability(
    contact: Contact,
    config: PrototypeConfig,
    settings: StochasticTransferSettings | None = None,
) -> float:
    settings = settings or StochasticTransferSettings.from_config(config)
    if not settings.enabled:
        return 0.0

    health = float(config.runtime_defaults.health)
    raw = (
        settings.base_failure_probability
        + settings.weather_weight * float(contact.weather_risk)
        + settings.health_weight * (1.0 - health)
        + settings.reliability_weight * (1.0 - float(contact.reliability))
    )
    return max(0.0, min(settings.max_failure_probability, raw))


def sample_transfer(
    *,
    stochastic_seed: int,
    bundle_id: str,
    contact: Contact,
    attempt_ordinal: int,
    size_bytes: int,
    now_s: float,
    config: PrototypeConfig,
    settings: StochasticTransferSettings | None = None,
) -> TransferOutcome:
    """Sample one atomic transfer attempt.

    A failed attempt does *not* partially deliver the bundle. It consumes the
    fraction of link capacity/time already spent before the failure is detected,
    then leaves the full bundle at the sender. This is conservative and makes
    retries materially costly without requiring fragment-level reassembly.
    """
    if attempt_ordinal < 1:
        raise ValueError("attempt_ordinal must start at 1")
    if size_bytes < 1:
        raise ValueError("size_bytes must be positive")

    settings = settings or StochasticTransferSettings.from_config(config)
    p_fail = failure_probability(contact, config, settings)
    depart_s = max(float(now_s), float(contact.start_s))
    tx_time_s = float(contact.transmission_time_s(size_bytes))
    identity = _contact_identity(contact)

    success_draw = _uniform01(
        "transfer-success",
        int(stochastic_seed),
        bundle_id,
        identity,
        int(attempt_ordinal),
    )
    success = success_draw >= p_fail

    if success:
        transfer_end_s = depart_s + tx_time_s
        arrival_s = transfer_end_s + float(contact.propagation_delay_s)
        return TransferOutcome(
            success=True,
            failure_probability=p_fail,
            success_draw=success_draw,
            depart_s=depart_s,
            event_time_s=arrival_s,
            arrival_s=arrival_s,
            transfer_progress=1.0,
            capacity_bytes_consumed=int(size_bytes),
            bundle_bytes_delivered=int(size_bytes),
        )

    progress_draw = _uniform01(
        "failure-progress",
        int(stochastic_seed),
        bundle_id,
        identity,
        int(attempt_ordinal),
    )
    progress = settings.min_failure_progress + progress_draw * (
        settings.max_failure_progress - settings.min_failure_progress
    )
    progress = max(settings.min_failure_progress, min(settings.max_failure_progress, progress))
    consumed = max(1, min(int(size_bytes), int(math.ceil(size_bytes * progress))))
    failure_time_s = depart_s + tx_time_s * progress

    return TransferOutcome(
        success=False,
        failure_probability=p_fail,
        success_draw=success_draw,
        depart_s=depart_s,
        event_time_s=failure_time_s,
        arrival_s=None,
        transfer_progress=progress,
        capacity_bytes_consumed=consumed,
        bundle_bytes_delivered=0,
    )


class TransferOracle:
    """Stateful ordinal tracker around deterministic seeded transfer draws."""

    def __init__(
        self,
        stochastic_seed: int,
        config: PrototypeConfig,
        settings: StochasticTransferSettings | None = None,
    ):
        self.stochastic_seed = int(stochastic_seed)
        self.config = config
        self.settings = settings or StochasticTransferSettings.from_config(config)
        self._counts: dict[tuple[str, str], int] = {}

    def attempt(
        self,
        *,
        bundle_id: str,
        contact: Contact,
        size_bytes: int,
        now_s: float,
    ) -> TransferOutcome:
        identity = _contact_identity(contact)
        key = (str(bundle_id), identity)
        ordinal = self._counts.get(key, 0) + 1
        self._counts[key] = ordinal
        return sample_transfer(
            stochastic_seed=self.stochastic_seed,
            bundle_id=bundle_id,
            contact=contact,
            attempt_ordinal=ordinal,
            size_bytes=size_bytes,
            now_s=now_s,
            config=self.config,
            settings=self.settings,
        )

    def attempts_for(self, bundle_id: str, contact: Contact) -> int:
        return self._counts.get((str(bundle_id), _contact_identity(contact)), 0)
