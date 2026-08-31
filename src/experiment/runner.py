"""Single execution engine shared by the final benchmark and visual demo.

If a run is labelled as part of the reported experiment, it must pass through
this module.  That prevents the frontend from simulating a simplified or
non-stochastic approximation of the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Callable, Sequence

from src.experiment.spec import FinalExperimentSpec
from src.integration.capacity import CapacityLedger
from src.integration.config import GROUND_IDS, PrototypeConfig, load_config
from src.integration.contact_plan import build_contact_plan
from src.integration.rl_bridge import MaskablePPOPolicy, best_feasible_contact, build_observation
from src.integration.scenario import simulate_snapshots
from src.integration.stochastic_transfer import StochasticTransferSettings, TransferOracle
from src.integration.traffic import generate_bundles
from src.models.bundle import DataBundle
from src.models.contact import ContactPlan
from src.routing.temporal_baseline import earliest_arrival

Algorithm = str


@dataclass(frozen=True)
class ExperimentWorld:
    config: PrototypeConfig
    satellites: tuple[object, ...]
    snapshots: tuple[object, ...]
    plan: ContactPlan
    diagnostics: object


@dataclass(frozen=True)
class AttemptRecord:
    bundle_id: str
    attempt_index: int
    holder_id: int
    destination_id: int
    requested_algorithm: str
    actual_algorithm: str
    fallback_used: bool
    fallback_reason: str | None
    contact_start_s: float
    contact_end_s: float
    failure_probability: float
    success_draw: float
    success: bool
    transfer_progress: float
    capacity_bytes_consumed: int
    depart_s: float
    event_time_s: float
    arrival_s: float | None


@dataclass(frozen=True)
class ExperimentResult:
    bundle_id: str
    source_id: int
    policy: str
    requested_algorithm: str
    actual_algorithms: tuple[str, ...]
    delivered: bool
    on_time: bool
    arrival_s: float | None
    latency_s: float | None
    hops: int
    attempts: int
    transfer_failures: int
    retries: int
    wasted_capacity_bytes: int
    path: tuple[int, ...]
    fallbacks: int
    reason: str
    science_priority: float
    attempt_trace: tuple[AttemptRecord, ...]


def clone_bundle(bundle: DataBundle) -> DataBundle:
    return DataBundle(
        bundle_id=bundle.bundle_id,
        source_id=bundle.source_id,
        size_bytes=bundle.size_bytes,
        created_s=bundle.created_s,
        science_priority=bundle.science_priority,
        deadline_s=bundle.deadline_s,
        data_type=bundle.data_type,
        destination_class=bundle.destination_class,
    )


def build_world(spec: FinalExperimentSpec) -> ExperimentWorld:
    spec.verify_config()
    config = load_config(spec.scenario_config)
    satellites, snapshots = simulate_snapshots(config)
    plan, diagnostics = build_contact_plan(snapshots, config)
    return ExperimentWorld(
        config=config,
        satellites=tuple(satellites),
        snapshots=tuple(snapshots),
        plan=plan,
        diagnostics=diagnostics,
    )


def load_final_policy(spec: FinalExperimentSpec) -> MaskablePPOPolicy:
    spec.verify_model(require_exists=True)
    return MaskablePPOPolicy(spec.model)


def bundles_for_seed(spec: FinalExperimentSpec, config: PrototypeConfig, seed_offset: int) -> tuple[list[DataBundle], int, int]:
    traffic_seed, stochastic_seed = spec.benchmark.seeds(seed_offset)
    bundles = generate_bundles(config, count=spec.benchmark.bundles_per_seed, seed=traffic_seed)
    return bundles, traffic_seed, stochastic_seed


def _temporal_next_hop(ledger: CapacityLedger, bundle: DataBundle, now_s: float) -> int | None:
    plan = ledger.planning_plan()
    route = earliest_arrival(
        plan,
        bundle.current_holder,
        GROUND_IDS,
        bundle.remaining_bytes,
        start_s=now_s,
        deadline_s=bundle.deadline_s,
    )
    if route is None:
        route = earliest_arrival(
            plan,
            bundle.current_holder,
            GROUND_IDS,
            bundle.remaining_bytes,
            start_s=now_s,
            deadline_s=None,
        )
    return None if route is None or not route.hops else int(route.next_hop())


def _rl_next_hop(
    ledger: CapacityLedger,
    config: PrototypeConfig,
    bundle: DataBundle,
    now_s: float,
    policy: MaskablePPOPolicy,
) -> tuple[int | None, str | None]:
    observation, mask = build_observation(ledger.planning_plan(), bundle, now_s, config)
    if not mask.any():
        return None, "no_feasible_action"
    try:
        action = int(policy.choose(observation, mask))
    except Exception as exc:  # runtime failures are part of pure-RL evaluation
        return None, f"policy_error:{type(exc).__name__}"
    if action < 0 or action >= len(mask) or not mask[action]:
        return None, "invalid_action"
    return action, None


def _choose_action(
    requested: str,
    *,
    ledger: CapacityLedger,
    config: PrototypeConfig,
    bundle: DataBundle,
    now_s: float,
    policy: MaskablePPOPolicy | None,
    allow_temporal_fallback: bool,
) -> tuple[int | None, str, bool, str | None]:
    if requested == "temporal":
        return _temporal_next_hop(ledger, bundle, now_s), "temporal", False, None
    if requested != "rl":
        raise ValueError(f"unknown requested algorithm: {requested}")
    if policy is None:
        if not allow_temporal_fallback:
            return None, "rl", False, "model_unavailable"
        return _temporal_next_hop(ledger, bundle, now_s), "temporal", True, "model_unavailable"

    action, error = _rl_next_hop(ledger, config, bundle, now_s, policy)
    if action is not None:
        return action, "rl", False, None
    if not allow_temporal_fallback:
        return None, "rl", False, error
    return _temporal_next_hop(ledger, bundle, now_s), "temporal", True, error


def run_algorithm(
    *,
    algorithm: str,
    plan: ContactPlan,
    config: PrototypeConfig,
    bundles: Sequence[DataBundle],
    policy: MaskablePPOPolicy | None,
    stochastic_seed: int,
    max_hops: int = 8,
    max_attempts: int = 24,
    requested_at: Callable[[float], str] | None = None,
) -> list[ExperimentResult]:
    """Execute one algorithm with the exact stochastic/capacity semantics.

    Reported benchmark modes are `temporal` and `rl_pure`.  The optional
    `rl_with_temporal_fallback` and time-varying `requested_at` modes are useful
    for operations demos but are explicitly not the headline benchmark.
    """
    valid = {"temporal", "rl_pure", "rl_with_temporal_fallback", "scheduled"}
    if algorithm not in valid:
        raise ValueError(f"unknown algorithm {algorithm}")
    if algorithm.startswith("rl") and policy is None:
        raise ValueError("RL algorithm requires the final PPO policy")
    if algorithm == "scheduled" and requested_at is None:
        raise ValueError("scheduled mode requires requested_at(time_s)")

    ledger = CapacityLedger(plan)
    settings = StochasticTransferSettings.from_config(config)
    oracle = TransferOracle(stochastic_seed, config, settings)
    results: list[ExperimentResult] = []

    for original in sorted(bundles, key=lambda item: (item.created_s, item.bundle_id)):
        bundle = clone_bundle(original)
        now_s = bundle.created_s
        path = [bundle.source_id]
        actual_algorithms: list[str] = []
        attempt_trace: list[AttemptRecord] = []
        attempts = failures = fallbacks = wasted = successful_hops = 0
        reason = "no_route"

        while attempts < max_attempts and successful_hops < max_hops:
            if bundle.current_holder in GROUND_IDS:
                reason = "delivered"
                break

            if algorithm == "temporal":
                requested = "temporal"
                allow_fallback = False
            elif algorithm == "rl_pure":
                requested = "rl"
                allow_fallback = False
            elif algorithm == "rl_with_temporal_fallback":
                requested = "rl"
                allow_fallback = True
            else:
                requested = str(requested_at(now_s))  # type: ignore[misc]
                allow_fallback = True

            action, actual, fallback_used, fallback_reason = _choose_action(
                requested,
                ledger=ledger,
                config=config,
                bundle=bundle,
                now_s=now_s,
                policy=policy,
                allow_temporal_fallback=allow_fallback,
            )
            if fallback_used:
                fallbacks += 1
            if action is None:
                reason = fallback_reason or "no_route"
                break

            contact = best_feasible_contact(
                ledger.planning_plan(), bundle.current_holder, int(action), now_s, bundle.remaining_bytes
            )
            if contact is None and allow_fallback and actual == "rl":
                fallbacks += 1
                fallback_used = True
                fallback_reason = "contact_drift"
                action = _temporal_next_hop(ledger, bundle, now_s)
                actual = "temporal"
                if action is not None:
                    contact = best_feasible_contact(
                        ledger.planning_plan(), bundle.current_holder, int(action), now_s, bundle.remaining_bytes
                    )
            if contact is None:
                reason = "contact_drift"
                break

            outcome = oracle.attempt(
                bundle_id=bundle.bundle_id,
                contact=contact,
                size_bytes=bundle.remaining_bytes,
                now_s=now_s,
            )
            ledger.reserve_contact(contact, outcome.capacity_bytes_consumed)
            attempts += 1
            actual_algorithms.append(actual)
            attempt_trace.append(
                AttemptRecord(
                    bundle_id=bundle.bundle_id,
                    attempt_index=attempts,
                    holder_id=bundle.current_holder,
                    destination_id=int(action),
                    requested_algorithm=requested,
                    actual_algorithm=actual,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason if fallback_used else None,
                    contact_start_s=float(contact.start_s),
                    contact_end_s=float(contact.end_s),
                    failure_probability=outcome.failure_probability,
                    success_draw=outcome.success_draw,
                    success=outcome.success,
                    transfer_progress=outcome.transfer_progress,
                    capacity_bytes_consumed=outcome.capacity_bytes_consumed,
                    depart_s=outcome.depart_s,
                    event_time_s=outcome.event_time_s,
                    arrival_s=outcome.arrival_s,
                )
            )

            if not outcome.success:
                failures += 1
                wasted += outcome.wasted_capacity_bytes
                now_s = outcome.event_time_s
                reason = "transmission_failed"
                if bundle.deadline_s is not None and now_s > bundle.deadline_s:
                    reason = "missed_deadline_after_failure"
                    break
                continue

            now_s = float(outcome.arrival_s)
            bundle.current_holder = int(action)
            bundle.route_history.append(int(action))
            path.append(int(action))
            successful_hops += 1

            if bundle.current_holder in GROUND_IDS:
                reason = "delivered"
                break
            if bundle.deadline_s is not None and now_s > bundle.deadline_s:
                reason = "missed_deadline"
                break
        else:
            reason = "max_attempts" if attempts >= max_attempts else "max_hops"

        delivered = bundle.current_holder in GROUND_IDS
        on_time = delivered and (bundle.deadline_s is None or now_s <= bundle.deadline_s)
        if delivered:
            reason = "delivered_on_time" if on_time else "delivered_late"

        requested_label = (
            "temporal" if algorithm == "temporal" else
            "rl" if algorithm.startswith("rl") else
            "scheduled"
        )
        results.append(
            ExperimentResult(
                bundle_id=bundle.bundle_id,
                source_id=bundle.source_id,
                policy=algorithm,
                requested_algorithm=requested_label,
                actual_algorithms=tuple(actual_algorithms),
                delivered=delivered,
                on_time=on_time,
                arrival_s=now_s if delivered else None,
                latency_s=(now_s - bundle.created_s) if delivered else None,
                hops=successful_hops,
                attempts=attempts,
                transfer_failures=failures,
                retries=failures,
                wasted_capacity_bytes=wasted,
                path=tuple(path),
                fallbacks=fallbacks,
                reason=reason,
                science_priority=bundle.science_priority,
                attempt_trace=tuple(attempt_trace),
            )
        )

    return results


def aggregate(rows: Sequence[ExperimentResult]) -> dict[str, float]:
    if not rows:
        keys = [
            "delivery_ratio", "deadline_success", "priority_weighted_timely",
            "mean_latency_s", "mean_hops", "mean_attempts", "transfer_failure_rate",
            "mean_failures", "mean_wasted_mb", "fallback_rate",
        ]
        return {key: float("nan") for key in keys}
    delivered = [row for row in rows if row.delivered]
    total_priority = sum(row.science_priority for row in rows)
    timely_priority = sum(row.science_priority for row in rows if row.on_time)
    attempts = sum(row.attempts for row in rows)
    failures = sum(row.transfer_failures for row in rows)
    executed_hops = sum(row.hops for row in rows)
    fallbacks = sum(row.fallbacks for row in rows)
    return {
        "delivery_ratio": len(delivered) / len(rows),
        "deadline_success": sum(row.on_time for row in rows) / len(rows),
        "priority_weighted_timely": timely_priority / total_priority if total_priority else 0.0,
        "mean_latency_s": fmean(row.latency_s for row in delivered) if delivered else float("nan"),
        "mean_hops": fmean(row.hops for row in rows),
        "mean_attempts": fmean(row.attempts for row in rows),
        "transfer_failure_rate": failures / attempts if attempts else 0.0,
        "mean_failures": fmean(row.transfer_failures for row in rows),
        "mean_wasted_mb": fmean(row.wasted_capacity_bytes / 1_000_000 for row in rows),
        "fallback_rate": fallbacks / executed_hops if executed_hops else 0.0,
    }
