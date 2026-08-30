"""End-to-end bundle execution on one shared physical contact plan."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Protocol, Sequence

from src.models.bundle import DataBundle
from src.models.contact import ContactPlan
from src.routing.temporal_baseline import earliest_arrival

from .capacity import CapacityLedger
from .config import GROUND_IDS, PrototypeConfig
from .rl_bridge import action_mask, best_feasible_contact, build_observation


@dataclass(frozen=True)
class DeliveryResult:
    bundle_id: str
    policy: str
    delivered: bool
    on_time: bool
    arrival_s: float | None
    latency_s: float | None
    hops: int
    path: tuple[int, ...]
    fallbacks: int
    reason: str
    science_priority: float


class NextHopPolicy(Protocol):
    def choose(self, observation, mask) -> int: ...


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


class IntegratedSimulator:
    def __init__(self, contact_plan: ContactPlan, config: PrototypeConfig):
        self.contact_plan = contact_plan
        self.config = config

    def _baseline_route(self, ledger: CapacityLedger, bundle: DataBundle, now_s: float):
        plan = ledger.planning_plan()
        # First try to preserve scientific value. If no on-time route exists,
        # still allow best-effort late delivery rather than silently dropping.
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
        return route

    def run_baseline(self, bundles: Sequence[DataBundle]) -> list[DeliveryResult]:
        ledger = CapacityLedger(self.contact_plan)
        results: list[DeliveryResult] = []

        for original in sorted(bundles, key=lambda b: (b.created_s, b.bundle_id)):
            bundle = clone_bundle(original)
            route = self._baseline_route(ledger, bundle, bundle.created_s)
            if route is None:
                results.append(
                    DeliveryResult(
                        bundle_id=bundle.bundle_id,
                        policy="baseline",
                        delivered=False,
                        on_time=False,
                        arrival_s=None,
                        latency_s=None,
                        hops=0,
                        path=(bundle.source_id,),
                        fallbacks=0,
                        reason="no_route",
                        science_priority=bundle.science_priority,
                    )
                )
                continue

            ledger.reserve_route(route, bundle.remaining_bytes)
            arrival = route.arrival_s
            deadline = bundle.deadline_s
            on_time = deadline is None or arrival <= deadline
            results.append(
                DeliveryResult(
                    bundle_id=bundle.bundle_id,
                    policy="baseline",
                    delivered=True,
                    on_time=on_time,
                    arrival_s=arrival,
                    latency_s=arrival - bundle.created_s,
                    hops=len(route.hops),
                    path=tuple(route.path_ids),
                    fallbacks=0,
                    reason="delivered_on_time" if on_time else "delivered_late",
                    science_priority=bundle.science_priority,
                )
            )

        return results

    def run_rl_with_fallback(
        self,
        bundles: Sequence[DataBundle],
        policy: NextHopPolicy,
        *,
        max_hops: int = 8,
    ) -> list[DeliveryResult]:
        ledger = CapacityLedger(self.contact_plan)
        results: list[DeliveryResult] = []

        for original in sorted(bundles, key=lambda b: (b.created_s, b.bundle_id)):
            bundle = clone_bundle(original)
            now_s = bundle.created_s
            path = [bundle.source_id]
            fallbacks = 0
            reason = "no_route"

            for _ in range(max_hops):
                if bundle.current_holder in GROUND_IDS:
                    reason = "delivered"
                    break

                planning_plan = ledger.planning_plan()
                obs, mask = build_observation(planning_plan, bundle, now_s, self.config)

                action = None
                if mask.any():
                    try:
                        candidate = int(policy.choose(obs, mask))
                        if 0 <= candidate < len(mask) and mask[candidate]:
                            action = candidate
                    except Exception:
                        # A model loading/runtime error is not allowed to become
                        # a data-loss mode. The deterministic router is the safety path.
                        action = None

                if action is None:
                    fallbacks += 1
                    route = self._baseline_route(ledger, bundle, now_s)
                    if route is None or not route.hops:
                        reason = "no_route"
                        break
                    action = route.next_hop()

                planning_plan = ledger.planning_plan()
                contact = best_feasible_contact(
                    planning_plan,
                    bundle.current_holder,
                    action,
                    now_s,
                    bundle.remaining_bytes,
                )
                if contact is None:
                    # Policy and current plan drifted between decision and
                    # commit; recompute through the baseline once.
                    fallbacks += 1
                    route = self._baseline_route(ledger, bundle, now_s)
                    if route is None or not route.hops:
                        reason = "no_route"
                        break
                    action = route.next_hop()
                    contact = best_feasible_contact(
                        ledger.planning_plan(),
                        bundle.current_holder,
                        action,
                        now_s,
                        bundle.remaining_bytes,
                    )
                    if contact is None:
                        reason = "commit_race"
                        break

                ledger.reserve_contact(contact, bundle.remaining_bytes)
                depart = max(now_s, contact.start_s)
                now_s = (
                    depart
                    + contact.transmission_time_s(bundle.remaining_bytes)
                    + contact.propagation_delay_s
                )
                bundle.current_holder = int(action)
                bundle.route_history.append(int(action))
                path.append(int(action))

                if action in GROUND_IDS:
                    reason = "delivered"
                    break

            delivered = bundle.current_holder in GROUND_IDS
            deadline = bundle.deadline_s
            on_time = delivered and (deadline is None or now_s <= deadline)
            results.append(
                DeliveryResult(
                    bundle_id=bundle.bundle_id,
                    policy="rl_fallback",
                    delivered=delivered,
                    on_time=on_time,
                    arrival_s=now_s if delivered else None,
                    latency_s=(now_s - bundle.created_s) if delivered else None,
                    hops=len(path) - 1,
                    path=tuple(path),
                    fallbacks=fallbacks,
                    reason=(
                        "delivered_on_time"
                        if delivered and on_time
                        else "delivered_late"
                        if delivered
                        else reason
                    ),
                    science_priority=bundle.science_priority,
                )
            )

        return results


def aggregate_results(results: Sequence[DeliveryResult]) -> dict[str, float]:
    if not results:
        return {
            "bundles": 0,
            "delivery_ratio": 0.0,
            "deadline_success": 0.0,
            "priority_weighted_timely": 0.0,
            "mean_latency_s": float("nan"),
            "mean_hops": float("nan"),
            "fallback_rate": 0.0,
        }

    delivered = [r for r in results if r.delivered]
    total_priority = sum(r.science_priority for r in results)
    timely_priority = sum(r.science_priority for r in results if r.on_time)
    total_hops = sum(r.hops for r in results)
    total_fallbacks = sum(r.fallbacks for r in results)

    return {
        "bundles": float(len(results)),
        "delivery_ratio": len(delivered) / len(results),
        "deadline_success": sum(1 for r in results if r.on_time) / len(results),
        "priority_weighted_timely": (
            timely_priority / total_priority if total_priority else 0.0
        ),
        "mean_latency_s": (
            fmean(r.latency_s for r in delivered if r.latency_s is not None)
            if delivered
            else float("nan")
        ),
        "mean_hops": fmean(r.hops for r in results),
        "fallback_rate": total_fallbacks / total_hops if total_hops else 0.0,
    }
