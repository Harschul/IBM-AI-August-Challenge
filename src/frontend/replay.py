"""Replay generation for the integrated orbital/network frontend.

The replay records both the routing algorithm the operator *requested* and the
algorithm that *actually* selected every hop.  This is intentionally explicit:
an RL request that falls back to the temporal router is displayed and exported
as temporal execution, never as RL execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Literal

import numpy as np

from src.integration.capacity import CapacityLedger
from src.integration.config import GEO_IDS, GROUND_IDS, LEO_IDS, SCIENCE_IDS, PrototypeConfig, load_config
from src.integration.contact_plan import build_contact_plan, ground_position
from src.integration.rl_bridge import MaskablePPOPolicy, best_feasible_contact, build_observation
from src.integration.scenario import simulate_snapshots
from src.integration.traffic import generate_bundles
from src.models.bundle import DataBundle
from src.models.contact import Contact, ContactPlan
from src.routing.temporal_baseline import earliest_arrival

AlgorithmName = Literal["temporal", "rl"]


@dataclass(frozen=True)
class PacketHop:
    bundle_id: str
    hop_index: int
    source_id: int
    destination_id: int
    depart_s: float
    transfer_end_s: float
    arrival_s: float
    requested_algorithm: AlgorithmName
    actual_algorithm: AlgorithmName
    fallback_used: bool
    fallback_reason: str | None
    data_rate_bps: float
    priority: float
    contact_start_s: float
    contact_end_s: float


@dataclass(frozen=True)
class BundlePlayback:
    bundle_id: str
    source_id: int
    created_s: float
    size_bytes: int
    science_priority: float
    deadline_s: float | None
    data_type: str
    path: tuple[int, ...]
    delivered: bool
    on_time: bool
    arrival_s: float | None
    reason: str
    fallbacks: int
    hops: tuple[PacketHop, ...]

    @property
    def requested_algorithms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(hop.requested_algorithm for hop in self.hops))

    @property
    def actual_algorithms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(hop.actual_algorithm for hop in self.hops))


@dataclass(frozen=True)
class ActivePacket:
    bundle_id: str
    source_id: int
    destination_id: int
    xyz: tuple[float, float, float]
    topology_xy: tuple[float, float]
    progress: float
    requested_algorithm: AlgorithmName
    actual_algorithm: AlgorithmName
    priority: float
    fallback_used: bool
    fallback_reason: str | None


@dataclass(frozen=True)
class ReplaySummary:
    bundles: int
    delivered: int
    on_time: int
    mean_latency_s: float | None
    mean_hops: float
    total_fallbacks: int
    delivery_ratio: float
    deadline_success: float
    priority_weighted_timely: float
    rl_requested_hops: int
    rl_executed_hops: int
    temporal_executed_hops: int


@dataclass(frozen=True)
class ReplayData:
    config: PrototypeConfig
    satellites: tuple[object, ...]
    snapshots: tuple[object, ...]
    plan: ContactPlan
    diagnostics: object
    bundle_runs: tuple[BundlePlayback, ...]
    before_policy: AlgorithmName
    after_policy: AlgorithmName
    switch_time_s: float | None
    selected_bundle_id: str | None = None
    model_loaded: bool = False
    model_path: str | None = None

    @property
    def times(self) -> tuple[float, ...]:
        return tuple(float(snapshot.time) for snapshot in self.snapshots)

    @property
    def horizon_s(self) -> float:
        return float(self.times[-1]) if self.times else 0.0

    @property
    def topology_positions(self) -> dict[int, tuple[float, float]]:
        from src.frontend.layout import topology_positions

        return topology_positions()

    def node_position_3d(self, node_id: int, time_s: float) -> np.ndarray:
        if node_id in GROUND_IDS:
            return ground_position(self.config, node_id, time_s)
        frame_index = self.frame_index_at(time_s)
        return np.asarray(self.snapshots[frame_index].positions[node_id], dtype=float)

    def frame_index_at(self, time_s: float) -> int:
        if not self.snapshots:
            return 0
        step = self.config.sample_step_s
        index = int(round(max(0.0, min(self.horizon_s, float(time_s))) / step))
        return max(0, min(len(self.snapshots) - 1, index))

    def active_contacts(self, time_s: float) -> list[Contact]:
        t = float(time_s)
        return [c for c in self.plan.contacts if c.start_s <= t <= c.end_s]

    def active_packets(self, time_s: float) -> list[ActivePacket]:
        t = float(time_s)
        topo = self.topology_positions
        packets: list[ActivePacket] = []
        for bundle in self.bundle_runs:
            for hop in bundle.hops:
                if hop.depart_s <= t <= hop.arrival_s:
                    duration = max(1e-9, hop.arrival_s - hop.depart_s)
                    alpha = max(0.0, min(1.0, (t - hop.depart_s) / duration))
                    src_xyz = self.node_position_3d(hop.source_id, t)
                    dst_xyz = self.node_position_3d(hop.destination_id, t)
                    xyz = src_xyz + alpha * (dst_xyz - src_xyz)
                    sx, sy = topo[hop.source_id]
                    dx, dy = topo[hop.destination_id]
                    topo_xy = (sx + alpha * (dx - sx), sy + alpha * (dy - sy))
                    packets.append(
                        ActivePacket(
                            bundle_id=hop.bundle_id,
                            source_id=hop.source_id,
                            destination_id=hop.destination_id,
                            xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                            topology_xy=topo_xy,
                            progress=alpha,
                            requested_algorithm=hop.requested_algorithm,
                            actual_algorithm=hop.actual_algorithm,
                            priority=hop.priority,
                            fallback_used=hop.fallback_used,
                            fallback_reason=hop.fallback_reason,
                        )
                    )
                    break
        return packets

    def requested_algorithm_at(self, time_s: float) -> AlgorithmName:
        if self.switch_time_s is None:
            return self.before_policy
        return self.before_policy if float(time_s) < self.switch_time_s else self.after_policy

    # Backward-compatible alias for old frontend callers. It is explicitly the
    # requested policy, not the algorithm that actually executed.
    def current_policy_label(self, time_s: float) -> str:
        return self.requested_algorithm_at(time_s)

    def actual_algorithm_at(self, time_s: float) -> str:
        """Report actual execution at the current replay point.

        Active transfers take precedence.  Between transfers, the most recent
        routing decision is reported.  Multiple simultaneous transfers can
        legitimately use different algorithms, in which case both are shown.
        """
        active = self.active_packets(time_s)
        if active:
            values = sorted({packet.actual_algorithm for packet in active})
            return " + ".join(values)

        latest: PacketHop | None = None
        t = float(time_s)
        for bundle in self.bundle_runs:
            for hop in bundle.hops:
                if hop.depart_s <= t and (latest is None or hop.depart_s > latest.depart_s):
                    latest = hop
        return latest.actual_algorithm if latest is not None else "none"

    def selected_bundle(self, bundle_id: str | None = None) -> BundlePlayback | None:
        wanted = bundle_id or self.selected_bundle_id
        if wanted is None and self.bundle_runs:
            return self.bundle_runs[0]
        for bundle in self.bundle_runs:
            if bundle.bundle_id == wanted:
                return bundle
        return None

    def summary(self) -> ReplaySummary:
        rows = self.bundle_runs
        delivered = [r for r in rows if r.delivered]
        on_time = [r for r in rows if r.on_time]
        all_hops = [hop for row in rows for hop in row.hops]
        total_priority = sum(r.science_priority for r in rows)
        on_time_priority = sum(r.science_priority for r in rows if r.on_time)
        return ReplaySummary(
            bundles=len(rows),
            delivered=len(delivered),
            on_time=len(on_time),
            mean_latency_s=(fmean(r.arrival_s - r.created_s for r in delivered) if delivered else None),
            mean_hops=fmean(len(r.hops) for r in rows) if rows else 0.0,
            total_fallbacks=sum(1 for hop in all_hops if hop.fallback_used),
            delivery_ratio=(len(delivered) / len(rows) if rows else 0.0),
            deadline_success=(len(on_time) / len(rows) if rows else 0.0),
            priority_weighted_timely=(on_time_priority / total_priority if total_priority else 0.0),
            rl_requested_hops=sum(1 for hop in all_hops if hop.requested_algorithm == "rl"),
            rl_executed_hops=sum(1 for hop in all_hops if hop.actual_algorithm == "rl"),
            temporal_executed_hops=sum(1 for hop in all_hops if hop.actual_algorithm == "temporal"),
        )

    def event_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for bundle in self.bundle_runs:
            rows.append(
                {
                    "time_s": round(bundle.created_s, 2),
                    "bundle_id": bundle.bundle_id,
                    "event": "created",
                    "detail": f"source={node_label(bundle.source_id, self.config)} {bundle.data_type} priority={bundle.science_priority:.2f}",
                    "requested_algorithm": "",
                    "actual_algorithm": "",
                    "fallback": False,
                }
            )
            for hop in bundle.hops:
                rows.append(
                    {
                        "time_s": round(hop.depart_s, 2),
                        "bundle_id": bundle.bundle_id,
                        "event": "depart",
                        "detail": f"{node_label(hop.source_id, self.config)} -> {node_label(hop.destination_id, self.config)}",
                        "requested_algorithm": hop.requested_algorithm,
                        "actual_algorithm": hop.actual_algorithm,
                        "fallback": hop.fallback_used,
                    }
                )
                rows.append(
                    {
                        "time_s": round(hop.arrival_s, 2),
                        "bundle_id": bundle.bundle_id,
                        "event": "arrive",
                        "detail": f"at {node_label(hop.destination_id, self.config)}",
                        "requested_algorithm": hop.requested_algorithm,
                        "actual_algorithm": hop.actual_algorithm,
                        "fallback": hop.fallback_used,
                    }
                )
            rows.append(
                {
                    "time_s": round(bundle.arrival_s if bundle.arrival_s is not None else self.horizon_s, 2),
                    "bundle_id": bundle.bundle_id,
                    "event": bundle.reason,
                    "detail": f"delivered={bundle.delivered} on_time={bundle.on_time}",
                    "requested_algorithm": "",
                    "actual_algorithm": "",
                    "fallback": False,
                }
            )
        rows.sort(key=lambda row: (float(row["time_s"]), str(row["bundle_id"]), str(row["event"])))
        return rows


def node_label(node_id: int, config: PrototypeConfig) -> str:
    if node_id in SCIENCE_IDS:
        return f"SCI-{SCIENCE_IDS.index(node_id)}"
    if node_id in LEO_IDS:
        return f"LEO-{node_id}"
    if node_id in GEO_IDS:
        return f"GEO-{node_id}"
    if node_id in GROUND_IDS:
        station = next(gs for gs in config.ground_stations if gs.node_id == node_id)
        return station.name
    return str(node_id)


def _clone_bundle(bundle: DataBundle) -> DataBundle:
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


def _baseline_route(ledger: CapacityLedger, bundle: DataBundle, now_s: float):
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
    return route


def _temporal_action(ledger: CapacityLedger, bundle: DataBundle, now_s: float) -> int | None:
    route = _baseline_route(ledger, bundle, now_s)
    if route is None or not route.hops:
        return None
    return int(route.next_hop())


def _choose_action(
    requested_algorithm: AlgorithmName,
    policy: MaskablePPOPolicy | None,
    ledger: CapacityLedger,
    bundle: DataBundle,
    now_s: float,
    config: PrototypeConfig,
) -> tuple[int | None, AlgorithmName, str | None]:
    """Return action, actual algorithm, and optional fallback reason."""
    if requested_algorithm == "temporal":
        return _temporal_action(ledger, bundle, now_s), "temporal", None

    if policy is None:
        return _temporal_action(ledger, bundle, now_s), "temporal", "rl_model_unavailable"

    planning = ledger.planning_plan()
    obs, mask = build_observation(planning, bundle, now_s, config)
    if not mask.any():
        return _temporal_action(ledger, bundle, now_s), "temporal", "rl_no_legal_action"

    try:
        candidate = int(policy.choose(obs, mask))
    except Exception:
        return _temporal_action(ledger, bundle, now_s), "temporal", "rl_inference_error"

    if not (0 <= candidate < len(mask) and mask[candidate]):
        return _temporal_action(ledger, bundle, now_s), "temporal", "rl_invalid_action"
    return candidate, "rl", None


def _policy_for_time(
    time_s: float,
    before_policy: AlgorithmName,
    after_policy: AlgorithmName,
    switch_time_s: float | None,
) -> AlgorithmName:
    if switch_time_s is None:
        return before_policy
    return before_policy if float(time_s) < switch_time_s else after_policy


def _load_rl_policy(
    model_path: str | Path | None,
    allow_missing_model: bool,
) -> tuple[MaskablePPOPolicy | None, bool, str | None]:
    if not model_path:
        return None, False, None
    path = Path(model_path)
    if not path.exists():
        if allow_missing_model:
            return None, False, str(path)
        raise FileNotFoundError(path)
    try:
        return MaskablePPOPolicy(path), True, str(path)
    except Exception:
        if allow_missing_model:
            return None, False, str(path)
        raise



def _resolve_hop_decision(
    requested: AlgorithmName,
    policy: MaskablePPOPolicy | None,
    ledger: CapacityLedger,
    bundle: DataBundle,
    now_s: float,
    config: PrototypeConfig,
) -> tuple[int | None, Contact | None, AlgorithmName, str | None]:
    """Choose an action and bind it to the contact that would be committed."""
    action, actual, fallback_reason = _choose_action(
        requested, policy, ledger, bundle, now_s, config
    )
    if action is None:
        return None, None, actual, fallback_reason

    contact = best_feasible_contact(
        ledger.planning_plan(),
        bundle.current_holder,
        action,
        now_s,
        bundle.remaining_bytes,
    )
    if contact is not None:
        return action, contact, actual, fallback_reason

    # Replan deterministically at commit time. If the requested algorithm was
    # RL this becomes an explicit RL -> temporal fallback.
    action = _temporal_action(ledger, bundle, now_s)
    actual = "temporal"
    fallback_reason = (
        "rl_contact_commit_failed" if requested == "rl" else "temporal_contact_replan"
    )
    if action is None:
        return None, None, actual, fallback_reason
    contact = best_feasible_contact(
        ledger.planning_plan(),
        bundle.current_holder,
        action,
        now_s,
        bundle.remaining_bytes,
    )
    return action, contact, actual, fallback_reason

def build_replay(
    *,
    config_path: str | Path = "config/prototype.yaml",
    bundles: int = 24,
    traffic_seed: int = 20260830,
    before_policy: AlgorithmName = "temporal",
    after_policy: AlgorithmName = "temporal",
    switch_time_s: float | None = None,
    model_path: str | Path | None = "RL/rl_env_v0/models/rl_agent_seed_42.zip",
    allow_missing_model: bool = True,
) -> ReplayData:
    """Build the complete synchronized replay used by the UI."""
    config = load_config(config_path)
    satellites, snapshots = simulate_snapshots(config)
    plan, diagnostics = build_contact_plan(snapshots, config)
    visual_bundles = generate_bundles(config, count=bundles, seed=traffic_seed)
    rl_policy, model_loaded, model_path_resolved = _load_rl_policy(model_path, allow_missing_model)

    ledger = CapacityLedger(plan)
    results: list[BundlePlayback] = []

    for original in visual_bundles:
        bundle = _clone_bundle(original)
        now_s = bundle.created_s
        path = [bundle.source_id]
        hop_records: list[PacketHop] = []
        reason = "no_route"

        for hop_index in range(1, 9):
            if bundle.current_holder in GROUND_IDS:
                reason = "delivered"
                break

            requested = _policy_for_time(now_s, before_policy, after_policy, switch_time_s)
            action, contact, actual, fallback_reason = _resolve_hop_decision(
                requested, rl_policy, ledger, bundle, now_s, config
            )
            if action is None or contact is None:
                reason = "no_route" if action is None else "commit_race"
                break

            depart_s = max(now_s, contact.start_s)

            # If the bundle is waiting for a future contact and the operator's
            # switch happens during that wait, discard the uncommitted choice
            # and make a fresh routing decision at the switch instant. In-flight
            # transfers are not interrupted.
            if (
                switch_time_s is not None
                and before_policy != after_policy
                and now_s < switch_time_s <= depart_s
            ):
                now_s = float(switch_time_s)
                requested = after_policy
                action, contact, actual, fallback_reason = _resolve_hop_decision(
                    requested, rl_policy, ledger, bundle, now_s, config
                )
                if action is None or contact is None:
                    reason = "no_route" if action is None else "commit_race"
                    break
                depart_s = max(now_s, contact.start_s)
            transfer_end_s = depart_s + contact.transmission_time_s(bundle.remaining_bytes)
            arrival_s = transfer_end_s + contact.propagation_delay_s
            ledger.reserve_contact(contact, bundle.remaining_bytes)
            fallback_used = requested != actual
            hop_records.append(
                PacketHop(
                    bundle_id=bundle.bundle_id,
                    hop_index=hop_index,
                    source_id=bundle.current_holder,
                    destination_id=action,
                    depart_s=depart_s,
                    transfer_end_s=transfer_end_s,
                    arrival_s=arrival_s,
                    requested_algorithm=requested,
                    actual_algorithm=actual,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason if fallback_used else None,
                    data_rate_bps=contact.data_rate_bps,
                    priority=bundle.science_priority,
                    contact_start_s=contact.start_s,
                    contact_end_s=contact.end_s,
                )
            )
            now_s = arrival_s
            bundle.current_holder = action
            bundle.route_history.append(action)
            path.append(action)

            if action in GROUND_IDS:
                reason = "delivered"
                break
            if bundle.deadline_s is not None and now_s > bundle.deadline_s:
                reason = "missed_deadline"
                break
        else:
            reason = "max_hops"

        delivered = bundle.current_holder in GROUND_IDS
        arrival = now_s if delivered else None
        on_time = bool(delivered and (bundle.deadline_s is None or now_s <= bundle.deadline_s))
        if delivered and not on_time:
            reason = "delivered_late"
        elif delivered:
            reason = "delivered_on_time"

        results.append(
            BundlePlayback(
                bundle_id=bundle.bundle_id,
                source_id=bundle.source_id,
                created_s=bundle.created_s,
                size_bytes=bundle.size_bytes,
                science_priority=bundle.science_priority,
                deadline_s=bundle.deadline_s,
                data_type=bundle.data_type,
                path=tuple(path),
                delivered=delivered,
                on_time=on_time,
                arrival_s=arrival,
                reason=reason,
                fallbacks=sum(1 for hop in hop_records if hop.fallback_used),
                hops=tuple(hop_records),
            )
        )

    default_selected = None
    if results:
        default_selected = max(results, key=lambda row: (row.science_priority, len(row.hops))).bundle_id

    return ReplayData(
        config=config,
        satellites=tuple(satellites),
        snapshots=tuple(snapshots),
        plan=plan,
        diagnostics=diagnostics,
        bundle_runs=tuple(results),
        before_policy=before_policy,
        after_policy=after_policy,
        switch_time_s=switch_time_s,
        selected_bundle_id=default_selected,
        model_loaded=model_loaded,
        model_path=model_path_resolved,
    )
