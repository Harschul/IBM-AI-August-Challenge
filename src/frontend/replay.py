"""Replay generation for the integrated orbital/network frontend.

This module precomputes one coherent routing run, then exposes synchronized 3D
and 2D frame state for the UI.  The same underlying physical contact plan drives
both the orbital view and the topology graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Literal, Sequence

import numpy as np

from src.integration.capacity import CapacityLedger
from src.integration.config import GEO_IDS, GROUND_IDS, LEO_IDS, NUM_NODES, SCIENCE_ID, PrototypeConfig, load_config
from src.integration.contact_plan import build_contact_plan, ground_position
from src.integration.rl_bridge import MaskablePPOPolicy, best_feasible_contact, build_observation
from src.integration.scenario import build_satellites, simulate_snapshots
from src.integration.traffic import generate_bundles
from src.models.bundle import DataBundle
from src.models.contact import Contact, ContactPlan
from src.routing.temporal_baseline import earliest_arrival

PolicyName = Literal["temporal", "rl"]


@dataclass(frozen=True)
class PacketHop:
    bundle_id: str
    hop_index: int
    source_id: int
    destination_id: int
    depart_s: float
    transfer_end_s: float
    arrival_s: float
    policy_used: str
    fallback_used: bool
    data_rate_bps: float
    priority: float
    contact_start_s: float
    contact_end_s: float


@dataclass(frozen=True)
class BundlePlayback:
    bundle_id: str
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


@dataclass(frozen=True)
class ActivePacket:
    bundle_id: str
    source_id: int
    destination_id: int
    xyz: tuple[float, float, float]
    topology_xy: tuple[float, float]
    progress: float
    policy_used: str
    priority: float
    fallback_used: bool


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


@dataclass(frozen=True)
class ReplayData:
    config: PrototypeConfig
    satellites: tuple[object, ...]
    snapshots: tuple[object, ...]
    plan: ContactPlan
    diagnostics: object
    bundle_runs: tuple[BundlePlayback, ...]
    before_policy: PolicyName
    after_policy: PolicyName
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
                            policy_used=hop.policy_used,
                            priority=hop.priority,
                            fallback_used=hop.fallback_used,
                        )
                    )
                    break
        return packets

    def current_policy_label(self, time_s: float) -> str:
        if self.switch_time_s is None:
            return self.before_policy
        return self.before_policy if float(time_s) < self.switch_time_s else self.after_policy

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
        total_priority = sum(r.science_priority for r in rows)
        on_time_priority = sum(r.science_priority for r in rows if r.on_time)
        return ReplaySummary(
            bundles=len(rows),
            delivered=len(delivered),
            on_time=len(on_time),
            mean_latency_s=(fmean(r.arrival_s - r.created_s for r in delivered) if delivered else None),
            mean_hops=fmean(len(r.hops) for r in rows) if rows else 0.0,
            total_fallbacks=sum(r.fallbacks for r in rows),
            delivery_ratio=(len(delivered) / len(rows) if rows else 0.0),
            deadline_success=(len(on_time) / len(rows) if rows else 0.0),
            priority_weighted_timely=(on_time_priority / total_priority if total_priority else 0.0),
        )

    def event_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for bundle in self.bundle_runs:
            rows.append(
                {
                    "time_s": round(bundle.created_s, 2),
                    "bundle_id": bundle.bundle_id,
                    "event": "created",
                    "detail": f"{bundle.data_type} priority={bundle.science_priority:.2f}",
                }
            )
            for hop in bundle.hops:
                rows.append(
                    {
                        "time_s": round(hop.depart_s, 2),
                        "bundle_id": bundle.bundle_id,
                        "event": "depart",
                        "detail": f"{hop.source_id} -> {hop.destination_id} via {hop.policy_used}",
                    }
                )
                rows.append(
                    {
                        "time_s": round(hop.arrival_s, 2),
                        "bundle_id": bundle.bundle_id,
                        "event": "arrive",
                        "detail": f"at node {hop.destination_id}",
                    }
                )
            rows.append(
                {
                    "time_s": round(bundle.arrival_s if bundle.arrival_s is not None else self.horizon_s, 2),
                    "bundle_id": bundle.bundle_id,
                    "event": bundle.reason,
                    "detail": f"delivered={bundle.delivered} on_time={bundle.on_time}",
                }
            )
        rows.sort(key=lambda row: (float(row["time_s"]), str(row["bundle_id"]), str(row["event"])))
        return rows


def node_label(node_id: int, config: PrototypeConfig) -> str:
    if node_id == SCIENCE_ID:
        return "SCI-0"
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


def _baseline_route(ledger: CapacityLedger, bundle: DataBundle, now_s: float) -> object | None:
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


def _choose_action(
    policy_name: PolicyName,
    policy: MaskablePPOPolicy | None,
    ledger: CapacityLedger,
    bundle: DataBundle,
    now_s: float,
    config: PrototypeConfig,
) -> tuple[int | None, bool, str]:
    if policy_name == "rl" and policy is not None:
        planning = ledger.planning_plan()
        obs, mask = build_observation(planning, bundle, now_s, config)
        if mask.any():
            try:
                candidate = int(policy.choose(obs, mask))
                if 0 <= candidate < len(mask) and mask[candidate]:
                    return candidate, False, "rl"
            except Exception:
                pass

    route = _baseline_route(ledger, bundle, now_s)
    if route is None or not route.hops:
        return None, True, "temporal"
    return int(route.next_hop()), policy_name == "rl", "temporal"


def _policy_for_time(time_s: float, before_policy: PolicyName, after_policy: PolicyName, switch_time_s: float | None) -> PolicyName:
    if switch_time_s is None:
        return before_policy
    return before_policy if float(time_s) < switch_time_s else after_policy


def _load_rl_policy(model_path: str | Path | None, allow_missing_model: bool) -> tuple[MaskablePPOPolicy | None, bool, str | None]:
    if not model_path:
        return None, False, None
    path = Path(model_path)
    if not path.exists():
        if allow_missing_model:
            return None, False, str(path)
        raise FileNotFoundError(path)
    return MaskablePPOPolicy(path), True, str(path)


def build_replay(
    *,
    config_path: str | Path = "config/prototype.yaml",
    bundles: int = 24,
    traffic_seed: int = 20260830,
    before_policy: PolicyName = "temporal",
    after_policy: PolicyName = "temporal",
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
        fallbacks = 0
        reason = "no_route"

        for hop_index in range(1, 9):
            if bundle.current_holder in GROUND_IDS:
                reason = "delivered"
                break
            active_policy = _policy_for_time(now_s, before_policy, after_policy, switch_time_s)
            action, fallback_used, policy_used = _choose_action(
                active_policy,
                rl_policy,
                ledger,
                bundle,
                now_s,
                config,
            )
            if fallback_used:
                fallbacks += 1
            if action is None:
                reason = "no_route"
                break

            contact = best_feasible_contact(
                ledger.planning_plan(),
                bundle.current_holder,
                action,
                now_s,
                bundle.remaining_bytes,
            )
            if contact is None:
                fallbacks += 1
                route = _baseline_route(ledger, bundle, now_s)
                if route is None or not route.hops:
                    reason = "commit_race"
                    break
                action = int(route.next_hop())
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
                fallback_used = True
                policy_used = "temporal"

            depart_s = max(now_s, contact.start_s)
            transfer_end_s = depart_s + contact.transmission_time_s(bundle.remaining_bytes)
            arrival_s = transfer_end_s + contact.propagation_delay_s
            ledger.reserve_contact(contact, bundle.remaining_bytes)
            hop_records.append(
                PacketHop(
                    bundle_id=bundle.bundle_id,
                    hop_index=hop_index,
                    source_id=bundle.current_holder,
                    destination_id=action,
                    depart_s=depart_s,
                    transfer_end_s=transfer_end_s,
                    arrival_s=arrival_s,
                    policy_used=policy_used,
                    fallback_used=fallback_used,
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
                fallbacks=fallbacks,
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
