"""Frontend replay built from the exact final experiment runner.

Reported demo modes do not have their own routing simulator. They call the same
`src.experiment.runner.run_algorithm` function used by the final benchmark,
with the same frozen config, checkpoint, traffic seed, stochastic seed, bundle
count, capacity semantics, and transfer oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Literal

import numpy as np

from src.experiment.runner import AttemptRecord, ExperimentResult, aggregate, build_world, load_final_policy, run_algorithm
from src.experiment.spec import FinalExperimentSpec, load_final_spec
from src.integration.config import GEO_IDS, GROUND_IDS, LEO_IDS, SCIENCE_IDS, PrototypeConfig
from src.integration.contact_plan import ground_position
from src.integration.traffic import generate_bundles
from src.models.contact import Contact

AlgorithmName = Literal["temporal", "rl"]
DemoMode = Literal["reported_rl", "reported_temporal", "interactive_switch"]


@dataclass(frozen=True)
class PacketHop:
    bundle_id: str
    hop_index: int
    source_id: int
    destination_id: int
    depart_s: float
    transfer_end_s: float
    arrival_s: float
    requested_algorithm: str
    actual_algorithm: str
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
    attempts: tuple[AttemptRecord, ...]
    transfer_failures: int
    wasted_capacity_bytes: int


@dataclass(frozen=True)
class ActivePacket:
    bundle_id: str
    source_id: int
    destination_id: int
    xyz: tuple[float, float, float]
    topology_xy: tuple[float, float]
    progress: float
    requested_algorithm: str
    actual_algorithm: str
    priority: float
    fallback_used: bool
    fallback_reason: str | None
    will_succeed: bool
    failure_probability: float
    success_draw: float


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
    transfer_failure_rate: float
    mean_wasted_mb: float
    rl_requested_attempts: int
    rl_executed_attempts: int
    temporal_executed_attempts: int


@dataclass(frozen=True)
class ReplayData:
    spec: FinalExperimentSpec
    config: PrototypeConfig
    satellites: tuple[object, ...]
    snapshots: tuple[object, ...]
    plan: object
    diagnostics: object
    bundle_runs: tuple[BundlePlayback, ...]
    raw_results: tuple[ExperimentResult, ...]
    mode: DemoMode
    seed_offset: int
    traffic_seed: int
    stochastic_seed: int
    before_policy: AlgorithmName
    after_policy: AlgorithmName
    switch_time_s: float | None
    selected_bundle_id: str | None
    model_loaded: bool
    model_path: str
    reported_experiment: bool

    @property
    def times(self) -> tuple[float, ...]:
        return tuple(float(snapshot.time) for snapshot in self.snapshots)

    @property
    def horizon_s(self) -> float:
        return self.times[-1] if self.times else 0.0

    @property
    def topology_positions(self) -> dict[int, tuple[float, float]]:
        from src.frontend.layout import topology_positions
        return topology_positions()

    def frame_index_at(self, time_s: float) -> int:
        if not self.snapshots:
            return 0
        index = int(round(max(0.0, min(self.horizon_s, float(time_s))) / self.config.sample_step_s))
        return max(0, min(len(self.snapshots) - 1, index))

    def node_position_3d(self, node_id: int, time_s: float) -> np.ndarray:
        if node_id in GROUND_IDS:
            return ground_position(self.config, node_id, time_s)
        return np.asarray(self.snapshots[self.frame_index_at(time_s)].positions[node_id], dtype=float)

    def active_contacts(self, time_s: float) -> list[Contact]:
        t = float(time_s)
        return [contact for contact in self.plan.contacts if contact.start_s <= t <= contact.end_s]

    def active_packets(self, time_s: float) -> list[ActivePacket]:
        t = float(time_s)
        topo = self.topology_positions
        packets: list[ActivePacket] = []
        priorities = {run.bundle_id: run.science_priority for run in self.bundle_runs}
        for result in self.raw_results:
            for attempt in result.attempt_trace:
                end_s = attempt.arrival_s if attempt.success else attempt.event_time_s
                if attempt.depart_s <= t <= float(end_s):
                    elapsed = max(0.0, t - attempt.depart_s)
                    duration = max(1e-9, float(end_s) - attempt.depart_s)
                    fraction_of_attempt = max(0.0, min(1.0, elapsed / duration))
                    final_progress = 1.0 if attempt.success else attempt.transfer_progress
                    alpha = fraction_of_attempt * final_progress
                    src_xyz = self.node_position_3d(attempt.holder_id, t)
                    dst_xyz = self.node_position_3d(attempt.destination_id, t)
                    xyz = src_xyz + alpha * (dst_xyz - src_xyz)
                    sx, sy = topo[attempt.holder_id]
                    dx, dy = topo[attempt.destination_id]
                    packets.append(ActivePacket(
                        bundle_id=attempt.bundle_id,
                        source_id=attempt.holder_id,
                        destination_id=attempt.destination_id,
                        xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                        topology_xy=(sx + alpha * (dx - sx), sy + alpha * (dy - sy)),
                        progress=alpha,
                        requested_algorithm=attempt.requested_algorithm,
                        actual_algorithm=attempt.actual_algorithm,
                        priority=priorities[attempt.bundle_id],
                        fallback_used=attempt.fallback_used,
                        fallback_reason=attempt.fallback_reason,
                        will_succeed=attempt.success,
                        failure_probability=attempt.failure_probability,
                        success_draw=attempt.success_draw,
                    ))
                    break
        return packets

    def requested_algorithm_at(self, time_s: float) -> str:
        if self.mode == "reported_rl":
            return "rl"
        if self.mode == "reported_temporal":
            return "temporal"
        if self.switch_time_s is None:
            return self.before_policy
        return self.before_policy if float(time_s) < self.switch_time_s else self.after_policy

    def current_policy_label(self, time_s: float) -> str:
        return self.requested_algorithm_at(time_s)

    def actual_algorithm_at(self, time_s: float) -> str:
        active = self.active_packets(time_s)
        if active:
            return " + ".join(sorted({packet.actual_algorithm for packet in active}))
        latest: AttemptRecord | None = None
        for result in self.raw_results:
            for attempt in result.attempt_trace:
                if attempt.depart_s <= time_s and (latest is None or attempt.depart_s > latest.depart_s):
                    latest = attempt
        return latest.actual_algorithm if latest else "none"

    def selected_bundle(self, bundle_id: str | None = None) -> BundlePlayback | None:
        wanted = bundle_id or self.selected_bundle_id
        return next((bundle for bundle in self.bundle_runs if bundle.bundle_id == wanted), None)

    def summary(self) -> ReplaySummary:
        metrics = aggregate(self.raw_results)
        all_attempts = [attempt for row in self.raw_results for attempt in row.attempt_trace]
        return ReplaySummary(
            bundles=len(self.raw_results),
            delivered=sum(row.delivered for row in self.raw_results),
            on_time=sum(row.on_time for row in self.raw_results),
            mean_latency_s=None if np.isnan(metrics["mean_latency_s"]) else metrics["mean_latency_s"],
            mean_hops=metrics["mean_hops"],
            total_fallbacks=sum(row.fallbacks for row in self.raw_results),
            delivery_ratio=metrics["delivery_ratio"],
            deadline_success=metrics["deadline_success"],
            priority_weighted_timely=metrics["priority_weighted_timely"],
            transfer_failure_rate=metrics["transfer_failure_rate"],
            mean_wasted_mb=metrics["mean_wasted_mb"],
            rl_requested_attempts=sum(attempt.requested_algorithm == "rl" for attempt in all_attempts),
            rl_executed_attempts=sum(attempt.actual_algorithm == "rl" for attempt in all_attempts),
            temporal_executed_attempts=sum(attempt.actual_algorithm == "temporal" for attempt in all_attempts),
        )

    def event_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        lookup = {run.bundle_id: run for run in self.bundle_runs}
        for result in self.raw_results:
            bundle = lookup[result.bundle_id]
            rows.append({
                "time_s": round(bundle.created_s, 3),
                "bundle_id": bundle.bundle_id,
                "event": "created",
                "detail": f"source={node_label(bundle.source_id, self.config)} priority={bundle.science_priority:.2f}",
                "requested_algorithm": "",
                "actual_algorithm": "",
                "fallback": False,
            })
            for attempt in result.attempt_trace:
                rows.append({
                    "time_s": round(attempt.depart_s, 3),
                    "bundle_id": bundle.bundle_id,
                    "event": "attempt",
                    "detail": (
                        f"{node_label(attempt.holder_id, self.config)} -> {node_label(attempt.destination_id, self.config)} "
                        f"p_fail={attempt.failure_probability:.3f} draw={attempt.success_draw:.3f}"
                    ),
                    "requested_algorithm": attempt.requested_algorithm,
                    "actual_algorithm": attempt.actual_algorithm,
                    "fallback": attempt.fallback_used,
                })
                if not attempt.success:
                    rows.append({
                        "time_s": round(attempt.event_time_s, 3),
                        "bundle_id": bundle.bundle_id,
                        "event": "transfer_failed",
                        "detail": f"failed at {attempt.transfer_progress:.0%}; capacity consumed={attempt.capacity_bytes_consumed / 1e6:.2f} MB",
                        "requested_algorithm": attempt.requested_algorithm,
                        "actual_algorithm": attempt.actual_algorithm,
                        "fallback": attempt.fallback_used,
                    })
            rows.append({
                "time_s": round(result.arrival_s if result.arrival_s is not None else self.horizon_s, 3),
                "bundle_id": bundle.bundle_id,
                "event": result.reason,
                "detail": f"delivered={result.delivered} on_time={result.on_time}",
                "requested_algorithm": result.requested_algorithm,
                "actual_algorithm": "|".join(result.actual_algorithms),
                "fallback": bool(result.fallbacks),
            })
        rows.sort(key=lambda row: (float(row["time_s"]), str(row["bundle_id"])))
        return rows


def node_label(node_id: int, config: PrototypeConfig) -> str:
    if node_id in SCIENCE_IDS:
        return f"SCI-{SCIENCE_IDS.index(node_id)}"
    if node_id in LEO_IDS:
        return f"LEO-{node_id}"
    if node_id in GEO_IDS:
        return f"GEO-{node_id}"
    if node_id in GROUND_IDS:
        return next(gs.name for gs in config.ground_stations if gs.node_id == node_id)
    return str(node_id)


def _bundle_lookup(bundles):
    return {bundle.bundle_id: bundle for bundle in bundles}


def _to_playback(results: list[ExperimentResult], bundles) -> tuple[BundlePlayback, ...]:
    lookup = _bundle_lookup(bundles)
    playback: list[BundlePlayback] = []
    for result in results:
        original = lookup[result.bundle_id]
        successful = [attempt for attempt in result.attempt_trace if attempt.success]
        hops = tuple(PacketHop(
            bundle_id=result.bundle_id,
            hop_index=index + 1,
            source_id=attempt.holder_id,
            destination_id=attempt.destination_id,
            depart_s=attempt.depart_s,
            transfer_end_s=float(attempt.arrival_s),
            arrival_s=float(attempt.arrival_s),
            requested_algorithm=attempt.requested_algorithm,
            actual_algorithm=attempt.actual_algorithm,
            fallback_used=attempt.fallback_used,
            fallback_reason=attempt.fallback_reason,
            data_rate_bps=0.0,
            priority=original.science_priority,
            contact_start_s=attempt.contact_start_s,
            contact_end_s=attempt.contact_end_s,
        ) for index, attempt in enumerate(successful))
        playback.append(BundlePlayback(
            bundle_id=result.bundle_id,
            source_id=result.source_id,
            created_s=original.created_s,
            size_bytes=original.size_bytes,
            science_priority=original.science_priority,
            deadline_s=original.deadline_s,
            data_type=original.data_type,
            path=result.path,
            delivered=result.delivered,
            on_time=result.on_time,
            arrival_s=result.arrival_s,
            reason=result.reason,
            fallbacks=result.fallbacks,
            hops=hops,
            attempts=result.attempt_trace,
            transfer_failures=result.transfer_failures,
            wasted_capacity_bytes=result.wasted_capacity_bytes,
        ))
    return tuple(playback)


def build_replay(
    *,
    spec_path: str = "config/final_experiment.json",
    mode: DemoMode = "reported_rl",
    seed_offset: int | None = None,
    before_policy: AlgorithmName = "temporal",
    after_policy: AlgorithmName = "rl",
    switch_time_s: float | None = 900.0,
) -> ReplayData:
    spec = load_final_spec(spec_path)
    world = build_world(spec)
    offset = spec.demo.default_seed_offset if seed_offset is None else int(seed_offset)
    traffic_seed, stochastic_seed = spec.benchmark.seeds(offset)
    bundles = generate_bundles(world.config, count=spec.benchmark.bundles_per_seed, seed=traffic_seed)

    needs_rl = mode == "reported_rl" or (mode == "interactive_switch" and "rl" in {before_policy, after_policy})
    policy = load_final_policy(spec) if needs_rl else None

    if mode == "reported_rl":
        results = run_algorithm(
            algorithm="rl_pure", plan=world.plan, config=world.config, bundles=bundles,
            policy=policy, stochastic_seed=stochastic_seed,
            max_hops=spec.benchmark.max_hops, max_attempts=spec.benchmark.max_attempts,
        )
        before_policy = after_policy = "rl"
        switch_time_s = None
        reported = True
    elif mode == "reported_temporal":
        results = run_algorithm(
            algorithm="temporal", plan=world.plan, config=world.config, bundles=bundles,
            policy=None, stochastic_seed=stochastic_seed,
            max_hops=spec.benchmark.max_hops, max_attempts=spec.benchmark.max_attempts,
        )
        before_policy = after_policy = "temporal"
        switch_time_s = None
        reported = True
    elif mode == "interactive_switch":
        def requested_at(time_s: float) -> str:
            if switch_time_s is None:
                return before_policy
            return before_policy if time_s < switch_time_s else after_policy
        results = run_algorithm(
            algorithm="scheduled", plan=world.plan, config=world.config, bundles=bundles,
            policy=policy, stochastic_seed=stochastic_seed,
            max_hops=spec.benchmark.max_hops, max_attempts=spec.benchmark.max_attempts,
            requested_at=requested_at,
        )
        reported = False
    else:
        raise ValueError(f"unknown demo mode {mode}")

    playback = _to_playback(results, bundles)
    selected = max(playback, key=lambda item: (item.science_priority, item.transfer_failures, len(item.hops))).bundle_id if playback else None
    return ReplayData(
        spec=spec,
        config=world.config,
        satellites=world.satellites,
        snapshots=world.snapshots,
        plan=world.plan,
        diagnostics=world.diagnostics,
        bundle_runs=playback,
        raw_results=tuple(results),
        mode=mode,
        seed_offset=offset,
        traffic_seed=traffic_seed,
        stochastic_seed=stochastic_seed,
        before_policy=before_policy,
        after_policy=after_policy,
        switch_time_s=switch_time_s,
        selected_bundle_id=selected,
        model_loaded=policy is not None,
        model_path=str(spec.model),
        reported_experiment=reported,
    )
