"""MaskablePPO environment with physical contacts and stochastic transfers.

The environment preserves the frozen 14-action / 158-observation policy
interface, but transmission outcomes are now sampled from contact weather,
health and reliability. Failed attempts consume contact capacity and time,
leave the full bundle at the sender, and force the agent to retry or reroute.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.integration.capacity import CapacityLedger
from src.integration.config import GROUND_IDS, NUM_NODES, PrototypeConfig, load_config
from src.integration.contact_plan import build_contact_plan
from src.integration.rl_bridge import OBS_LEN, action_mask, best_feasible_contact, build_observation
from src.integration.scenario import simulate_snapshots
from src.integration.stochastic_transfer import (
    StochasticTransferSettings,
    TransferOracle,
)
from src.integration.traffic import generate_bundles
from src.models.bundle import DataBundle

MAX_HOPS = 8
MAX_ATTEMPTS = 24


@dataclass(frozen=True)
class EpisodeStats:
    bundles_total: int
    bundles_delivered: int
    bundles_on_time: int
    bundles_failed: int
    transfer_attempts: int
    transfer_failures: int
    wasted_capacity_bytes: int


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


class PhysicalRoutingEnv(gym.Env):
    """Physical multi-science routing environment for MaskablePPO.

    Stochasticity is deterministic under a seed. Each episode gets its own
    transfer oracle, so repeated training episodes see different failure draws
    while remaining reproducible for a fixed training seed.
    """

    metadata = {"render_modes": []}
    MAX_RESET_ATTEMPTS = 30

    def __init__(
        self,
        config_path: str | Path = "config/prototype.yaml",
        *,
        bundles_per_episode: int = 32,
        seed: int = 42,
        max_hops: int = MAX_HOPS,
        max_attempts: int = MAX_ATTEMPTS,
        stochastic: bool = True,
    ):
        super().__init__()
        if bundles_per_episode < 1:
            raise ValueError("bundles_per_episode must be positive")
        if max_hops < 1 or max_attempts < 1:
            raise ValueError("max_hops/max_attempts must be positive")

        self.config_path = str(config_path)
        self.config: PrototypeConfig = load_config(config_path)
        _, snapshots = simulate_snapshots(self.config)
        self.contact_plan, self.diagnostics = build_contact_plan(snapshots, self.config)

        settings = StochasticTransferSettings.from_config(self.config)
        if not stochastic:
            settings = StochasticTransferSettings(
                enabled=False,
                base_failure_probability=settings.base_failure_probability,
                weather_weight=settings.weather_weight,
                health_weight=settings.health_weight,
                reliability_weight=settings.reliability_weight,
                max_failure_probability=settings.max_failure_probability,
                min_failure_progress=settings.min_failure_progress,
                max_failure_progress=settings.max_failure_progress,
                risk_shaping_weight=settings.risk_shaping_weight,
                failure_penalty_base=settings.failure_penalty_base,
                failure_penalty_priority=settings.failure_penalty_priority,
            )
        self.stochastic_settings = settings

        self.bundles_per_episode = int(bundles_per_episode)
        self.max_hops = int(max_hops)
        self.max_attempts = int(max_attempts)
        self._base_seed = int(seed)
        self._episode_index = 0
        self._rng = random.Random(self._base_seed)

        self.action_space = spaces.Discrete(NUM_NODES)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_LEN,),
            dtype=np.float32,
        )

        self.ledger: CapacityLedger | None = None
        self.transfer_oracle: TransferOracle | None = None
        self._episode_bundles: list[DataBundle] = []
        self._bundle_index = -1
        self.bundle: DataBundle | None = None
        self.t = 0.0
        self.hops = 0
        self.attempts = 0

        self.delivered_count = 0
        self.on_time_count = 0
        self.failed_count = 0
        self.transfer_attempt_count = 0
        self.transfer_failure_count = 0
        self.wasted_capacity_bytes = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._base_seed = int(seed)
            self._episode_index = 0
            self._rng = random.Random(self._base_seed)

        for _ in range(self.MAX_RESET_ATTEMPTS):
            self._episode_index += 1
            traffic_seed = self._base_seed * 1_000_000 + self._episode_index
            transfer_seed = self._base_seed * 10_000_000 + self._episode_index * 7_919

            self.ledger = CapacityLedger(self.contact_plan)
            self.transfer_oracle = TransferOracle(
                transfer_seed,
                self.config,
                self.stochastic_settings,
            )
            self._episode_bundles = generate_bundles(
                self.config,
                count=self.bundles_per_episode,
                seed=traffic_seed,
            )
            self._bundle_index = -1
            self.bundle = None
            self.t = 0.0
            self.hops = 0
            self.attempts = 0
            self.delivered_count = 0
            self.on_time_count = 0
            self.failed_count = 0
            self.transfer_attempt_count = 0
            self.transfer_failure_count = 0
            self.wasted_capacity_bytes = 0

            if self._advance_to_next_routable_bundle():
                return self._build_observation(), {
                    "action_mask": self.action_masks(),
                    "traffic_seed": traffic_seed,
                    "transfer_seed": transfer_seed,
                    "episode_index": self._episode_index,
                }

        raise RuntimeError("could not produce a routable physical training episode")

    def step(self, action: int):
        if self.bundle is None or self.ledger is None or self.transfer_oracle is None:
            raise RuntimeError("call reset() before step()")

        reward = 0.0
        info: dict[str, object] = {}
        mask = self.action_masks()

        if action < 0 or action >= NUM_NODES or mask[action] == 0:
            reward -= 15.0
            info["event"] = "invalid_action"
            self.failed_count += 1
            return self._finish_or_advance(reward, info)

        contact = best_feasible_contact(
            self.ledger.planning_plan(),
            self.bundle.current_holder,
            int(action),
            self.t,
            self.bundle.remaining_bytes,
        )
        if contact is None:
            reward -= 15.0
            info["event"] = "contact_drift"
            self.failed_count += 1
            return self._finish_or_advance(reward, info)

        nominal_capacity = max(
            1.0,
            (contact.end_s - contact.start_s) * contact.data_rate_bps / 8.0,
        )
        utilization = max(
            0.0,
            min(1.0, 1.0 - contact.residual_capacity_bytes / nominal_capacity),
        )

        outcome = self.transfer_oracle.attempt(
            bundle_id=self.bundle.bundle_id,
            contact=contact,
            size_bytes=self.bundle.remaining_bytes,
            now_s=self.t,
        )
        self.ledger.reserve_contact(contact, outcome.capacity_bytes_consumed)
        self.attempts += 1
        self.transfer_attempt_count += 1

        # Every attempted transfer costs routing/energy budget. The expected
        # risk term is deliberately small because failures are now sampled for
        # real; it is only a shaping signal, not a substitute for stochasticity.
        reward -= 2.0
        reward -= 5.0 * utilization
        reward -= self.stochastic_settings.risk_shaping_weight * outcome.failure_probability
        reward -= 2.0 * (1.0 - self.config.runtime_defaults.battery) * outcome.transfer_progress

        info.update(
            {
                "source_id": self.bundle.source_id,
                "attempted_destination_id": int(action),
                "contact_utilization": utilization,
                "failure_probability": outcome.failure_probability,
                "success_draw": outcome.success_draw,
                "transfer_progress": outcome.transfer_progress,
                "capacity_bytes_consumed": outcome.capacity_bytes_consumed,
                "depart_s": outcome.depart_s,
                "event_time_s": outcome.event_time_s,
                "transfer_success": outcome.success,
            }
        )

        if not outcome.success:
            self.t = outcome.event_time_s
            self.transfer_failure_count += 1
            self.wasted_capacity_bytes += outcome.wasted_capacity_bytes
            reward -= self.stochastic_settings.failure_penalty_base
            reward -= (
                self.stochastic_settings.failure_penalty_priority
                * float(self.bundle.science_priority)
            )
            info["event"] = "transmission_failed"
            info["holder_id"] = self.bundle.current_holder

            if self.bundle.deadline_s is not None and self.t > self.bundle.deadline_s:
                self.failed_count += 1
                reward -= 25.0
                info["event"] = "missed_deadline_after_failure"
                return self._finish_or_advance(reward, info)
            if self.attempts >= self.max_attempts:
                self.failed_count += 1
                reward -= 25.0
                info["event"] = "max_attempts"
                return self._finish_or_advance(reward, info)
            if not self.action_masks().any():
                self.failed_count += 1
                reward -= 25.0
                info["event"] = "no_feasible_contact_after_failure"
                return self._finish_or_advance(reward, info)

            # Continue the same bundle at the same holder. The next policy
            # decision can retry this contact if capacity/time still allow it,
            # or choose a different route.
            return self._build_observation(), reward, False, False, self._next_info(info)

        # Successful atomic transfer: now the bundle changes holder.
        self.t = float(outcome.arrival_s)
        self.bundle.current_holder = int(action)
        self.bundle.route_history.append(int(action))
        self.hops += 1
        info["holder_id"] = self.bundle.current_holder
        info["arrival_s"] = self.t
        info["event"] = "hop_succeeded"

        if int(action) in GROUND_IDS:
            self.delivered_count += 1
            on_time = self.bundle.deadline_s is None or self.t <= self.bundle.deadline_s
            reward += 100.0
            if on_time:
                self.on_time_count += 1
                reward += 100.0 * self.bundle.science_priority
                info["event"] = "delivered_on_time"
            else:
                reward -= 25.0
                info["event"] = "delivered_late"
            reward -= 0.05 * max(0.0, self.t - self.bundle.created_s)
            return self._finish_or_advance(reward, info)

        if self.bundle.deadline_s is not None and self.t > self.bundle.deadline_s:
            self.failed_count += 1
            reward -= 25.0
            info["event"] = "missed_deadline"
            return self._finish_or_advance(reward, info)
        if self.hops >= self.max_hops:
            self.failed_count += 1
            reward -= 25.0
            info["event"] = "max_hops"
            return self._finish_or_advance(reward, info)
        if self.attempts >= self.max_attempts:
            self.failed_count += 1
            reward -= 25.0
            info["event"] = "max_attempts"
            return self._finish_or_advance(reward, info)
        if not self.action_masks().any():
            self.failed_count += 1
            reward -= 25.0
            info["event"] = "no_feasible_contact"
            return self._finish_or_advance(reward, info)

        return self._build_observation(), reward, False, False, self._next_info(info)

    def action_masks(self) -> np.ndarray:
        if self.bundle is None or self.ledger is None:
            return np.zeros(NUM_NODES, dtype=np.int8)
        return action_mask(self.ledger.planning_plan(), self.bundle, self.t)

    def _advance_to_next_routable_bundle(self) -> bool:
        if self.ledger is None:
            return False

        while True:
            self._bundle_index += 1
            if self._bundle_index >= len(self._episode_bundles):
                self.bundle = None
                return False

            candidate = clone_bundle(self._episode_bundles[self._bundle_index])
            self.bundle = candidate
            self.t = candidate.created_s
            self.hops = 0
            self.attempts = 0
            if self.action_masks().any():
                return True
            self.failed_count += 1

    def _finish_or_advance(self, reward: float, info: dict[str, object]):
        episode_done = not self._advance_to_next_routable_bundle()
        if episode_done:
            return self._terminal_observation(), reward, True, False, self._final_info(info)
        return self._build_observation(), reward, False, False, self._next_info(info)

    def _build_observation(self) -> np.ndarray:
        if self.bundle is None or self.ledger is None:
            return self._terminal_observation()
        obs, _ = build_observation(
            self.ledger.planning_plan(),
            self.bundle,
            self.t,
            self.config,
        )
        return obs

    def _terminal_observation(self) -> np.ndarray:
        return np.zeros((OBS_LEN,), dtype=np.float32)

    def _next_info(self, info: dict[str, object]) -> dict[str, object]:
        info = dict(info)
        info["action_mask"] = self.action_masks()
        if self.bundle is not None:
            info["next_bundle_id"] = self.bundle.bundle_id
            info["next_source_id"] = self.bundle.source_id
        return info

    def _final_info(self, info: dict[str, object]) -> dict[str, object]:
        info = dict(info)
        stats = EpisodeStats(
            bundles_total=len(self._episode_bundles),
            bundles_delivered=self.delivered_count,
            bundles_on_time=self.on_time_count,
            bundles_failed=self.failed_count,
            transfer_attempts=self.transfer_attempt_count,
            transfer_failures=self.transfer_failure_count,
            wasted_capacity_bytes=self.wasted_capacity_bytes,
        )
        info["episode_stats"] = stats.__dict__
        info["action_mask"] = np.zeros(NUM_NODES, dtype=np.int8)
        return info
