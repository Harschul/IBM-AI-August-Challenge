"""RoutingEnv: the frozen RL interface for next-hop bundle routing.

This is the artifact for the 20 Aug milestone ("RL API fixed"): the
observation layout, action space and reward signature below are meant to be
STABLE from here on. Only the contact-plan source (`mock_graph.py` today,
Serafin's real generator later) should change underneath it.

Action space:      Discrete(14) -- pick one of the 14 fixed node IDs as the
                    next hop for the bundle currently held by the agent.
Observation space:  Box, fixed length 158 =
                        4   bundle features
                      + 14 candidate nodes * 11 features each
Reward:             simplified version of the R formula in section 5.5
                    (delivered / on-time / latency / deadline miss / hop cost /
                    congestion / failed-transmission risk / energy cost).

Ablation support:
    `ablation=None`             -> full observation (default / production policy)
    `ablation="no_priority"`    -> bundle's science_priority feature zeroed out
    `ablation="no_weather_health"` -> per-candidate weather_risk and health
                                      features zeroed out
    In both ablation modes the *environment dynamics and reward* still use
    the true underlying values -- only what the agent gets to OBSERVE is
    restricted. This isolates the effect of the agent losing access to that
    information, per section 11.4's ablation requirement.
"""

from __future__ import annotations

import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .mock_graph import (
    NUM_NODES,
    SCIENCE_ID,
    GROUND_IDS,
    MockContactPlan,
    Bundle,
)

BUNDLE_FEATURES = 4
CANDIDATE_FEATURES = 11
OBS_LEN = BUNDLE_FEATURES + NUM_NODES * CANDIDATE_FEATURES

MAX_HOPS = 8
STEP_S = 5.0  # sim tick while waiting for a contact, matches section 8.1


class RoutingEnv(gym.Env):
    """Custom Gymnasium env matching the plan's fixed 14-node interface."""

    metadata = {"render_modes": []}

    VALID_ABLATIONS = (None, "no_priority", "no_weather_health")
    MAX_RESET_ATTEMPTS = 50

    def __init__(self, horizon_s: float = 1800.0, seed: int | None = None, ablation: str | None = None):
        super().__init__()
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"ablation must be one of {self.VALID_ABLATIONS}, got {ablation!r}")
        self.horizon_s = horizon_s
        self._base_seed = seed
        self.ablation = ablation
        self.action_space = spaces.Discrete(NUM_NODES)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_LEN,), dtype=np.float32
        )

        self.rng = random.Random(seed)
        self.contact_plan: MockContactPlan | None = None
        self.bundle: Bundle | None = None
        self.t = 0.0
        self.hops = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        # Redraw until the science satellite has at least one feasible hop.
        # Without this, feasibility filtering occasionally produces an episode
        # that is over before it starts, which MaskablePPO cannot sample from.
        for _ in range(self.MAX_RESET_ATTEMPTS):
            self.contact_plan = MockContactPlan(
                horizon_s=self.horizon_s, seed=self.rng.randint(0, 10**9))
            self.t = 0.0
            self.bundle = self.contact_plan.sample_bundle(self.t, self.rng)
            self.bundle.current_holder = SCIENCE_ID
            self.hops = 0
            mask = self._action_mask()
            if mask.any():
                break
        else:
            raise RuntimeError(
                "no routable scenario in %d attempts" % self.MAX_RESET_ATTEMPTS)

        obs = self._build_observation()
        info = {"action_mask": mask}
        return obs, info

    def step(self, action: int):
        mask = self._action_mask()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if mask[action] == 0:
            # Invalid pick: heavily penalised, episode ends (mirrors "invalid
            # action" handling in section 9.2 -- the masked policy should
            # never actually do this once trained).
            reward -= 15.0
            terminated = True
            info["event"] = "invalid_action"
            obs = self._build_observation()
            info["action_mask"] = mask
            return obs, reward, terminated, truncated, info

        contact = self._feasible_contact_to(action)
        depart = max(self.t, contact.start_s)
        tx_time = self.bundle.remaining_bytes * 8 / contact.data_rate_bps
        arrival = depart + tx_time

        self.t = arrival
        self.bundle.current_holder = action
        self.hops += 1
        reward -= 2.0  # per-hop cost, section 5.5
        reward -= 5.0 * contact.queue_norm  # congestion penalty proxy

        # failed_transmission risk, section 5.5 (-15 * failed_transmission).
        # No hard failure state yet -- modelled as expected cost from the
        # contact's own risk signals, same style as the congestion penalty
        # above. weather dominates, node health and link reliability soften
        # or worsen it. This is what makes weather_risk/health actually
        # matter to the policy (see VALID_ABLATIONS docstring above).
        p_fail = max(0.0, min(
            0.5 * contact.weather_risk + 0.3 * (1.0 - contact.health) + 0.2 * (1.0 - contact.reliability),
            0.9,
        ))
        reward -= 15.0 * p_fail
        info["failed_transmission_risk"] = p_fail

        # energy_penalty, section 5.5 (-2 * energy_penalty). Cost of routing
        # through a relay with a depleted battery.
        energy_penalty = 1.0 - contact.battery
        reward -= 2.0 * energy_penalty
        info["energy_penalty"] = energy_penalty

        if action in GROUND_IDS:
            terminated = True
            on_time = self.t <= self.bundle.deadline_s
            reward += 100.0
            if on_time:
                reward += 100.0 * self.bundle.science_priority
                info["event"] = "delivered_on_time"
            else:
                reward -= 25.0
                info["event"] = "delivered_late"
            reward -= 0.05 * self.t
        elif self.t > self.bundle.deadline_s:
            terminated = True
            reward -= 25.0
            info["event"] = "missed_deadline"
        elif self.hops >= MAX_HOPS:
            truncated = True
            reward -= 25.0
            info["event"] = "max_hops"

        obs = self._build_observation()
        next_mask = self._action_mask()

        # With feasibility enforced, a bundle can now reach a node from which
        # nothing can carry it onward. MaskablePPO cannot sample from an
        # all-zero mask, so the episode has to end here rather than be offered
        # to the policy as an impossible choice.
        if not terminated and not truncated and not next_mask.any():
            terminated = True
            reward -= 25.0
            info["event"] = "no_feasible_contact"

        info["action_mask"] = next_mask
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO convention."""
        return self._action_mask()

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_NODES, dtype=np.int8)
        holder = self.bundle.current_holder

        # A destination is legal only if some contact to it can carry the whole
        # bundle before its window shuts. Existence of a contact is no longer
        # enough -- see _feasible_contact_to.
        for dest_id in range(NUM_NODES):
            if dest_id == holder:
                continue
            if self._feasible_contact_to(dest_id) is not None:
                mask[dest_id] = 1

        return mask

    def _feasible_contact_to(self, dest_id: int):
        """Best contact to dest_id that the transfer can actually COMPLETE in.

        The old version only required that a contact exist and not be over yet.
        That let a bundle be pushed through a window far too short to carry it:
        a measured 61% of hops finished after `contact.end_s` (issue #4). The
        transmission must fit, so the check is:

            depart = max(now, contact.start_s)     # waiting is legal
            depart + tx_time <= contact.end_s      # ...but the transfer must fit

        Returns None when no contact to dest_id can carry the bundle. Callers
        treat None as "not a legal next hop", which is what keeps the action
        mask and step() in agreement -- they now ask the same question through
        the same code path.
        """
        holder = self.bundle.current_holder
        tx_time = self.bundle.remaining_bytes * 8

        best = None
        best_arrival = None
        for c in self.contact_plan.contacts:
            if c.source_id != holder or c.destination_id != dest_id:
                continue
            if c.end_s < self.t:
                continue

            depart = max(self.t, c.start_s)
            if depart >= c.end_s:
                continue

            arrival = depart + tx_time / c.data_rate_bps
            if arrival > c.end_s:
                continue  # the window closes mid-transfer

            if best_arrival is None or arrival < best_arrival:
                best, best_arrival = c, arrival

        return best

    def _best_contact_to(self, dest_id: int):
        """Backwards-compatible alias. Raises if the hop is not feasible, which
        should be unreachable now that the mask filters on feasibility."""
        contact = self._feasible_contact_to(dest_id)
        if contact is None:
            raise ValueError(
                "no feasible contact from %d to %d at t=%.1f"
                % (self.bundle.current_holder, dest_id, self.t))
        return contact

    def _build_observation(self) -> np.ndarray:
        b = self.bundle
        bundle_feats = [
            b.science_priority,
            min(b.remaining_bytes / 1e9, 1.0),
            max(0.0, min((b.deadline_s - self.t) / self.horizon_s, 1.0)),
            min(self.t / self.horizon_s, 1.0),
        ]
        if self.ablation == "no_priority":
            bundle_feats[0] = 0.0

        mask = self._action_mask()
        holder = self.bundle.current_holder
        rows = []
        for node_id in range(NUM_NODES):
            valid = mask[node_id]
            if valid:
                c = self._best_contact_to(node_id)
                remaining = max(0.0, c.end_s - max(self.t, c.start_s))
                health_obs = 0.0 if self.ablation == "no_weather_health" else c.health
                weather_obs = 0.0 if self.ablation == "no_weather_health" else c.weather_risk
                rows.extend([
                    1.0,
                    min(c.data_rate_bps / 5e7, 1.0),
                    min(remaining / self.horizon_s, 1.0),
                    min(c.range_km / 40000.0, 1.0),
                    c.queue_norm,
                    c.storage_free_norm,
                    health_obs,
                    c.battery,
                    weather_obs,
                    max(0.0, min((self.horizon_s - self.t) / self.horizon_s, 1.0)),
                    c.reliability,
                ])
            else:
                rows.extend([0.0] * CANDIDATE_FEATURES)

        obs = np.array(bundle_feats + rows, dtype=np.float32)
        return obs