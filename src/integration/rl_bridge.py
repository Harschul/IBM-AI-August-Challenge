"""Use the existing 14-action MaskablePPO checkpoints on canonical contacts.

The trained policy's frozen interface is preserved: 4 bundle features plus
14 candidates x 11 features = 158 floats, and a 14-entry action mask. The
physical scenario now uses three science sources while keeping 14 total IDs, so
existing checkpoints remain shape-compatible. Their training distribution was
single-source, so multi-source results should be treated as transfer evaluation
until the policy is retrained on the new source distribution.
No import from `RL/rl_env_v0/src` is required, avoiding the repository's two
competing packages named `src`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.models.bundle import DataBundle
from src.models.contact import Contact, ContactPlan

from .config import NUM_NODES, PrototypeConfig

BUNDLE_FEATURES = 4
CANDIDATE_FEATURES = 11
OBS_LEN = BUNDLE_FEATURES + NUM_NODES * CANDIDATE_FEATURES


def best_feasible_contact(
    plan: ContactPlan,
    holder: int,
    destination: int,
    now_s: float,
    size_bytes: int,
) -> Contact | None:
    best = None
    best_arrival = None
    for c in plan.from_node(holder):
        if c.destination_id != destination:
            continue
        depart = max(float(now_s), c.start_s)
        if depart >= c.end_s:
            continue
        if c.residual_capacity_bytes < size_bytes:
            continue
        arrival = depart + c.transmission_time_s(size_bytes) + c.propagation_delay_s
        # The entire transmission (not propagation) must fit in the contact.
        if depart + c.transmission_time_s(size_bytes) > c.end_s:
            continue
        if best_arrival is None or arrival < best_arrival:
            best, best_arrival = c, arrival
    return best


def action_mask(plan: ContactPlan, bundle: DataBundle, now_s: float) -> np.ndarray:
    mask = np.zeros(NUM_NODES, dtype=np.int8)
    for node_id in range(NUM_NODES):
        if node_id == bundle.current_holder:
            continue
        if best_feasible_contact(
            plan,
            bundle.current_holder,
            node_id,
            now_s,
            bundle.remaining_bytes,
        ) is not None:
            mask[node_id] = 1
    return mask


def build_observation(
    plan: ContactPlan,
    bundle: DataBundle,
    now_s: float,
    config: PrototypeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = config.horizon_s
    deadline = bundle.deadline_s if bundle.deadline_s is not None else horizon
    bundle_features = [
        float(bundle.science_priority),
        min(float(bundle.remaining_bytes) / 1e9, 1.0),
        max(0.0, min((float(deadline) - now_s) / horizon, 1.0)),
        min(now_s / horizon, 1.0),
    ]

    mask = action_mask(plan, bundle, now_s)
    defaults = config.runtime_defaults
    rows: list[float] = []
    for node_id in range(NUM_NODES):
        if mask[node_id]:
            c = best_feasible_contact(
                plan,
                bundle.current_holder,
                node_id,
                now_s,
                bundle.remaining_bytes,
            )
            assert c is not None
            depart = max(now_s, c.start_s)
            remaining = max(0.0, c.end_s - depart)
            nominal_capacity = max(1.0, (c.end_s - c.start_s) * c.data_rate_bps / 8.0)
            utilization = max(0.0, min(1.0, 1.0 - c.residual_capacity_bytes / nominal_capacity))
            queue_norm = max(defaults.queue_norm, utilization)
            storage_free = max(0.0, min(1.0, defaults.storage_free_norm * (1.0 - 0.5 * utilization)))
            rows.extend(
                [
                    1.0,
                    min(c.data_rate_bps / 5e7, 1.0),
                    min(remaining / horizon, 1.0),
                    min(c.range_km / 40000.0, 1.0),
                    queue_norm,
                    storage_free,
                    defaults.health,
                    defaults.battery,
                    c.weather_risk,
                    max(0.0, min((horizon - now_s) / horizon, 1.0)),
                    c.reliability,
                ]
            )
        else:
            rows.extend([0.0] * CANDIDATE_FEATURES)

    observation = np.asarray(bundle_features + rows, dtype=np.float32)
    if observation.shape != (OBS_LEN,):
        raise AssertionError(f"RL observation drifted: expected {OBS_LEN}, got {observation.shape}")
    return observation, mask


class MaskablePPOPolicy:
    """Thin lazy loader for the existing trained MaskablePPO checkpoints."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise RuntimeError(
                "RL policy requested but sb3-contrib is not installed. "
                "Install requirements-integration.txt."
            ) from exc
        self._model = MaskablePPO.load(str(self.model_path))

    def choose(self, observation: np.ndarray, mask: np.ndarray) -> int:
        if not mask.any():
            raise ValueError("cannot choose an RL action from an all-zero mask")
        action, _ = self._model.predict(
            observation,
            deterministic=True,
            action_masks=mask,
        )
        return int(action)
