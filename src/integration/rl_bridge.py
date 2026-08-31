"""Bridge the canonical physical ContactPlan to the final 14-action MaskablePPO policy.

The frozen final interface is 4 bundle features plus 14 candidates × 11 features
(158 floats) with a 14-entry action mask. The canonical checkpoint shipped with
this release was retrained on the 3-science / 6-LEO / 2-GEO / 3-ground physical
stochastic environment. No legacy RL package import is required at runtime.
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


def _numpy2_compat() -> None:
    """Let NumPy 1.x load checkpoints pickled under NumPy 2.x.

    The saved models reference `numpy._core.*`, NumPy 2's internal module
    layout; NumPy 1.26 calls the same modules `numpy.core.*`. Without this,
    loading dies with `ModuleNotFoundError: No module named
    'numpy._core.numeric'` on any machine that resolved an older NumPy --
    which run_integrated_demo.py currently does, so the baseline prints and
    then the RL half crashes.

    This compatibility shim is kept local to the final integration bridge so
    checkpoint loading does not depend on any legacy RL package.
    """
    import sys

    import numpy.core  # noqa: F401

    for suffix in ("", ".numeric", ".multiarray", ".umath", ".numerictypes",
                   ".overrides", "._multiarray_umath"):
        target = "numpy._core" + suffix
        if target in sys.modules:
            continue
        try:
            sys.modules[target] = __import__("numpy.core" + suffix, fromlist=["_"])
        except Exception:
            pass


def _space_overrides() -> dict:
    """Supply the spaces rather than unpickling them.

    A pickled gymnasium Space carries a numpy Generator, which does not survive
    the NumPy 2 -> 1.x boundary even with the module aliases above. The
    interface is frozen anyway (158 observations, 14 actions), so handing them
    to sb3 directly is both safe and faster than round-tripping them.
    """
    from gymnasium import spaces

    return {
        "observation_space": spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_LEN,), dtype=np.float32),
        "action_space": spaces.Discrete(NUM_NODES),
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }


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
                "Install requirements.txt."
            ) from exc

        _numpy2_compat()
        self._model = MaskablePPO.load(
            str(self.model_path),
            device="cpu",
            custom_objects=_space_overrides(),
        )

    def choose(self, observation: np.ndarray, mask: np.ndarray) -> int:
        if not mask.any():
            raise ValueError("cannot choose an RL action from an all-zero mask")
        action, _ = self._model.predict(
            observation,
            deterministic=True,
            action_masks=mask,
        )
        return int(action)
