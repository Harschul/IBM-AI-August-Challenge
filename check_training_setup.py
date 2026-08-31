#!/usr/bin/env python3
"""Pre-flight check for the frozen physical stochastic PPO environment."""

from __future__ import annotations

from pathlib import Path

from src.integration.config import GROUND_IDS, GEO_IDS, LEO_IDS, NUM_NODES, SCIENCE_IDS, load_config
from src.integration.physical_rl_env import PhysicalRoutingEnv
from src.integration.rl_bridge import OBS_LEN
from src.integration.stochastic_transfer import StochasticTransferSettings


def main() -> None:
    config_path = Path("config/prototype.yaml")
    if not config_path.exists():
        raise SystemExit("config/prototype.yaml not found; run this from the repository root")

    config = load_config(config_path)
    settings = StochasticTransferSettings.from_config(config)
    env = PhysicalRoutingEnv(config_path=config_path, bundles_per_episode=4, seed=123, stochastic=True)
    obs, info = env.reset()
    mask = env.action_masks()

    print("Frozen physical PPO pre-flight")
    print("------------------------------")
    print(f"SCIENCE_IDS : {SCIENCE_IDS}")
    print(f"LEO_IDS     : {LEO_IDS}")
    print(f"GEO_IDS     : {GEO_IDS}")
    print(f"GROUND_IDS  : {GROUND_IDS}")
    print(f"NUM_NODES   : {NUM_NODES}")
    print(f"OBS_LEN     : {OBS_LEN}")
    print(f"obs shape   : {obs.shape}")
    print(f"mask shape  : {mask.shape}")
    print(f"legal hops  : {int(mask.sum())}")
    print(f"stochastic  : {settings.enabled}")
    print(f"base p_fail : {settings.base_failure_probability:.3f}")
    print(f"traffic seed: {info.get('traffic_seed')}")
    print(f"xfer seed   : {info.get('transfer_seed')}")

    expected_science = (0, 1, 2)
    expected_leo = (3, 4, 5, 6, 7, 8)
    expected_geo = (9, 10)
    expected_ground = (11, 12, 13)
    if tuple(SCIENCE_IDS) != expected_science:
        raise SystemExit(f"SCIENCE_IDS drifted: expected {expected_science}, got {SCIENCE_IDS}")
    if tuple(LEO_IDS) != expected_leo:
        raise SystemExit(f"LEO_IDS drifted: expected {expected_leo}, got {LEO_IDS}")
    if tuple(GEO_IDS) != expected_geo:
        raise SystemExit(f"GEO_IDS drifted: expected {expected_geo}, got {GEO_IDS}")
    if tuple(GROUND_IDS) != expected_ground:
        raise SystemExit(f"GROUND_IDS drifted: expected {expected_ground}, got {GROUND_IDS}")
    if NUM_NODES != 14 or OBS_LEN != 158:
        raise SystemExit("Frozen PPO contract drifted; expected 14 actions and 158 observations")
    if obs.shape != (158,) or mask.shape != (14,):
        raise SystemExit("Environment observation/action-mask shape mismatch")
    if not settings.enabled:
        raise SystemExit("stochastic_transfer.enabled must be true for stochastic retraining")

    print("\nPASS: environment is ready for stochastic physical PPO training.")


if __name__ == "__main__":
    main()
