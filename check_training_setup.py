#!/usr/bin/env python3
"""Pre-flight verifier for the exact final PPO experiment."""

from __future__ import annotations

from src.experiment.spec import load_final_spec
from src.integration.config import GROUND_IDS, GEO_IDS, LEO_IDS, NUM_NODES, SCIENCE_IDS
from src.integration.physical_rl_env import PhysicalRoutingEnv
from src.integration.rl_bridge import OBS_LEN
from src.integration.stochastic_transfer import StochasticTransferSettings
from src.integration.config import load_config


def main() -> None:
    spec = load_final_spec()
    config = load_config(spec.scenario_config)
    settings = StochasticTransferSettings.from_config(config)
    env = PhysicalRoutingEnv(config_path=spec.scenario_config, bundles_per_episode=4, seed=123, stochastic=True)
    obs, info = env.reset()
    mask = env.action_masks()
    print("Final experiment pre-flight")
    print("---------------------------")
    print(f"experiment   : {spec.name}")
    print(f"SCIENCE_IDS  : {SCIENCE_IDS}")
    print(f"LEO_IDS      : {LEO_IDS}")
    print(f"GEO_IDS      : {GEO_IDS}")
    print(f"GROUND_IDS   : {GROUND_IDS}")
    print(f"NUM_NODES    : {NUM_NODES}")
    print(f"OBS_LEN      : {OBS_LEN}")
    print(f"obs shape    : {obs.shape}")
    print(f"mask shape   : {mask.shape}")
    print(f"stochastic   : {settings.enabled}")
    print(f"config SHA   : {spec.scenario_config_sha256}")
    print(f"final model  : {spec.model}")
    print(f"traffic seed : {info.get('traffic_seed')}")
    print(f"xfer seed    : {info.get('transfer_seed')}")
    assert tuple(SCIENCE_IDS) == (0, 1, 2)
    assert tuple(LEO_IDS) == (3, 4, 5, 6, 7, 8)
    assert tuple(GEO_IDS) == (9, 10)
    assert tuple(GROUND_IDS) == (11, 12, 13)
    assert NUM_NODES == 14 and OBS_LEN == 158
    assert obs.shape == (158,) and mask.shape == (14,)
    assert settings.enabled
    print("\nPASS: locked training environment is ready.")
    if spec.model.exists():
        spec.verify_model(require_exists=True)
        print("PASS: final checkpoint exists and matches the locked config.")
    else:
        print("NOTE: final checkpoint does not exist yet; training can still start.")


if __name__ == "__main__":
    main()
