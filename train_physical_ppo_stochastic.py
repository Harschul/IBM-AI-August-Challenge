#!/usr/bin/env python3
"""Train MaskablePPO on the frozen physical environment with real transfer failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from src.integration.config import GROUND_IDS, GEO_IDS, LEO_IDS, NUM_NODES, SCIENCE_IDS, load_config
from src.integration.physical_rl_env import PhysicalRoutingEnv
from src.integration.rl_bridge import OBS_LEN
from src.integration.stochastic_transfer import StochasticTransferSettings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def make_env(config_path: str, bundles_per_episode: int, max_attempts: int, seed: int):
    def _factory():
        return PhysicalRoutingEnv(
            config_path=config_path,
            bundles_per_episode=bundles_per_episode,
            max_attempts=max_attempts,
            seed=seed,
            stochastic=True,
        )

    return _factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/prototype.yaml")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--bundles-per-episode", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out",
        default="RL/rl_env_v0/models/physical_multisource_stochastic_ppo.zip",
    )
    parser.add_argument(
        "--tensorboard-log",
        default="RL/rl_env_v0/logs/physical_multisource_stochastic",
    )
    args = parser.parse_args()

    if args.n_envs < 1:
        raise SystemExit("--n-envs must be >= 1")

    config_path = Path(args.config)
    config = load_config(config_path)
    stochastic = StochasticTransferSettings.from_config(config)

    print("Frozen stochastic physical scenario")
    print("-----------------------------------")
    print(f"science IDs       : {SCIENCE_IDS}")
    print(f"LEO IDs           : {LEO_IDS}")
    print(f"GEO IDs           : {GEO_IDS}")
    print(f"ground IDs        : {GROUND_IDS}")
    print(f"actions           : {NUM_NODES}")
    print(f"observation       : {OBS_LEN}")
    print(f"config SHA        : {sha256_file(config_path)}")
    print(f"stochastic        : {stochastic.enabled}")
    print(f"base p_fail       : {stochastic.base_failure_probability:.3f}")
    print(f"max attempts      : {args.max_attempts}")

    env_fns = [
        make_env(
            str(config_path),
            args.bundles_per_episode,
            args.max_attempts,
            args.seed + rank * 10_000,
        )
        for rank in range(args.n_envs)
    ]
    env = VecMonitor(DummyVecEnv(env_fns))

    rollout_size = 1024
    batch_size = min(256, rollout_size * args.n_envs)
    batch_size = max(64, batch_size)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=rollout_size,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
        device=args.device,
        policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_path.parent / f"{out_path.stem}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callback = CheckpointCallback(
        save_freq=max(1, 250_000 // args.n_envs),
        save_path=str(checkpoint_dir),
        name_prefix=out_path.stem,
    )

    print(f"\nTraining for {args.timesteps:,} timesteps across {args.n_envs} environments...")
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    model.save(str(out_path))

    saved_model = out_path if out_path.suffix == ".zip" else Path(str(out_path) + ".zip")
    metadata_path = saved_model.with_suffix(".metadata.json")
    metadata = {
        "model": str(saved_model),
        "training_timesteps": args.timesteps,
        "training_seed": args.seed,
        "n_envs": args.n_envs,
        "bundles_per_episode": args.bundles_per_episode,
        "max_attempts": args.max_attempts,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "git_commit": git_commit(),
        "science_ids": list(SCIENCE_IDS),
        "leo_ids": list(LEO_IDS),
        "geo_ids": list(GEO_IDS),
        "ground_ids": list(GROUND_IDS),
        "num_nodes": NUM_NODES,
        "observation_length": OBS_LEN,
        "horizon_s": config.horizon_s,
        "sample_step_s": config.sample_step_s,
        "algorithm": "MaskablePPO",
        "policy": "MlpPolicy [256,256] actor/value",
        "stochastic_transfer": stochastic.__dict__,
        "stochastic_failure_semantics": {
            "failed_bundle_delivery": "atomic; bundle remains fully at sender",
            "failed_capacity": "partial capacity consumed up to sampled failure point",
            "failed_time": "time advances to sampled failure detection point",
            "next_decision": "retry or reroute from same holder",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved model    : {saved_model}")
    print(f"Saved metadata : {metadata_path}")
    print("Benchmark with benchmark_temporal_vs_rl_stochastic.py on held-out seeds.")


if __name__ == "__main__":
    main()
