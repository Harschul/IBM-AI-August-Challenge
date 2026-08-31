#!/usr/bin/env python3
"""Train the final MaskablePPO checkpoint on the locked physical stochastic environment."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from src.experiment.spec import load_final_spec, sha256_file
from src.integration.config import GROUND_IDS, GEO_IDS, LEO_IDS, NUM_NODES, SCIENCE_IDS, load_config
from src.integration.physical_rl_env import PhysicalRoutingEnv
from src.integration.rl_bridge import OBS_LEN
from src.integration.stochastic_transfer import StochasticTransferSettings


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def make_env(config_path: Path, bundles_per_episode: int, max_attempts: int, seed: int):
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
    spec = load_final_spec()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=spec.training.timesteps)
    parser.add_argument("--n-envs", type=int, default=spec.training.n_envs)
    parser.add_argument("--bundles-per-episode", type=int, default=spec.training.bundles_per_episode)
    parser.add_argument("--max-attempts", type=int, default=spec.training.max_attempts)
    parser.add_argument("--seed", type=int, default=spec.training.seed)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=spec.model)
    parser.add_argument("--tensorboard-log", type=Path, default=spec.training.tensorboard_dir)
    args = parser.parse_args()

    if args.n_envs < 1:
        raise SystemExit("--n-envs must be >= 1")
    config = load_config(spec.scenario_config)
    stochastic = StochasticTransferSettings.from_config(config)

    print("Locked final stochastic physical scenario")
    print("-----------------------------------------")
    print(f"science IDs       : {SCIENCE_IDS}")
    print(f"LEO IDs           : {LEO_IDS}")
    print(f"GEO IDs           : {GEO_IDS}")
    print(f"ground IDs        : {GROUND_IDS}")
    print(f"actions           : {NUM_NODES}")
    print(f"observation       : {OBS_LEN}")
    print(f"config SHA        : {sha256_file(spec.scenario_config)}")
    print(f"stochastic        : {stochastic.enabled}")
    print(f"final model path  : {args.out}")

    env_fns = [
        make_env(spec.scenario_config, args.bundles_per_episode, args.max_attempts, args.seed + rank * 10_000)
        for rank in range(args.n_envs)
    ]
    env = VecMonitor(DummyVecEnv(env_fns))
    rollout_size = 1024
    batch_size = max(64, min(256, rollout_size * args.n_envs))

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
        tensorboard_log=str(args.tensorboard_log),
        seed=args.seed,
        device=args.device,
        policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
    )

    out_path = args.out
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
        "experiment": spec.name,
        "model": str(saved_model),
        "training_timesteps": args.timesteps,
        "training_seed": args.seed,
        "n_envs": args.n_envs,
        "bundles_per_episode": args.bundles_per_episode,
        "max_attempts": args.max_attempts,
        "config_path": str(spec.scenario_config),
        "config_sha256": spec.scenario_config_sha256,
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
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved model    : {saved_model}")
    print(f"Saved metadata : {metadata_path}")
    print("The final demo and benchmark now resolve this exact checkpoint through config/final_experiment.json.")


if __name__ == "__main__":
    main()
