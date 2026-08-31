"""Repository-root-safe paths and the lock file for the final experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = REPO_ROOT / "config" / "final_experiment.json"


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BenchmarkSpec:
    algorithms: tuple[str, ...]
    num_seeds: int
    traffic_seed_base: int
    stochastic_seed_base: int
    bundles_per_seed: int
    max_hops: int
    max_attempts: int
    output_dir: Path

    def seeds(self, offset: int) -> tuple[int, int]:
        if not 0 <= offset < self.num_seeds:
            raise ValueError(f"seed offset must be in 0..{self.num_seeds - 1}")
        return self.traffic_seed_base + offset, self.stochastic_seed_base + offset


@dataclass(frozen=True)
class DemoSpec:
    default_seed_offset: int
    default_algorithm: str
    output_dir: Path
    reported_modes: tuple[str, ...]
    interactive_switch_is_reported: bool


@dataclass(frozen=True)
class TrainingSpec:
    timesteps: int
    n_envs: int
    bundles_per_episode: int
    max_attempts: int
    seed: int
    tensorboard_dir: Path


@dataclass(frozen=True)
class FinalExperimentSpec:
    name: str
    scenario_config: Path
    scenario_config_sha256: str
    model: Path
    benchmark: BenchmarkSpec
    demo: DemoSpec
    training: TrainingSpec
    raw: Mapping[str, Any]

    def verify_config(self) -> str:
        actual = sha256_file(self.scenario_config)
        if actual != self.scenario_config_sha256:
            raise RuntimeError(
                "Final experiment config drifted. "
                f"Expected {self.scenario_config_sha256}, got {actual}. "
                f"File: {self.scenario_config}"
            )
        return actual

    @property
    def model_metadata(self) -> Path:
        return self.model.with_suffix(".metadata.json")

    def verify_model(self, *, require_exists: bool = True) -> dict[str, Any] | None:
        self.verify_config()
        if not self.model.exists():
            if require_exists:
                raise FileNotFoundError(
                    f"Final retrained PPO checkpoint not found: {self.model}. "
                    "Run the final trainer or copy the trained checkpoint to this exact path."
                )
            return None
        if not self.model_metadata.exists():
            if require_exists:
                raise FileNotFoundError(
                    f"Model metadata not found: {self.model_metadata}. "
                    "The final demo requires metadata so it cannot silently use a stale checkpoint."
                )
            return None
        metadata = json.loads(self.model_metadata.read_text(encoding="utf-8"))
        trained_sha = metadata.get("config_sha256")
        if trained_sha != self.scenario_config_sha256:
            raise RuntimeError(
                "Final PPO checkpoint was not trained on the locked scenario config. "
                f"metadata config SHA={trained_sha!r}; expected={self.scenario_config_sha256}."
            )
        if int(metadata.get("num_nodes", 14)) != 14 or int(metadata.get("observation_length", 158)) != 158:
            raise RuntimeError("Final PPO metadata does not match the frozen 14-action / 158-observation contract.")
        science_ids = tuple(metadata.get("science_ids", (0, 1, 2)))
        if science_ids != (0, 1, 2):
            raise RuntimeError(f"Final PPO metadata has unexpected science IDs: {science_ids}")
        return metadata


def load_final_spec(path: str | Path = DEFAULT_SPEC_PATH) -> FinalExperimentSpec:
    path = repo_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    benchmark = payload["benchmark"]
    demo = payload["demo"]
    training = payload["training"]
    spec = FinalExperimentSpec(
        name=str(payload["name"]),
        scenario_config=repo_path(payload["scenario_config"]),
        scenario_config_sha256=str(payload["scenario_config_sha256"]),
        model=repo_path(payload["model"]),
        benchmark=BenchmarkSpec(
            algorithms=tuple(str(v) for v in benchmark["algorithms"]),
            num_seeds=int(benchmark["num_seeds"]),
            traffic_seed_base=int(benchmark["traffic_seed_base"]),
            stochastic_seed_base=int(benchmark["stochastic_seed_base"]),
            bundles_per_seed=int(benchmark["bundles_per_seed"]),
            max_hops=int(benchmark.get("max_hops", 8)),
            max_attempts=int(benchmark["max_attempts"]),
            output_dir=repo_path(benchmark["output_dir"]),
        ),
        demo=DemoSpec(
            default_seed_offset=int(demo["default_seed_offset"]),
            default_algorithm=str(demo["default_algorithm"]),
            output_dir=repo_path(demo["output_dir"]),
            reported_modes=tuple(str(v) for v in demo["reported_modes"]),
            interactive_switch_is_reported=bool(demo["interactive_switch_is_reported"]),
        ),
        training=TrainingSpec(
            timesteps=int(training["timesteps"]),
            n_envs=int(training["n_envs"]),
            bundles_per_episode=int(training["bundles_per_episode"]),
            max_attempts=int(training["max_attempts"]),
            seed=int(training["seed"]),
            tensorboard_dir=repo_path(training["tensorboard_dir"]),
        ),
        raw=payload,
    )
    spec.verify_config()
    return spec
