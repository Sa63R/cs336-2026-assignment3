from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

from cs336_scaling.local.database import LocalDatabase
from cs336_scaling.local.schemas import LocalExperimentConfig, MetricRecord


def atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def collect_environment() -> dict[str, object]:
    packages = {}
    for package in ("jax", "jaxlib", "equinox", "optax", "flash-hog", "wandb"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    git_status = command_output(["git", "status", "--porcelain"])
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(git_status),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


class RunRecorder:
    def __init__(
        self,
        *,
        experiment_id: int,
        output_dir: Path,
        database: LocalDatabase,
    ):
        self.experiment_id = experiment_id
        self.output_dir = output_dir
        self.database = database
        self.metrics_path = output_dir / "metrics.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)

    def initialize(
        self,
        config: LocalExperimentConfig,
        *,
        config_hash: str,
        dataset_id: str,
    ) -> None:
        atomic_write_json(
            self.output_dir / "config.json", config.model_dump(mode="json")
        )
        atomic_write_json(
            self.output_dir / "environment.json",
            collect_environment()
            | {"config_hash": config_hash, "dataset_id": dataset_id},
        )

    def metric(self, metric: MetricRecord) -> None:
        line = metric.model_dump_json() + "\n"
        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(line)
            file.flush()
            os.fsync(file.fileno())
        self.database.add_metric(metric)

    def result(self, value: dict[str, object]) -> None:
        atomic_write_json(self.output_dir / "result.json", value)
