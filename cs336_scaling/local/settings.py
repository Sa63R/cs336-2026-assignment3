from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LocalSettings:
    """Filesystem and service settings for the local experiment system."""

    home: Path
    database_path: Path
    runs_dir: Path
    datasets_dir: Path
    worker_lock_path: Path
    api_host: str
    api_port: int
    worker_poll_seconds: float
    scaling_budget_seconds: float
    final_budget_seconds: float

    @classmethod
    def from_env(cls) -> "LocalSettings":
        home = Path(
            os.environ.get("LOCAL_SCALING_HOME", PROJECT_ROOT / ".local_scaling")
        ).expanduser()
        return cls(
            home=home,
            database_path=Path(
                os.environ.get("LOCAL_SCALING_DB", home / "experiments.sqlite3")
            ).expanduser(),
            runs_dir=Path(
                os.environ.get("LOCAL_SCALING_RUNS_DIR", home / "runs")
            ).expanduser(),
            datasets_dir=Path(
                os.environ.get("LOCAL_SCALING_DATASETS_DIR", home / "datasets")
            ).expanduser(),
            worker_lock_path=Path(
                os.environ.get("LOCAL_SCALING_WORKER_LOCK", home / "worker.lock")
            ).expanduser(),
            api_host=os.environ.get("LOCAL_SCALING_API_HOST", "127.0.0.1"),
            api_port=int(os.environ.get("LOCAL_SCALING_API_PORT", "8765")),
            worker_poll_seconds=float(
                os.environ.get("LOCAL_SCALING_WORKER_POLL_SECONDS", "1")
            ),
            scaling_budget_seconds=float(
                os.environ.get("LOCAL_SCALING_BUDGET_SECONDS", str(3 * 3600))
            ),
            final_budget_seconds=float(
                os.environ.get("LOCAL_FINAL_BUDGET_SECONDS", str(12 * 3600))
            ),
        )

    def ensure_directories(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.worker_lock_path.parent.mkdir(parents=True, exist_ok=True)
