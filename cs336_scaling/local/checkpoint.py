from __future__ import annotations

import datetime as dt
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx


@dataclass(frozen=True)
class CheckpointMetadata:
    format_version: int
    experiment_id: int
    config_hash: str
    optimizer_step: int
    eval_index: int
    tokens_seen: int
    validation_loss: float
    elapsed_training_seconds: float
    created_at: str

    @classmethod
    def load(cls, path: Path) -> "CheckpointMetadata":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class CheckpointManager:
    def __init__(
        self,
        root: Path,
        *,
        experiment_id: int,
        config_hash: str,
        keep_last: int,
        keep_best: bool,
    ):
        self.root = root
        self.experiment_id = experiment_id
        self.config_hash = config_hash
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        payload: object,
        *,
        optimizer_step: int,
        eval_index: int,
        tokens_seen: int,
        validation_loss: float,
        elapsed_training_seconds: float,
    ) -> Path:
        destination = self.root / f"step_{optimizer_step:09d}"
        if destination.exists():
            return destination

        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=self.root))
        try:
            eqx.tree_serialise_leaves(temporary / "state.eqx", payload)
            metadata = CheckpointMetadata(
                format_version=1,
                experiment_id=self.experiment_id,
                config_hash=self.config_hash,
                optimizer_step=optimizer_step,
                eval_index=eval_index,
                tokens_seen=tokens_seen,
                validation_loss=validation_loss,
                elapsed_training_seconds=elapsed_training_seconds,
                created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
            metadata_path = temporary / "metadata.json"
            metadata_path.write_text(
                json.dumps(metadata.__dict__, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self.prune()
        return destination

    def list_checkpoints(self) -> list[tuple[Path, CheckpointMetadata]]:
        checkpoints: list[tuple[Path, CheckpointMetadata]] = []
        for directory in self.root.glob("step_*"):
            metadata_path = directory / "metadata.json"
            payload_path = directory / "state.eqx"
            if metadata_path.is_file() and payload_path.is_file():
                checkpoints.append((directory, CheckpointMetadata.load(metadata_path)))
        return sorted(checkpoints, key=lambda item: item[1].optimizer_step)

    def latest(self) -> tuple[Path, CheckpointMetadata] | None:
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def prune(self) -> None:
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self.keep_last:
            return
        retained = {path for path, _ in checkpoints[-self.keep_last :]}
        if self.keep_best:
            best_path, _ = min(checkpoints, key=lambda item: item[1].validation_loss)
            retained.add(best_path)
        for path, _ in checkpoints:
            if path not in retained:
                shutil.rmtree(path)


def load_checkpoint(path: Path, template: object) -> tuple[object, CheckpointMetadata]:
    metadata = CheckpointMetadata.load(path / "metadata.json")
    payload = eqx.tree_deserialise_leaves(path / "state.eqx", template)
    return payload, metadata


def find_latest_checkpoint(run_dir: Path) -> tuple[Path, CheckpointMetadata] | None:
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return None
    candidates: list[tuple[Path, CheckpointMetadata]] = []
    for directory in checkpoint_root.glob("step_*"):
        metadata_path = directory / "metadata.json"
        payload_path = directory / "state.eqx"
        if metadata_path.is_file() and payload_path.is_file():
            candidates.append((directory, CheckpointMetadata.load(metadata_path)))
    return max(candidates, key=lambda item: item[1].optimizer_step, default=None)
