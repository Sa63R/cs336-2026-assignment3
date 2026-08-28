from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from cs336_scaling.schemas.base import FrozenForbidExtraModel
from cs336_scaling.training.model.config import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig, OptimizerConfig, SGDConfig
from cs336_scaling.training.training_config import TrainingConfig


BudgetGroup = Literal["scaling", "final"]
ExperimentStatus = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
]
WandbMode = Literal["disabled", "offline", "online"]
ComputeParameterBasis = Literal["non_embedding"]


class LocalTrainingConfig(FrozenForbidExtraModel):
    """Hardware-aware training configuration for one local JAX run.

    ``micro_batch_size`` is the number of sequences resident on the GPU for one
    forward/backward pass. The effective optimizer batch is the micro batch
    multiplied by ``gradient_accumulation_steps``.
    """

    architecture_config: BasicTransformerConfig
    optimizer_config: OptimizerConfig = Field(default_factory=AdamWConfig)
    micro_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    val_batch_size: int = Field(default=4, gt=0)
    seq_len: Literal[512] = 512
    validation_tokens: int = Field(default=2**18, gt=0)
    total_train_tokens: int = Field(gt=0, le=500_000_000_000)
    n_evals: int = Field(default=16, gt=0)
    max_runtime_seconds: float = Field(gt=0)
    model_seed: int = Field(default=0, ge=0)
    data_seed: int = Field(default=67, ge=0)

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.seq_len * self.effective_batch_size

    @property
    def total_optimizer_steps(self) -> int:
        return self.total_train_tokens // self.tokens_per_optimizer_step

    @property
    def optimizer_steps_per_eval(self) -> int:
        return self.total_optimizer_steps // self.n_evals

    def optimizer_training_config(self) -> TrainingConfig:
        """Return the course config shape expected by the optimizer builder."""

        return TrainingConfig(
            architecture_config=self.architecture_config,
            optimizer_config=self.optimizer_config,
            train_batch_size=self.effective_batch_size,
            val_batch_size=self.val_batch_size,
            n_evals=self.n_evals,
            total_train_tokens=self.total_train_tokens,
            max_runtime_seconds=self.max_runtime_seconds,
            model_seed=self.model_seed,
        )

    @model_validator(mode="after")
    def validate_training_shape(self) -> Self:
        if self.total_train_tokens % self.tokens_per_optimizer_step != 0:
            raise ValueError(
                "total_train_tokens must be divisible by "
                "seq_len * micro_batch_size * gradient_accumulation_steps"
            )
        if self.total_optimizer_steps % self.n_evals != 0:
            raise ValueError("total optimizer steps must be divisible by n_evals")
        if self.validation_tokens % (self.seq_len * self.val_batch_size) != 0:
            raise ValueError(
                "validation_tokens must be divisible by seq_len * val_batch_size"
            )

        # Reuse the course validator for architectural and optimizer invariants.
        self.optimizer_training_config()
        return self


class CheckpointConfig(FrozenForbidExtraModel):
    every_n_evals: int = Field(default=1, gt=0)
    keep_last: int = Field(default=2, ge=1)
    keep_best: bool = True


class WandbConfig(FrozenForbidExtraModel):
    mode: WandbMode = "disabled"
    project: str = "cs336-local-scaling"
    entity: str | None = None
    group: str | None = None


class LocalExperimentConfig(FrozenForbidExtraModel):
    training: LocalTrainingConfig
    dataset_manifest: Path
    target_compute_flops: float | None = Field(default=None, gt=0)
    compute_parameter_basis: ComputeParameterBasis | None = None
    budget_group: BudgetGroup = "scaling"
    priority: int = Field(default=0, ge=-100, le=100)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    notes: str | None = Field(default=None, max_length=2_000)

    def semantic_hash(self, dataset_id: str) -> str:
        """Hash inputs that can affect model weights or measured losses."""

        semantic_value = {
            "training": self.training.model_dump(mode="json"),
            "dataset_id": dataset_id,
        }
        encoded = json.dumps(
            semantic_value, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.blake2s(encoded, digest_size=16).hexdigest()


class DatasetManifest(FrozenForbidExtraModel):
    format_version: Literal[1] = 1
    dataset_id: str
    source: str
    source_revision: str | None = None
    tokenizer: str
    tokenizer_revision: str | None = None
    vocab_size: int = Field(gt=1, le=65_536)
    eos_token_id: int = Field(ge=0, le=65_535)
    seed: int = Field(ge=0)
    train_tokens_path: Path
    validation_tokens_path: Path
    train_tokens: int = Field(gt=0)
    validation_tokens: int = Field(gt=0)
    train_sha256: str
    validation_sha256: str
    created_at: dt.datetime

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        manifest = cls.model_validate_json(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
        updates: dict[str, Path] = {}
        if not manifest.train_tokens_path.is_absolute():
            updates["train_tokens_path"] = base / manifest.train_tokens_path
        if not manifest.validation_tokens_path.is_absolute():
            updates["validation_tokens_path"] = base / manifest.validation_tokens_path
        return manifest.model_copy(update=updates)


class SubmitExperimentRequest(FrozenForbidExtraModel):
    config: LocalExperimentConfig


class SubmitExperimentResponse(FrozenForbidExtraModel):
    experiment_id: int
    config_hash: str


class RetryExperimentRequest(FrozenForbidExtraModel):
    resume: bool = True


class BudgetUsage(FrozenForbidExtraModel):
    budget_group: BudgetGroup
    used_seconds: float
    reserved_seconds: float
    remaining_seconds: float
    total_seconds: float


class MetricRecord(FrozenForbidExtraModel):
    experiment_id: int
    phase: Literal["train", "validation", "system"]
    optimizer_step: int = Field(ge=0)
    tokens_seen: int = Field(ge=0)
    values: dict[str, float]
    created_at: dt.datetime

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if any(not math.isfinite(value) for value in self.values.values()):
            raise ValueError("metric values must be finite")
        return self


class ExperimentResponse(FrozenForbidExtraModel):
    experiment_id: int
    attempt: int
    config_hash: str
    status: ExperimentStatus
    config: LocalExperimentConfig
    dataset_id: str
    output_dir: Path
    created_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    heartbeat_at: dt.datetime | None = None
    worker_pid: int | None = None
    used_runtime_seconds: float = 0.0
    cancel_requested: bool = False
    resume_from_experiment_id: int | None = None
    result: dict[str, object] | None = None
    error_message: str | None = None


OptimizerConfigField = Annotated[
    AdamWConfig | SGDConfig,
    Field(union_mode="left_to_right"),
]
