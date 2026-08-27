from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cs336_scaling.local.integrity import sha256_file
from cs336_scaling.local.schemas import DatasetManifest, LocalTrainingConfig
from cs336_scaling.training.data import Batch


class LocalTokenDataset:
    """Memory-mapped deterministic token stream for local training."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path.expanduser().resolve()
        self.manifest = DatasetManifest.load(self.manifest_path)
        self.train_tokens = np.load(
            self.manifest.train_tokens_path, mmap_mode="r", allow_pickle=False
        )
        self.validation_tokens = np.load(
            self.manifest.validation_tokens_path, mmap_mode="r", allow_pickle=False
        )
        self._validate_array(self.train_tokens, self.manifest.train_tokens, "training")
        self._validate_array(
            self.validation_tokens, self.manifest.validation_tokens, "validation"
        )

    @staticmethod
    def _validate_array(array: NDArray, expected_tokens: int, name: str) -> None:
        if array.ndim != 1:
            raise ValueError(f"{name} token array must be one-dimensional")
        if array.dtype != np.uint16:
            raise ValueError(f"{name} token array must use uint16, got {array.dtype}")
        if array.size != expected_tokens:
            raise ValueError(
                f"{name} manifest records {expected_tokens} tokens, "
                f"but the file contains {array.size}"
            )

    def verify_checksums(self) -> None:
        train_hash = sha256_file(self.manifest.train_tokens_path)
        validation_hash = sha256_file(self.manifest.validation_tokens_path)
        if train_hash != self.manifest.train_sha256:
            raise ValueError("training token file checksum does not match its manifest")
        if validation_hash != self.manifest.validation_sha256:
            raise ValueError(
                "validation token file checksum does not match its manifest"
            )

    def validate_for(self, config: LocalTrainingConfig) -> None:
        if config.data_seed != self.manifest.seed:
            raise ValueError(
                f"training data_seed {config.data_seed} must match dataset seed "
                f"{self.manifest.seed}"
            )
        if config.architecture_config.vocab_size != self.manifest.vocab_size:
            raise ValueError("model and dataset vocabulary sizes do not match")
        if self.train_tokens.size < config.total_train_tokens + 1:
            raise ValueError("training token file is too short for this experiment")
        if self.validation_tokens.size < config.validation_tokens + 1:
            raise ValueError("validation token file is too short for this experiment")
        max_train_token = int(
            np.max(self.train_tokens[: config.total_train_tokens + 1])
        )
        max_validation_token = int(
            np.max(self.validation_tokens[: config.validation_tokens + 1])
        )
        if max(max_train_token, max_validation_token) >= self.manifest.vocab_size:
            raise ValueError(
                "token file contains an id outside the declared vocabulary"
            )

    def train_chunk(
        self,
        *,
        optimizer_step: int,
        optimizer_steps: int,
        config: LocalTrainingConfig,
    ) -> Batch[NDArray]:
        """Read a contiguous chunk shaped as updates/accumulation/microbatch/sequence."""

        start = optimizer_step * config.tokens_per_optimizer_step
        token_count = optimizer_steps * config.tokens_per_optimizer_step
        stop = start + token_count
        if stop > config.total_train_tokens:
            raise ValueError(
                f"requested training tokens [{start}, {stop}) exceed configured "
                f"total {config.total_train_tokens}"
            )
        source = np.asarray(self.train_tokens[start : stop + 1], dtype=np.int32)
        input_ids = source[:-1].reshape(
            optimizer_steps,
            config.gradient_accumulation_steps,
            config.micro_batch_size,
            config.seq_len,
        )
        labels = source[1:].reshape(input_ids.shape)
        return Batch(input_ids=input_ids, labels=labels)

    def validation_batches(self, config: LocalTrainingConfig) -> Batch[NDArray]:
        source = np.asarray(
            self.validation_tokens[: config.validation_tokens + 1], dtype=np.int32
        )
        input_ids = source[:-1].reshape(
            -1,
            config.val_batch_size,
            config.seq_len,
        )
        labels = source[1:].reshape(input_ids.shape)
        return Batch(input_ids=input_ids, labels=labels)
