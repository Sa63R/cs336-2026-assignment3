from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cs336_scaling.local.api import create_app
from cs336_scaling.local.cli import app as local_cli
from cs336_scaling.local.data import LocalTokenDataset
from cs336_scaling.local.database import (
    DuplicateExperimentError,
    LocalDatabase,
)
from cs336_scaling.local.integrity import sha256_file
from cs336_scaling.local.schemas import (
    DatasetManifest,
    LocalExperimentConfig,
    LocalTrainingConfig,
    MetricRecord,
)
from cs336_scaling.local.planning import (
    estimate_parameter_count,
    estimate_parameter_counts,
    round_tokens_for_compute,
    runtime_limit_for_compute,
)
from cs336_scaling.local.settings import LocalSettings
from cs336_scaling.training.model.config import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig


def local_settings(tmp_path: Path) -> LocalSettings:
    home = tmp_path / "local"
    return LocalSettings(
        home=home,
        database_path=home / "experiments.sqlite3",
        runs_dir=home / "runs",
        datasets_dir=home / "datasets",
        worker_lock_path=home / "worker.lock",
        api_host="127.0.0.1",
        api_port=8765,
        worker_poll_seconds=0.01,
        scaling_budget_seconds=100,
        final_budget_seconds=200,
    )


def write_dataset(settings: LocalSettings) -> Path:
    directory = settings.datasets_dir / "test"
    directory.mkdir(parents=True)
    train_path = directory / "train.npy"
    validation_path = directory / "validation.npy"
    np.save(train_path, np.arange(8_193, dtype=np.uint16) % 256)
    np.save(validation_path, np.arange(2_049, dtype=np.uint16) % 256)
    manifest = DatasetManifest(
        dataset_id="test-dataset-id",
        source="unit-test",
        tokenizer="unit-test",
        vocab_size=256,
        eos_token_id=0,
        seed=67,
        train_tokens_path=Path("train.npy"),
        validation_tokens_path=Path("validation.npy"),
        train_tokens=8_193,
        validation_tokens=2_049,
        train_sha256=sha256_file(train_path),
        validation_sha256=sha256_file(validation_path),
        created_at=dt.datetime.now(dt.timezone.utc),
    )
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest_path


def experiment_config(manifest_path: Path, *, model_seed: int = 0):
    return LocalExperimentConfig(
        training=LocalTrainingConfig(
            architecture_config=BasicTransformerConfig(
                attention_bias=False,
                head_dim=32,
                hidden_size=64,
                intermediate_size=128,
                num_attention_heads=2,
                num_hidden_layers=1,
                num_key_value_heads=2,
                rms_norm_eps=1e-6,
                rope_theta=10_000,
                tie_word_embeddings=True,
                dtype="bfloat16",
                vocab_size=256,
            ),
            optimizer_config=AdamWConfig(),
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            val_batch_size=1,
            validation_tokens=1_024,
            total_train_tokens=2_048,
            n_evals=2,
            max_runtime_seconds=10,
            model_seed=model_seed,
            data_seed=67,
        ),
        dataset_manifest=manifest_path,
    )


def test_database_state_machine_deduplication_and_budget(tmp_path: Path):
    settings = local_settings(tmp_path)
    manifest_path = write_dataset(settings)
    config = experiment_config(manifest_path)
    database = LocalDatabase(settings)
    database.initialize()

    experiment_id = database.submit(
        config,
        config_hash="config-hash",
        dataset_id="test-dataset-id",
    )
    assert experiment_id == 1
    assert database.budget_usage("scaling").reserved_seconds == 10

    with pytest.raises(DuplicateExperimentError) as duplicate:
        database.submit(
            config,
            config_hash="config-hash",
            dataset_id="test-dataset-id",
        )
    assert duplicate.value.experiment_id == experiment_id

    claimed = database.claim_next(worker_pid=123)
    assert claimed is not None
    assert claimed.experiment_id == experiment_id
    database.add_metric(
        MetricRecord(
            experiment_id=experiment_id,
            phase="validation",
            optimizer_step=1,
            tokens_seen=512,
            values={"validation_loss": 5.0},
            created_at=dt.datetime.now(dt.timezone.utc),
        )
    )
    database.finish(
        experiment_id,
        status="failed",
        used_runtime_seconds=4,
        error_message="test failure",
    )

    usage = database.budget_usage("scaling")
    assert usage.used_seconds == 4
    assert usage.reserved_seconds == 0
    assert usage.remaining_seconds == 96
    assert database.list_metrics(experiment_id)[0].values == {"validation_loss": 5.0}

    retry_id = database.retry(experiment_id, resume=True)
    retry = database.get_experiment(retry_id)
    assert retry.attempt == 2
    assert retry.resume_from_experiment_id == experiment_id
    assert database.budget_usage("scaling").reserved_seconds == 10


def test_api_submit_query_cancel_and_budget(tmp_path: Path):
    settings = local_settings(tmp_path)
    manifest_path = write_dataset(settings)
    config = experiment_config(manifest_path)

    with TestClient(create_app(settings)) as client:
        submit = client.post(
            "/experiments",
            json={"config": config.model_dump(mode="json")},
        )
        assert submit.status_code == 201
        experiment_id = submit.json()["experiment_id"]

        queried = client.get(f"/experiments/{experiment_id}")
        assert queried.status_code == 200
        assert queried.json()["status"] == "queued"

        budget = client.get("/budget")
        assert budget.status_code == 200
        assert budget.json()[0]["reserved_seconds"] == 10

        cancelled = client.post(f"/experiments/{experiment_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"


def test_memory_mapped_token_batches_are_shifted_and_shaped(tmp_path: Path):
    settings = local_settings(tmp_path)
    manifest_path = write_dataset(settings)
    config = experiment_config(manifest_path).training
    dataset = LocalTokenDataset(manifest_path)
    dataset.verify_checksums()
    dataset.validate_for(config)

    chunk = dataset.train_chunk(
        optimizer_step=1,
        optimizer_steps=1,
        config=config,
    )
    assert chunk.input_ids.shape == (1, 1, 1, 512)
    assert chunk.labels.shape == chunk.input_ids.shape
    np.testing.assert_array_equal(chunk.input_ids[..., 1:], chunk.labels[..., :-1])

    validation = dataset.validation_batches(config)
    assert validation.input_ids.shape == (2, 1, 512)
    np.testing.assert_array_equal(
        validation.input_ids.reshape(-1)[1:], validation.labels.reshape(-1)[:-1]
    )


def test_checksum_verification_detects_file_change(tmp_path: Path):
    settings = local_settings(tmp_path)
    manifest_path = write_dataset(settings)
    dataset = LocalTokenDataset(manifest_path)
    dataset.verify_checksums()
    dataset.verify_checksums()

    with dataset.manifest.train_tokens_path.open("r+b") as file:
        file.seek(-1, 2)
        final_byte = file.read(1)
        file.seek(-1, 2)
        file.write(bytes([final_byte[0] ^ 1]))

    with pytest.raises(ValueError, match="checksum"):
        dataset.verify_checksums()


def test_parameter_estimate_and_compute_rounding(tmp_path: Path):
    settings = local_settings(tmp_path)
    manifest_path = write_dataset(settings)
    architecture = experiment_config(manifest_path).training.architecture_config

    parameters = estimate_parameter_count(architecture)
    counts = estimate_parameter_counts(architecture)
    assert parameters == 57_600
    assert counts.total == parameters
    assert counts.embedding == 16_384
    assert counts.non_embedding == 41_216
    assert counts.approximate_non_embedding == 49_152
    target_compute = 6 * counts.non_embedding * 2_048
    assert (
        round_tokens_for_compute(
            target_compute,
            parameters=counts.non_embedding,
            token_quantum=512,
            maximum_tokens=2_048,
        )
        == 2_048
    )
    assert (
        runtime_limit_for_compute(
            1e12,
            reference_flops_per_second=2e11,
            margin=1.3,
            minimum_seconds=10,
        )
        == 10
    )


def test_make_sweep_and_export_completed_runs(tmp_path: Path, monkeypatch):
    settings = local_settings(tmp_path)
    manifest_path = write_dataset(settings)
    monkeypatch.setenv("LOCAL_SCALING_HOME", str(settings.home))
    output_dir = tmp_path / "sweep"
    runner = CliRunner()
    generated = runner.invoke(
        local_cli,
        [
            "make-sweep",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--compute-budget",
            "5e8",
            "--hidden-size",
            "128",
            "--gradient-accumulation-steps",
            "1",
            "--n-evals",
            "1",
            "--validation-tokens",
            "2048",
            "--max-runtime-seconds",
            "10",
            "--wandb-mode",
            "disabled",
        ],
    )
    assert generated.exit_code == 0, generated.output
    plan = json.loads((output_dir / "plan.json").read_text())
    assert plan["parameter_basis"] == "non_embedding"
    assert len(plan["configs"]) == 1
    row = plan["configs"][0]

    config = LocalExperimentConfig.model_validate_json(
        (output_dir / row["config"]).read_text()
    )
    assert config.compute_parameter_basis == "non_embedding"
    database = LocalDatabase(settings)
    database.initialize()
    experiment_id = database.submit(
        config,
        config_hash="profiled-config",
        dataset_id="test-dataset-id",
    )
    assert database.claim_next(worker_pid=123) is not None
    database.finish(
        experiment_id,
        status="completed",
        used_runtime_seconds=1,
        result={
            "model_parameters": row["total_parameters"],
            "non_embedding_parameters": row["non_embedding_parameters"],
            "embedding_parameters": row["embedding_parameters"],
            "approximate_non_embedding_parameters": row[
                "approximate_non_embedding_parameters"
            ],
            "estimated_flops": row["actual_compute_flops"],
            "train_tokens": row["train_tokens"],
            "validation_losses": [4.5, 4.25],
            "final_validation_loss": 4.25,
            "runtime_seconds": 1.0,
            "compile_seconds": 0.5,
            "wall_clock_seconds": 1.5,
            "estimated_flops_per_second": row["actual_compute_flops"],
            "tokens_per_second": row["train_tokens"],
            "estimated_memory_bytes": 1024,
        },
    )

    export_path = tmp_path / "isoflops.json"
    exported = runner.invoke(
        local_cli, ["export-isoflops", "--output", str(export_path)]
    )
    assert exported.exit_code == 0, exported.output
    rows = json.loads(export_path.read_text())
    assert len(rows) == 1
    assert rows[0]["parameters"] == row["non_embedding_parameters"]
    assert rows[0]["total_parameters"] == row["total_parameters"]
    assert rows[0]["compute_budget"] == 5e8
    assert rows[0]["final_loss"] == 4.25
    assert rows[0]["actual_compute"] == row["actual_compute_flops"]
    assert rows[0]["chain_runtime_seconds"] == 1.0
    assert rows[0]["attempts"] == 1
    assert rows[0]["experiment_id"] == experiment_id
    refused = runner.invoke(
        local_cli, ["export-isoflops", "--output", str(export_path)]
    )
    assert refused.exit_code == 2
    replaced = runner.invoke(
        local_cli,
        ["export-isoflops", "--output", str(export_path), "--force"],
    )
    assert replaced.exit_code == 0, replaced.output


def test_make_lr_pilot_is_excluded_from_scaling_profiles(tmp_path: Path):
    settings = local_settings(tmp_path)
    manifest_path = write_dataset(settings)
    output_dir = tmp_path / "pilot"
    runner = CliRunner()
    generated = runner.invoke(
        local_cli,
        [
            "make-lr-pilot",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--peak-learning-rate",
            "1e-4",
            "--peak-learning-rate",
            "3e-4",
            "--hidden-size",
            "128",
            "--num-hidden-layers",
            "2",
            "--train-tokens",
            "8192",
            "--validation-tokens",
            "2048",
            "--max-runtime-seconds",
            "10",
            "--wandb-mode",
            "disabled",
        ],
    )
    assert generated.exit_code == 0, generated.output
    plan = json.loads((output_dir / "plan.json").read_text())
    assert plan["included_in_scaling_fit"] is False
    assert len(plan["configs"]) == 2
    for row in plan["configs"]:
        config = LocalExperimentConfig.model_validate_json(
            (output_dir / row["config"]).read_text()
        )
        assert config.target_compute_flops is None
        assert config.compute_parameter_basis is None
