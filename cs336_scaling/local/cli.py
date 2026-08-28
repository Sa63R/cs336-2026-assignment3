from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from rich.console import Console
from rich.table import Table

from cs336_scaling.local.client import LocalScalingClient
from cs336_scaling.local.database import LocalDatabase
from cs336_scaling.local.planning import (
    ParameterCounts,
    estimate_parameter_counts,
    round_tokens_for_compute,
    runtime_limit_for_compute,
)
from cs336_scaling.local.prepare_data import app as data_app
from cs336_scaling.local.schemas import (
    CheckpointConfig,
    DatasetManifest,
    LocalExperimentConfig,
    LocalTrainingConfig,
    WandbConfig,
    WandbMode,
)
from cs336_scaling.local.settings import LocalSettings
from cs336_scaling.training.model.config import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig, WarmupCosineDecay


app = typer.Typer(
    help="Manage local single-GPU CS336 scaling experiments.", no_args_is_help=True
)
console = Console()
app.add_typer(data_app, name="data")


def print_json(value: object) -> None:
    console.print_json(json.dumps(value, default=str))


def scaling_architecture(
    manifest: DatasetManifest,
    *,
    hidden_size: int,
    num_hidden_layers: int | None = None,
) -> BasicTransformerConfig:
    """Build one width-scaled architecture shared by pilot and sweep configs."""

    return BasicTransformerConfig(
        attention_bias=False,
        head_dim=64,
        hidden_size=hidden_size,
        intermediate_size=math.ceil((8 * hidden_size / 3) / 64) * 64,
        num_attention_heads=hidden_size // 64,
        num_hidden_layers=num_hidden_layers or max(2, hidden_size // 48),
        num_key_value_heads=hidden_size // 64,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        tie_word_embeddings=True,
        dtype="bfloat16",
        vocab_size=manifest.vocab_size,
    )


@app.command("init")
def initialize() -> None:
    """Create local directories and initialize the SQLite database."""

    settings = LocalSettings.from_env()
    LocalDatabase(settings).initialize()
    console.print(f"Initialized local scaling home: [bold]{settings.home}[/bold]")


@app.command()
def serve() -> None:
    """Run the loopback-only FastAPI service."""

    from cs336_scaling.local.api import main

    main()


@app.command()
def worker() -> None:
    """Run the single-GPU FIFO worker."""

    from cs336_scaling.local.worker import main

    main()


@app.command("make-config")
def make_config(
    dataset_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Path = typer.Option(Path("experiment.json"), dir_okay=False),
    optimizer_steps: int = typer.Option(64, min=1),
    n_evals: int = typer.Option(8, min=1),
    micro_batch_size: int = typer.Option(1, min=1),
    gradient_accumulation_steps: int = typer.Option(8, min=1),
    max_runtime_seconds: float = typer.Option(600, min=1),
) -> None:
    """Write a conservative RTX 4060 calibration configuration."""

    if optimizer_steps % n_evals != 0:
        raise typer.BadParameter("optimizer_steps must be divisible by n_evals")
    manifest_path = dataset_manifest.expanduser().resolve()
    manifest = DatasetManifest.load(manifest_path)
    if manifest.vocab_size <= 1_024:
        architecture = BasicTransformerConfig(
            attention_bias=False,
            head_dim=32,
            hidden_size=128,
            intermediate_size=384,
            num_attention_heads=4,
            num_hidden_layers=2,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000,
            tie_word_embeddings=True,
            dtype="bfloat16",
            vocab_size=manifest.vocab_size,
        )
    else:
        architecture = BasicTransformerConfig(
            attention_bias=False,
            head_dim=64,
            hidden_size=384,
            intermediate_size=1_024,
            num_attention_heads=6,
            num_hidden_layers=8,
            num_key_value_heads=6,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000,
            tie_word_embeddings=True,
            dtype="bfloat16",
            vocab_size=manifest.vocab_size,
        )
    effective_batch_size = micro_batch_size * gradient_accumulation_steps
    total_train_tokens = optimizer_steps * 512 * effective_batch_size
    validation_tokens = min(manifest.validation_tokens - 1, 2**18)
    validation_tokens -= validation_tokens % (512 * min(4, effective_batch_size))
    if validation_tokens <= 0:
        raise typer.BadParameter("dataset does not contain enough validation tokens")
    config = LocalExperimentConfig(
        training=LocalTrainingConfig(
            architecture_config=architecture,
            optimizer_config=AdamWConfig(),
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            val_batch_size=min(4, effective_batch_size),
            validation_tokens=validation_tokens,
            total_train_tokens=total_train_tokens,
            n_evals=n_evals,
            max_runtime_seconds=max_runtime_seconds,
            model_seed=0,
            data_seed=manifest.seed,
        ),
        dataset_manifest=manifest_path,
        checkpoint=CheckpointConfig(),
        wandb=WandbConfig(mode="online"),
        notes="RTX 4060 calibration run generated by local-scaling make-config",
    )
    output = output.expanduser().resolve()
    if output.exists():
        raise typer.BadParameter(f"refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(f"Wrote calibration config: [bold]{output}[/bold]")


@app.command("make-lr-pilot")
def make_lr_pilot(
    dataset_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Path = typer.Option(Path("configs/local_lr_pilot"), file_okay=False),
    peak_learning_rates: Annotated[
        list[float] | None,
        typer.Option("--peak-learning-rate", help="Repeat for each candidate LR."),
    ] = None,
    hidden_size: int = typer.Option(384, min=64),
    num_hidden_layers: int = typer.Option(8, min=1),
    train_tokens: int = typer.Option(262_144, min=8_192),
    validation_tokens: int = typer.Option(65_536, min=2_048),
    max_runtime_seconds: float = typer.Option(300, min=1),
    wandb_mode: str = typer.Option("offline"),
) -> None:
    """Create a short LR calibration that is excluded from scaling-law fits."""

    peak_learning_rates = peak_learning_rates or [1e-4, 3e-4, 6e-4]
    if any(
        not math.isfinite(learning_rate) or learning_rate <= 0
        for learning_rate in peak_learning_rates
    ):
        raise typer.BadParameter("peak learning rates must be positive and finite")
    if hidden_size % 64 != 0:
        raise typer.BadParameter("hidden-size must be a multiple of 64")
    if wandb_mode not in {"disabled", "offline", "online"}:
        raise typer.BadParameter("wandb-mode must be disabled, offline, or online")

    manifest_path = dataset_manifest.expanduser().resolve()
    manifest = DatasetManifest.load(manifest_path)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise typer.BadParameter(
            f"refusing to overwrite existing directory: {output_dir}"
        )
    if train_tokens + 1 > manifest.train_tokens:
        raise typer.BadParameter("dataset does not contain enough training tokens")
    if validation_tokens + 1 > manifest.validation_tokens:
        raise typer.BadParameter("dataset does not contain enough validation tokens")

    architecture = scaling_architecture(
        manifest,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
    )
    parameter_counts = estimate_parameter_counts(architecture)
    rows: list[dict[str, object]] = []
    payloads: list[tuple[str, str]] = []
    for learning_rate in peak_learning_rates:
        config = LocalExperimentConfig(
            training=LocalTrainingConfig(
                architecture_config=architecture,
                optimizer_config=AdamWConfig(
                    lr_scheduler=WarmupCosineDecay(peak_value=learning_rate)
                ),
                micro_batch_size=1,
                gradient_accumulation_steps=4,
                val_batch_size=4,
                validation_tokens=validation_tokens,
                total_train_tokens=train_tokens,
                n_evals=2,
                max_runtime_seconds=max_runtime_seconds,
                model_seed=0,
                data_seed=manifest.seed,
            ),
            dataset_manifest=manifest_path,
            checkpoint=CheckpointConfig(),
            wandb=WandbConfig(mode=cast(WandbMode, wandb_mode), group="lr-pilot-v1"),
            notes=(
                "LR pilot only; excluded from scaling-law fits; "
                f"peak_lr={learning_rate:.6e}"
            ),
        )
        filename = f"lr{learning_rate:.0e}_h{hidden_size}_l{num_hidden_layers}.json"
        payloads.append((filename, config.model_dump_json(indent=2) + "\n"))
        rows.append(
            {
                "config": filename,
                "peak_learning_rate": learning_rate,
                "non_embedding_parameters": parameter_counts.non_embedding,
                "total_parameters": parameter_counts.total,
                "train_tokens": train_tokens,
                "estimated_flops": (6 * parameter_counts.non_embedding * train_tokens),
                "max_runtime_seconds": max_runtime_seconds,
            }
        )

    output_dir.mkdir(parents=True)
    for filename, payload in payloads:
        (output_dir / filename).write_text(payload, encoding="utf-8")
    plan = {
        "format_version": 1,
        "method": "fixed-architecture peak learning-rate pilot",
        "included_in_scaling_fit": False,
        "selection_rule": (
            "choose the stable run with the lowest final validation loss; "
            "inspect the full loss trajectory for divergence"
        ),
        "maximum_reserved_seconds": len(rows) * max_runtime_seconds,
        "configs": rows,
    }
    (output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    console.print(
        f"Wrote {len(rows)} LR pilot configs to [bold]{output_dir}[/bold]; "
        f"maximum reserved {len(rows) * max_runtime_seconds / 60:.1f}min"
    )


@app.command("make-sweep")
def make_sweep(
    dataset_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Path = typer.Option(Path("configs/local_isoflops"), file_okay=False),
    compute_budgets: Annotated[
        list[float] | None,
        typer.Option("--compute-budget", help="Repeat for each target FLOPs profile."),
    ] = None,
    hidden_sizes: Annotated[
        list[int] | None,
        typer.Option("--hidden-size", help="Repeat for each model width."),
    ] = None,
    gradient_accumulation_steps: int = typer.Option(4, min=1),
    n_evals: int = typer.Option(4, min=1),
    validation_tokens: int = typer.Option(65_536, min=512),
    max_runtime_seconds: float | None = typer.Option(
        None,
        min=1,
        help="Fixed limit for every run; defaults to compute-aware limits.",
    ),
    reference_flops_per_second: float = typer.Option(2.4e11, min=1),
    reference_is_measured: bool = typer.Option(
        False,
        help="Mark the throughput reference as measured rather than provisional.",
    ),
    runtime_margin: float = typer.Option(1.3, min=1),
    minimum_runtime_seconds: float = typer.Option(300, min=1),
    peak_learning_rate: float = typer.Option(3e-4, min=1e-8),
    model_seed: int = typer.Option(0, min=0),
    wandb_mode: str = typer.Option("offline"),
) -> None:
    """Create a non-embedding-parameter RTX 4060 IsoFLOPs sweep."""

    compute_budgets = compute_budgets or [3e13, 6e13, 1.2e14, 2.4e14]
    hidden_sizes = hidden_sizes or [256, 320, 384, 512]
    if any(not math.isfinite(value) or value <= 0 for value in compute_budgets):
        raise typer.BadParameter("compute budgets must be positive and finite")
    if any(hidden <= 0 or hidden % 64 != 0 for hidden in hidden_sizes):
        raise typer.BadParameter("hidden sizes must be positive multiples of 64")
    if wandb_mode not in {"disabled", "offline", "online"}:
        raise typer.BadParameter("wandb-mode must be disabled, offline, or online")

    manifest_path = dataset_manifest.expanduser().resolve()
    manifest = DatasetManifest.load(manifest_path)
    val_batch_size = 4
    validation_quantum = 512 * val_batch_size
    if validation_tokens % validation_quantum != 0:
        raise typer.BadParameter(
            f"validation-tokens must be divisible by {validation_quantum:,}"
        )
    if validation_tokens + 1 > manifest.validation_tokens:
        raise typer.BadParameter("dataset does not contain enough validation tokens")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise typer.BadParameter(
            f"refusing to overwrite existing directory: {output_dir}"
        )

    model_specs: list[tuple[BasicTransformerConfig, ParameterCounts]] = []
    for hidden_size in hidden_sizes:
        architecture = scaling_architecture(manifest, hidden_size=hidden_size)
        model_specs.append((architecture, estimate_parameter_counts(architecture)))

    rows: list[dict[str, object]] = []
    config_payloads: list[tuple[str, str]] = []
    reserved = 0.0
    total_actual_compute = 0.0
    token_quantum = 512 * gradient_accumulation_steps * n_evals
    for compute_budget in sorted(compute_budgets):
        computed_runtime_limit = runtime_limit_for_compute(
            compute_budget,
            reference_flops_per_second=reference_flops_per_second,
            margin=runtime_margin,
            minimum_seconds=minimum_runtime_seconds,
        )
        runtime_limit = max_runtime_seconds or computed_runtime_limit
        for architecture, parameter_counts in model_specs:
            non_embedding_parameters = parameter_counts.non_embedding
            try:
                train_tokens = round_tokens_for_compute(
                    compute_budget,
                    parameters=non_embedding_parameters,
                    token_quantum=token_quantum,
                    maximum_tokens=manifest.train_tokens - 1,
                )
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            actual_compute = 6 * non_embedding_parameters * train_tokens
            reserved += runtime_limit
            total_actual_compute += actual_compute
            config = LocalExperimentConfig(
                training=LocalTrainingConfig(
                    architecture_config=architecture,
                    optimizer_config=AdamWConfig(
                        lr_scheduler=WarmupCosineDecay(peak_value=peak_learning_rate)
                    ),
                    micro_batch_size=1,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    val_batch_size=val_batch_size,
                    validation_tokens=validation_tokens,
                    total_train_tokens=train_tokens,
                    n_evals=n_evals,
                    max_runtime_seconds=runtime_limit,
                    model_seed=model_seed,
                    data_seed=manifest.seed,
                ),
                dataset_manifest=manifest_path,
                target_compute_flops=compute_budget,
                compute_parameter_basis="non_embedding",
                checkpoint=CheckpointConfig(),
                wandb=WandbConfig(
                    mode=cast(WandbMode, wandb_mode),
                    group="isoflops-nonembedding-v1",
                ),
                notes=(
                    f"RTX 4060 IsoFLOPs profile C={compute_budget:.6e}; "
                    f"exact non-embedding N={non_embedding_parameters}; "
                    f"model_seed={model_seed}"
                ),
            )
            filename = (
                f"c{compute_budget:.0e}_h{architecture.hidden_size}"
                f"_l{architecture.num_hidden_layers}.json"
            ).replace("+", "")
            config_payloads.append((filename, config.model_dump_json(indent=2) + "\n"))
            rows.append(
                {
                    "config": filename,
                    "target_compute_flops": compute_budget,
                    "actual_compute_flops": actual_compute,
                    "relative_compute_error": actual_compute / compute_budget - 1,
                    "parameter_basis": "non_embedding",
                    "non_embedding_parameters": non_embedding_parameters,
                    "approximate_non_embedding_parameters": (
                        parameter_counts.approximate_non_embedding
                    ),
                    "total_parameters": parameter_counts.total,
                    "embedding_parameters": parameter_counts.embedding,
                    "hidden_size": architecture.hidden_size,
                    "intermediate_size": architecture.intermediate_size,
                    "num_hidden_layers": architecture.num_hidden_layers,
                    "train_tokens": train_tokens,
                    "max_runtime_seconds": runtime_limit,
                }
            )

    filenames = [filename for filename, _ in config_payloads]
    if len(filenames) != len(set(filenames)):
        raise typer.BadParameter(
            "compute budgets produce duplicate filenames; use more separated values"
        )
    output_dir.mkdir(parents=True)
    for filename, payload in config_payloads:
        (output_dir / filename).write_text(payload, encoding="utf-8")
    expected_training_seconds = total_actual_compute / reference_flops_per_second
    plan = {
        "format_version": 2,
        "method": "IsoFLOPs with C = 6*N_non_embedding*D",
        "parameter_basis": "non_embedding",
        "dataset_id": manifest.dataset_id,
        "reference_flops_per_second": reference_flops_per_second,
        "runtime_estimate_status": (
            "measured_reference"
            if reference_is_measured
            else "provisional_until_lr_pilot_completes"
        ),
        "runtime_reference_note": (
            "reference supplied from a completed local calibration"
            if reference_is_measured
            else "regenerate with measured non-embedding FLOPs/s before submission"
        ),
        "runtime_margin": runtime_margin,
        "peak_learning_rate": peak_learning_rate,
        "model_seed": model_seed,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": gradient_accumulation_steps,
        "n_evals": n_evals,
        "validation_tokens": validation_tokens,
        "expected_training_seconds": expected_training_seconds,
        "maximum_reserved_seconds": reserved,
        "submission_order": "compute ascending, then model size ascending",
        "configs": rows,
    }
    (output_dir / "plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    console.print(
        f"Wrote {len(rows)} configs to [bold]{output_dir}[/bold]; "
        f"expected training {expected_training_seconds / 3600:.2f}h, "
        f"maximum reserved {reserved / 3600:.2f}h"
    )


@app.command("submit-dir")
def submit_directory(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Submit every experiment JSON in a generated sweep directory."""

    directory = directory.resolve()
    plan_path = directory / "plan.json"
    configs: list[Path]
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_rows = plan.get("configs") if isinstance(plan, dict) else None
        if isinstance(plan_rows, list):
            configs = []
            for row in plan_rows:
                if not isinstance(row, dict) or not isinstance(row.get("config"), str):
                    raise typer.BadParameter("plan.json contains an invalid config row")
                config_path = (directory / row["config"]).resolve()
                if config_path.parent != directory:
                    raise typer.BadParameter("plan.json config escapes its directory")
                configs.append(config_path)
        else:
            configs = sorted(
                path for path in directory.glob("*.json") if path.name != "plan.json"
            )
    else:
        configs = sorted(directory.glob("*.json"))
    if not configs:
        raise typer.BadParameter(f"no experiment JSON files found in {directory}")
    client = LocalScalingClient()
    for config_path in configs:
        response = client.submit(config_path.resolve())
        console.print(f"{config_path.name}: experiment {response['experiment_id']}")


@app.command("export-isoflops")
def export_isoflops(
    output: Path = typer.Option(
        Path(".local_scaling/analysis/isoflops_runs.json"), dir_okay=False
    ),
    force: bool = typer.Option(False, help="Replace an earlier derived export."),
) -> None:
    """Export completed, profiled runs for ``chinchilla_isoflops.py``."""

    settings = LocalSettings.from_env()
    database = LocalDatabase(settings)
    database.initialize()
    all_experiments = database.list_experiments(limit=1_000)
    runtime_by_config_hash: dict[str, float] = {}
    attempts_by_config_hash: dict[str, int] = {}
    for item in all_experiments:
        runtime_by_config_hash[item.config_hash] = (
            runtime_by_config_hash.get(item.config_hash, 0.0)
            + item.used_runtime_seconds
        )
        attempts_by_config_hash[item.config_hash] = (
            attempts_by_config_hash.get(item.config_hash, 0) + 1
        )
    rows: list[dict[str, object]] = []
    for experiment in reversed(
        [item for item in all_experiments if item.status == "completed"]
    ):
        result = experiment.result
        target_compute = experiment.config.target_compute_flops
        if (
            result is None
            or target_compute is None
            or experiment.config.compute_parameter_basis != "non_embedding"
        ):
            continue
        try:
            raw_parameters = result["non_embedding_parameters"]
            raw_final_loss = result["final_validation_loss"]
            raw_actual_compute = result["estimated_flops"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"experiment {experiment.experiment_id} has an invalid result"
            ) from exc
        if (
            isinstance(raw_parameters, bool)
            or not isinstance(raw_parameters, (int, float))
            or not float(raw_parameters).is_integer()
            or isinstance(raw_final_loss, bool)
            or not isinstance(raw_final_loss, (int, float))
            or isinstance(raw_actual_compute, bool)
            or not isinstance(raw_actual_compute, (int, float))
        ):
            raise RuntimeError(
                f"experiment {experiment.experiment_id} has non-numeric results"
            )
        typed_result = cast(dict[str, Any], result)
        rows.append(
            {
                "parameters": int(raw_parameters),
                "total_parameters": int(typed_result["model_parameters"]),
                "embedding_parameters": int(typed_result["embedding_parameters"]),
                "approximate_non_embedding_parameters": int(
                    typed_result["approximate_non_embedding_parameters"]
                ),
                "compute_budget": target_compute,
                "final_loss": float(raw_final_loss),
                "actual_compute": float(raw_actual_compute),
                "target_compute_relative_error": (
                    float(raw_actual_compute) / target_compute - 1.0
                ),
                "train_tokens": int(typed_result["train_tokens"]),
                "validation_losses": [
                    float(loss) for loss in typed_result["validation_losses"]
                ],
                "runtime_seconds": float(typed_result["runtime_seconds"]),
                "chain_runtime_seconds": runtime_by_config_hash[experiment.config_hash],
                "attempts": attempts_by_config_hash[experiment.config_hash],
                "compile_seconds": float(typed_result["compile_seconds"]),
                "wall_clock_seconds": float(typed_result["wall_clock_seconds"]),
                "estimated_flops_per_second": float(
                    typed_result["estimated_flops_per_second"]
                ),
                "tokens_per_second": float(typed_result["tokens_per_second"]),
                "estimated_memory_bytes": int(typed_result["estimated_memory_bytes"]),
                "hidden_size": experiment.config.training.architecture_config.hidden_size,
                "intermediate_size": (
                    experiment.config.training.architecture_config.intermediate_size
                ),
                "num_hidden_layers": (
                    experiment.config.training.architecture_config.num_hidden_layers
                ),
                "num_attention_heads": (
                    experiment.config.training.architecture_config.num_attention_heads
                ),
                "peak_learning_rate": (
                    experiment.config.training.optimizer_config.lr_scheduler.peak_value
                ),
                "model_seed": experiment.config.training.model_seed,
                "data_seed": experiment.config.training.data_seed,
                "config_hash": experiment.config_hash,
                "experiment_id": experiment.experiment_id,
            }
        )
    if not rows:
        raise typer.BadParameter(
            "no completed runs with target_compute_flops are available"
        )
    output = output.expanduser().resolve()
    if output.exists() and not force:
        raise typer.BadParameter(f"refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    console.print(f"Exported {len(rows)} runs to [bold]{output}[/bold]")


@app.command()
def submit(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Submit an immutable JSON experiment configuration."""

    print_json(LocalScalingClient().submit(config.resolve()))


@app.command("list")
def list_experiments(limit: int = typer.Option(100, min=1, max=1_000)) -> None:
    """List recent experiments."""

    experiments = LocalScalingClient().list_experiments(limit=limit)
    table = Table("ID", "Status", "Budget", "Attempt", "Runtime", "Created")
    for experiment in experiments:
        table.add_row(
            str(experiment["experiment_id"]),
            str(experiment["status"]),
            str(experiment["config"]["budget_group"]),
            str(experiment["attempt"]),
            f"{float(experiment['used_runtime_seconds']):.1f}s",
            str(experiment["created_at"]),
        )
    console.print(table)


@app.command()
def show(experiment_id: int) -> None:
    """Show one experiment and its current result."""

    print_json(LocalScalingClient().get(experiment_id))


@app.command()
def metrics(experiment_id: int) -> None:
    """Show persisted metrics for one experiment."""

    print_json(LocalScalingClient().metrics(experiment_id))


@app.command()
def cancel(experiment_id: int) -> None:
    """Cancel a queued or running experiment."""

    print_json(LocalScalingClient().cancel(experiment_id))


@app.command()
def retry(experiment_id: int, resume: bool = True) -> None:
    """Queue a new attempt, optionally resuming the latest checkpoint."""

    print_json(LocalScalingClient().retry(experiment_id, resume=resume))


@app.command()
def budget() -> None:
    """Show scaling and final-run GPU-time budgets."""

    usage = LocalScalingClient().budget()
    table = Table("Budget", "Used", "Reserved", "Remaining", "Total")
    for item in usage:
        table.add_row(
            str(item["budget_group"]),
            f"{float(item['used_seconds']) / 3600:.3f}h",
            f"{float(item['reserved_seconds']) / 3600:.3f}h",
            f"{float(item['remaining_seconds']) / 3600:.3f}h",
            f"{float(item['total_seconds']) / 3600:.3f}h",
        )
    console.print(table)


if __name__ == "__main__":
    app()
