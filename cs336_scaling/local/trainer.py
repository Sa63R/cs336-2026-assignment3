from __future__ import annotations

import os

# These defaults must be established before Equinox/JAX is imported by the local
# loop or checkpoint modules. Users can override either setting explicitly.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/cs336-local-jax-cache")

import argparse
import signal
import subprocess
import time
import traceback
from typing import Any, cast

from cs336_scaling.local.database import LocalDatabase, utc_now
from cs336_scaling.local.schemas import MetricRecord
from cs336_scaling.local.settings import LocalSettings


class StopController:
    def __init__(self):
        self.requested = False

    def request(self, _signum: int, _frame: object) -> None:
        self.requested = True
        print("Stop requested; waiting for the current compiled chunk to finish")


def gpu_memory_limit_bytes() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        first_device_mib = float(result.stdout.splitlines()[0].strip())
        return int(first_device_mib * 1024**2)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def compile_memory_estimate(compiled) -> int:
    analysis = compiled.memory_analysis()
    arguments = analysis.argument_size_in_bytes or 0
    outputs = analysis.output_size_in_bytes or 0
    temporary = analysis.temp_size_in_bytes or 0
    aliases = analysis.alias_size_in_bytes or 0
    return int(max(arguments, outputs) + temporary - aliases)


def run_experiment(experiment_id: int, settings: LocalSettings) -> int:
    import jax
    from equinox import nn
    from jax.sharding import AxisType, NamedSharding
    from jax.sharding import PartitionSpec as P

    from cs336_scaling.local.checkpoint import (
        CheckpointManager,
        find_latest_checkpoint,
        load_checkpoint,
    )
    from cs336_scaling.local.data import LocalTokenDataset
    from cs336_scaling.local.loop import train_and_evaluate_chunk
    from cs336_scaling.local.recording import RunRecorder
    from cs336_scaling.local.wandb_sink import SafeWandbSink
    from cs336_scaling.training.model.basic_model import BasicCausalLM
    from cs336_scaling.training.model.jax_utils import count_params

    database = LocalDatabase(settings)
    database.initialize()
    experiment = database.get_experiment(experiment_id)
    if experiment.status != "running":
        raise RuntimeError(
            f"trainer requires a running experiment, got {experiment.status!r}"
        )
    config = experiment.config
    training = config.training
    output_dir = experiment.output_dir
    recorder = RunRecorder(
        experiment_id=experiment_id,
        output_dir=output_dir,
        database=database,
    )
    recorder.initialize(
        config,
        config_hash=experiment.config_hash,
        dataset_id=experiment.dataset_id,
    )
    wandb_sink = SafeWandbSink(
        config=config,
        experiment_id=experiment_id,
        attempt=experiment.attempt,
        output_dir=output_dir,
    )
    stop = StopController()
    signal.signal(signal.SIGINT, stop.request)
    signal.signal(signal.SIGTERM, stop.request)
    process_started_at = time.perf_counter()
    training_started_at: float | None = None

    checkpoint_manager = CheckpointManager(
        output_dir / "checkpoints",
        experiment_id=experiment_id,
        config_hash=experiment.config_hash,
        keep_last=config.checkpoint.keep_last,
        keep_best=config.checkpoint.keep_best,
    )

    model = None
    state = None
    opt_state = None
    last_validation_loss = float("inf")
    optimizer_step = 0
    completed_evals = 0
    estimated_memory = 0
    val_losses: list[float] = []

    def elapsed_training() -> float:
        if training_started_at is None:
            return 0.0
        return time.perf_counter() - training_started_at

    def elapsed_wall_clock() -> float:
        return time.perf_counter() - process_started_at

    try:
        dataset = LocalTokenDataset(config.dataset_manifest)
        dataset.verify_checksums()
        dataset.validate_for(training)

        if jax.default_backend() != "gpu":
            raise RuntimeError(
                f"local training requires the CUDA JAX backend, got {jax.default_backend()!r}"
            )
        if jax.device_count() != 1:
            raise RuntimeError(
                "the local worker requires exactly one visible GPU; set "
                "CUDA_VISIBLE_DEVICES to one device"
            )
        mesh = jax.make_mesh((1,), ("fsdp",), axis_types=(AxisType.Explicit,))
        jax.set_mesh(mesh)

        @jax.jit
        def make_model() -> tuple[BasicCausalLM, nn.State]:
            created_model, created_state = cast(
                tuple[BasicCausalLM, nn.State],
                nn.make_with_state(BasicCausalLM)(
                    training.architecture_config,
                    key=jax.random.PRNGKey(training.model_seed),
                ),
            )
            return created_model.apply_sharding(mesh), created_state

        model, state = make_model()
        optimizer = training.optimizer_config.build(
            training.optimizer_training_config()
        )
        opt_state = optimizer.init(model)
        model_params = count_params(model)

        if experiment.resume_from_experiment_id is not None:
            source = database.get_experiment(experiment.resume_from_experiment_id)
            latest = find_latest_checkpoint(source.output_dir)
            if latest is None:
                raise RuntimeError(
                    f"experiment {source.experiment_id} has no checkpoint to resume"
                )
            checkpoint_path, metadata = latest
            if metadata.config_hash != experiment.config_hash:
                raise RuntimeError("checkpoint configuration hash does not match")
            # The serialized Equinox state contains a materialized RoPE cache.
            # Recreate that structure before deserializing so every leaf has the
            # same path and shape as the checkpoint payload.
            state = model.layers.self_attn.rotary.update_cache(
                state, seq_len=training.seq_len
            )
            payload, metadata = load_checkpoint(
                checkpoint_path, (model, state, opt_state)
            )
            model, state, opt_state = cast(tuple[Any, Any, Any], payload)
            optimizer_step = metadata.optimizer_step
            completed_evals = metadata.eval_index
            last_validation_loss = metadata.validation_loss
            val_losses = [
                metric.values["validation_loss"]
                for metric in database.list_metrics(source.experiment_id)
                if metric.phase == "validation" and "validation_loss" in metric.values
            ]
            print(
                f"Resumed experiment {source.experiment_id} from optimizer step "
                f"{optimizer_step}"
            )

        starting_optimizer_step = optimizer_step
        validation_data = dataset.validation_batches(training).to_jax()
        validation_data = jax.device_put(
            validation_data,
            NamedSharding(mesh, P(None, "fsdp", None)),
        )

        if completed_evals >= training.n_evals:
            raise RuntimeError(
                "checkpoint already completed all configured evaluations"
            )
        first_chunk = dataset.train_chunk(
            optimizer_step=optimizer_step,
            optimizer_steps=training.optimizer_steps_per_eval,
            config=training,
        ).to_jax()
        first_chunk = jax.device_put(
            first_chunk,
            NamedSharding(mesh, P(None, None, "fsdp", None)),
        )
        compile_started = time.perf_counter()
        compiled_chunk = cast(Any, train_and_evaluate_chunk)
        compiled = compiled_chunk.lower(
            model,
            state,
            first_chunk,
            validation_data,
            training,
            opt_state,
        ).compile()
        compile_seconds = time.perf_counter() - compile_started
        estimated_memory = compile_memory_estimate(compiled)
        memory_limit = gpu_memory_limit_bytes()
        if memory_limit is not None and estimated_memory > memory_limit * 0.92:
            raise MemoryError(
                f"compiled program estimates {estimated_memory / 1024**3:.2f} GiB, "
                f"above the 92% safety limit of {memory_limit / 1024**3:.2f} GiB"
            )
        system_metric = MetricRecord(
            experiment_id=experiment_id,
            phase="system",
            optimizer_step=optimizer_step,
            tokens_seen=optimizer_step * training.tokens_per_optimizer_step,
            values={
                "compile_seconds": compile_seconds,
                "estimated_memory_bytes": float(estimated_memory),
                "model_parameters": float(model_params),
            },
            created_at=utc_now(),
        )
        recorder.metric(system_metric)
        wandb_sink.log(system_metric)
        print(
            f"Compiled {model_params:,}-parameter model in {compile_seconds:.1f}s; "
            f"estimated device memory {estimated_memory / 1024**3:.2f} GiB"
        )

        previous_chunk_seconds: float | None = None
        training_started_at = time.perf_counter()
        for eval_index in range(completed_evals, training.n_evals):
            if stop.requested or database.cancel_requested(experiment_id):
                raise InterruptedError("experiment cancellation requested")
            remaining = training.max_runtime_seconds - elapsed_training()
            if remaining <= 0 or (
                previous_chunk_seconds is not None
                and previous_chunk_seconds * 1.1 > remaining
            ):
                raise TimeoutError(
                    "insufficient runtime budget for another training chunk"
                )

            if eval_index == completed_evals:
                train_data = first_chunk
            else:
                train_data = dataset.train_chunk(
                    optimizer_step=optimizer_step,
                    optimizer_steps=training.optimizer_steps_per_eval,
                    config=training,
                ).to_jax()
                train_data = jax.device_put(
                    train_data,
                    NamedSharding(mesh, P(None, None, "fsdp", None)),
                )

            chunk_started = time.perf_counter()
            result = compiled_chunk(
                model,
                state,
                train_data,
                validation_data,
                training,
                opt_state,
            )
            jax.block_until_ready(result)
            previous_chunk_seconds = time.perf_counter() - chunk_started
            model = result.model
            state = result.state
            opt_state = result.opt_state
            optimizer_step += training.optimizer_steps_per_eval
            completed_evals = eval_index + 1
            tokens_seen = optimizer_step * training.tokens_per_optimizer_step
            last_validation_loss = float(result.val_loss.item())
            train_loss = float(result.train_losses.mean().item())
            val_losses.append(last_validation_loss)
            throughput = (
                training.optimizer_steps_per_eval
                * training.tokens_per_optimizer_step
                / previous_chunk_seconds
            )
            metric = MetricRecord(
                experiment_id=experiment_id,
                phase="validation",
                optimizer_step=optimizer_step,
                tokens_seen=tokens_seen,
                values={
                    "train_loss": train_loss,
                    "validation_loss": last_validation_loss,
                    "chunk_seconds": previous_chunk_seconds,
                    "tokens_per_second": throughput,
                },
                created_at=utc_now(),
            )
            recorder.metric(metric)
            wandb_sink.log(metric)
            database.heartbeat(experiment_id, os.getpid())
            print(
                f"eval {completed_evals}/{training.n_evals}: step={optimizer_step}, "
                f"train_loss={train_loss:.5f}, val_loss={last_validation_loss:.5f}, "
                f"throughput={throughput:.0f} tokens/s"
            )

            should_checkpoint = (
                completed_evals % config.checkpoint.every_n_evals == 0
                or completed_evals == training.n_evals
                or stop.requested
                or database.cancel_requested(experiment_id)
            )
            if should_checkpoint:
                checkpoint_manager.save(
                    (model, state, opt_state),
                    optimizer_step=optimizer_step,
                    eval_index=completed_evals,
                    tokens_seen=tokens_seen,
                    validation_loss=last_validation_loss,
                    elapsed_training_seconds=elapsed_training(),
                )
            if stop.requested or database.cancel_requested(experiment_id):
                raise InterruptedError("experiment cancellation requested")

        runtime = elapsed_training()
        wall_clock = elapsed_wall_clock()
        final_tokens = optimizer_step * training.tokens_per_optimizer_step
        attempt_tokens = (
            optimizer_step - starting_optimizer_step
        ) * training.tokens_per_optimizer_step
        result_value: dict[str, object] = {
            "status": "completed",
            "model_parameters": model_params,
            "optimizer_steps": optimizer_step,
            "train_tokens": final_tokens,
            "estimated_flops": 6 * model_params * final_tokens,
            "attempt_train_tokens": attempt_tokens,
            "attempt_estimated_flops": 6 * model_params * attempt_tokens,
            "runtime_seconds": runtime,
            "wall_clock_seconds": wall_clock,
            "compile_seconds": compile_seconds,
            "tokens_per_second": attempt_tokens / runtime,
            "estimated_memory_bytes": estimated_memory,
            "validation_losses": val_losses,
            "final_validation_loss": last_validation_loss,
            "resumed_from_experiment_id": experiment.resume_from_experiment_id,
        }
        recorder.result(result_value)
        database.finish(
            experiment_id,
            status="completed",
            used_runtime_seconds=runtime,
            result=result_value,
        )
        wandb_sink.finish(exit_code=0)
        return 0
    except InterruptedError as exc:
        runtime = elapsed_training()
        if (
            model is not None
            and state is not None
            and opt_state is not None
            and completed_evals
        ):
            checkpoint_manager.save(
                (model, state, opt_state),
                optimizer_step=optimizer_step,
                eval_index=completed_evals,
                tokens_seen=optimizer_step * training.tokens_per_optimizer_step,
                validation_loss=last_validation_loss,
                elapsed_training_seconds=runtime,
            )
        database.finish(
            experiment_id,
            status="cancelled",
            used_runtime_seconds=runtime,
            error_message=str(exc),
        )
        wandb_sink.finish(exit_code=130)
        return 130
    except Exception as exc:
        runtime = elapsed_training()
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        try:
            database.finish(
                experiment_id,
                status="failed",
                used_runtime_seconds=runtime,
                error_message=error[:4_000],
            )
        except Exception:
            traceback.print_exc()
        wandb_sink.finish(exit_code=1)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one claimed local experiment")
    parser.add_argument("--experiment-id", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run_experiment(args.experiment_id, LocalSettings.from_env()))


if __name__ == "__main__":
    main()
