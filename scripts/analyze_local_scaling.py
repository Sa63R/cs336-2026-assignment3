# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10,<4",
#     "numpy>=2.2,<3",
# ]
# ///
"""Analyze completed local RTX 4060 IsoFLOPs experiments.

The input is produced by ``local-scaling export-isoflops``.  Within each fixed
compute profile, final validation loss is fit as a quadratic in log model size.
The clipped vertex gives a continuous N_opt estimate and C=6ND gives D_opt.
Power laws are then fit to the per-profile optima in log-log space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / ".local_scaling/analysis/isoflops_runs.json"
DEFAULT_RUNS_DIR = PROJECT_ROOT / ".local_scaling/runs"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/local_scaling_4060"


@dataclass(frozen=True)
class Run:
    experiment_id: int
    parameters: int
    total_parameters: int
    compute_budget: float
    actual_compute: float
    train_tokens: int
    final_loss: float
    runtime_seconds: float
    chain_runtime_seconds: float
    compile_seconds: float
    wall_clock_seconds: float
    flops_per_second: float
    tokens_per_second: float
    estimated_memory_bytes: int
    hidden_size: int
    num_hidden_layers: int
    peak_learning_rate: float
    model_seed: int


@dataclass(frozen=True)
class ProfileFit:
    compute_budget: float
    optimal_parameters: float
    optimal_dataset_tokens: float
    predicted_optimal_loss: float
    observed_best_parameters: int
    observed_best_loss: float
    quadratic_a: float
    quadratic_b: float
    quadratic_c: float
    r_squared: float
    rmse: float
    optimum_at_boundary: bool


@dataclass(frozen=True)
class PowerLaw:
    coefficient: float
    exponent: float
    r_squared: float

    def predict(self, value: float | np.ndarray) -> float | np.ndarray:
        return self.coefficient * np.power(value, self.exponent)


@dataclass(frozen=True)
class LossLaw:
    irreducible_loss: float
    coefficient: float
    exponent: float
    r_squared: float

    def predict(self, value: float | np.ndarray) -> float | np.ndarray:
        return self.irreducible_loss + self.coefficient * np.power(
            value, -self.exponent
        )


def finite_number(row: dict[str, Any], field: str, index: int) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"row {index}: {field} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"row {index}: {field} must be finite")
    return value


def load_runs(path: Path) -> list[Run]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("input must be a non-empty JSON array")
    runs: list[Run] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"row {index}: expected an object")
        typed_row = cast(dict[str, Any], row)
        run = Run(
            experiment_id=int(finite_number(typed_row, "experiment_id", index)),
            parameters=int(finite_number(typed_row, "parameters", index)),
            total_parameters=int(finite_number(typed_row, "total_parameters", index)),
            compute_budget=finite_number(typed_row, "compute_budget", index),
            actual_compute=finite_number(typed_row, "actual_compute", index),
            train_tokens=int(finite_number(typed_row, "train_tokens", index)),
            final_loss=finite_number(typed_row, "final_loss", index),
            runtime_seconds=finite_number(typed_row, "runtime_seconds", index),
            chain_runtime_seconds=finite_number(
                typed_row, "chain_runtime_seconds", index
            ),
            compile_seconds=finite_number(typed_row, "compile_seconds", index),
            wall_clock_seconds=finite_number(typed_row, "wall_clock_seconds", index),
            flops_per_second=finite_number(
                typed_row, "estimated_flops_per_second", index
            ),
            tokens_per_second=finite_number(typed_row, "tokens_per_second", index),
            estimated_memory_bytes=int(
                finite_number(typed_row, "estimated_memory_bytes", index)
            ),
            hidden_size=int(finite_number(typed_row, "hidden_size", index)),
            num_hidden_layers=int(finite_number(typed_row, "num_hidden_layers", index)),
            peak_learning_rate=finite_number(typed_row, "peak_learning_rate", index),
            model_seed=int(finite_number(typed_row, "model_seed", index)),
        )
        if (
            min(
                run.parameters,
                run.total_parameters,
                run.compute_budget,
                run.actual_compute,
                run.train_tokens,
                run.final_loss,
                run.runtime_seconds,
                run.chain_runtime_seconds,
                run.flops_per_second,
                run.tokens_per_second,
                run.estimated_memory_bytes,
            )
            <= 0
        ):
            raise ValueError(f"row {index}: positive values required")
        runs.append(run)
    return runs


def summarize_repeats(runs: Sequence[Run]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int], list[Run]] = {}
    for run in runs:
        grouped.setdefault((run.compute_budget, run.parameters), []).append(run)
    rows: list[dict[str, Any]] = []
    for (compute_budget, parameters), repeats in sorted(grouped.items()):
        if len(repeats) < 2:
            continue
        losses = np.asarray([run.final_loss for run in repeats])
        rows.append(
            {
                "compute_budget": compute_budget,
                "parameters": parameters,
                "count": len(repeats),
                "model_seeds": [run.model_seed for run in repeats],
                "experiment_ids": [run.experiment_id for run in repeats],
                "final_losses": [run.final_loss for run in repeats],
                "mean_final_loss": float(np.mean(losses)),
                "sample_standard_deviation": float(np.std(losses, ddof=1)),
                "loss_range": float(np.ptp(losses)),
            }
        )
    return rows


def coefficient_of_determination(observed: np.ndarray, fitted: np.ndarray) -> float:
    residual = float(np.sum(np.square(observed - fitted)))
    total = float(np.sum(np.square(observed - np.mean(observed))))
    return 1.0 - residual / total if total > 0 else 1.0


def fit_profiles(runs: Sequence[Run]) -> list[ProfileFit]:
    budgets = sorted({run.compute_budget for run in runs})
    if len(budgets) < 2:
        raise ValueError("at least two compute profiles are required")
    profiles: list[ProfileFit] = []
    for budget in budgets:
        profile_runs = sorted(
            (run for run in runs if run.compute_budget == budget),
            key=lambda run: run.parameters,
        )
        losses_by_parameters: dict[int, list[float]] = {}
        for run in profile_runs:
            losses_by_parameters.setdefault(run.parameters, []).append(run.final_loss)
        if len(losses_by_parameters) < 3:
            raise ValueError(f"C={budget:.4e}: at least three model sizes required")
        parameter_values = sorted(losses_by_parameters)
        log_n = np.log(parameter_values)
        losses = np.asarray(
            [np.mean(losses_by_parameters[value]) for value in parameter_values],
            dtype=np.float64,
        )
        a, b, c = np.polyfit(log_n, losses, deg=2)
        fitted = np.polyval((a, b, c), log_n)
        lower, upper = float(np.min(log_n)), float(np.max(log_n))
        if a > 0:
            raw_optimum = float(-b / (2.0 * a))
        else:
            raw_optimum = float(log_n[int(np.argmin(losses))])
        clipped_optimum = float(np.clip(raw_optimum, lower, upper))
        optimum_at_boundary = not lower < clipped_optimum < upper
        optimal_parameters = float(np.exp(clipped_optimum))
        observed_best_index = int(np.argmin(losses))
        profiles.append(
            ProfileFit(
                compute_budget=budget,
                optimal_parameters=optimal_parameters,
                optimal_dataset_tokens=budget / (6.0 * optimal_parameters),
                predicted_optimal_loss=float(np.polyval((a, b, c), clipped_optimum)),
                observed_best_parameters=parameter_values[observed_best_index],
                observed_best_loss=float(losses[observed_best_index]),
                quadratic_a=float(a),
                quadratic_b=float(b),
                quadratic_c=float(c),
                r_squared=coefficient_of_determination(losses, fitted),
                rmse=float(np.sqrt(np.mean(np.square(losses - fitted)))),
                optimum_at_boundary=optimum_at_boundary,
            )
        )
    return profiles


def fit_power_law(x_values: Sequence[float], y_values: Sequence[float]) -> PowerLaw:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    log_x, log_y = np.log(x), np.log(y)
    intercept, exponent = np.linalg.lstsq(
        np.column_stack((np.ones_like(log_x), log_x)), log_y, rcond=None
    )[0]
    fitted = intercept + exponent * log_x
    return PowerLaw(
        coefficient=float(np.exp(intercept)),
        exponent=float(exponent),
        r_squared=coefficient_of_determination(log_y, fitted),
    )


def fit_loss_law(profiles: Sequence[ProfileFit]) -> LossLaw:
    """Fit L(C)=L_inf+K*C^-gamma by profiling over L_inf."""

    compute = np.asarray([fit.compute_budget for fit in profiles], dtype=np.float64)
    losses = np.asarray(
        [fit.predicted_optimal_loss for fit in profiles], dtype=np.float64
    )
    upper = float(losses.min()) - 1e-6
    lower = max(0.0, upper - max(2.0, float(np.ptp(losses)) * 20.0))
    best: tuple[float, float, float, float] | None = None
    log_compute = np.log(compute)
    for irreducible in np.linspace(lower, upper, 20_000, endpoint=True):
        log_excess = np.log(losses - irreducible)
        intercept, slope = np.linalg.lstsq(
            np.column_stack((np.ones_like(log_compute), log_compute)),
            log_excess,
            rcond=None,
        )[0]
        coefficient = float(np.exp(intercept))
        exponent = float(-slope)
        if exponent <= 0:
            continue
        predicted = irreducible + coefficient * np.power(compute, -exponent)
        squared_error = float(np.sum(np.square(losses - predicted)))
        candidate = (squared_error, irreducible, coefficient, exponent)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("could not fit a decreasing optimal-loss law")
    _, irreducible, coefficient, exponent = best
    predicted = irreducible + coefficient * np.power(compute, -exponent)
    return LossLaw(
        irreducible_loss=irreducible,
        coefficient=coefficient,
        exponent=exponent,
        r_squared=coefficient_of_determination(losses, predicted),
    )


def leave_one_budget_out(
    profiles: Sequence[ProfileFit], target_compute: float
) -> list[dict[str, float]]:
    if len(profiles) < 4:
        return []
    rows: list[dict[str, float]] = []
    for omitted in profiles:
        retained = [fit for fit in profiles if fit is not omitted]
        model_law = fit_power_law(
            [fit.compute_budget for fit in retained],
            [fit.optimal_parameters for fit in retained],
        )
        data_law = fit_power_law(
            [fit.compute_budget for fit in retained],
            [fit.optimal_dataset_tokens for fit in retained],
        )
        loss_law = fit_loss_law(retained)
        rows.append(
            {
                "omitted_compute_budget": omitted.compute_budget,
                "predicted_parameters": float(model_law.predict(target_compute)),
                "predicted_dataset_tokens": float(data_law.predict(target_compute)),
                "predicted_final_loss": float(loss_law.predict(target_compute)),
            }
        )
    return rows


def solve_time_budget(
    model_law: PowerLaw,
    runs: Sequence[Run],
    target_hours: float,
) -> tuple[float, float, list[dict[str, float]]]:
    """Jointly solve C=t(N_opt(C))*seconds using measured throughput."""

    throughput_by_parameters: dict[int, list[float]] = {}
    for run in runs:
        throughput_by_parameters.setdefault(run.parameters, []).append(
            run.flops_per_second
        )
    hardware_points = [
        {
            "parameters": float(parameters),
            "median_flops_per_second": float(np.median(values)),
        }
        for parameters, values in sorted(throughput_by_parameters.items())
    ]
    log_parameters = np.log([row["parameters"] for row in hardware_points])
    throughput = np.asarray([row["median_flops_per_second"] for row in hardware_points])

    def interpolate(model_size: float) -> float:
        return float(np.interp(np.log(model_size), log_parameters, throughput))

    seconds = target_hours * 3600.0
    target_compute = float(np.median(throughput)) * seconds
    target_throughput = float(np.median(throughput))
    for _ in range(100):
        target_model_size = float(model_law.predict(target_compute))
        target_throughput = interpolate(target_model_size)
        updated_compute = target_throughput * seconds
        if math.isclose(updated_compute, target_compute, rel_tol=1e-10):
            target_compute = updated_compute
            break
        target_compute = updated_compute
    return target_compute, target_throughput, hardware_points


def architecture_candidates(
    target_parameters: float, target_compute: float
) -> list[dict[str, int | float | bool | str]]:
    """Map a continuous N prediction back to the measured architecture family."""

    candidates: list[dict[str, int | float | bool | str]] = []
    for hidden in range(192, 2_049, 64):
        layers = max(2, hidden // 48)
        intermediate = math.ceil((8 * hidden / 3) / 64) * 64
        parameters_per_layer = (
            4 * hidden * hidden + 3 * hidden * intermediate + 2 * hidden + 2 * 64
        )
        non_embedding = layers * parameters_per_layer + hidden
        embedding = 32_000 * hidden
        train_tokens = max(
            2_048,
            round(target_compute / (6 * non_embedding) / 2_048) * 2_048,
        )
        candidates.append(
            {
                "hidden_size": hidden,
                "intermediate_size": intermediate,
                "num_hidden_layers": layers,
                "num_attention_heads": hidden // 64,
                "num_key_value_heads": hidden // 64,
                "head_dim": 64,
                "non_embedding_parameters": non_embedding,
                "approximate_non_embedding_parameters": 12 * layers * hidden**2,
                "embedding_parameters": embedding,
                "total_parameters": non_embedding + embedding,
                "train_tokens": train_tokens,
                "actual_compute_flops": 6 * non_embedding * train_tokens,
                "requires_memory_probe": True,
                "status": "extrapolated_candidate_not_yet_validated",
                "log_distance_from_continuous_optimum": abs(
                    math.log(non_embedding / target_parameters)
                ),
            }
        )
    return sorted(
        candidates,
        key=lambda row: float(row["log_distance_from_continuous_optimum"]),
    )[:3]


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    figure.tight_layout()
    figure.savefig(output_stem.with_suffix(".png"), dpi=220)
    figure.savefig(output_stem.with_suffix(".svg"))
    plt.close(figure)


def plot_profiles(
    runs: Sequence[Run], profiles: Sequence[ProfileFit], output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(9.6, 6.2))
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(profiles)))
    for fit, color in zip(profiles, colors, strict=True):
        rows = sorted(
            (run for run in runs if run.compute_budget == fit.compute_budget),
            key=lambda run: run.parameters,
        )
        n_values = np.asarray([row.parameters for row in rows], dtype=np.float64)
        losses = np.asarray([row.final_loss for row in rows], dtype=np.float64)
        curve_n = np.geomspace(float(n_values.min()), float(n_values.max()), 180)
        curve_loss = np.polyval(
            (fit.quadratic_a, fit.quadratic_b, fit.quadratic_c), np.log(curve_n)
        )
        label = f"C={fit.compute_budget:.1e}"
        axis.scatter(n_values, losses, color=color, s=42, zorder=3)
        axis.plot(curve_n, curve_loss, color=color, linewidth=2, label=label)
        axis.scatter(
            fit.optimal_parameters,
            fit.predicted_optimal_loss,
            color=color,
            marker="*",
            edgecolor="black",
            linewidth=0.5,
            s=145,
            zorder=4,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Non-embedding parameters N")
    axis.set_ylabel("Final validation loss")
    axis.set_title("RTX 4060 IsoFLOPs profiles (stars: clipped quadratic optima)")
    axis.grid(True, which="both", color="#e5e7eb")
    axis.legend(frameon=False, ncol=2)
    save_figure(figure, output / "isoflops_profiles")


def plot_power_law(
    profiles: Sequence[ProfileFit],
    law: PowerLaw,
    target_compute: float,
    field: str,
    ylabel: str,
    output_stem: Path,
) -> None:
    compute = np.asarray([fit.compute_budget for fit in profiles])
    values = np.asarray([getattr(fit, field) for fit in profiles])
    domain = np.geomspace(float(compute.min()), target_compute, 300)
    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    axis.scatter(compute, values, color="#172554", s=65, label="Profile optima")
    in_range = domain <= compute.max()
    axis.plot(domain[in_range], law.predict(domain[in_range]), color="#0284c7", lw=2.2)
    axis.plot(
        domain[~in_range],
        law.predict(domain[~in_range]),
        color="#ea580c",
        lw=2.2,
        linestyle="--",
        label="12 h extrapolation",
    )
    target_value = float(law.predict(target_compute))
    axis.scatter(target_compute, target_value, marker="X", color="#ea580c", s=100)
    axis.annotate(
        f"12 h: {target_value:.3e}",
        (target_compute, target_value),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Compute C = 6ND (FLOPs)")
    axis.set_ylabel(ylabel)
    axis.set_title(
        f"{ylabel} scaling: y={law.coefficient:.3e} C^{law.exponent:.4f}, "
        f"log-space R²={law.r_squared:.4f}"
    )
    axis.grid(True, which="both", color="#e5e7eb")
    axis.legend(frameon=False)
    save_figure(figure, output_stem)


def plot_loss_law(
    profiles: Sequence[ProfileFit],
    law: LossLaw,
    target_compute: float,
    output: Path,
) -> None:
    compute = np.asarray([fit.compute_budget for fit in profiles])
    losses = np.asarray([fit.predicted_optimal_loss for fit in profiles])
    domain = np.geomspace(float(compute.min()), target_compute, 300)
    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    axis.scatter(compute, losses, color="#172554", s=65, label="Profile optima")
    in_range = domain <= compute.max()
    axis.plot(domain[in_range], law.predict(domain[in_range]), color="#0284c7", lw=2.2)
    axis.plot(
        domain[~in_range],
        law.predict(domain[~in_range]),
        color="#ea580c",
        lw=2.2,
        linestyle="--",
        label="12 h extrapolation",
    )
    target_loss = float(law.predict(target_compute))
    axis.scatter(target_compute, target_loss, marker="X", color="#ea580c", s=100)
    axis.annotate(
        f"12 h: {target_loss:.4f}",
        (target_compute, target_loss),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Compute C = 6ND (FLOPs)")
    axis.set_ylabel("Compute-optimal validation loss")
    axis.set_title(
        f"L_opt(C)={law.irreducible_loss:.4f}+{law.coefficient:.3e} "
        f"C^-{law.exponent:.4f}, R²={law.r_squared:.4f}"
    )
    axis.grid(True, which="both", color="#e5e7eb")
    axis.legend(frameon=False)
    save_figure(figure, output / "optimal_loss_scaling")


def plot_sensitivity(
    rows: Sequence[dict[str, float]],
    full_parameters: float,
    full_tokens: float,
    output: Path,
) -> None:
    if not rows:
        return
    labels = [f"omit {row['omitted_compute_budget']:.1e}" for row in rows]
    positions = np.arange(len(rows))
    width = 0.38
    model_ratios = [row["predicted_parameters"] / full_parameters for row in rows]
    data_ratios = [row["predicted_dataset_tokens"] / full_tokens for row in rows]
    figure, axis = plt.subplots(figsize=(9.4, 5.4))
    axis.bar(
        positions - width / 2, model_ratios, width, label="N prediction / full fit"
    )
    axis.bar(positions + width / 2, data_ratios, width, label="D prediction / full fit")
    axis.axhline(1.0, color="#111827", linewidth=1, linestyle="--")
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_ylabel("Relative 12 h prediction")
    axis.set_title("Leave-one-compute-budget-out extrapolation sensitivity")
    axis.grid(True, axis="y", color="#e5e7eb")
    axis.legend(frameon=False)
    save_figure(figure, output / "extrapolation_sensitivity")


def plot_fit_quality(
    runs: Sequence[Run], profiles: Sequence[ProfileFit], output: Path
) -> None:
    fit_by_budget = {fit.compute_budget: fit for fit in profiles}
    observed = np.asarray([run.final_loss for run in runs])
    predicted = np.asarray(
        [
            np.polyval(
                (
                    fit_by_budget[run.compute_budget].quadratic_a,
                    fit_by_budget[run.compute_budget].quadratic_b,
                    fit_by_budget[run.compute_budget].quadratic_c,
                ),
                np.log(run.parameters),
            )
            for run in runs
        ]
    )
    lower = float(min(observed.min(), predicted.min()))
    upper = float(max(observed.max(), predicted.max()))
    figure, axis = plt.subplots(figsize=(6.5, 6.2))
    axis.scatter(
        observed,
        predicted,
        c=np.log10([run.compute_budget for run in runs]),
        cmap="viridis",
        s=55,
    )
    axis.plot([lower, upper], [lower, upper], color="#dc2626", linestyle="--")
    axis.set_xlabel("Observed final loss")
    axis.set_ylabel("Quadratic fitted loss")
    axis.set_title(
        f"Within-profile fit quality (overall RMSE={np.sqrt(np.mean((observed - predicted) ** 2)):.4f})"
    )
    axis.grid(True, color="#e5e7eb")
    save_figure(figure, output / "loss_fit_quality")


def plot_hardware(runs: Sequence[Run], output: Path) -> None:
    figure, axes_grid = plt.subplots(2, 2, figsize=(11.5, 9.0))
    axes = axes_grid.ravel()
    sizes = np.asarray([run.parameters for run in runs])
    compute = np.log10([run.compute_budget for run in runs])
    axes[0].scatter(
        sizes,
        np.asarray([run.flops_per_second for run in runs]) / 1e12,
        c=compute,
        cmap="plasma",
    )
    axes[1].scatter(
        sizes, [run.tokens_per_second for run in runs], c=compute, cmap="plasma"
    )
    axes[2].scatter(
        sizes,
        np.asarray([run.chain_runtime_seconds for run in runs]) / 60.0,
        c=compute,
        cmap="plasma",
    )
    axes[3].scatter(
        sizes,
        np.asarray([run.estimated_memory_bytes for run in runs]) / 1024**3,
        c=compute,
        cmap="plasma",
    )
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(True, which="both", color="#e5e7eb")
        axis.set_xlabel("Non-embedding parameters N")
    axes[0].set_ylabel("Course-accounting TFLOPs/s")
    axes[1].set_ylabel("Training tokens/s")
    axes[2].set_ylabel("Training runtime (minutes)")
    axes[3].set_ylabel("XLA estimated device memory (GiB)")
    axes[0].set_title("Measured C=6ND throughput")
    axes[1].set_title("End-to-end training throughput")
    axes[2].set_title("Runtime by model and compute profile")
    axes[3].set_title("Compiled memory estimate")
    save_figure(figure, output / "hardware_efficiency")


def load_lr_pilots(runs_dir: Path) -> list[dict[str, float]]:
    pilots: list[dict[str, float]] = []
    for result_path in sorted(runs_dir.glob("*/result.json")):
        config_path = result_path.with_name("config.json")
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get(
            "target_compute_flops"
        ) is not None or "LR pilot" not in config.get("notes", ""):
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            continue
        training = config["training"]
        pilots.append(
            {
                "peak_learning_rate": float(
                    training["optimizer_config"]["lr_scheduler"]["peak_value"]
                ),
                "train_tokens": float(training["total_train_tokens"]),
                "final_loss": float(result["final_validation_loss"]),
                "runtime_seconds": float(result["runtime_seconds"]),
            }
        )
    return pilots


def plot_lr_pilots(pilots: Sequence[dict[str, float]], output: Path) -> None:
    if not pilots:
        return
    figure, axis = plt.subplots(figsize=(8.6, 5.4))
    for tokens in sorted({row["train_tokens"] for row in pilots}):
        rows = sorted(
            (row for row in pilots if row["train_tokens"] == tokens),
            key=lambda row: row["peak_learning_rate"],
        )
        axis.plot(
            [row["peak_learning_rate"] for row in rows],
            [row["final_loss"] for row in rows],
            marker="o",
            linewidth=2,
            label=f"{tokens / 1e6:.3g}M train tokens",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Peak learning rate")
    axis.set_ylabel("Final validation loss")
    axis.set_title("Learning-rate calibration")
    axis.grid(True, which="both", color="#e5e7eb")
    axis.legend(frameon=False)
    save_figure(figure, output / "learning_rate_calibration")


def plot_seed_repeats(repeats: Sequence[dict[str, Any]], output: Path) -> None:
    if not repeats:
        return
    labels = [
        f"C={row['compute_budget']:.1e}\nN={row['parameters'] / 1e6:.2f}M"
        for row in repeats
    ]
    positions = np.arange(len(repeats), dtype=np.float64)
    figure, axis = plt.subplots(figsize=(8.8, 5.5))
    for index, row in enumerate(repeats):
        losses = np.asarray(row["final_losses"], dtype=np.float64)
        mean_loss = float(row["mean_final_loss"])
        residuals = 1_000.0 * (losses - mean_loss)
        offsets = np.linspace(-0.08, 0.08, losses.size)
        axis.scatter(
            np.full(losses.shape, positions[index]) + offsets,
            residuals,
            color="#172554",
            s=55,
            zorder=3,
        )
    deviations = [1_000.0 * row["sample_standard_deviation"] for row in repeats]
    axis.errorbar(
        positions,
        np.zeros(len(repeats)),
        yerr=deviations,
        fmt="D",
        color="#ea580c",
        capsize=5,
        label="Group mean ± sample std",
    )
    axis.axhline(0.0, color="#64748b", linewidth=1, linestyle="--")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Deviation from group mean (milliloss)")
    axis.set_title("Adaptive model-seed repeat stability")
    axis.grid(True, axis="y", color="#e5e7eb")
    axis.legend(frameon=False)
    save_figure(figure, output / "seed_repeat_stability")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary: dict[str, Any]) -> str:
    model = summary["model_size_law"]
    data = summary["dataset_size_law"]
    loss = summary["optimal_loss_law"]
    prediction = summary["twelve_hour_prediction"]
    candidate = prediction["architecture_candidates"][0]
    sensitivity = summary["leave_one_budget_out"]
    repeats = summary["seed_repeats"]
    boundary_count = sum(
        bool(row["optimum_at_boundary"]) for row in summary["profile_fits"]
    )
    if sensitivity:
        model_values = [row["predicted_parameters"] for row in sensitivity]
        data_values = [row["predicted_dataset_tokens"] for row in sensitivity]
        loss_values = [row["predicted_final_loss"] for row in sensitivity]
        sensitivity_sentence = (
            f"Leave-one-budget-out 的 12 小时 N 预测范围为 "
            f"{min(model_values):.3e}–{max(model_values):.3e}，D 预测范围为 "
            f"{min(data_values):.3e}–{max(data_values):.3e}，loss 预测范围为 "
            f"{min(loss_values):.4f}–{max(loss_values):.4f}。"
        )
    else:
        sensitivity_sentence = "预算点不足四个，尚未计算 leave-one-budget-out 敏感性。"
    if repeats:
        max_seed_std = max(row["sample_standard_deviation"] for row in repeats)
        repeat_sentence = (
            f"{len(repeats)} 个候选做了自适应 seed 复验，最大样本标准差为 "
            f"{max_seed_std:.4f}。"
        )
    else:
        repeat_sentence = "没有完成可比较的 model-seed 复验。"
    quality_issues: list[str] = []
    if boundary_count:
        quality_issues.append(f"{boundary_count} 条 profile 的最优点仍在边界")
    if model["r_squared"] < 0.9:
        quality_issues.append(f"N 幂律 R² 仅 {model['r_squared']:.3f}")
    if data["r_squared"] < 0.9:
        quality_issues.append(f"D 幂律 R² 仅 {data['r_squared']:.3f}")
    if loss["irreducible_loss"] <= 1e-8:
        quality_issues.append("loss 拟合的不可约项落在搜索下界 0")
    quality_sentence = "；".join(quality_issues) or "所有主要质量检查均通过"
    return f"""# RTX 4060 本地 Scaling Law 报告

## 方法

固定每个计算预算 `C`，用多个模型规模构成 IsoFLOPs profile；训练 token 数由
`D = C / (6N_non_embedding)` 决定。每个 profile 对最终验证损失与 `log(N)` 做二次拟合，
将区间内（必要时裁剪到边界）的极小值作为 `N_opt(C)`，再由 `C=6ND` 得到 `D_opt(C)`。
最后在 log-log 空间拟合两条幂律。

## 拟合结果

- `N_opt(C) = {model["coefficient"]:.6e} * C^{model["exponent"]:.6f}`，R²={model["r_squared"]:.4f}
- `D_opt(C) = {data["coefficient"]:.6e} * C^{data["exponent"]:.6f}`，R²={data["r_squared"]:.4f}
- `L_opt(C) = {loss["irreducible_loss"]:.6f} + {loss["coefficient"]:.6e} * C^-{loss["exponent"]:.6f}`，R²={loss["r_squared"]:.4f}
- {len(summary["profile_fits"])} 个 profile 中有 {boundary_count} 个最优点位于搜索边界；边界点越多，外推风险越高。
- 主搜索累计训练计时 {summary["aggregate"]["runtime_hours"]:.3f} GPU 小时，编译和验证墙钟时间另列，不混入 `6ND` 吞吐。
- 包含学习率 pilot、失败后 checkpoint 续跑等在内的严格预算账本为 {summary["aggregate"]["strict_budget_ledger_hours"]:.3f} GPU 小时。
- {sensitivity_sentence}
- {repeat_sentence}

## 可信度结论

本次 12 小时预测属于**低置信度条件外推**：{quality_sentence}。中心估计用于指导下一轮
边界搜索和显存验证，而不是足以直接启动 12 小时最终训练的定论。model seed 复验稳定只说明
初始化噪声很小，不能抵消搜索边界和幂律拟合不足造成的结构性不确定性。

## 12 小时 4060 外推

联合 `N_opt(C)` 与按模型规模插值（区间外裁剪）的实测 `6ND` 吞吐，估算 12 小时
计算预算为 `{prediction["compute_budget"]:.6e}` FLOPs，对应吞吐
`{prediction["estimated_flops_per_second"]:.6e}` FLOPs/s；
连续幂律预测 `N_opt={prediction["optimal_parameters"]:.6e}` 个非 embedding 参数、
`D_opt={prediction["optimal_dataset_tokens"]:.6e}` 个训练 token、最终验证 loss
`{prediction["final_validation_loss"]:.4f}`。这里是超出观测 C 范围的外推，
正式 12 小时运行前应把连续 N 映射到可整除的层数/宽度，并做短时显存与稳定性验证。
当前架构族中最近的离散候选是 hidden={candidate["hidden_size"]}、
layers={candidate["num_hidden_layers"]}、精确非 embedding 参数
{candidate["non_embedding_parameters"]:,}、训练 token {candidate["train_tokens"]:,}；
它尚未做该规模的显存与学习率验证，不能直接视为最终可运行配置。

## 为什么这样搜索

IsoFLOPs 在固定 C 下同时改变 N 与 D，能直接观察“模型太小/数据太多”和“模型太大/数据太少”
两侧的损失变化。先做学习率校准、再固定优化器，是为了让不同 N/D 运行的损失主要反映规模分配，
而不是每个实验又引入一套超参数。图中的 profile 曲率、R²、边界最优点和吞吐变化共同用于判断
缩放律是否可信；不能只看一条拟合直线。

## 与课程问题的对应

1. **固定因素：** 数据 revision/tokenizer/顺序、context=512、effective batch=4、AdamW、
   weight decay、warmup/cosine、gradient clipping、BF16、validation prefix、评估次数和主网格 seed
   均固定；只改变模型规模 N 和由固定 C 决定的数据量 D。学习率先用独立 pilot 校准，主拟合中固定。
2. **拟合方法：** 每个固定 C 上拟合 `L(log N)` 二次曲线并保留观测最低点/边界标记，随后对
   `N_opt(C)`、`D_opt(C)` 做 log-log OLS，对 `L_opt(C)` 做带不可约项的非线性剖面拟合。
3. **拟合质量：** 不能只引用 R²；还要报告边界点、profile RMSE、leave-one-budget-out 范围、
   seed 标准差和外推倍数。本次 seed 很稳定，但边界与留一法非常不稳定，所以总体低置信度。
4. **迁移到 48 B200 小时：** 先用课程 API 实测 B200 上目标模型族的 `6ND` 吞吐，求解
   `C_48h = throughput_B200(N_opt(C_48h)) * 48h`，再代入同样的 N/D/loss 拟合。不能把本报告
   的 12h 4060 计算预算直接乘 4，因为 GPU 架构、利用率和模型相关吞吐都不同。
5. **最终离散架构：** 将连续 N 映射到满足 head/layer 整除约束的配置；课程建议的
   `12 * layers * hidden²` 可作近似，但提交和 FLOPs 分组应继续使用精确非 embedding 参数。
   若候选接近，先做短时显存/学习率验证，再用不同 seed 复验，而不是凭外推小数直接选择。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-hours", type=float, default=12.0)
    parser.add_argument(
        "--ledger-runtime-seconds",
        type=float,
        default=None,
        help="strict API budget ledger, including pilots and failed attempts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.target_hours) or args.target_hours <= 0:
        raise ValueError("target-hours must be positive")
    runs = load_runs(args.input)
    profiles = fit_profiles(runs)
    model_law = fit_power_law(
        [fit.compute_budget for fit in profiles],
        [fit.optimal_parameters for fit in profiles],
    )
    dataset_law = fit_power_law(
        [fit.compute_budget for fit in profiles],
        [fit.optimal_dataset_tokens for fit in profiles],
    )
    loss_law = fit_loss_law(profiles)
    median_throughput = float(np.median([run.flops_per_second for run in runs]))
    target_compute, target_throughput, hardware_points = solve_time_budget(
        model_law, runs, args.target_hours
    )
    target_parameters = float(model_law.predict(target_compute))
    candidates = architecture_candidates(target_parameters, target_compute)
    sensitivity = leave_one_budget_out(profiles, target_compute)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_profiles(runs, profiles, args.output_dir)
    plot_power_law(
        profiles,
        model_law,
        target_compute,
        "optimal_parameters",
        "Compute-optimal model size N",
        args.output_dir / "model_size_scaling",
    )
    plot_power_law(
        profiles,
        dataset_law,
        target_compute,
        "optimal_dataset_tokens",
        "Compute-optimal dataset size D",
        args.output_dir / "dataset_size_scaling",
    )
    plot_loss_law(profiles, loss_law, target_compute, args.output_dir)
    plot_sensitivity(
        sensitivity,
        float(model_law.predict(target_compute)),
        float(dataset_law.predict(target_compute)),
        args.output_dir,
    )
    plot_fit_quality(runs, profiles, args.output_dir)
    plot_hardware(runs, args.output_dir)
    pilots = load_lr_pilots(args.runs_dir)
    plot_lr_pilots(pilots, args.output_dir)
    repeats = summarize_repeats(runs)
    plot_seed_repeats(repeats, args.output_dir)
    main_runtime_seconds = sum(run.chain_runtime_seconds for run in runs)
    strict_ledger_seconds = (
        args.ledger_runtime_seconds
        if args.ledger_runtime_seconds is not None
        else main_runtime_seconds + sum(row["runtime_seconds"] for row in pilots)
    )
    if not math.isfinite(strict_ledger_seconds) or strict_ledger_seconds <= 0:
        raise ValueError("ledger-runtime-seconds must be positive and finite")
    profile_rows = [asdict(profile) for profile in profiles]
    write_csv(args.output_dir / "profile_fits.csv", profile_rows)
    summary: dict[str, Any] = {
        "method": "quadratic final_loss versus log(N) inside each IsoFLOPs profile; OLS power laws in log-log space",
        "compute_relation": "C = 6 * N_non_embedding * D",
        "model_size_law": asdict(model_law),
        "dataset_size_law": asdict(dataset_law),
        "optimal_loss_law": asdict(loss_law),
        "profile_fits": profile_rows,
        "aggregate": {
            "experiments": len(runs),
            "runtime_seconds": main_runtime_seconds,
            "runtime_hours": main_runtime_seconds / 3600.0,
            "strict_budget_ledger_seconds": strict_ledger_seconds,
            "strict_budget_ledger_hours": strict_ledger_seconds / 3600.0,
            "compile_seconds": sum(run.compile_seconds for run in runs),
            "wall_clock_seconds": sum(run.wall_clock_seconds for run in runs),
            "median_flops_per_second": median_throughput,
            "throughput_by_model_size": hardware_points,
        },
        "twelve_hour_prediction": {
            "hours": args.target_hours,
            "compute_budget": target_compute,
            "estimated_flops_per_second": target_throughput,
            "optimal_parameters": target_parameters,
            "optimal_dataset_tokens": float(dataset_law.predict(target_compute)),
            "final_validation_loss": float(loss_law.predict(target_compute)),
            "architecture_candidates": candidates,
        },
        "leave_one_budget_out": sensitivity,
        "seed_repeats": repeats,
        "learning_rate_pilots": pilots,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "twelve_hour_candidates.json").write_text(
        json.dumps(candidates, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    print(
        f"Analyzed {len(runs)} main runs across {len(profiles)} profiles; "
        f"outputs: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
