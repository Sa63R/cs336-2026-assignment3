# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10,<4",
#     "numpy>=2.2,<3",
# ]
# ///
"""Fit Chinchilla-style IsoFLOPs scaling laws from completed training runs.

For each compute budget, the script selects the run with the lowest final loss.
It then computes the corresponding dataset size from C = 6ND and fits the two
power laws N_opt(C) = k_N C^a and D_opt(C) = k_D C^b in log-log space.

This is deliberately a CPU-only analysis.  A GPU, including an RTX 4060, is not
needed because the input contains completed training runs rather than models to
train.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "isoflops_curves.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chinchilla_isoflops"
DEFAULT_TARGETS = (1e23, 1e24)


@dataclass(frozen=True)
class TrainingRun:
    """One completed training run from the input JSON file."""

    parameters: int
    compute_budget: float
    final_loss: float


@dataclass(frozen=True)
class OptimalPoint:
    """The lowest-loss run in one IsoFLOPs profile."""

    compute_budget: float
    parameters: int
    dataset_tokens: float
    final_loss: float


@dataclass(frozen=True)
class PowerLaw:
    """A fitted relationship y = coefficient * x**exponent."""

    coefficient: float
    exponent: float
    r_squared: float

    def predict(self, compute_budget: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the fitted power law at one or more compute budgets."""

        return self.coefficient * np.power(compute_budget, self.exponent)


def positive_finite_number(value: object, field: str, row_index: int) -> float:
    """Validate a numeric JSON field and return it as a float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"row {row_index}: {field!r} must be numeric")

    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"row {row_index}: {field!r} must be positive and finite")
    return number


def load_runs(path: Path) -> list[TrainingRun]:
    """Load and validate the synthetic training-run records."""

    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list) or not payload:
        raise ValueError("the input JSON must be a non-empty array")

    runs: list[TrainingRun] = []
    required_fields = {"parameters", "compute_budget", "final_loss"}
    for row_index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"row {row_index}: expected an object")

        missing_fields = required_fields.difference(row)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"row {row_index}: missing fields: {missing}")

        parameters = positive_finite_number(row["parameters"], "parameters", row_index)
        if not parameters.is_integer():
            raise ValueError(f"row {row_index}: 'parameters' must be an integer")

        runs.append(
            TrainingRun(
                parameters=int(parameters),
                compute_budget=positive_finite_number(
                    row["compute_budget"], "compute_budget", row_index
                ),
                final_loss=positive_finite_number(
                    row["final_loss"], "final_loss", row_index
                ),
            )
        )

    return runs


def select_optimal_points(runs: Sequence[TrainingRun]) -> list[OptimalPoint]:
    """Take the lowest-loss run from each fixed-compute IsoFLOPs profile."""

    best_by_budget: dict[float, TrainingRun] = {}
    for run in runs:
        current_best = best_by_budget.get(run.compute_budget)
        # The parameter-count tie-breaker makes the result deterministic if two
        # runs happen to have exactly the same final loss.
        if current_best is None or (run.final_loss, run.parameters) < (
            current_best.final_loss,
            current_best.parameters,
        ):
            best_by_budget[run.compute_budget] = run

    if len(best_by_budget) < 2:
        raise ValueError("at least two distinct compute budgets are required")

    return [
        OptimalPoint(
            compute_budget=run.compute_budget,
            parameters=run.parameters,
            dataset_tokens=run.compute_budget / (6.0 * run.parameters),
            final_loss=run.final_loss,
        )
        for run in sorted(best_by_budget.values(), key=lambda item: item.compute_budget)
    ]


def fit_power_law(x_values: Sequence[float], y_values: Sequence[float]) -> PowerLaw:
    """Fit y = k*x**p using ordinary least squares on natural logarithms."""

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError(
            "x and y must be equally sized 1-D arrays with at least 2 values"
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("power-law fitting requires positive x and y values")

    log_x = np.log(x)
    log_y = np.log(y)
    design_matrix = np.column_stack((np.ones_like(log_x), log_x))
    log_coefficient, exponent = np.linalg.lstsq(design_matrix, log_y, rcond=None)[0]

    fitted_log_y = log_coefficient + exponent * log_x
    residual_sum_squares = float(np.sum(np.square(log_y - fitted_log_y)))
    total_sum_squares = float(np.sum(np.square(log_y - np.mean(log_y))))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else 1.0
    )

    return PowerLaw(
        coefficient=float(np.exp(log_coefficient)),
        exponent=float(exponent),
        r_squared=r_squared,
    )


def format_scientific(value: float) -> str:
    """Format a positive number compactly for console output and annotations."""

    return f"{value:.4e}"


def plot_scaling_law(
    points: Sequence[OptimalPoint],
    law: PowerLaw,
    targets: Sequence[float],
    value_name: str,
    value_symbol: str,
    value_getter: str,
    output_stem: Path,
) -> None:
    """Plot observed optima, the in-range fit, and the extrapolated power law."""

    compute_budgets = np.asarray(
        [point.compute_budget for point in points], dtype=np.float64
    )
    observed_values = np.asarray(
        [getattr(point, value_getter) for point in points], dtype=np.float64
    )
    target_budgets = np.asarray(targets, dtype=np.float64)

    observed_min = float(np.min(compute_budgets))
    observed_max = float(np.max(compute_budgets))
    plot_max = max(observed_max, float(np.max(target_budgets)))
    fitted_budget_range = np.geomspace(observed_min, observed_max, 240)
    extrapolated_budget_range = np.geomspace(observed_max, plot_max, 240)

    figure, axis = plt.subplots(figsize=(9.6, 6.0))
    axis.scatter(
        compute_budgets,
        observed_values,
        color="#13213c",
        edgecolor="white",
        linewidth=0.8,
        s=68,
        zorder=3,
        label="Lowest-loss run at each budget",
    )
    axis.plot(
        fitted_budget_range,
        law.predict(fitted_budget_range),
        color="#0072b2",
        linewidth=2.2,
        label="Power-law fit",
    )
    if plot_max > observed_max:
        axis.plot(
            extrapolated_budget_range,
            law.predict(extrapolated_budget_range),
            color="#d55e00",
            linewidth=2.2,
            linestyle="--",
            label="Extrapolation",
        )

    target_values = np.asarray(law.predict(target_budgets), dtype=np.float64)
    axis.scatter(
        target_budgets,
        target_values,
        marker="X",
        color="#d55e00",
        edgecolor="white",
        linewidth=0.8,
        s=100,
        zorder=4,
        label="Requested predictions",
    )
    largest_prediction = float(np.max(target_values))
    for compute_budget, predicted_value in zip(
        target_budgets, target_values, strict=True
    ):
        is_top_prediction = math.isclose(predicted_value, largest_prediction)
        axis.annotate(
            f"C={compute_budget:.0e}\n{value_symbol}={format_scientific(predicted_value)}",
            xy=(compute_budget, predicted_value),
            xytext=(-8, -14 if is_top_prediction else 12),
            textcoords="offset points",
            ha="right",
            va="top" if is_top_prediction else "bottom",
            fontsize=8.5,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.margins(x=0.04, y=0.08)
    axis.set_xlabel("Compute budget C (FLOPs)")
    axis.set_ylabel(f"Compute-optimal {value_name}")
    axis.set_title(
        f"IsoFLOPs scaling law for {value_name}\n"
        f"{value_symbol}(C) = {law.coefficient:.3e} C^{law.exponent:.4f}  "
        f"(log-space R² = {law.r_squared:.4f})"
    )
    axis.grid(which="major", color="#d7dce2", linewidth=0.8)
    axis.grid(which="minor", color="#edf0f2", linewidth=0.5)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()

    for suffix in ("png", "svg"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=220)
    plt.close(figure)


def write_fit_points(path: Path, points: Sequence[OptimalPoint]) -> None:
    """Write the exact data points used by both fitted scaling laws."""

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(points[0])))
        writer.writeheader()
        writer.writerows(asdict(point) for point in points)


def build_summary(
    points: Sequence[OptimalPoint],
    model_law: PowerLaw,
    dataset_law: PowerLaw,
    targets: Sequence[float],
) -> dict[str, object]:
    """Build a machine-readable record of the method, fits, and predictions."""

    predictions = [
        {
            "compute_budget": target,
            "optimal_parameters": float(model_law.predict(target)),
            "optimal_dataset_tokens": float(dataset_law.predict(target)),
        }
        for target in targets
    ]
    return {
        "method": "lowest final loss per compute budget; OLS in log-log space",
        "compute_relation": "C = 6*N*D",
        "model_size_law": asdict(model_law),
        "dataset_size_law": asdict(dataset_law),
        "optimal_points": [asdict(point) for point in points],
        "predictions": predictions,
    }


def write_answer_sentences(
    path: Path,
    model_law: PowerLaw,
    dataset_law: PowerLaw,
    targets: Sequence[float],
) -> None:
    """Write the one-sentence responses requested in parts (a) and (b)."""

    target_text = " and ".join(format_scientific(target) for target in targets)
    model_predictions = " and ".join(
        format_scientific(float(model_law.predict(target))) for target in targets
    )
    dataset_predictions = " and ".join(
        format_scientific(float(dataset_law.predict(target))) for target in targets
    )
    sentences = (
        f"(a) At compute budgets {target_text} FLOPs, the fitted IsoFLOPs law "
        f"predicts optimal model sizes of {model_predictions} parameters, respectively.\n"
        f"(b) At compute budgets {target_text} FLOPs, the fitted IsoFLOPs law "
        f"predicts optimal dataset sizes of {dataset_predictions} tokens, respectively.\n"
    )
    path.write_text(sentences, encoding="utf-8")


def print_results(
    points: Sequence[OptimalPoint],
    model_law: PowerLaw,
    dataset_law: PowerLaw,
    targets: Sequence[float],
    output_dir: Path,
) -> None:
    """Print selected minima, fitted equations, and requested predictions."""

    print("Selected IsoFLOPs minima:")
    print(f"{'C (FLOPs)':>13}  {'N_opt':>13}  {'D_opt (tokens)':>15}  {'loss':>9}")
    for point in points:
        print(
            f"{point.compute_budget:13.4e}  {point.parameters:13d}  "
            f"{point.dataset_tokens:15.4e}  {point.final_loss:9.5f}"
        )

    print("\nFitted laws:")
    print(
        f"N_opt(C) = {model_law.coefficient:.6e} * C^{model_law.exponent:.6f} "
        f"(R^2={model_law.r_squared:.6f})"
    )
    print(
        f"D_opt(C) = {dataset_law.coefficient:.6e} * C^{dataset_law.exponent:.6f} "
        f"(R^2={dataset_law.r_squared:.6f})"
    )

    print("\nPredictions:")
    for target in targets:
        print(
            f"C={target:.4e}: N_opt={float(model_law.predict(target)):.4e} parameters, "
            f"D_opt={float(dataset_law.predict(target)):.4e} tokens"
        )
    print(f"\nWrote plots and tables to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"training-run JSON (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for plots and tables (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=DEFAULT_TARGETS,
        help="compute budgets to extrapolate to (default: 1e23 1e24)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete IsoFLOPs fitting and plotting pipeline."""

    args = parse_args()
    if any(not math.isfinite(target) or target <= 0 for target in args.targets):
        raise ValueError("all target compute budgets must be positive and finite")

    runs = load_runs(args.data)
    points = select_optimal_points(runs)
    model_law = fit_power_law(
        [point.compute_budget for point in points],
        [point.parameters for point in points],
    )
    dataset_law = fit_power_law(
        [point.compute_budget for point in points],
        [point.dataset_tokens for point in points],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_scaling_law(
        points=points,
        law=model_law,
        targets=args.targets,
        value_name="model size",
        value_symbol="N_opt",
        value_getter="parameters",
        output_stem=args.output_dir / "model_size_scaling",
    )
    plot_scaling_law(
        points=points,
        law=dataset_law,
        targets=args.targets,
        value_name="dataset size (tokens)",
        value_symbol="D_opt",
        value_getter="dataset_tokens",
        output_stem=args.output_dir / "dataset_size_scaling",
    )
    write_fit_points(args.output_dir / "fit_points.csv", points)
    summary = build_summary(points, model_law, dataset_law, args.targets)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_answer_sentences(
        args.output_dir / "answer_sentences.txt",
        model_law,
        dataset_law,
        args.targets,
    )
    print_results(points, model_law, dataset_law, args.targets, args.output_dir)


if __name__ == "__main__":
    main()
