from __future__ import annotations

import math
from dataclasses import dataclass

from cs336_scaling.training.model.config import BasicTransformerConfig


@dataclass(frozen=True)
class ParameterCounts:
    """Parameter counts needed for scaling-law and hardware accounting."""

    total: int
    non_embedding: int
    embedding: int
    approximate_non_embedding: int


def estimate_parameter_counts(config: BasicTransformerConfig) -> ParameterCounts:
    """Exactly count ``BasicCausalLM`` scalars and the course's 12Ld² proxy."""

    hidden = config.hidden_size
    intermediate = config.intermediate_size
    head_dim = config.head_dim
    embedding_matrices = 1 if config.tie_word_embeddings else 2
    embedding_parameters = embedding_matrices * config.vocab_size * hidden
    attention_biases = 4 * hidden if config.attention_bias else 0
    parameters_per_layer = (
        4 * hidden * hidden
        + 3 * hidden * intermediate
        + 2 * hidden
        + 2 * head_dim
        + attention_biases
    )
    non_embedding_parameters = config.num_hidden_layers * parameters_per_layer + hidden
    return ParameterCounts(
        total=embedding_parameters + non_embedding_parameters,
        non_embedding=non_embedding_parameters,
        embedding=embedding_parameters,
        approximate_non_embedding=(
            12 * config.num_hidden_layers * config.hidden_size**2
        ),
    )


def estimate_parameter_count(config: BasicTransformerConfig) -> int:
    """Exactly count trainable scalars in ``BasicCausalLM`` from its config."""

    return estimate_parameter_counts(config).total


def runtime_limit_for_compute(
    target_compute_flops: float,
    *,
    reference_flops_per_second: float,
    margin: float,
    minimum_seconds: float,
) -> float:
    """Convert a FLOPs profile into a conservative per-run runtime limit."""

    values = (
        target_compute_flops,
        reference_flops_per_second,
        margin,
        minimum_seconds,
    )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("runtime planning values must be positive and finite")
    return max(
        minimum_seconds,
        target_compute_flops / reference_flops_per_second * margin,
    )


def round_tokens_for_compute(
    target_compute_flops: float,
    *,
    parameters: int,
    token_quantum: int,
    maximum_tokens: int,
) -> int:
    """Round ``D = C/(6N)`` to a valid optimizer/evaluation boundary."""

    if not math.isfinite(target_compute_flops) or target_compute_flops <= 0:
        raise ValueError("target compute must be positive and finite")
    if parameters <= 0 or token_quantum <= 0 or maximum_tokens <= 0:
        raise ValueError(
            "parameters, token_quantum, and maximum_tokens must be positive"
        )
    ideal_tokens = target_compute_flops / (6 * parameters)
    rounded_tokens = max(
        token_quantum, round(ideal_tokens / token_quantum) * token_quantum
    )
    if rounded_tokens > maximum_tokens:
        raise ValueError(
            f"target compute needs {rounded_tokens:,} tokens for a {parameters:,}-parameter "
            f"model, but the dataset allows {maximum_tokens:,}"
        )
    return rounded_tokens
