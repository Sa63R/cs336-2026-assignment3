from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
from equinox import nn
from jax import Array

from cs336_scaling.local.schemas import LocalTrainingConfig
from cs336_scaling.training.data import Batch
from cs336_scaling.training.loop import sharded_sequence_loss, val_model
from cs336_scaling.training.model.basic_model import BasicCausalLM


class LocalChunkResult(eqx.Module):
    model: BasicCausalLM
    state: nn.State
    opt_state: object
    train_losses: Array
    val_loss: Array


def _add_trees(left, right):
    return jax.tree.map(lambda x, y: x + y, left, right)


def _scale_tree(tree, scale: float):
    return jax.tree.map(lambda value: value * scale, tree)


def accumulated_train_updates(
    model: BasicCausalLM,
    state: nn.State,
    train_data: Batch[Array],
    config: LocalTrainingConfig,
    opt_state: object,
) -> tuple[BasicCausalLM, nn.State, Array, object]:
    """Apply optimizer updates with gradients averaged across microbatches."""

    optimizer_config = config.optimizer_training_config()
    optimizer = config.optimizer_config.build(optimizer_config)
    state = model.layers.self_attn.rotary.update_cache(state, seq_len=config.seq_len)

    def update_step(carry, accumulation_batches: Batch[Array]):
        current_model, current_opt_state = carry
        first_batch = jax.tree.map(lambda value: value[0], accumulation_batches)
        first_loss, gradient_sum = jax.value_and_grad(sharded_sequence_loss)(
            current_model, state, first_batch
        )

        def accumulate(next_carry, microbatch: Batch[Array]):
            loss_sum, gradients = next_carry
            loss, next_gradients = jax.value_and_grad(sharded_sequence_loss)(
                current_model, state, microbatch
            )
            return (loss_sum + loss, _add_trees(gradients, next_gradients)), None

        (loss_sum, gradient_sum), _ = jax.lax.scan(
            accumulate,
            (first_loss, gradient_sum),
            jax.tree.map(lambda value: value[1:], accumulation_batches),
        )
        inverse_accumulation = 1.0 / config.gradient_accumulation_steps
        mean_gradients = _scale_tree(gradient_sum, inverse_accumulation)
        mean_loss = loss_sum * inverse_accumulation
        updates, next_opt_state = optimizer.update(
            mean_gradients, current_opt_state, current_model
        )
        next_model = eqx.apply_updates(current_model, updates)
        return (next_model, next_opt_state), mean_loss

    (model, opt_state), losses = jax.lax.scan(
        update_step, (model, opt_state), train_data
    )
    return model, state, losses, opt_state


@partial(jax.jit, static_argnames=("config",))
def train_and_evaluate_chunk(
    model: BasicCausalLM,
    state: nn.State,
    train_data: Batch[Array],
    validation_data: Batch[Array],
    config: LocalTrainingConfig,
    opt_state: object,
) -> LocalChunkResult:
    model, state, losses, opt_state = accumulated_train_updates(
        model,
        state,
        train_data,
        config,
        opt_state,
    )
    validation_loss, state = val_model(model, state, validation_data)
    return LocalChunkResult(
        model=model,
        state=state,
        opt_state=opt_state,
        train_losses=losses,
        val_loss=validation_loss,
    )
