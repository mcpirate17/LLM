"""Stateless eval helpers for non-mutating screening paths."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call

from .training_core import run_training_loop
from .utils import language_model_loss


TensorMap = Dict[str, torch.Tensor]


def clone_module_state(model: nn.Module) -> Tuple[TensorMap, TensorMap]:
    """Clone parameters and buffers for stateless execution."""
    params = {
        name: param.detach().clone().requires_grad_(param.requires_grad)
        for name, param in model.named_parameters()
    }
    buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    return params, buffers


def reset_parameters_(params: TensorMap, template_params: TensorMap) -> None:
    """Restore parameter tensors from an immutable template state."""
    for name, template in template_params.items():
        params[name].data.copy_(template.data)


def functional_logits(
    model: nn.Module,
    params: TensorMap,
    buffers: TensorMap,
    batch: torch.Tensor,
) -> torch.Tensor:
    """Run the module with external parameters and buffers."""
    return functional_call(model, (params, buffers), (batch,))


def functional_compute_perplexity(
    model: nn.Module,
    params: TensorMap,
    buffers: TensorMap,
    batches: List[torch.Tensor],
    vocab_size: int,
) -> Optional[float]:
    """Compute perplexity without mutating the live module state."""
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in batches:
            logits = functional_logits(model, params, buffers, batch)
            loss = language_model_loss(logits, batch, vocab_size, reduction="sum")
            if torch.isfinite(loss):
                total_loss += float(loss.item())
                total_tokens += batch[:, 1:].numel()
    if total_tokens == 0:
        return None
    return math.exp(min(total_loss / total_tokens, 20.0))


def functional_micro_train_loop(
    model: nn.Module,
    params: TensorMap,
    buffers: TensorMap,
    batches: List[torch.Tensor],
    vocab_size: int,
    n_steps: int = 200,
    lr: float = 3e-4,
    clip_grad: float = 1.0,
    warmup_steps: int = 10,
    loss_trajectory: Optional[dict] = None,
    step_callback=None,
) -> float:
    """Train cloned parameters only; leave the live module untouched."""
    if not batches:
        return float("inf")

    template_params = {name: tensor.detach().clone() for name, tensor in params.items()}

    def _run(run_lr: float) -> float:
        param_values = list(params.values())

        def compute_loss(step: int) -> torch.Tensor:
            if step_callback is not None:
                step_callback(step, n_steps)
            batch = batches[step % len(batches)]
            logits = functional_logits(model, params, buffers, batch)
            return language_model_loss(logits, batch, vocab_size)

        result = run_training_loop(
            param_values,
            compute_loss,
            n_steps=n_steps,
            optimizer_name="adamw",
            lr=run_lr,
            clip_grad=clip_grad,
            warmup_steps=warmup_steps,
            loss_trajectory=loss_trajectory,
        )
        return result.final_loss

    result = _run(lr)
    if hasattr(model, "set_routing_progress"):
        model.set_routing_progress(1.0)
    if math.isfinite(result):
        return result

    if loss_trajectory is not None:
        loss_trajectory.clear()
    reset_parameters_(params, template_params)
    for tensor in params.values():
        tensor.grad = None
    result = _run(lr * 0.1)
    if hasattr(model, "set_routing_progress"):
        model.set_routing_progress(1.0)
    return result
