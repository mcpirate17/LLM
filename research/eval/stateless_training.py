"""Stateless eval helpers for non-mutating screening paths."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


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
            sl = logits[:, :-1].contiguous()
            if sl.shape[-1] > vocab_size:
                sl = sl[..., :vocab_size]
            loss = F.cross_entropy(
                sl.reshape(-1, sl.shape[-1]),
                batch[:, 1:].reshape(-1),
                reduction="sum",
            )
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
) -> float:
    """Train cloned parameters only; leave the live module untouched."""
    if not batches:
        return float("inf")

    template_params = {name: tensor.detach().clone() for name, tensor in params.items()}

    def _run(run_lr: float) -> float:
        optimizer = torch.optim.AdamW(list(params.values()), lr=run_lr)
        final_loss = float("inf")
        for step in range(n_steps):
            if step < warmup_steps:
                warmup_factor = (step + 1) / warmup_steps
                for group in optimizer.param_groups:
                    group["lr"] = run_lr * warmup_factor

            batch = batches[step % len(batches)]
            optimizer.zero_grad(set_to_none=True)
            logits = functional_logits(model, params, buffers, batch)
            sl = logits[:, :-1].contiguous()
            if sl.shape[-1] > vocab_size:
                sl = sl[..., :vocab_size]
            loss = F.cross_entropy(
                sl.reshape(-1, sl.shape[-1]),
                batch[:, 1:].reshape(-1),
            )
            if not torch.isfinite(loss):
                return float("inf")
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(list(params.values()), clip_grad)
            optimizer.step()
            final_loss = float(loss.item())
            if loss_trajectory is not None:
                loss_trajectory[step + 1] = final_loss
        return final_loss

    result = _run(lr)
    if math.isfinite(result):
        return result

    if loss_trajectory is not None:
        loss_trajectory.clear()
    reset_parameters_(params, template_params)
    for tensor in params.values():
        tensor.grad = None
    return _run(lr * 0.1)
