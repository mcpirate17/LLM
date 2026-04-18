from __future__ import annotations

import logging
from functools import lru_cache

import torch
import torch.nn as nn

from ._loss_native import load_loss_native

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _native_loss_available() -> bool:
    try:
        native = load_loss_native()
        return hasattr(native, "next_token_cross_entropy") and hasattr(
            native, "clip_grad_norm_"
        )
    except Exception as exc:
        logger.warning(
            "Native loss extension unavailable; using torch fallback (%s)", exc
        )
        return False


def next_token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    if _native_loss_available():
        return load_loss_native().next_token_cross_entropy(
            logits,
            targets,
            int(vocab_size),
            "mean",
        )

    score_logits = logits[:, :-1]
    if score_logits.size(-1) > int(vocab_size):
        score_logits = score_logits[..., : int(vocab_size)]
    return torch.nn.functional.cross_entropy(
        score_logits.reshape(-1, score_logits.size(-1)),
        targets[:, 1:].reshape(-1),
    )


def clip_grad_norm_(
    params_or_model,
    max_norm: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if isinstance(params_or_model, nn.Module):
        params = list(params_or_model.parameters())
    else:
        params = list(params_or_model)
    grads = [param.grad for param in params if param.grad is not None]

    if _native_loss_available():
        return load_loss_native().clip_grad_norm_(grads, float(max_norm), float(eps))

    return torch.nn.utils.clip_grad_norm_(params, float(max_norm))
