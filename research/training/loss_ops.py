"""Thin wrappers around research.eval.utils for the micro-training tools.

Kept as a stable import surface for the scripts under research/tools/. The
native fast path lives in research.training._native and is reached
through language_model_loss / clip_grad_norm.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from research.eval.utils import clip_grad_norm as _clip_grad_norm
from research.eval.utils import language_model_loss as _language_model_loss


def next_token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    return _language_model_loss(logits, targets, int(vocab_size))


def clip_grad_norm_(
    params_or_model,
    max_norm: float = 1.0,
) -> torch.Tensor:
    if isinstance(params_or_model, nn.Module):
        params: Iterable[torch.Tensor] = params_or_model.parameters()
    else:
        params = params_or_model
    return _clip_grad_norm(params, float(max_norm))
