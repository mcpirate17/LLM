from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from ._loss_native import load_loss_native

LOG_PROB_COMPONENTS = frozenset(
    {
        "label_smoothed_ce",
        "rank_weighted_ce",
        "tropical_ce",
        "contrastive_push",
        "kl_uniform",
        "entropy_reg",
    }
)


def compute_spectral_component(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() < 3:
        return logits.new_zeros(())
    freq = torch.fft.rfft(logits.float(), dim=1)
    high_freq_energy = freq[:, freq.shape[1] // 2 :].abs().mean()
    return high_freq_energy * 0.01


def loss_cross_entropy(flat_logits, flat_targets, log_probs):
    return F.cross_entropy(flat_logits, flat_targets)


def loss_label_smoothed_ce(flat_logits, flat_targets, log_probs):
    if log_probs is None:
        log_probs = F.log_softmax(flat_logits, dim=-1)
    smooth = 0.1
    nll = -log_probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
    smooth_loss = -log_probs.mean(dim=-1)
    return ((1 - smooth) * nll + smooth * smooth_loss).mean()


def loss_rank_weighted_ce(flat_logits, flat_targets, log_probs):
    if log_probs is None:
        log_probs = F.log_softmax(flat_logits, dim=-1)
    return load_loss_native().rank_weighted_ce(flat_logits, flat_targets, log_probs)


def loss_tropical_ce(flat_logits, flat_targets, log_probs):
    if flat_logits.shape[-1] <= 1:
        return flat_logits.new_zeros(())
    if log_probs is None:
        log_probs = F.log_softmax(flat_logits, dim=-1)
    return load_loss_native().tropical_ce(flat_targets, log_probs)


def loss_contrastive_push(flat_logits, flat_targets, log_probs):
    target_logits = flat_logits.gather(1, flat_targets.unsqueeze(1))
    topk_width = min(6, flat_logits.shape[-1])
    if topk_width <= 1:
        return flat_logits.new_zeros(())
    topk, _ = flat_logits.topk(topk_width, dim=-1)
    return F.relu(topk[:, 1:] - target_logits + 0.5).mean()


def loss_entropy_reg(flat_logits, flat_targets, log_probs):
    if log_probs is None:
        log_probs = F.log_softmax(flat_logits, dim=-1)
    return load_loss_native().entropy_reg(log_probs)


def loss_gradient_penalty(flat_logits, flat_targets, log_probs):
    return flat_logits.pow(2).mean() * 0.001


def loss_kl_uniform(flat_logits, flat_targets, log_probs):
    if log_probs is None:
        log_probs = F.log_softmax(flat_logits, dim=-1)
    vocab_size = flat_logits.shape[-1]
    return -(log_probs.mean(dim=-1) + math.log(vocab_size)).mean()


LOSS_DISPATCH: dict[str, Callable[..., torch.Tensor]] = {
    "cross_entropy": loss_cross_entropy,
    "label_smoothed_ce": loss_label_smoothed_ce,
    "rank_weighted_ce": loss_rank_weighted_ce,
    "tropical_ce": loss_tropical_ce,
    "contrastive_push": loss_contrastive_push,
    "entropy_reg": loss_entropy_reg,
    "gradient_penalty": loss_gradient_penalty,
    "kl_uniform": loss_kl_uniform,
}


def compute_component_fast(
    name: str,
    flat_logits: torch.Tensor,
    flat_targets: torch.Tensor,
    log_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    fn = LOSS_DISPATCH.get(name, loss_cross_entropy)
    return fn(flat_logits, flat_targets, log_probs)
