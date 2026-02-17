"""
One-shot pruning utilities.

Provides lightweight offline pruning routines (Wanda-style / SparseGPT-style)
for evaluating quality-retention without retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OneShotPruneResult:
    method: str
    target_sparsity: float
    actual_sparsity: float
    n_params_total: int
    n_params_pruned: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "method": self.method,
            "target_sparsity": self.target_sparsity,
            "actual_sparsity": self.actual_sparsity,
            "n_params_total": self.n_params_total,
            "n_params_pruned": self.n_params_pruned,
        }


def _iter_prunable_params(model: nn.Module) -> Iterable[torch.nn.Parameter]:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:
            continue
        if "embed" in name.lower():
            continue
        yield param


def _score_tensor(weight: torch.Tensor, method: str) -> torch.Tensor:
    w = weight.detach().abs()
    method_norm = str(method or "wanda").strip().lower()
    if method_norm == "sparsegpt":
        return w
    row_scale = torch.sqrt(torch.clamp((weight.detach() ** 2).mean(dim=-1, keepdim=True), min=1e-12))
    return w * row_scale


def apply_one_shot_pruning(
    model: nn.Module,
    target_sparsity: float = 0.5,
    method: str = "wanda",
) -> OneShotPruneResult:
    """In-place one-shot pruning over prunable 2D+ parameters."""
    target = float(max(0.0, min(0.95, target_sparsity)))
    total_params = 0
    total_pruned = 0

    for param in _iter_prunable_params(model):
        numel = int(param.numel())
        total_params += numel
        if numel <= 1 or target <= 0.0:
            continue

        score = _score_tensor(param, method=method).reshape(-1)
        k_prune = int(round(numel * target))
        k_prune = max(0, min(k_prune, numel - 1))
        if k_prune <= 0:
            continue

        prune_idx = torch.topk(score, k=k_prune, largest=False).indices
        mask = torch.ones(numel, device=param.device, dtype=param.dtype)
        mask[prune_idx] = 0.0
        mask = mask.reshape_as(param)

        with torch.no_grad():
            param.mul_(mask)

        total_pruned += k_prune

    actual = (total_pruned / total_params) if total_params > 0 else 0.0
    return OneShotPruneResult(
        method=str(method or "wanda").strip().lower() or "wanda",
        target_sparsity=target,
        actual_sparsity=float(actual),
        n_params_total=total_params,
        n_params_pruned=total_pruned,
    )


def estimate_lm_ce_loss(
    model: nn.Module,
    input_batches: List[torch.Tensor],
    device: torch.device,
) -> Optional[float]:
    """Estimate average CE loss on provided token batches without training."""
    if not input_batches:
        return None

    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in input_batches:
            input_ids = batch.to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                input_ids[:, 1:].reshape(-1),
            )
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            losses.append(float(loss.item()))

    if not losses:
        return None
    return float(sum(losses) / len(losses))
