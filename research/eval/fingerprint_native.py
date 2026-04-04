"""Native-backed metric helpers for behavioral fingerprinting."""

from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn.functional as F

from research.env import aria_core


def _cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    detached = tensor.detach()
    if detached.device.type != "cpu":
        detached = detached.cpu()
    return detached.contiguous()


def interaction_metrics(
    influence_matrix: torch.Tensor,
    positions: torch.Tensor,
) -> Dict[str, float]:
    native = aria_core.interaction_metrics_f32(
        _cpu_contiguous(influence_matrix),
        _cpu_contiguous(positions),
    )
    return {
        "locality": float(native[0].item()),
        "sparsity": float(native[1].item()),
        "symmetry": float(native[2].item()),
        "hierarchy": float(native[3].item()),
    }


def sensitivity_metrics(sens_matrix: torch.Tensor) -> Dict[str, float]:
    native = aria_core.sensitivity_metrics_f32(_cpu_contiguous(sens_matrix))
    return {
        "spectral_norm": float(native[0].item()),
        "uniformity": float(native[1].item()),
        "effective_rank": float(native[2].item()),
    }


def geometry_metrics(
    reps: torch.Tensor,
    *,
    max_rows: int = 500,
) -> Dict[str, float] | None:
    if aria_core is None or not hasattr(aria_core, "geometry_metrics_f32"):
        return None
    native = aria_core.geometry_metrics_f32(
        _cpu_contiguous(reps.float()), int(max_rows)
    )
    return {
        "intrinsic_dim": float(native[0].item()),
        "isotropy": float(native[1].item()),
        "rank_ratio": float(native[2].item()),
    }


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    if (
        X.device.type == "cpu"
        and Y.device.type == "cpu"
        and aria_core is not None
        and not os.environ.get("ARIA_DISABLE_NATIVE_CKA")
    ):
        return aria_core.linear_cka_f32(X.contiguous(), Y.contiguous())

    X = X - X.mean()
    Y = Y - Y.mean()
    hsic_xy = (X * Y).sum()
    hsic_xx = (X * X).sum()
    hsic_yy = (Y * Y).sum()
    denom = (hsic_xx * hsic_yy).clamp(min=1e-30).sqrt()
    return (hsic_xy / denom).clamp(0, 1).item()


def sequence_self_similarity(reps: torch.Tensor) -> torch.Tensor:
    if (
        aria_core is not None
        and hasattr(aria_core, "sequence_self_similarity_f32")
        and not os.environ.get("ARIA_DISABLE_NATIVE_CKA")
    ):
        native_in = _cpu_contiguous(reps.float())
        return aria_core.sequence_self_similarity_f32(native_in)

    if reps.dim() > 2:
        norm = F.normalize(reps.float(), dim=-1)
        return torch.bmm(norm, norm.transpose(1, 2)).mean(dim=0)

    norm = F.normalize(reps.float(), dim=-1)
    width = norm.shape[-1]
    flat = norm.reshape(-1, width)
    return torch.mm(flat, flat.t())


def collect_sensitivity_rows(
    x: torch.Tensor,
    embed: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor | None:
    if (
        aria_core is not None
        and hasattr(aria_core, "sensitivity_collect_f32")
        and not os.environ.get("ARIA_DISABLE_NATIVE_CKA")
    ):
        return aria_core.sensitivity_collect_f32(
            x.contiguous(),
            embed.contiguous(),
            positions.contiguous(),
        )
    return None


def mean_abs_linear_delta(
    delta: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor | None:
    if (
        aria_core is not None
        and hasattr(aria_core, "mean_abs_linear_delta_f32")
        and not os.environ.get("ARIA_DISABLE_NATIVE_CKA")
    ):
        return aria_core.mean_abs_linear_delta_f32(
            _cpu_contiguous(delta.float()),
            _cpu_contiguous(weight.float()),
        )
    return None
