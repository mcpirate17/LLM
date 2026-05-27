"""Native-backed metric helpers for behavioral fingerprinting."""

from __future__ import annotations

import math
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
    influence = _cpu_contiguous(influence_matrix.float())
    pos = _cpu_contiguous(positions.to(dtype=torch.int64))
    if aria_core is not None and hasattr(aria_core, "interaction_metrics_f32"):
        native = aria_core.interaction_metrics_f32(influence, pos)
        return {
            "locality": float(native[0].item()),
            "sparsity": float(native[1].item()),
            "symmetry": float(native[2].item()),
            "hierarchy": float(native[3].item()),
        }

    if influence.numel() == 0:
        return {"locality": 0.5, "sparsity": 0.5, "symmetry": 0.5, "hierarchy": 0.5}
    if influence.ndim != 2:
        influence = influence.reshape(influence.shape[0], -1)

    influence = torch.nan_to_num(influence, nan=0.0, posinf=0.0, neginf=0.0)
    n_pos, seq_len = influence.shape
    if n_pos == 0 or seq_len == 0:
        return {"locality": 0.5, "sparsity": 0.5, "symmetry": 0.5, "hierarchy": 0.5}

    row_sum = influence.sum(dim=1)
    valid_rows = row_sum > 1e-8
    if valid_rows.any():
        columns = torch.arange(seq_len, dtype=influence.dtype)
        clipped_pos = (
            pos[:n_pos].clamp(min=0, max=max(seq_len - 1, 0)).to(dtype=influence.dtype)
        )
        distances = (columns.unsqueeze(0) - clipped_pos.unsqueeze(1)).abs()
        weighted_distance = (influence * distances).sum(dim=1)
        locality_rows = 1.0 - (
            weighted_distance[valid_rows] / row_sum[valid_rows]
        ) / float(seq_len)
        locality = float(locality_rows.mean().item())
    else:
        locality = 0.5

    total = influence.sum()
    total_n = int(n_pos * seq_len)
    if float(total.item()) > 1e-8:
        probs = influence.reshape(-1) / total
        nz = probs > 1e-10
        entropy = -(probs[nz] * probs[nz].log()).sum()
    else:
        entropy = influence.new_tensor(0.0)
    max_entropy = math.log(float(total_n if total_n > 1 else 2))
    sparsity = (
        float((1.0 - entropy / max_entropy).item()) if max_entropy > 1e-8 else 0.5
    )

    sq = min(n_pos, seq_len)
    if sq >= 2:
        square = influence[:sq, :sq]
        upper = torch.triu(square, diagonal=1)
        lower_t = torch.tril(square, diagonal=-1).transpose(0, 1)
        upper_sq = upper.square().sum()
        if float(upper_sq.item()) > 1e-8:
            sym_diff_sq = (upper - lower_t).square().sum()
            symmetry = float((1.0 - sym_diff_sq.sqrt() / upper_sq.sqrt()).item())
        else:
            symmetry = 0.5
    else:
        symmetry = 0.5

    fine_var = influence.var(unbiased=False)
    if float(fine_var.item()) > 1e-10:
        pool = 4
        coarse_cols = max(seq_len // pool, 1)
        coarse_values = []
        for col_idx in range(coarse_cols):
            start = col_idx * pool
            end = min(start + pool, seq_len)
            coarse_values.append(influence[:, start:end].mean(dim=1))
        coarse = torch.stack(coarse_values, dim=1)
        hierarchy = float(
            torch.clamp(coarse.var(unbiased=False) / fine_var, max=1.0).item()
        )
    else:
        hierarchy = 0.5

    return {
        "locality": locality,
        "sparsity": sparsity,
        "symmetry": symmetry,
        "hierarchy": hierarchy,
    }


def sensitivity_metrics(sens_matrix: torch.Tensor) -> Dict[str, float]:
    matrix = _cpu_contiguous(sens_matrix.float())
    if aria_core is not None and hasattr(aria_core, "sensitivity_metrics_f32"):
        native = aria_core.sensitivity_metrics_f32(matrix)
        return {
            "spectral_norm": float(native[0].item()),
            "uniformity": float(native[1].item()),
            "effective_rank": float(native[2].item()),
        }

    if matrix.numel() == 0:
        return {"spectral_norm": 0.0, "uniformity": 0.0, "effective_rank": 0.0}
    if matrix.ndim == 1:
        matrix = matrix.unsqueeze(0)
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    singular_values = torch.linalg.svdvals(matrix)
    spectral_norm = float(singular_values[0].item()) if singular_values.numel() else 0.0
    total = singular_values.sum()
    if float(total.item()) > 0.0:
        probs = singular_values / total
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
        effective_rank = float(torch.exp(entropy).item())
    else:
        effective_rank = 0.0
    row_norms = matrix.norm(dim=-1)
    mean_norm = row_norms.mean()
    if float(mean_norm.item()) > 0.0 and row_norms.numel() > 1:
        cv = row_norms.std(unbiased=False) / mean_norm
        uniformity = float(torch.clamp(1.0 - cv, min=0.0, max=1.0).item())
    else:
        uniformity = 1.0 if float(mean_norm.item()) > 0.0 else 0.0
    return {
        "spectral_norm": spectral_norm,
        "uniformity": uniformity,
        "effective_rank": effective_rank,
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
