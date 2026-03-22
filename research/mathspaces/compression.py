"""
Weight Compression Primitives

Parameterized ops that achieve similar expressivity to full D×D linear
projections with far fewer parameters. Gives the synthesis grammar
vocabulary for discovering inherently parameter-efficient layers.

Operations:
- Low-rank factored linear (rank=D/4): D²/2 params
- Block-diagonal grouped linear (4 groups): D²/4 params
- Squeeze-expand bottleneck (D→D/4→D): D²/2 params
- Shared-basis projection (8 basis vectors): 16·D params

All ops preserve (B, S, D) shape and require learnable parameters
initialized by the compiler via custom _init_params branches.

- Tied projection (down+up share transposed weights): D·r params
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def execute_low_rank_proj(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Low-rank factored linear: x @ U @ V where U:(D,r), V:(r,D), r=D//4.

    Uses D²/2 params instead of D² — captures the dominant subspace
    of a full linear projection.
    """
    if not hasattr(module, "U") or not hasattr(module, "V"):
        return x
    # x: (B, S, D), U: (D, r), V: (r, D) -> (B, S, D)
    return x @ module.U @ module.V


def execute_grouped_linear(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Block-diagonal linear with g=4 groups.

    Each group transforms D/g dims independently.
    Uses D²/4 params — captures independent feature subspace transformations.
    """
    if not hasattr(module, "weight"):
        return x
    B, S, D = x.shape
    g = module.n_groups
    group_dim = D // g
    # Truncate to evenly divisible size
    usable = group_dim * g
    x_groups = x[..., :usable].view(B, S, g, group_dim)  # (B, S, g, D/g)
    # weight: (g, D/g, D/g) — per-group linear
    out_groups = torch.einsum("bsgd,gde->bsge", x_groups, module.weight)
    out = out_groups.reshape(B, S, usable)
    # Pass through any remainder dims unchanged
    if usable < D:
        out = torch.cat([out, x[..., usable:]], dim=-1)
    return out


def execute_bottleneck_proj(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Squeeze-expand bottleneck: D → r → D with r=D//4.

    Down-project, GELU nonlinearity, up-project. Classic adapter/inverted
    residual pattern. Uses D²/2 params.
    """
    if not hasattr(module, "down") or not hasattr(module, "up"):
        return x
    # down: (r, D), up: (D, r)
    hidden = F.gelu(F.linear(x, module.down))  # (B, S, r)
    return F.linear(hidden, module.up)  # (B, S, D)


def execute_shared_basis_proj(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Shared-basis projection: x @ M @ B with M:(D,k), B:(k,D), k=8.

    Learn a small basis of k vectors and per-dimension mixing coefficients.
    Dramatically fewer params: 16·D instead of D².
    """
    if not hasattr(module, "mixing") or not hasattr(module, "basis"):
        return x
    # mixing: (D, k), basis: (k, D) -> x @ mixing @ basis = (B, S, D)
    return x @ module.mixing @ module.basis


def execute_tied_proj(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Tied down/up projection: down = W, up = W^T. Only D·r params.

    Projects D → r via W then r → D via W^T with GELU between.
    The up-projection reuses the transposed down-projection matrix,
    halving parameters vs bottleneck_proj. Classic autoencoder tie.
    """
    if not hasattr(module, "tied_weight"):
        return x
    # tied_weight: (r, D)
    hidden = F.gelu(F.linear(x, module.tied_weight))  # (B, S, r)
    return F.linear(hidden, module.tied_weight.t())  # (B, S, D)
