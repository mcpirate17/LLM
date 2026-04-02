"""
Hyperbolic Geometry Operations (Poincare Ball Model)

In hyperbolic space, distances grow exponentially — ideal for modeling
hierarchical structures (trees, taxonomies, part-whole relationships).

Operations:
- Mobius addition (non-commutative, non-associative vector addition)
- Exponential map (Euclidean -> Hyperbolic)
- Logarithmic map (Hyperbolic -> Euclidean)
- Hyperbolic distance
- Hyperbolic linear (gyroplane)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.env import aria_core, HAS_ARIA_CORE as _HAS_ARIA_CORE


# Curvature parameter (negative curvature)
DEFAULT_C = 1.0
EPS = 1e-5
MAX_CURVATURE = 10.0


def _curvature_raw_init(c: float) -> float:
    clamped = min(max(float(c), EPS), MAX_CURVATURE)
    return math.log(math.expm1(clamped))


def _positive_curvature(raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw).clamp(min=EPS, max=MAX_CURVATURE)


def _module_curvature(module: nn.Module | None, default: float = DEFAULT_C) -> float:
    raw = getattr(module, "_c_raw", None)
    if isinstance(raw, torch.Tensor):
        return float(_positive_curvature(raw).item())
    value = getattr(module, "c", default)
    return float(min(max(value, EPS), MAX_CURVATURE))


def _clamp_norm(x: torch.Tensor, max_norm: float = 1.0 - 1e-3) -> torch.Tensor:
    """Clamp vectors to stay inside the Poincare ball."""
    norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    scale = (max_norm / norm).clamp(max=1.0)
    return x * scale


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = DEFAULT_C) -> torch.Tensor:
    """Mobius addition in the Poincare ball.

    The hyperbolic analog of vector addition. Non-commutative!
    """
    if (
        _HAS_ARIA_CORE
        and x.is_contiguous()
        and y.is_contiguous()
        and x.device.type == "cpu"
    ):
        shape = x.shape
        x_flat = x.view(-1, shape[-1])
        y_flat = y.view(-1, shape[-1])
        out = torch.empty_like(x_flat)
        aria_core.hyperbolic_mobius_add_f32(x_flat, y_flat, out, c)
        return out.view(shape)

    x = _clamp_norm(x)
    y = _clamp_norm(y)

    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
    denom = 1 + 2 * c * xy + c * c * x_sq * y_sq
    return _clamp_norm(num / denom.clamp(min=EPS))


def exp_map(v: torch.Tensor, c: float = DEFAULT_C) -> torch.Tensor:
    """Exponential map: tangent space (Euclidean) -> Poincare ball.

    Maps a Euclidean vector to a point on the hyperbolic manifold.
    """
    sqrt_c = math.sqrt(c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return _clamp_norm(torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm))


def log_map(y: torch.Tensor, c: float = DEFAULT_C) -> torch.Tensor:
    """Logarithmic map: Poincare ball -> tangent space (Euclidean).

    Maps a hyperbolic point back to Euclidean space.
    """
    y = _clamp_norm(y)
    sqrt_c = math.sqrt(c)
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return torch.atanh(sqrt_c * y_norm).clamp(-10, 10) * y / (sqrt_c * y_norm)


def hyperbolic_distance(
    x: torch.Tensor, y: torch.Tensor, c: float = DEFAULT_C
) -> torch.Tensor:
    """Poincare ball distance between two points.

    Returns per-element distances: (B, S, 1)
    """
    if (
        _HAS_ARIA_CORE
        and x.is_contiguous()
        and y.is_contiguous()
        and x.device.type == "cpu"
    ):
        shape = x.shape
        x_flat = x.view(-1, shape[-1])
        y_flat = y.view(-1, shape[-1])
        out = aria_core.hyperbolic_distance_f32(x_flat, y_flat, c)
        return out.view(*shape[:-1], 1)

    x = _clamp_norm(x)
    y = _clamp_norm(y)
    diff = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return (2.0 / math.sqrt(c)) * torch.atanh(math.sqrt(c) * diff_norm).clamp(-10, 10)


class HyperbolicLinear(nn.Module):
    """Linear transformation in hyperbolic space.

    Maps through: log_map -> Euclidean linear -> exp_map
    """

    __slots__ = ()

    def __init__(self, dim: int, c: float = DEFAULT_C):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))
        self._c_raw = nn.Parameter(
            torch.tensor(_curvature_raw_init(c), dtype=torch.float32)
        )

    @property
    def c(self) -> float:
        return float(_positive_curvature(self._c_raw).item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map to tangent space, transform, map back
        c = self.c
        euclidean = log_map(x, c=c)
        transformed = F.linear(euclidean, self.weight.to(euclidean.dtype))
        return exp_map(transformed, c=c)


# ── Primitive execution functions ─────────────────────────────────────


def execute_poincare_add(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply Mobius addition with a learnable bias in hyperbolic space."""
    if not hasattr(module, "hyp_bias"):
        return x
    return mobius_add(
        x, module.hyp_bias.unsqueeze(0).unsqueeze(0), c=_module_curvature(module)
    )


def execute_exp_map(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Map from Euclidean to Poincare ball."""
    return exp_map(x, c=_module_curvature(module))


def execute_log_map(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Map from Poincare ball to Euclidean."""
    return log_map(x, c=_module_curvature(module))


def execute_hyp_distance(
    module: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Hyperbolic distance between two tensors."""
    return hyperbolic_distance(x, y, c=_module_curvature(module))


def execute_hyp_linear(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic linear transformation."""
    c = _module_curvature(module)
    euclidean = log_map(x, c=c)
    transformed = F.linear(euclidean, module.weight.to(euclidean.dtype))
    return exp_map(transformed, c=c)


def execute_hyp_tangent_nonlinear(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Stable tangent-space nonlinearity in hyperbolic coordinates."""
    c = _module_curvature(module)
    euclidean = log_map(x, c=c)
    bounded = torch.tanh(euclidean)
    return exp_map(bounded, c=c)


def execute_hyperbolic_norm(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Normalization that respects manifold structure: log-map → LayerNorm → exp-map.

    Standard LayerNorm distorts hyperbolic geometry. This compound op
    maps to tangent space first, normalizes there, then maps back.
    """
    c = _module_curvature(module)
    euclidean = log_map(x, c=c)
    if hasattr(module, "weight") and hasattr(module, "bias"):
        D = euclidean.shape[-1]
        weight = module.weight[:D]
        bias = module.bias[:D]
        normed = F.layer_norm(euclidean, [D], weight, bias)
    else:
        normed = F.layer_norm(euclidean, [euclidean.shape[-1]])
    return exp_map(normed, c=c)
