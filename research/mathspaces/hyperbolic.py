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

try:
    import aria_core
    _HAS_ARIA_CORE = True
except ImportError:
    _HAS_ARIA_CORE = False


# Curvature parameter (negative curvature)
DEFAULT_C = 1.0
EPS = 1e-5


def _clamp_norm(x: torch.Tensor, max_norm: float = 1.0 - 1e-3) -> torch.Tensor:
    """Clamp vectors to stay inside the Poincare ball."""
    norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return x / norm * norm.clamp(max=max_norm)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = DEFAULT_C) -> torch.Tensor:
    """Mobius addition in the Poincare ball.

    The hyperbolic analog of vector addition. Non-commutative!
    """
    if _HAS_ARIA_CORE and x.is_contiguous() and y.is_contiguous():
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


def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor,
                        c: float = DEFAULT_C) -> torch.Tensor:
    """Poincare ball distance between two points.

    Returns per-element distances: (B, S, 1)
    """
    if _HAS_ARIA_CORE and x.is_contiguous() and y.is_contiguous():
        shape = x.shape
        x_flat = x.view(-1, shape[-1])
        y_flat = y.view(-1, shape[-1])
        out = torch.empty(x_flat.size(0), device=x.device, dtype=x.dtype)
        aria_core.hyperbolic_distance_f32(x_flat, y_flat, out, c)
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

    def __init__(self, dim: int, c: float = DEFAULT_C):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map to tangent space, transform, map back
        euclidean = log_map(x, self.c)
        transformed = F.linear(euclidean, self.weight)
        return exp_map(transformed, self.c)


# ── Primitive execution functions ─────────────────────────────────────

def execute_poincare_add(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply Mobius addition with a learnable bias in hyperbolic space."""
    if not hasattr(module, 'hyp_bias'):
        return x
    return mobius_add(x, module.hyp_bias.unsqueeze(0).unsqueeze(0))


def execute_exp_map(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Map from Euclidean to Poincare ball."""
    return exp_map(x)


def execute_log_map(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Map from Poincare ball to Euclidean."""
    return log_map(x)


def execute_hyp_distance(module: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Hyperbolic distance between two tensors."""
    return hyperbolic_distance(x, y)


def execute_hyp_linear(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic linear transformation."""
    euclidean = log_map(x)
    transformed = F.linear(euclidean, module.weight)
    return exp_map(transformed)


def execute_hyp_tangent_nonlinear(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Stable tangent-space nonlinearity in hyperbolic coordinates."""
    euclidean = log_map(x)
    bounded = torch.tanh(euclidean)
    return exp_map(bounded)


class PoincareDistanceRouting(nn.Module):
    """Routes information based on hyperbolic distance in the Poincare ball.

    Learns centroids in the Poincare ball and computes routing weights based
    on hyperbolic distance from each token to each centroid. Tokens closer
    to a centroid (in hyperbolic distance) get stronger routing to that head.

    This naturally captures hierarchical structure: centroids near the origin
    route broadly, centroids near the boundary route narrowly.

    Input: (B, S, D)
    Output: (B, S, D) — features weighted by centroid routing

    Reference: ARIA_NEXT_GEN_ARCHITECTURE.md §1.2
    """

    def __init__(self, dim: int, n_heads: int = 8, c: float = DEFAULT_C):
        super().__init__()
        self.c = c
        self.n_heads = n_heads
        self.dim = dim
        # Initialize centroids near origin for stability
        self.centroids = nn.Parameter(torch.randn(n_heads, dim) * 1e-3)
        # Per-head projection back to dim
        self.head_proj = nn.Parameter(torch.randn(n_heads, dim) / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # Clamp input and centroids inside the ball
        x_clamped = _clamp_norm(x)
        centroids = _clamp_norm(self.centroids)

        # Compute hyperbolic distance from each token to each centroid
        # Use inline Mobius addition to avoid aria_core 2D restriction
        x_neg = -x_clamped  # (B, S, D)
        x_exp = x_neg.unsqueeze(2)       # (B, S, 1, D)
        c_exp = centroids.unsqueeze(0).unsqueeze(0)  # (1, 1, H, D)

        # Inline Mobius add: (-x) +_M centroid
        x_sq = (x_exp * x_exp).sum(dim=-1, keepdim=True)
        c_sq = (c_exp * c_exp).sum(dim=-1, keepdim=True)
        xc = (x_exp * c_exp).sum(dim=-1, keepdim=True)
        c = self.c
        num = (1 + 2 * c * xc + c * c_sq) * x_exp + (1 - c * x_sq) * c_exp
        denom = (1 + 2 * c * xc + c * c * x_sq * c_sq).clamp(min=EPS)
        diff = _clamp_norm(num / denom)

        diff_norm = diff.norm(dim=-1).clamp(min=EPS)  # (B, S, H)
        sqrt_c = math.sqrt(self.c)
        dist = (2.0 / sqrt_c) * torch.atanh(
            (sqrt_c * diff_norm).clamp(max=1.0 - EPS)
        ).clamp(-10, 10)  # (B, S, H)

        # Routing weights: closer = stronger
        routing_weights = torch.softmax(-dist, dim=-1)  # (B, S, H)

        # Weighted combination via head projections
        # routing_weights: (B, S, H), head_proj: (H, D)
        out = torch.einsum('bsh,hd->bsd', routing_weights, self.head_proj)
        return x + out  # Residual connection


def execute_poincare_routing(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply Poincare distance routing with learned centroids."""
    B, S, D = x.shape
    n_heads = getattr(module, '_routing_heads', 8)
    router = PoincareDistanceRouting(dim=D, n_heads=n_heads).to(x.device)
    if hasattr(module, 'weight') and module.weight.numel() >= router.centroids.numel():
        n = router.centroids.numel()
        router.centroids.data = module.weight[:n].reshape(router.centroids.shape)
    return router(x)


def execute_hyperbolic_norm(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Normalization that respects manifold structure: log-map → LayerNorm → exp-map.

    Standard LayerNorm distorts hyperbolic geometry. This compound op
    maps to tangent space first, normalizes there, then maps back.
    """
    euclidean = log_map(x)
    if hasattr(module, 'weight') and hasattr(module, 'bias'):
        D = euclidean.shape[-1]
        weight = module.weight[:D]
        bias = module.bias[:D]
        normed = F.layer_norm(euclidean, [D], weight, bias)
    else:
        normed = F.layer_norm(euclidean, [euclidean.shape[-1]])
    return exp_map(normed)
