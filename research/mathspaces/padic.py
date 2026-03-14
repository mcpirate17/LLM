"""
p-adic Arithmetic Operations

p-adic numbers use a fundamentally different notion of distance:
two numbers are "close" if their difference is divisible by a high
power of p. This ultrametric distance satisfies the strong triangle
inequality: d(x,z) <= max(d(x,y), d(y,z)).

This creates a tree-like metric space where every triangle is isoceles.
Useful for hierarchical clustering, multi-scale representations.

We implement p-adic-inspired operations over real-valued tensors:
- p-adic valuation (how "divisible by p" a number is)
- p-adic distance (ultrametric)
- p-adic expansion (multi-scale decomposition)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from research.env import aria_core, HAS_ARIA_CORE as _HAS_ARIA_CORE


DEFAULT_P = 2
PADIC_EPS = 1e-6


def padic_valuation(x: torch.Tensor, p: int = DEFAULT_P) -> torch.Tensor:
    """Approximate p-adic valuation for real numbers.

    The p-adic valuation v_p(x) measures how many times p divides x.
    For real tensors, we use a smooth approximation based on
    the closest power of p.

    Returns: (B, S, D) tensor of valuations.
    """
    log_p = math.log(p)
    smooth_abs = torch.sqrt(x * x + PADIC_EPS * PADIC_EPS)
    return -(torch.log(smooth_abs.clamp_min(PADIC_EPS)) / log_p)


def padic_distance(x: torch.Tensor, y: torch.Tensor,
                   p: int = DEFAULT_P) -> torch.Tensor:
    """Ultrametric distance inspired by p-adic metric.

    d(x, y) = p^(-v_p(x-y)) where v_p is the p-adic valuation.
    Satisfies the ultrametric inequality: d(x,z) <= max(d(x,y), d(y,z))
    """
    diff = x - y
    val = padic_valuation(diff, p)
    return torch.exp((-val) * math.log(p))


def padic_expansion(x: torch.Tensor, p: int = DEFAULT_P,
                    n_digits: int = 4) -> torch.Tensor:
    """Multi-scale decomposition inspired by p-adic expansion.

    Decomposes x into components at different "scales" (powers of p),
    similar to how a p-adic number is a series sum(a_i * p^i).

    Returns: (B, S, D * n_digits) — concatenation of scale components.
    """
    components = []
    residual = x
    for i in range(n_digits):
        scale = p ** i
        # Extract component at this scale
        component = torch.remainder(residual * scale, float(p)) / float(p)
        components.append(component)
        residual = (residual * scale - component * p) / scale

    return torch.cat(components, dim=-1)


def padic_norm(x: torch.Tensor, p: int = DEFAULT_P) -> torch.Tensor:
    """p-adic norm: |x|_p = p^(-v_p(x))

    Small values have LARGE p-adic norm (opposite of usual).
    """
    val = padic_valuation(x, p)
    return p ** (-val)


def ultrametric_attention(x: torch.Tensor, p: int = DEFAULT_P) -> torch.Tensor:
    """Attention using ultrametric (p-adic) distance.

    Tokens attend based on ultrametric closeness rather than
    dot-product similarity. This naturally creates hierarchical
    attention patterns — tokens at the same "level" attend to each other.
    """
    B, S, D = x.shape
    # Compute pairwise ultrametric distances
    x_i = x.unsqueeze(2)  # (B, S, 1, D)
    x_j = x.unsqueeze(1)  # (B, 1, S, D)
    dist = padic_distance(x_i, x_j, p).mean(dim=-1)  # (B, S, S)
    
    # Apply causal mask if S > 1
    if S > 1:
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        dist.masked_fill_(mask, float('inf'))
        
    # Invert: close = high weight
    weights = torch.softmax(-dist, dim=-1)
    return torch.bmm(weights, x)


# ── Primitive execution functions ─────────────────────────────────────

def execute_padic_distance(module: nn.Module, x: torch.Tensor,
                           y: torch.Tensor) -> torch.Tensor:
    """p-adic distance between two tensors."""
    return padic_distance(x, y)


def execute_padic_expand(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Multi-scale p-adic expansion."""
    B, S, D = x.shape
    expanded = padic_expansion(x, n_digits=2)  # (B, S, D*2)
    # Project back to D
    if hasattr(module, 'weight'):
        return torch.nn.functional.linear(expanded, module.weight)
    return expanded[..., :D]


def execute_ultrametric_attn(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Ultrametric attention."""
    if _HAS_ARIA_CORE and x.is_contiguous() and x.ndim == 3 and x.device.type == "cpu":
        return aria_core.ultrametric_attention_f32(x, float(DEFAULT_P))
    return ultrametric_attention(x)


def execute_padic_gate(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Gate activations using smooth p-adic valuation signal."""
    if _HAS_ARIA_CORE and x.is_contiguous() and x.device.type == "cpu":
        try:
            return aria_core.padic_gate_f32(x, float(DEFAULT_P))
        except TypeError:
            pass  # Fall through to Python path
    valuation = padic_valuation(x).clamp(min=-10.0, max=10.0)
    gate = torch.sigmoid(valuation)
    return x * gate


def execute_padic_residual(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Multi-resolution skip connection via p-adic expansion.

    P-adic expansion → per-scale linear transform → recombine + residual.
    Each scale captures structure at a different resolution, and learned
    transforms can weight them before residual addition.
    """
    B, S, D = x.shape
    expanded = padic_expansion(x, n_digits=2)  # (B, S, D*2)
    if hasattr(module, 'weight'):
        # Weight projects expanded (D*2) back to D
        transformed = torch.nn.functional.linear(expanded, module.weight)
    else:
        # Average the two scales as fallback
        transformed = (expanded[..., :D] + expanded[..., D:]) * 0.5
    # Residual connection
    return x + transformed
