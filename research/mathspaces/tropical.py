"""
Tropical Semiring Operations

In tropical algebra, addition becomes min (or max) and multiplication
becomes addition. This gives shortest-path semantics:
"tropical matmul" computes shortest-path distances between tokens.

The tropical semiring (R ∪ {+∞}, min, +) replaces:
- Standard addition → min
- Standard multiplication → +

Applications: sequence alignment, shortest paths, parsing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math


def tropical_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Tropical addition: element-wise minimum."""
    return torch.minimum(x, y)


def tropical_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Tropical multiplication: element-wise standard addition."""
    return x + y


def tropical_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Tropical matrix multiplication.

    Instead of sum(a_ik * b_kj), computes min_k(a_ik + b_kj).
    This gives shortest-path distances: if a[i,k] is the distance from
    token i to intermediate k, and b[k,j] from k to j, then the result
    is the shortest path from i to j through any intermediate.

    Input: a (B, S, D), b (B, D, S) or (B, S, D)
    Output: (B, S, S) or similar
    """
    # a: (B, S, D), b: (B, S, D) -> we want (B, S, S) min-plus
    B, S, D = a.shape
    # a_ik + b_jk for all i,j via broadcasting
    # a: (B, S, 1, D), b: (B, 1, S, D)
    expanded_a = a.unsqueeze(2)  # (B, S, 1, D)
    expanded_b = b.unsqueeze(1)  # (B, 1, S, D)
    pairwise = expanded_a + expanded_b  # (B, S, S, D)
    # Min over D dimension (the "sum" in tropical semiring)
    result = pairwise.min(dim=-1).values  # (B, S, S)
    return result


def tropical_softmax(x: torch.Tensor, dim: int = -1,
                     temperature: float = 0.1) -> torch.Tensor:
    """Smooth approximation of tropical (min) using low-temperature softmax.

    As temperature -> 0, softmin -> argmin (tropical behavior).
    """
    return torch.softmax(-x / temperature, dim=dim)


def tropical_attention(q: torch.Tensor, k: torch.Tensor,
                       v: torch.Tensor) -> torch.Tensor:
    """Tropical attention: shortest-path distance as attention weights.

    Instead of softmax(QK^T/sqrt(d))V, computes:
    1. Tropical distance matrix between Q and K
    2. Softmin to get weights (closest = highest weight)
    3. Standard weighted sum of V

    This makes tokens attend to their "nearest neighbors" in a
    shortest-path sense rather than highest-dot-product sense.
    """
    # Distance matrix via tropical matmul
    distances = tropical_matmul(q, k)  # (B, S, S)
    # Softmin: attend to closest tokens
    weights = tropical_softmax(distances, dim=-1)  # (B, S, S)
    # Standard value aggregation
    return torch.bmm(weights, v)  # (B, S, D)


# ── Primitive execution functions ─────────────────────────────────────

def execute_tropical_matmul(module: nn.Module, x: torch.Tensor,
                            y: torch.Tensor) -> torch.Tensor:
    """Tropical matmul then project back to D dim."""
    B, S, D = x.shape
    scores = tropical_matmul(x, y)  # (B, S, S)
    weights = tropical_softmax(scores, dim=-1)
    return torch.bmm(weights, y)  # (B, S, D)


def execute_tropical_add(module: nn.Module, x: torch.Tensor,
                         y: torch.Tensor) -> torch.Tensor:
    """Element-wise tropical addition (min)."""
    return tropical_add(x, y)


def execute_tropical_attention(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Self-attention using tropical geometry."""
    # Simple: Q=K=V=x with learned projections
    if hasattr(module, 'weight'):
        q = torch.nn.functional.linear(x, module.weight)
    else:
        q = x
    return tropical_attention(q, x, x)


def execute_tropical_center(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Center features by tropical (min) sequence baseline."""
    baseline = x.amin(dim=1, keepdim=True)
    return x - baseline


def execute_tropical_gate(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Shortest-path routing as a gating mechanism.

    Tropical distance scores → sigmoid gate → elementwise multiply
    with linear projection. Routes information based on tropical
    (shortest-path) proximity rather than learned attention.
    """
    B, S, D = x.shape
    # Tropical distance scores: pairwise min-plus distances
    distances = tropical_matmul(x, x)  # (B, S, S)
    gate_scores = tropical_softmax(distances, dim=-1)  # (B, S, S)
    gated = torch.bmm(gate_scores, x)  # (B, S, D)
    # Linear projection if params available
    if hasattr(module, 'weight'):
        gated = torch.nn.functional.linear(gated, module.weight)
    # Sigmoid gate blending with residual
    gate = torch.sigmoid(gated)
    return x * gate
