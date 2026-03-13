"""
Tropical Semiring Operations

In tropical algebra, addition becomes min (or max) and multiplication
becomes addition. This gives shortest-path semantics:
"tropical matmul" computes shortest-path distances between tokens.

The tropical semiring (R ∪ {+∞}, min, +) replaces:
- Standard addition → min
- Standard multiplication → +

Applications: sequence alignment, shortest paths, parsing.

Gradient fix (2026-03-12): Hard min/max kills gradient flow on the
non-selected branch.  We use log-sum-exp smooth-min:
  softmin(x, y, τ) = -τ · log(exp(-x/τ) + exp(-y/τ))
which converges to exact min as τ→0 while giving both branches
gradient proportional to their softmin weight.  τ=0.1 is small
enough to preserve tropical semantics while enabling learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from research.env import aria_core, HAS_ARIA_CORE as _HAS_ARIA_CORE

try:
    from ..synthesis.kernels import triton_tropical_matmul
    _HAS_TRITON_KERNELS = True
except ImportError:
    _HAS_TRITON_KERNELS = False


# ── Smooth min/max primitives ────────────────────────────────────────
# Hard min: gradient is 1 for selected branch, 0 for all others.
# Over a chain of tropical ops, gradient info is multiplicatively lost.
# Smooth min via log-sum-exp preserves gradient flow to both branches.

_SMOOTH_TAU: float = 0.1  # Temperature for smooth min/max


def _smooth_min(x: torch.Tensor, y: torch.Tensor, tau: float = _SMOOTH_TAU) -> torch.Tensor:
    """Smooth element-wise minimum via log-sum-exp.

    softmin(x, y, τ) = -τ · logsumexp(-x/τ, -y/τ)
    Converges to min(x, y) as τ→0.  With τ=0.1, both inputs receive
    gradient proportional to exp(-x_i/τ) / (exp(-x_i/τ) + exp(-y_i/τ)).
    """
    inv_tau = 1.0 / tau
    # Stack for logsumexp along new dim 0: shape (2, *input_shape)
    stacked = torch.stack([-x * inv_tau, -y * inv_tau], dim=0)
    return -tau * torch.logsumexp(stacked, dim=0)


def _smooth_min_dim(x: torch.Tensor, dim: int, tau: float = _SMOOTH_TAU) -> torch.Tensor:
    """Smooth minimum reduction along a dimension via log-sum-exp.

    Replaces x.min(dim=dim).values with a differentiable version.
    """
    inv_tau = 1.0 / tau
    return -tau * torch.logsumexp(-x * inv_tau, dim=dim)


def tropical_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Tropical addition: element-wise smooth minimum."""
    if _HAS_ARIA_CORE and x.is_contiguous() and y.is_contiguous() and x.device.type == "cpu":
        return aria_core.tropical_add_f32(x, y)
    return _smooth_min(x, y)


def tropical_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Tropical multiplication: element-wise standard addition."""
    return x + y


def tropical_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Tropical matrix multiplication.

    Instead of sum(a_ik * b_kj), computes min_k(a_ik + b_kj).
    Dispatch order: Triton (GPU) -> aria_core (CPU) -> chunked torch fallback.

    Input: a (B, S, D), b (B, D, S) or (B, S, D)
    Output: (B, S, S) or similar
    """
    # GPU fast path: Triton kernel
    if _HAS_TRITON_KERNELS and a.is_cuda and b.is_cuda:
        try:
            return triton_tropical_matmul(a, b)
        except Exception:
            pass

    # CPU fast path: native C kernel
    if _HAS_ARIA_CORE and a.is_contiguous() and b.is_contiguous() and a.ndim == 3 and b.ndim == 3 and a.device.type == "cpu":
        B, S, D = a.shape
        _, D2, _ = b.shape
        if D == D2:
            return torch.stack([aria_core.tropical_matmul_f32(a[i], b[i]) for i in range(B)])

    # Chunked torch fallback — avoids O(S^2 * D) peak memory
    B, S1, D1 = a.shape
    if b.ndim == 3 and b.shape[1] == D1:
        S2 = b.shape[2]
        b_val = b.transpose(1, 2)
    else:
        S2 = b.shape[1]
        b_val = b

    result = torch.empty((B, S1, S2), device=a.device, dtype=a.dtype)
    b_expanded = b_val.unsqueeze(1)  # (B, 1, S2, D)

    chunk_size = 32
    for i in range(0, S1, chunk_size):
        end = min(i + chunk_size, S1)
        a_chunk = a[:, i:end, :].unsqueeze(2)  # (B, chunk, 1, D)
        pairwise = a_chunk + b_expanded        # (B, chunk, S2, D)
        # Smooth min: all D dimensions contribute gradient, not just argmin
        result[:, i:end, :] = _smooth_min_dim(pairwise, dim=-1)

    return result


def tropical_softmax(x: torch.Tensor, dim: int = -1,
                     temperature: float = 0.1) -> torch.Tensor:
    """Smooth approximation of tropical (min) using low-temperature softmax.

    As temperature -> 0, softmin -> argmin (tropical behavior).
    Temperature scales adaptively with sqrt(S/128) to prevent
    numerical underflow for long sequences.
    """
    # Adaptive: scale temperature with sqrt(S/128) to prevent
    # numerical underflow for long sequences
    S = x.shape[1] if x.ndim >= 2 else 1
    adaptive_t = temperature * max(1.0, (S / 128.0) ** 0.5)
    return torch.softmax(-x / adaptive_t, dim=dim)


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
    
    # Apply causal mask if S > 1
    S = q.shape[1]
    if S > 1:
        mask = torch.triu(torch.ones(S, S, device=q.device), diagonal=1).bool()
        distances.masked_fill_(mask, float('inf'))
        
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
    
    # Apply causal mask if S > 1
    if S > 1:
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        # For tropical softmax (distance-based), we set future distances to infinity
        scores.masked_fill_(mask, float('inf'))
        
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
    """Center features by tropical (min) sequence baseline.

    Uses smooth cumulative min to preserve gradient flow to all
    preceding tokens, not just the argmin position.
    """
    if _HAS_ARIA_CORE and x.is_contiguous() and x.ndim == 3 and x.device.type == "cpu":
        return aria_core.tropical_center_f32(x)
    # Smooth causal min centering via log-sum-exp scan
    B, S, D = x.shape
    inv_tau = 1.0 / _SMOOTH_TAU
    # Compute smooth cumulative min in log-space
    # log_acc[t] = logsumexp(-x[0..t] / τ), then smooth_cmin[t] = -τ * log_acc[t]
    neg_x_scaled = -x * inv_tau  # (B, S, D)
    # cumulative logsumexp along dim=1
    log_acc = neg_x_scaled[:, :1, :]  # (B, 1, D) — first token
    chunks = [log_acc]
    for t in range(1, S):
        log_acc = torch.logaddexp(log_acc, neg_x_scaled[:, t:t+1, :])
        chunks.append(log_acc)
    cmin_smooth = -_SMOOTH_TAU * torch.cat(chunks, dim=1)  # (B, S, D)
    return x - cmin_smooth


def execute_tropical_gate(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Shortest-path routing as a gating mechanism.

    Tropical distance scores → sigmoid gate → elementwise multiply
    with linear projection. Routes information based on tropical
    (shortest-path) proximity rather than learned attention.
    """
    B, S, D = x.shape
    # Tropical distance scores: pairwise min-plus distances
    distances = tropical_matmul(x, x)  # (B, S, S)
    
    # Apply causal mask if S > 1
    if S > 1:
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        distances.masked_fill_(mask, float('inf'))
        
    gate_scores = tropical_softmax(distances, dim=-1)  # (B, S, S)
    gated = torch.bmm(gate_scores, x)  # (B, S, D)
    # Linear projection if params available
    if hasattr(module, 'weight'):
        gated = torch.nn.functional.linear(gated, module.weight)
    # Sigmoid gate blending with residual
    gate = torch.sigmoid(gated)
    return x * gate
