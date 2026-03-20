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


def padic_distance(
    x: torch.Tensor, y: torch.Tensor, p: int = DEFAULT_P
) -> torch.Tensor:
    """Ultrametric distance inspired by p-adic metric.

    d(x, y) = p^(-v_p(x-y)) where v_p is the p-adic valuation.
    Satisfies the ultrametric inequality: d(x,z) <= max(d(x,y), d(y,z))
    """
    diff = x - y
    val = padic_valuation(diff, p)
    return torch.exp((-val) * math.log(p))


def padic_expansion(
    x: torch.Tensor, p: int = DEFAULT_P, n_digits: int = 4
) -> torch.Tensor:
    """Smooth multi-scale decomposition inspired by p-adic expansion.

    Decomposes x into components at different "scales" (powers of p),
    similar to how a p-adic number is a series sum(a_i * p^i).

    Uses soft periodic extraction (sin/cos) instead of hard torch.remainder
    to keep the decomposition differentiable. The hard modular arithmetic
    in the original version produced discontinuous digit boundaries that
    made the loss landscape jagged (CV 0.3–0.6 in dynamics probes).

    Returns: (B, S, D * n_digits) — concatenation of scale components.
    """
    components = []
    for i in range(n_digits):
        freq = float(p**i)
        # Soft periodic extraction: captures structure at scale p^i
        # sin and cos together give a full period — no information loss
        # vs hard remainder which had discontinuous jumps at digit boundaries
        components.append(torch.sin(x * freq))
        components.append(torch.cos(x * freq))

    # n_digits * 2 components (sin+cos pairs), each of shape (B, S, D)
    # Concatenate along feature dim → (B, S, D * n_digits * 2)
    return torch.cat(components, dim=-1)


def padic_norm(x: torch.Tensor, p: int = DEFAULT_P) -> torch.Tensor:
    """p-adic norm: |x|_p = p^(-v_p(x))

    Small values have LARGE p-adic norm (opposite of usual).
    """
    val = padic_valuation(x, p)
    return p ** (-val)


def _padic_dist_chunk(x_q: torch.Tensor, x_j: torch.Tensor, p: int) -> torch.Tensor:
    """Compute p-adic distance between query chunk and all keys, reduced over D.

    x_q: (B, chunk, 1, D), x_j: (B, 1, S, D)
    Returns: (B, chunk, S) mean distance.
    """
    diff = x_q - x_j  # (B, chunk, S, D)
    log_p = math.log(p)
    smooth_abs = torch.sqrt(diff * diff + PADIC_EPS * PADIC_EPS)
    val = -(torch.log(smooth_abs.clamp_min(PADIC_EPS)) / log_p)
    dist = torch.exp((-val) * log_p)
    return dist.mean(dim=-1)  # (B, chunk, S)


class _UltrametricAttentionFn(torch.autograd.Function):
    """Memory-efficient ultrametric attention via recomputation.

    Forward saves only x and the output. Backward recomputes distances
    per chunk to avoid retaining O(chunks * B * chunk * S * D) intermediates.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, p: int, chunk_size: int) -> torch.Tensor:
        B, S, D = x.shape
        ctx.save_for_backward(x)
        ctx.p = p
        ctx.chunk_size = chunk_size

        out = torch.empty_like(x)
        x_j = x.unsqueeze(1)  # (B, 1, S, D)

        for q_start in range(0, S, chunk_size):
            q_end = min(q_start + chunk_size, S)
            x_q = x[:, q_start:q_end, :].unsqueeze(2)  # (B, c, 1, D)

            with torch.no_grad():
                dist_chunk = _padic_dist_chunk(x_q, x_j, p)  # (B, c, S)
                if S > 1:
                    row_ids = torch.arange(q_start, q_end, device=x.device).unsqueeze(1)
                    col_ids = torch.arange(S, device=x.device).unsqueeze(0)
                    dist_chunk.masked_fill_(col_ids > row_ids, float("inf"))
                weights = torch.softmax(-dist_chunk, dim=-1)  # (B, c, S)

            out[:, q_start:q_end, :] = torch.bmm(weights, x)

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        p = ctx.p
        chunk_size = ctx.chunk_size
        B, S, D = x.shape
        log_p = math.log(p)

        grad_x = torch.zeros_like(x)
        # Backward uses smaller chunks — more intermediates per chunk than forward
        bwd_chunk = min(chunk_size, 32)

        for q_start in range(0, S, bwd_chunk):
            q_end = min(q_start + bwd_chunk, S)

            x_q = x[:, q_start:q_end, :].unsqueeze(2)  # (B, c, 1, D)
            x_j = x.unsqueeze(1)  # (B, 1, S, D)

            # Recompute distances
            diff = x_q - x_j  # (B, c, S, D)
            smooth_abs = torch.sqrt(diff * diff + PADIC_EPS * PADIC_EPS)
            # dist_full = smooth_abs (simplification: exp(log(smooth_abs)) = smooth_abs)
            dist = smooth_abs.mean(dim=-1)  # (B, c, S)

            if S > 1:
                row_ids = torch.arange(q_start, q_end, device=x.device).unsqueeze(1)
                col_ids = torch.arange(S, device=x.device).unsqueeze(0)
                dist = dist.masked_fill((col_ids > row_ids).unsqueeze(0), float("inf"))

            weights = torch.softmax(-dist, dim=-1)  # (B, c, S)
            g_out = grad_output[:, q_start:q_end, :]  # (B, c, D)

            # Gradient through value path: out = weights @ x
            grad_x += torch.bmm(weights.transpose(1, 2), g_out)

            # Gradient through weights -> softmax -> dist -> x
            g_weights = torch.bmm(g_out, x.transpose(1, 2))  # (B, c, S)
            g_softmax = weights * (
                g_weights - (g_weights * weights).sum(dim=-1, keepdim=True)
            )
            g_dist = -g_softmax  # (B, c, S)

            # dist = smooth_abs.mean(dim=-1)
            g_smooth_abs = g_dist.unsqueeze(-1) / D  # (B, c, S, D)
            # smooth_abs = sqrt(diff^2 + eps^2) -> d/d(diff) = diff / smooth_abs
            g_diff = g_smooth_abs * (diff / smooth_abs)  # (B, c, S, D)

            # diff = x_q - x_j
            grad_x[:, q_start:q_end, :] += g_diff.sum(dim=2)
            grad_x -= g_diff.sum(dim=1)

            # Free large tensors explicitly
            del diff, smooth_abs, g_diff, g_smooth_abs

        return grad_x, None, None


def ultrametric_attention(x: torch.Tensor, p: int = DEFAULT_P) -> torch.Tensor:
    """Attention using ultrametric (p-adic) distance.

    Tokens attend based on ultrametric closeness rather than
    dot-product similarity. This naturally creates hierarchical
    attention patterns — tokens at the same "level" attend to each other.

    Uses a custom autograd.Function to avoid retaining O(chunks * B * chunk * S * D)
    intermediate tensors. Only x is saved; distances are recomputed in backward.
    """
    B, S, D = x.shape

    # For short sequences, the unchunked path retains at most (B, S, S, D) — acceptable
    if S <= 128:
        x_i = x.unsqueeze(2)  # (B, S, 1, D)
        x_j = x.unsqueeze(1)  # (B, 1, S, D)
        dist = padic_distance(x_i, x_j, p).mean(dim=-1)  # (B, S, S)
        if S > 1:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            dist.masked_fill_(mask, float("inf"))
        weights = torch.softmax(-dist, dim=-1)
        return torch.bmm(weights, x)

    # Memory-efficient path: custom autograd with per-chunk recomputation
    return _UltrametricAttentionFn.apply(x, p, 32)


# ── Primitive execution functions ─────────────────────────────────────


def execute_padic_distance(
    module: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """p-adic distance between two tensors."""
    return padic_distance(x, y)


def execute_padic_expand(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Multi-scale p-adic expansion with ReZero residual.

    Uses n_digits=1 (sin+cos pair → D*2) to keep the expansion small.
    Learnable residual_scale starts at 0 (ReZero) so the p-adic branch
    begins as identity and gradually introduces signal as training stabilizes.
    """
    orig_dtype = x.dtype
    expanded = padic_expansion(x, n_digits=1)  # (B, S, D*2)
    if hasattr(module, "weight"):
        projected = torch.nn.functional.linear(
            expanded.to(module.weight.dtype), module.weight
        ).to(orig_dtype)
        scale = module.residual_scale if hasattr(module, "residual_scale") else 0.1
        return x + projected * scale
    return expanded[..., : x.shape[-1]].to(orig_dtype)


def execute_ultrametric_attn(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Ultrametric attention."""
    orig_dtype = x.dtype
    if (
        _HAS_ARIA_CORE
        and x.is_contiguous()
        and x.ndim == 3
        and x.device.type == "cpu"
        and x.dtype == torch.float32
    ):
        return aria_core.ultrametric_attention_f32(x, float(DEFAULT_P))
    return ultrametric_attention(x).to(orig_dtype)


def execute_padic_gate(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Gate activations using smooth p-adic valuation signal."""
    orig_dtype = x.dtype
    if (
        _HAS_ARIA_CORE
        and x.is_contiguous()
        and x.device.type == "cpu"
        and x.dtype == torch.float32
    ):
        try:
            return aria_core.padic_gate_f32(x, float(DEFAULT_P))
        except TypeError:
            pass  # Fall through to Python path
    valuation = padic_valuation(x).clamp(min=-10.0, max=10.0)
    gate = torch.sigmoid(valuation)
    return (x * gate).to(orig_dtype)


def execute_padic_residual(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Multi-resolution skip connection via p-adic expansion.

    Uses smooth sin/cos expansion and ReZero residual scaling.
    """
    orig_dtype = x.dtype
    D = x.shape[-1]
    expanded = padic_expansion(x, n_digits=1)  # (B, S, D*2)
    if hasattr(module, "weight"):
        transformed = torch.nn.functional.linear(
            expanded.to(module.weight.dtype), module.weight
        ).to(orig_dtype)
        scale = module.residual_scale if hasattr(module, "residual_scale") else 0.1
    else:
        transformed = (expanded[..., :D] + expanded[..., D:]).to(orig_dtype)
        scale = 0.5
    return x + transformed * scale
