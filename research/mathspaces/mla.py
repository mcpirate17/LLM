"""
Multi-Head Latent Attention (MLA)

DeepSeek V2/V3's asymmetric KV-cache compression, per
external_research_2026-05-10.md §1.1.

Standard multi-head attention caches K and V at full dimension D, which
dominates memory at long contexts. MLA caches only a shared low-rank
latent of size d_latent ≪ d_model, and decompresses K and V at attention
time via separate up-projections. The asymmetry is the key — K and V
share the *down* path but use distinct *up* paths.

Forward:

    kv_latent = kv_input @ W_down              # (B, S, d_latent)
    K = kv_latent @ W_up_K                     # (B, S, D)
    V = kv_latent @ W_up_V                     # (B, S, D)
    scores = (query @ K^T) / sqrt(D)           # (B, S, S)
    scores = mask_causal(scores)
    probs = softmax(scores, dim=-1)
    out = probs @ V                            # (B, S, D)

Param count: D·d_latent (down) + 2·d_latent·D (up_K + up_V) =
3·D·d_latent. At d_latent = D/8 that's 3·D²/8 = 0.375·D² — vs a
standard attention block's D² (Q/K/V) = 1.0·D². ~62% fewer attention
params, plus the ~93% KV cache reduction that makes MLA worth using.

Hot path is pure torch primitives — matmul, softmax, bmm, masked_fill —
which dispatch to native C++/CUDA kernels. No custom Triton kernel is
warranted at this granularity; for a future fused MLA + RoPE kernel,
add against the existing aria_core abi.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ._utils import causal_mask


_DEFAULT_LATENT_DIV = 8  # d_latent = d_model // 8 by default


def execute_mla_attention(
    module: nn.Module, query: torch.Tensor, kv_input: torch.Tensor
) -> torch.Tensor:
    """Asymmetric shared-latent attention. Identity if module not parameterized."""
    if not hasattr(module, "W_down"):
        return query

    B, S, D = query.shape
    if kv_input.shape != query.shape:
        # Allow broadcastable kv_input; otherwise fall back to self-attention.
        try:
            kv_input = kv_input.expand_as(query)
        except RuntimeError:
            kv_input = query

    dtype = query.dtype
    W_down = module.W_down.to(dtype)
    W_up_K = module.W_up_K.to(dtype)
    W_up_V = module.W_up_V.to(dtype)

    kv_latent = kv_input @ W_down  # (B, S, d_latent)
    K = kv_latent @ W_up_K  # (B, S, D)
    V = kv_latent @ W_up_V  # (B, S, D)

    scale = 1.0 / math.sqrt(max(D, 1))
    scores = torch.bmm(query, K.transpose(-2, -1)) * scale  # (B, S, S)
    if S > 1:
        scores = scores.masked_fill(causal_mask(S, query.device), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.bmm(probs, V)  # (B, S, D)
