"""
True Token-Routing Operations

These ops implement genuine token-level routing using gather-scatter dispatch.
Unlike gated mixture ops (moe_topk, adaptive_lane_mixer, etc.) where all paths
are the same operation type with different weights, these route tokens to
fundamentally different compute types:

- hetero_moe: attention expert + conv expert + SSM expert
- arch_router: transformer-style + mamba-style + MLP-only blocks
- compute_budget_router: cheap linear + medium conv + expensive attention

Each uses the proven gather-scatter pattern from moe_topk: gate → sort by
expert → process each group through its expert type → unsort back.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compiler_op_utils import (
    _record_routing_telemetry,
)
from .compiler_ops_routing import _apply_moe_load_balance


# ── Gather-scatter dispatch helper ────────────────────────────────


def _dispatch_to_experts(
    x: torch.Tensor,
    module: nn.Module,
    n_experts: int,
    expert_fns: list[Callable],
) -> torch.Tensor:
    """Gather-scatter dispatch: gate → sort → per-expert compute → unsort.

    Args:
        x: (B, S, D) input tensor.
        module: CompiledOp with gate_weight parameter.
        n_experts: Number of expert types.
        expert_fns: List of callables, one per expert. Each takes (chunk, module) → chunk.

    Returns:
        (B, S, D) output with each token processed by its assigned expert.
    """
    B, S, D = x.shape

    if not hasattr(module, "gate_weight"):
        raise RuntimeError(
            f"{type(module).__name__} missing gate_weight for true routing op"
        )

    # 1. Gate: learned routing decision per token (pointwise — causal-safe)
    logits = F.linear(x, module.gate_weight)  # (B, S, n_experts)
    logits = _apply_moe_load_balance(module, logits, n_experts)
    weights, indices = logits.topk(1, dim=-1)  # top-1 hard routing
    weights = torch.sigmoid(weights)  # (B, S, 1) — gate confidence

    # Record telemetry
    _record_routing_telemetry(module, n_experts, indices, logits=logits)

    # 2. Batched gather-scatter dispatch: flatten B*S tokens, sort by expert,
    # run each expert on its contiguous chunk, unsort back.  O(E) Python
    # iterations instead of O(B*E).
    idx_flat = indices.reshape(-1)  # (B*S, 1) → (B*S,) after squeeze
    if idx_flat.dim() > 1:
        idx_flat = idx_flat.squeeze(-1)
    w_flat = weights.reshape(-1, 1)  # (B*S, 1)
    x_flat = x.reshape(-1, D)  # (B*S, D)

    # Sort tokens by expert assignment for contiguous expert chunks
    sort_order = idx_flat.argsort(stable=True)
    x_sorted = x_flat[sort_order]  # (B*S, D)
    w_sorted = w_flat[sort_order]  # (B*S, 1)

    # Find per-expert chunk boundaries
    expert_counts = torch.bincount(idx_flat, minlength=n_experts).tolist()
    x_chunks = x_sorted.split(expert_counts, dim=0)
    w_chunks = w_sorted.split(expert_counts, dim=0)

    # Run each expert on its contiguous chunk (E iterations, not B*E)
    result_chunks = []
    for e_idx in range(n_experts):
        if expert_counts[e_idx] == 0:
            result_chunks.append(x_sorted.new_empty(0, D))
            continue
        out = expert_fns[e_idx](x_chunks[e_idx], module)
        result_chunks.append(out.to(x.dtype) * w_chunks[e_idx].to(x.dtype))

    # Unsort back to original token order
    result_sorted = torch.cat(result_chunks, dim=0)  # (B*S, D)
    result_flat = torch.zeros_like(x_flat)
    result_flat[sort_order] = result_sorted

    return result_flat.reshape(B, S, D)


# ── Mini-expert implementations ──────────────────────────────────


def _mini_attention(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Token-wise projection (expensive tier) on a token subset.

    Uses QKV projection + output projection per-token (no cross-token attention)
    to preserve causality when tokens from different positions are grouped.

    chunk: (N, D) where N is the number of tokens routed to this expert.
    Returns: (N, D)
    """
    # QKV projection → gated combination (per-token, no cross-token interaction)
    qkv = F.linear(chunk, module.attn_qkv)  # (N, 3*D)
    q, k, v = qkv.chunk(3, dim=-1)
    # Per-token gated mixing: sigmoid(q) * v (element-wise, causal-safe)
    gate = torch.sigmoid(q * k)
    out = gate * v
    return F.linear(out, module.attn_out)


def _mini_conv(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Per-token GELU-gated projection (medium tier).

    Cross-token conv is non-causal when chunk membership depends on future
    gate decisions. Uses GELU-gated linear: proj → GELU → element-wise gate.
    Cost: 1 D×D matmul + activation + element-wise mul (~2x cheap tier).

    chunk: (N, D). Returns: (N, D)
    """
    projected = F.linear(chunk, module.conv_proj)
    return F.gelu(projected) * chunk


def _mini_ssm(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Per-token dual-gated projection with skip (expensive SSM-style tier).

    Cross-token recurrence is non-causal when chunk membership changes.
    Uses per-token B_proj gate × C_proj gate + skip connection.
    Cost: 2 D×D matmuls + activations + skip (~3x cheap tier).

    chunk: (N, D). Returns: (N, D)
    """
    gate_b = torch.sigmoid(F.linear(chunk, module.ssm_B_proj))
    gate_c = torch.sigmoid(F.linear(chunk, module.ssm_C_proj))
    return gate_b * gate_c * chunk + module.ssm_D * chunk


def _mini_mlp(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Simple up-project → GELU → down-project.

    chunk: (N, D). Returns: (N, D)
    """
    hidden = F.gelu(F.linear(chunk, module.mlp_up))
    return F.linear(hidden, module.mlp_down)


def _mini_cheap_linear(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Single linear projection (cheapest tier).

    chunk: (N, D). Returns: (N, D)
    """
    return F.linear(chunk, module.cheap_proj)


# ── Op 1: hetero_moe ─────────────────────────────────────────────


def _op_hetero_moe(module, inputs, config):
    """Heterogeneous MoE: routes tokens to attention, conv, or SSM experts."""
    x = inputs[0]
    return _dispatch_to_experts(x, module, 3, [_mini_attention, _mini_conv, _mini_ssm])


# ── Op 2: arch_router ────────────────────────────────────────────


def _mini_transformer_block(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Transformer-style: attention → linear proj."""
    out = _mini_attention(chunk, module)
    return F.linear(F.gelu(out), module.arch_ffn)


def _mini_mamba_block(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Mamba-style: conv1d → SSM → linear proj."""
    out = _mini_conv(chunk, module)
    out = _mini_ssm(out, module)
    return F.linear(out, module.arch_proj)


def _op_arch_router(module, inputs, config):
    """Architecture router: tokens choose transformer, mamba, or MLP style."""
    x = inputs[0]
    return _dispatch_to_experts(
        x,
        module,
        3,
        [_mini_transformer_block, _mini_mamba_block, _mini_mlp],
    )


# ── Op 3: compute_budget_router ──────────────────────────────────


def _op_compute_budget_router(module, inputs, config):
    """Adaptive compute budget: easy → cheap linear, medium → conv, hard → attention."""
    x = inputs[0]
    return _dispatch_to_experts(
        x,
        module,
        3,
        [_mini_cheap_linear, _mini_conv, _mini_attention],
    )


# ── Op implementations dict ──────────────────────────────────────

OP_IMPLS: Dict[str, Callable] = {
    "hetero_moe": _op_hetero_moe,
    "arch_router": _op_arch_router,
    "compute_budget_router": _op_compute_budget_router,
}
