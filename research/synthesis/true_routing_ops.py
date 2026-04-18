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

import logging
import os
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compiler_op_utils import (
    _record_routing_telemetry,
    _safe_linear,
)
from .compiler_ops_routing import _apply_moe_load_balance

logger = logging.getLogger(__name__)


def _true_routing_flag(name: str) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _maybe_true_routing_sync(x: torch.Tensor) -> None:
    if x.is_cuda and _true_routing_flag("ARIA_TRUE_ROUTING_SYNC"):
        torch.cuda.synchronize(x.device)


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
    debug_enabled = _true_routing_flag("ARIA_TRUE_ROUTING_DEBUG")

    if not hasattr(module, "gate_weight"):
        raise RuntimeError(
            f"{type(module).__name__} missing gate_weight for true routing op"
        )

    try:
        _maybe_true_routing_sync(x)

        # 1. Gate: learned routing decision per token (pointwise — causal-safe)
        logits = _safe_linear(x, module.gate_weight)  # (B, S, n_experts)
        logits = _apply_moe_load_balance(module, logits, n_experts)
        weights, indices = logits.topk(1, dim=-1)  # top-1 hard routing
        weights = torch.sigmoid(weights)  # (B, S, 1) — gate confidence

        # Record telemetry
        _record_routing_telemetry(module, n_experts, indices, logits=logits)

        # 2. Batched gather-scatter dispatch: flatten B*S tokens, sort by expert,
        # run each expert on its contiguous chunk, unsort back. O(E) Python
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

        # Find per-expert chunk boundaries. Avoid Tensor.split([..., 0, ...])
        # because repeated zero-length views have been implicated in crashes.
        expert_counts = torch.bincount(idx_flat, minlength=n_experts).tolist()

        # Run each expert on its contiguous chunk (E iterations, not B*E)
        result_chunks = []
        start = 0
        for e_idx, count in enumerate(expert_counts):
            if count == 0:
                result_chunks.append(x_sorted.new_empty(0, D))
                continue
            x_chunk = x_sorted.narrow(0, start, count)
            w_chunk = w_sorted.narrow(0, start, count)
            start += count
            out = expert_fns[e_idx](x_chunk, module)
            if out.shape != x_chunk.shape:
                raise RuntimeError(
                    f"true routing expert {e_idx} returned shape {tuple(out.shape)} "
                    f"for input chunk {tuple(x_chunk.shape)}"
                )
            result_chunks.append(out.to(x.dtype) * w_chunk.to(x.dtype))

        if start != x_sorted.shape[0]:
            raise RuntimeError(
                f"true routing chunk accounting mismatch: consumed={start} "
                f"sorted_tokens={x_sorted.shape[0]}"
            )

        # Unsort back to original token order without indexed assignment.
        result_sorted = torch.cat(result_chunks, dim=0)  # (B*S, D)
        inverse_sort = sort_order.argsort(stable=True)
        result_flat = result_sorted[inverse_sort]
        _maybe_true_routing_sync(result_flat)
        return result_flat.reshape(B, S, D)
    except Exception:
        logger.exception(
            "true routing dispatch failed op=%s shape=%s device=%s dtype=%s sync=%s debug=%s",
            getattr(module, "op_name", type(module).__name__),
            tuple(x.shape),
            x.device,
            x.dtype,
            _true_routing_flag("ARIA_TRUE_ROUTING_SYNC"),
            debug_enabled,
            extra={
                "routing_n_experts": n_experts,
                "routing_input_shape": tuple(x.shape),
            },
        )
        if debug_enabled:
            with torch.no_grad():
                logger.error(
                    "true routing failure context expert_counts=%s gate_shape=%s",
                    torch.bincount(
                        indices.reshape(-1).squeeze(-1)
                        if "indices" in locals()
                        else torch.zeros(0, device=x.device, dtype=torch.long),
                        minlength=n_experts,
                    ).tolist(),
                    tuple(getattr(module, "gate_weight").shape),
                )
        raise


# ── Mini-expert implementations ──────────────────────────────────


def _mini_attention(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Token-wise projection (expensive tier) on a token subset.

    Uses QKV projection + output projection per-token (no cross-token attention)
    to preserve causality when tokens from different positions are grouped.

    chunk: (N, D) where N is the number of tokens routed to this expert.
    Returns: (N, D)
    """
    # QKV projection → gated combination (per-token, no cross-token interaction)
    qkv = _safe_linear(chunk, module.attn_qkv)  # (N, 3*D)
    q, k, v = qkv.chunk(3, dim=-1)
    # Per-token gated mixing: sigmoid(q) * v (element-wise, causal-safe)
    gate = torch.sigmoid(q * k)
    out = gate * v
    return _safe_linear(out, module.attn_out)


def _mini_conv(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Per-token GELU-gated projection (medium tier).

    Cross-token conv is non-causal when chunk membership depends on future
    gate decisions. Uses GELU-gated linear: proj → GELU → element-wise gate.
    Cost: 1 D×D matmul + activation + element-wise mul (~2x cheap tier).

    chunk: (N, D). Returns: (N, D)
    """
    projected = _safe_linear(chunk, module.conv_proj)
    return F.gelu(projected) * chunk


def _mini_ssm(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Per-token dual-gated projection with skip (expensive SSM-style tier).

    Cross-token recurrence is non-causal when chunk membership changes.
    Uses per-token B_proj gate × C_proj gate + skip connection.
    Cost: 2 D×D matmuls + activations + skip (~3x cheap tier).

    chunk: (N, D). Returns: (N, D)
    """
    gate_b = torch.sigmoid(_safe_linear(chunk, module.ssm_B_proj))
    gate_c = torch.sigmoid(_safe_linear(chunk, module.ssm_C_proj))
    return gate_b * gate_c * chunk + module.ssm_D * chunk


def _mini_mlp(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Simple up-project → GELU → down-project.

    chunk: (N, D). Returns: (N, D)
    """
    hidden = F.gelu(_safe_linear(chunk, module.mlp_up))
    return _safe_linear(hidden, module.mlp_down)


def _mini_cheap_linear(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Single linear projection (cheapest tier).

    chunk: (N, D). Returns: (N, D)
    """
    return _safe_linear(chunk, module.cheap_proj)


# ── Op 1: hetero_moe ─────────────────────────────────────────────


def _op_hetero_moe(module, inputs, config):
    """Heterogeneous MoE: routes tokens to attention, conv, or SSM experts."""
    x = inputs[0]
    return _dispatch_to_experts(x, module, 3, [_mini_attention, _mini_conv, _mini_ssm])


# ── Op 2: arch_router ────────────────────────────────────────────


def _mini_transformer_block(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Transformer-style: attention → linear proj."""
    out = _mini_attention(chunk, module)
    return _safe_linear(F.gelu(out), module.arch_ffn)


def _mini_mamba_block(chunk: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Mamba-style: conv1d → SSM → linear proj."""
    out = _mini_conv(chunk, module)
    out = _mini_ssm(out, module)
    return _safe_linear(out, module.arch_proj)


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
