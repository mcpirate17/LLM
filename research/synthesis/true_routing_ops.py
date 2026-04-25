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


def _has_params(module: nn.Module, *names: str) -> bool:
    return all(hasattr(module, name) for name in names)


def _route_top1(
    x: torch.Tensor,
    module: nn.Module,
    n_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _safe_linear(x, module.gate_weight)
    logits = _apply_moe_load_balance(module, logits, n_experts)
    weights, indices = logits.topk(1, dim=-1)
    weights = torch.sigmoid(weights)
    _record_routing_telemetry(module, n_experts, indices, logits=logits)
    return weights, indices


def _select_dense_expert_outputs(
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    idx = indices.reshape(-1, 1)
    selected = torch.where(idx == 0, out0, torch.where(idx == 1, out1, out2))
    return selected * weights.reshape(-1, 1).to(selected.dtype)


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
        weights, indices = _route_top1(x, module, n_experts)

        # 2. Batched gather-scatter dispatch: flatten B*S tokens, sort by expert,
        # run each expert on its contiguous chunk, unsort back. O(E) Python
        # iterations instead of O(B*E).
        idx_flat = indices.reshape(-1)  # (B*S, 1) → (B*S,) after squeeze
        if idx_flat.dim() > 1:
            idx_flat = idx_flat.squeeze(-1)
        w_flat = weights.reshape(-1, 1)  # (B*S, 1)
        x_flat = x.reshape(-1, D)  # (B*S, D)

        if x.is_cuda:
            result_flat = torch.empty_like(x_flat)
            for e_idx, expert_fn in enumerate(expert_fns):
                token_idx = torch.nonzero(idx_flat == e_idx, as_tuple=False).flatten()
                x_chunk = x_flat.index_select(0, token_idx)
                w_chunk = w_flat.index_select(0, token_idx)
                out = expert_fn(x_chunk, module)
                if out.shape != x_chunk.shape:
                    raise RuntimeError(
                        f"true routing expert {e_idx} returned shape {tuple(out.shape)} "
                        f"for input chunk {tuple(x_chunk.shape)}"
                    )
                result_flat.index_copy_(
                    0, token_idx, out.to(x.dtype) * w_chunk.to(x.dtype)
                )
            _maybe_true_routing_sync(result_flat)
            return result_flat.reshape(B, S, D)

        # Sort tokens by expert assignment for contiguous expert chunks
        sort_order = idx_flat.argsort(stable=True)
        x_sorted = x_flat[sort_order]  # (B*S, D)
        w_sorted = w_flat[sort_order]  # (B*S, 1)

        # Find per-expert chunk boundaries. Avoid Tensor.split([..., 0, ...])
        # because repeated zero-length views have been implicated in crashes.
        expert_counts = torch.bincount(idx_flat, minlength=n_experts).tolist()

        # Run each expert on its contiguous chunk (E iterations, not B*E).
        # Fill the sorted output buffer directly; building chunk lists and
        # concatenating them adds avoidable allocation in this hot path.
        result_sorted = torch.empty_like(x_sorted)
        start = 0
        for e_idx, count in enumerate(expert_counts):
            if count == 0:
                continue
            x_chunk = x_sorted.narrow(0, start, count)
            w_chunk = w_sorted.narrow(0, start, count)
            out = expert_fns[e_idx](x_chunk, module)
            if out.shape != x_chunk.shape:
                raise RuntimeError(
                    f"true routing expert {e_idx} returned shape {tuple(out.shape)} "
                    f"for input chunk {tuple(x_chunk.shape)}"
                )
            result_sorted.narrow(0, start, count).copy_(
                out.to(x.dtype) * w_chunk.to(x.dtype)
            )
            start += count

        if start != x_sorted.shape[0]:
            raise RuntimeError(
                f"true routing chunk accounting mismatch: consumed={start} "
                f"sorted_tokens={x_sorted.shape[0]}"
            )

        inverse_sort = torch.empty_like(sort_order)
        inverse_sort[sort_order] = torch.arange(sort_order.numel(), device=x.device)
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


def _dense_cuda_attention(flat: torch.Tensor, module: nn.Module) -> torch.Tensor:
    qkv = _safe_linear(flat, module.attn_qkv)
    q, k, v = qkv.chunk(3, dim=-1)
    return _safe_linear(torch.sigmoid(q * k) * v, module.attn_out)


def _dense_cuda_conv(flat: torch.Tensor, module: nn.Module) -> torch.Tensor:
    return F.gelu(_safe_linear(flat, module.conv_proj)) * flat


def _dense_cuda_ssm(flat: torch.Tensor, module: nn.Module) -> torch.Tensor:
    gate_b = torch.sigmoid(_safe_linear(flat, module.ssm_B_proj))
    gate_c = torch.sigmoid(_safe_linear(flat, module.ssm_C_proj))
    return gate_b * gate_c * flat + module.ssm_D * flat


# ── Op 1: hetero_moe ─────────────────────────────────────────────


def _op_hetero_moe_dense_cuda(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    B, S, D = x.shape
    weights, indices = _route_top1(x, module, 3)
    flat = x.reshape(-1, D)
    out = _select_dense_expert_outputs(
        _dense_cuda_attention(flat, module),
        _dense_cuda_conv(flat, module),
        _dense_cuda_ssm(flat, module),
        indices,
        weights,
    )
    _maybe_true_routing_sync(out)
    return out.reshape(B, S, D)


def _op_hetero_moe(module, inputs, config):
    """Heterogeneous MoE: routes tokens to attention, conv, or SSM experts."""
    x = inputs[0]
    if x.is_cuda and _has_params(
        module,
        "gate_weight",
        "attn_qkv",
        "attn_out",
        "conv_proj",
        "ssm_B_proj",
        "ssm_C_proj",
        "ssm_D",
    ):
        return _op_hetero_moe_dense_cuda(module, x)
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
    if x.is_cuda and _has_params(
        module,
        "gate_weight",
        "attn_qkv",
        "attn_out",
        "arch_ffn",
        "conv_proj",
        "ssm_B_proj",
        "ssm_C_proj",
        "ssm_D",
        "arch_proj",
        "mlp_up",
        "mlp_down",
    ):
        B, S, D = x.shape
        weights, indices = _route_top1(x, module, 3)
        flat = x.reshape(-1, D)
        attn = _dense_cuda_attention(flat, module)
        transformer = _safe_linear(F.gelu(attn), module.arch_ffn)
        conv = _dense_cuda_conv(flat, module)
        ssm = _dense_cuda_ssm(conv, module)
        mamba = _safe_linear(ssm, module.arch_proj)
        mlp = _safe_linear(F.gelu(_safe_linear(flat, module.mlp_up)), module.mlp_down)
        out = _select_dense_expert_outputs(
            transformer,
            mamba,
            mlp,
            indices,
            weights,
        )
        _maybe_true_routing_sync(out)
        return out.reshape(B, S, D)
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
    if x.is_cuda and _has_params(
        module,
        "gate_weight",
        "cheap_proj",
        "conv_proj",
        "attn_qkv",
        "attn_out",
    ):
        B, S, D = x.shape
        weights, indices = _route_top1(x, module, 3)
        flat = x.reshape(-1, D)
        out = _select_dense_expert_outputs(
            _safe_linear(flat, module.cheap_proj),
            _dense_cuda_conv(flat, module),
            _dense_cuda_attention(flat, module),
            indices,
            weights,
        )
        _maybe_true_routing_sync(out)
        return out.reshape(B, S, D)
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
