from __future__ import annotations

import math
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from research.defaults import ROPE_THETA_BASE
from .compiler_op_utils import (
    HAS_KERNELS,
    aria_core,
    kernels,
    _c,
    record_kernel_fallback,
)


def _op_softmax_attention(module, inputs, _):
    """Standard causal multi-head softmax attention."""
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, _ = x.shape
    nh, hd = module.n_heads, module.head_dim
    q = module.q_proj(x).reshape(B, S, nh, hd).transpose(1, 2)
    k = module.k_proj(x).reshape(B, S, nh, hd).transpose(1, 2)
    v = module.v_proj(x).reshape(B, S, nh, hd).transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=True,
        scale=module.attn_scale,
    )
    out = out.transpose(1, 2).reshape(B, S, -1)
    return module.o_proj(out)


def _op_linear_attention(module, inputs, _):
    """Linear attention with ELU kernel (O(S*D) complexity).

    Causal linear attention: instead of materializing (B, H, S, D, D) outer
    products, use a running state matrix S_t = sum_{i<=t} k_i^T v_i and
    compute out_t = q_t @ S_t.  Sequential scan over S but O(D^2) per step
    instead of O(S*D^2) for the full cumsum approach.
    """
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, _ = x.shape
    nh, hd = module.n_heads, module.head_dim
    q = (
        F.elu(module.q_proj(x).reshape(B, S, nh, hd).transpose(1, 2)) + 1
    )  # (B, H, S, hd)
    k = F.elu(module.k_proj(x).reshape(B, S, nh, hd).transpose(1, 2)) + 1
    v = module.v_proj(x).reshape(B, S, nh, hd).transpose(1, 2)

    # Chunked causal linear attention: accumulate kv state across chunks
    # to avoid the (B, H, S, hd, hd) intermediate tensor.
    CHUNK = 32
    # Running state: S_t = sum_{i<=t} k_i^T @ v_i, shape (B, H, hd, hd)
    state = torch.zeros(B, nh, hd, hd, device=x.device, dtype=x.dtype)
    # Running key sum: z_t = sum_{i<=t} k_i, shape (B, H, hd)
    z_state = torch.zeros(B, nh, hd, device=x.device, dtype=x.dtype)

    out_chunks = []
    for c_start in range(0, S, CHUNK):
        c_end = min(c_start + CHUNK, S)
        q_c = q[:, :, c_start:c_end]  # (B, H, C, hd)
        k_c = k[:, :, c_start:c_end]
        v_c = v[:, :, c_start:c_end]

        # Within-chunk: compute causal kv cumsum for this chunk
        kv_c = torch.einsum("bhcd,bhce->bhcde", k_c, v_c)  # (B, H, C, hd, hd)
        kv_cum_c = kv_c.cumsum(dim=2)  # causal within chunk
        k_cum_c = k_c.cumsum(dim=2)

        # Add carried-over state from previous chunks
        total_kv = kv_cum_c + state.unsqueeze(2)  # (B, H, C, hd, hd)
        total_z = k_cum_c + z_state.unsqueeze(2)  # (B, H, C, hd)

        # Compute output for this chunk
        out_c = torch.einsum("bhcd,bhcde->bhce", q_c, total_kv)
        denom_c = (
            torch.einsum("bhcd,bhcd->bhc", q_c, total_z).unsqueeze(-1).clamp(min=1e-6)
        )
        out_chunks.append(out_c / denom_c)

        # Update running state with this chunk's total contribution
        state = state + kv_c.sum(dim=2)
        z_state = z_state + k_c.sum(dim=2)

    out = torch.cat(out_chunks, dim=2)  # (B, H, S, hd)
    return module.o_proj(out.transpose(1, 2).reshape(B, S, -1))


def _op_graph_attention(module, inputs, _):
    """Graph attention with learned edge features + causal softmax attention."""
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, _ = x.shape
    nh, hd = module.n_heads, module.head_dim
    x_e = x + module.edge_proj(x)
    q = module.q_proj(x_e).reshape(B, S, nh, hd).transpose(1, 2)
    k = module.k_proj(x_e).reshape(B, S, nh, hd).transpose(1, 2)
    v = module.v_proj(x_e).reshape(B, S, nh, hd).transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=True,
        scale=module.attn_scale,
    )
    out = out.transpose(1, 2).reshape(B, S, -1)
    return module.o_proj(out)


def _op_rmsnorm(module, inputs, _):
    if not hasattr(module, "weight"):
        return inputs[0]
    x = inputs[0]
    if _c(x):
        return aria_core.rmsnorm_f32(x, module.weight, 1e-6)
    if HAS_KERNELS and x.is_cuda:
        try:
            return kernels.triton_rmsnorm(x, module.weight)
        except (ImportError, RuntimeError, AttributeError) as e:
            record_kernel_fallback("triton_rmsnorm", e)
    eps = 1e-6
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return (x / rms) * module.weight


def _op_layernorm(module, inputs, _):
    if not hasattr(module, "weight"):
        return inputs[0]
    x = inputs[0]
    if _c(x):
        return aria_core.layernorm_f32(x, module.weight, module.bias, 1e-5)
    return F.layer_norm(x, [x.shape[-1]], module.weight, module.bias)


def _op_gated_linear(module, inputs, _):
    if not hasattr(module, "linear_weight"):
        return inputs[0]
    x = inputs[0]
    if _c(x) and x.ndim == 3:
        out = aria_core.gated_linear_f32(
            x,
            module.linear_weight,
            module.linear_bias,
            module.gate_weight,
            module.gate_bias,
        )
        if out.ndim == x.ndim:
            return out
    linear = F.linear(x, module.linear_weight, module.linear_bias)
    gate = torch.sigmoid(F.linear(x, module.gate_weight, module.gate_bias))
    return linear * gate


def _op_embedding_lookup(module, inputs, _):
    """Learnable codebook projection: soft-lookup into learned prototypes.

    Projects continuous (B, S, D) input into codebook similarity space,
    then reconstructs via weighted sum of codebook entries.
    Acts as a learnable discretization / vector quantization layer.
    """
    x = inputs[0]
    if not hasattr(module, "codebook"):
        return x
    # (V, D) codebook, (B, S, D) input → (B, S, V) similarity
    codebook = module.codebook  # (V, D)
    sim = torch.matmul(x, codebook.t())  # (B, S, V)
    weights = torch.softmax(sim, dim=-1)  # (B, S, V)
    # Weighted codebook lookup → (B, S, D)
    looked_up = torch.matmul(weights, codebook)  # (B, S, D)
    # Project back through learnable transform
    return torch.matmul(looked_up, module.codebook_proj)  # (B, S, D)


class _RopeRotateC(torch.autograd.Function):
    """C-accelerated RoPE rotation with analytical backward (rotate by -angle)."""

    __slots__ = ()

    @staticmethod
    def forward(ctx, x, theta_base):
        ctx.theta_base = theta_base
        ctx.shape = x.shape
        return aria_core.rope_rotate_f32(x.detach().contiguous(), theta_base)

    @staticmethod
    def backward(ctx, grad_output):
        B, S, D = ctx.shape
        g = grad_output.contiguous()
        pos = torch.arange(S, device=g.device, dtype=g.dtype).unsqueeze(1)
        freqs = 1.0 / (
            ctx.theta_base
            ** (torch.arange(0, D, 2, device=g.device, dtype=g.dtype) / D)
        )
        angles = pos * freqs.unsqueeze(0)
        cos_a = torch.cos(angles).unsqueeze(0)
        sin_a = torch.sin(angles).unsqueeze(0)
        g1, g2 = g[..., 0::2], g[..., 1::2]
        grad_in = torch.zeros_like(g)
        # Inverse rotation (transpose of rotation matrix)
        grad_in[..., 0::2] = g1 * cos_a + g2 * sin_a
        grad_in[..., 1::2] = -g1 * sin_a + g2 * cos_a
        return grad_in, None


def _op_rope_rotate(_, inputs, __):
    x = inputs[0]
    if _c(x) and hasattr(aria_core, "rope_rotate_f32"):
        return _RopeRotateC.apply(x, ROPE_THETA_BASE)
    B, S, D = x.shape
    pos = torch.arange(S, device=x.device, dtype=x.dtype).unsqueeze(1)
    freqs = 1.0 / (
        ROPE_THETA_BASE ** (torch.arange(0, D, 2, device=x.device, dtype=x.dtype) / D)
    )
    angles = pos * freqs.unsqueeze(0)
    cos_a = torch.cos(angles).unsqueeze(0)
    sin_a = torch.sin(angles).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.zeros_like(x)
    out[..., 0::2] = x1 * cos_a - x2 * sin_a
    out[..., 1::2] = x1 * sin_a + x2 * cos_a
    return out


def _op_cosine_similarity(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if (
        _c(a)
        and _c(b)
        and a.dim() == 3
        and b.dim() == 3
        and a.shape == b.shape
        and hasattr(aria_core, "cosine_similarity_f32")
    ):
        native_sim = aria_core.cosine_similarity_f32(a, b)
        if isinstance(native_sim, torch.Tensor):
            if native_sim.dim() == 2 and native_sim.shape == a.shape[:2]:
                return native_sim.unsqueeze(-1)
            if (
                native_sim.dim() == 3
                and native_sim.shape[-1] == 1
                and native_sim.shape[:2] == a.shape[:2]
            ):
                return native_sim
    sim = F.cosine_similarity(a, b, dim=-1)
    return sim.unsqueeze(-1)


def _op_gather_topk(_, inputs, config):
    x, scores = inputs[0], inputs[1]
    k = min(int(config.get("k", 4)), x.shape[1])
    if scores.dim() == 3:
        scores = scores.mean(dim=-1)  # (B, S)
    # Causal gather: for each position t, only consider scores at positions <= t
    S = scores.shape[1]
    causal_mask = torch.triu(
        torch.full((S, S), float("-inf"), device=scores.device, dtype=scores.dtype),
        diagonal=1,
    )  # upper triangular = -inf
    # (B, S_query, S_key)
    scores_expanded = scores.unsqueeze(1) + causal_mask.unsqueeze(0)
    k_eff = min(k, S)
    _, indices = scores_expanded.topk(k_eff, dim=-1)  # (B, S, k)
    # Best scoring causal position per query
    idx = indices[:, :, 0]  # (B, S)
    gathered = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return gathered


def _op_softmax_last(_, inputs, __):
    x = inputs[0]
    if _c(x):
        return aria_core.softmax_f32(x)
    return F.softmax(x, dim=-1)


def _op_causal_mask(_, inputs, __):
    """Causal integration: every token becomes the average of itself and all previous tokens.
    This is a strictly causal 'mixing' operation that prevents future lookahead.
    """
    x = inputs[0]  # (B, S, D)
    if _c(x) and hasattr(aria_core, "causal_mask_f32"):
        return aria_core.causal_mask_f32(x)
    # Using cumulative sum / counts is O(S) and strictly causal
    return torch.cumsum(x, dim=1) / torch.arange(
        1, x.shape[1] + 1, device=x.device
    ).view(1, -1, 1)


def _op_local_window_attn(_, inputs, config):
    x = inputs[0]
    B, S, D = x.shape
    W = min(config.get("window_size", 32), S)
    # Clamp window_size to prevent Triton shared memory overflow.
    # The Triton kernel's shared memory usage scales with W * D.
    # GPU shared memory is typically 96-100 KB; exceeding it raises
    # OutOfResources. Conservatively cap at 16 for D >= 256.
    if D >= 256 and W > 16:
        W = 16
    # Triton local attention: internal accumulation is fp32, so bf16/fp16 input
    # is safe via conservative cast. Cast input to fp32, run kernel, cast back.
    if HAS_KERNELS and x.is_cuda:
        try:
            x_f32 = x.float() if x.dtype != torch.float32 else x
            out = kernels.triton_local_attn(x_f32, W)
            if torch.isfinite(out).all():
                return out.to(x.dtype)
        except Exception as e:
            record_kernel_fallback("triton_local_attn", e)
    x_work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    scores = torch.bmm(x_work, x_work.transpose(-2, -1)) / math.sqrt(D)
    row_idx = torch.arange(S, device=x.device).unsqueeze(1)
    col_idx = torch.arange(S, device=x.device).unsqueeze(0)
    mask = (col_idx > row_idx) | (row_idx - col_idx >= W)
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    attn = F.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.bmm(attn, x_work)
    return out.to(dtype=x.dtype)


def _op_sliding_window_mask(_, inputs, config):
    x = inputs[0]
    B, S, D = x.shape
    W = int(config.get("window_size", 32))

    if _c(x):
        return aria_core.sliding_window_mask_f32(x, W)

    # Triton banded kernel: O(S*W*D) instead of O(S²*D)
    if HAS_KERNELS and x.is_cuda:
        try:
            x_f32 = x.float() if x.dtype != torch.float32 else x
            out = kernels.triton_banded_sliding_window(x_f32, W)
            return out.to(x.dtype)
        except Exception as e:
            record_kernel_fallback("banded_sliding_window", e)

    # Python Fallback: O(S^2) masking
    W_safe = min(W, S)
    row_idx = torch.arange(S, device=x.device).unsqueeze(1)
    col_idx = torch.arange(S, device=x.device).unsqueeze(0)
    dist = row_idx - col_idx

    # Causal sliding window: col <= row AND dist < W
    mask = (dist >= 0) & (dist < W_safe)
    decay = torch.exp(-dist.float().clamp(min=0) / max(W_safe / 4, 1.0))

    # Normalize per-position to maintain signal scale
    final_mask = mask.float() * decay
    final_mask = final_mask / final_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    return torch.bmm(final_mask.unsqueeze(0).expand(B, -1, -1), x)


def _op_token_pool_restore(_, inputs, __):
    x = inputs[0]
    if _c(x):
        return aria_core.token_pool_restore_f32(x)
    if x.shape[1] < 2:
        return x
    S_half = x.shape[1] // 2
    restored = (
        (x[:, 0::2, :][:, :S_half] + x[:, 1::2, :][:, :S_half]) / 2.0
    ).repeat_interleave(2, dim=1)
    if restored.shape[1] < x.shape[1]:
        restored = torch.cat([restored, x[:, -1:, :]], dim=1)
    return restored


# ── Routing Ops (Phase 1/2) ──────────────────────────────────────────

OP_IMPLS: Dict[str, Callable] = {
    "softmax_attention": _op_softmax_attention,
    "linear_attention": _op_linear_attention,
    "graph_attention": _op_graph_attention,
    "local_window_attn": _op_local_window_attn,
    "sliding_window_mask": _op_sliding_window_mask,
    "rmsnorm": _op_rmsnorm,
    "layernorm": _op_layernorm,
    "gated_linear": _op_gated_linear,
    "embedding_lookup": _op_embedding_lookup,
    "rope_rotate": _op_rope_rotate,
    "cosine_similarity": _op_cosine_similarity,
    "gather_topk": _op_gather_topk,
    "softmax_last": _op_softmax_last,
    "token_pool_restore": _op_token_pool_restore,
    "causal_mask": _op_causal_mask,
}
