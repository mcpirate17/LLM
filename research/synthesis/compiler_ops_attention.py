from __future__ import annotations

import math
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from research.defaults import ROPE_THETA_BASE
from .compiler_op_utils import (
    aria_core,
    kernels,
    _c,
    _t,
    _safe_linear,
    record_kernel_fallback,
)


def _project_qkv(module, x):
    """Multi-head Q/K/V projection shared by every standard-attention variant.

    Returns ``(q, k, v, B, S)`` with q/k/v shaped ``(B, n_heads, S, head_dim)``.
    Callers can apply a per-variant activation (e.g. ``F.elu`` for linear
    attention) on the returned tensors.
    """
    B, S, _ = x.shape
    nh, hd = module.n_heads, module.head_dim

    def _proj(linear):
        return (
            _safe_linear(x, linear.weight, linear.bias)
            .reshape(B, S, nh, hd)
            .transpose(1, 2)
        )

    return _proj(module.q_proj), _proj(module.k_proj), _proj(module.v_proj), B, S


def _apply_o_proj(module, multi_head_out, B, S):
    """Collapse ``(B, n_heads, S, head_dim)`` to ``(B, S, D)`` and apply o_proj."""
    out = multi_head_out.transpose(1, 2).reshape(B, S, -1)
    return _safe_linear(out, module.o_proj.weight, module.o_proj.bias)


def _op_softmax_attention(module, inputs, _):
    """Standard causal multi-head softmax attention."""
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    q, k, v, B, S = _project_qkv(module, x)
    out = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=True, scale=module.attn_scale
    )
    return _apply_o_proj(module, out, B, S)


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    shifted = shifted.clamp(min=-20.0)
    zs = torch.sort(shifted, dim=dim, descending=True).values
    range_shape = [1] * logits.ndim
    range_shape[dim] = logits.shape[dim]
    ks = torch.arange(
        1, logits.shape[dim] + 1, device=logits.device, dtype=logits.dtype
    )
    ks = ks.reshape(range_shape)
    bound = 1 + ks * zs
    cumsum_zs = zs.cumsum(dim)
    is_gt = bound > cumsum_zs
    k_z = is_gt.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cumsum_zs.gather(dim, k_z - 1) - 1) / k_z.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0)


def _entmax_bisect(
    logits: torch.Tensor, alpha: float = 1.5, dim: int = -1
) -> torch.Tensor:
    alpha = float(max(1.01, min(2.0, alpha)))
    if alpha >= 1.999:
        return _sparsemax(logits, dim=dim)
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    shifted = shifted.clamp(min=-20.0)
    scaled = shifted * (alpha - 1.0)
    tau_lo = scaled.min(dim=dim, keepdim=True).values - 1.0
    tau_hi = scaled.max(dim=dim, keepdim=True).values
    power = 1.0 / (alpha - 1.0)
    for _ in range(24):
        tau = (tau_lo + tau_hi) * 0.5
        probs = torch.clamp(scaled - tau, min=0).pow(power)
        too_large = probs.sum(dim=dim, keepdim=True) > 1.0
        tau_lo = torch.where(too_large, tau, tau_lo)
        tau_hi = torch.where(too_large, tau_hi, tau)
    probs = torch.clamp(scaled - tau_hi, min=0).pow(power)
    return probs / probs.sum(dim=dim, keepdim=True).clamp(min=1e-8)


def _causal_attention_scores(
    q: torch.Tensor, k: torch.Tensor, scale: float
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    seq_len = scores.shape[-1]
    causal = torch.triu(
        torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1
    )
    return scores.masked_fill(causal, -1e9)


def _op_sparsemax_attention(module, inputs, _):
    """Causal attention with sparsemax-normalized retrieval weights."""
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    q, k, v, B, S = _project_qkv(module, x)
    scores = _causal_attention_scores(q, k, module.attn_scale)
    weights = _sparsemax(scores, dim=-1)
    return _apply_o_proj(module, torch.matmul(weights, v), B, S)


def _op_entmax_attention(module, inputs, config):
    """Causal alpha-entmax attention; alpha=1.5 by default, alpha=2 sparsemax."""
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    alpha = float(config.get("alpha", 1.5)) if isinstance(config, dict) else 1.5
    q, k, v, B, S = _project_qkv(module, x)
    scores = _causal_attention_scores(q, k, module.attn_scale)
    weights = _entmax_bisect(scores, alpha=alpha, dim=-1)
    return _apply_o_proj(module, torch.matmul(weights, v), B, S)


def _op_learnable_semiring_attention(module, inputs, _):
    """Causal attention with a learned per-head value-aggregation semiring.

    Per head, β (``module.semiring_beta``) slides the value pooling from
    arithmetic mean (β→0, == softmax attention) to winner-take-all max (β>0)
    or min (β<0). See ``mathspaces.semiring``.
    """
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    from ..mathspaces.semiring import semiring_attention

    q, k, v, B, S = _project_qkv(module, x)
    out = semiring_attention(
        q, k, v, module.semiring_beta.to(x.dtype), module.attn_scale
    )
    return _apply_o_proj(module, out, B, S)


def _op_reciprocal_rank_attention(module, inputs, _):
    """Causal attention boosted by reciprocal content agreement.

    Standard attention asks "how much should query i read key j?"  This mixer
    adds the reverse compatibility "how much does token j point back at i?"
    over the same causal prefix.  The product favors mutual matches, which is
    the useful shape for binding/retrieval, while retaining a dense fallback at
    initialization because the boost scale starts near zero.
    """
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    q, k, v, B, S = _project_qkv(module, x)
    raw_scores = torch.matmul(q, k.transpose(-2, -1)) * module.attn_scale
    causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
    scores = raw_scores.masked_fill(causal, -1e9)

    reverse_scores = raw_scores.transpose(-2, -1).masked_fill(causal, -1e9)
    reciprocal = torch.softmax(reverse_scores, dim=-1).clamp(min=1e-6)
    boost = torch.tanh(module.reciprocal_logit_scale).to(x.dtype)
    weights = torch.softmax(scores + boost * torch.log(reciprocal), dim=-1)
    return _apply_o_proj(module, torch.matmul(weights, v), B, S)


def _op_phase_lock_attention(module, inputs, _):
    """Causal attention with phase-synchrony content matching.

    The extra score term compares bounded q/k phases through cos(q-k), so keys
    are favored when their channel-wise phase pattern synchronizes with the
    query.  This is deliberately not a value-pooling semiring or sparsemax
    variant: the novelty is in the content address used by the mixer.
    """
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    q, k, v, B, S = _project_qkv(module, x)
    dot_scores = _causal_attention_scores(q, k, module.attn_scale)
    phase_q = torch.tanh(q)
    phase_k = torch.tanh(k)
    phase_scores = torch.cos(phase_q.unsqueeze(-2) - phase_k.unsqueeze(-3)).mean(dim=-1)
    causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
    phase_scores = phase_scores.masked_fill(causal, 0.0)
    phase_scale = torch.tanh(module.phase_lock_scale).to(x.dtype)
    weights = torch.softmax(dot_scores + phase_scale * phase_scores, dim=-1)
    return _apply_o_proj(module, torch.matmul(weights, v), B, S)


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
    q, k, v, B, S = _project_qkv(module, x)
    nh, hd = module.n_heads, module.head_dim
    # ELU+1 feature map for the positive linear-attention kernel; applied to
    # q and k only — v stays raw so the running state matches standard
    # linear-attention formulations.
    q = F.elu(q) + 1
    k = F.elu(k) + 1

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
    return _safe_linear(
        out.transpose(1, 2).reshape(B, S, -1),
        module.o_proj.weight,
        module.o_proj.bias,
    )


def _op_graph_attention(module, inputs, _):
    """Graph attention with learned edge features + causal softmax attention."""
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, _ = x.shape
    nh, hd = module.n_heads, module.head_dim
    x_e = x + _safe_linear(x, module.edge_proj.weight, module.edge_proj.bias)
    q = (
        _safe_linear(x_e, module.q_proj.weight, module.q_proj.bias)
        .reshape(B, S, nh, hd)
        .transpose(1, 2)
    )
    k = (
        _safe_linear(x_e, module.k_proj.weight, module.k_proj.bias)
        .reshape(B, S, nh, hd)
        .transpose(1, 2)
    )
    v = (
        _safe_linear(x_e, module.v_proj.weight, module.v_proj.bias)
        .reshape(B, S, nh, hd)
        .transpose(1, 2)
    )
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=True,
        scale=module.attn_scale,
    )
    out = out.transpose(1, 2).reshape(B, S, -1)
    return _safe_linear(out, module.o_proj.weight, module.o_proj.bias)


def _op_rmsnorm(module, inputs, _):
    if not hasattr(module, "weight"):
        return inputs[0]
    x = inputs[0]
    orig_dtype = x.dtype
    if _c(x):
        return aria_core.rmsnorm_f32(x, module.weight, 1e-6)
    if _t(x):
        try:
            return kernels.triton_rmsnorm(x, module.weight)
        except (ImportError, RuntimeError, AttributeError) as e:
            record_kernel_fallback("triton_rmsnorm", e)
    eps = 1e-6
    compute_dtype = (
        torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    )
    x_work = x.to(compute_dtype) if x.dtype != compute_dtype else x
    weight = (
        module.weight.to(compute_dtype)
        if module.weight.dtype != compute_dtype
        else module.weight
    )
    rms = torch.sqrt(torch.mean(x_work**2, dim=-1, keepdim=True) + eps)
    return ((x_work / rms) * weight).to(orig_dtype)


def _op_qk_norm(module, inputs, _):
    """L2-normalize the feature dim, rescale by a learned per-channel gain.

    Table-stakes attention stabilizer: applied to Q/K (or any activation),
    it bounds the dot-product logit magnitude regardless of input scale, which
    kills attention-logit blow-up and improves long-context behavior. Parameter
    name ``qk_scale`` (per-channel). Identity-shape; falls through if unbuilt."""
    x = inputs[0]
    if not hasattr(module, "qk_scale"):
        return x
    orig_dtype = x.dtype
    xf = x.float()
    normed = xf / xf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (normed * module.qk_scale.float()).to(orig_dtype)


def _op_logit_softcap(module, inputs, _):
    """Soft cap ``cap * tanh(x / cap)`` with a learned POSITIVE cap.

    Smoothly squashes large magnitudes toward ``±cap`` while staying ~linear
    near 0, bounding logits/activations without a hard clip (Gemma-style). The
    cap is ``softplus(softcap_logit)`` so it stays > 0 and learnable. Identity-
    shape; falls through if unbuilt."""
    x = inputs[0]
    if not hasattr(module, "softcap_logit"):
        return x
    cap = torch.nn.functional.softplus(module.softcap_logit).clamp_min(1e-2).to(x.dtype)
    return cap * torch.tanh(x / cap)


def _op_layernorm(module, inputs, _):
    if not hasattr(module, "weight"):
        return inputs[0]
    x = inputs[0]
    orig_dtype = x.dtype
    if _c(x):
        return aria_core.layernorm_f32(x, module.weight, module.bias, 1e-5)
    compute_dtype = (
        torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    )
    x_work = x.to(compute_dtype) if x.dtype != compute_dtype else x
    weight = (
        module.weight.to(compute_dtype)
        if module.weight.dtype != compute_dtype
        else module.weight
    )
    bias = None
    if module.bias is not None:
        bias = (
            module.bias.to(compute_dtype)
            if module.bias.dtype != compute_dtype
            else module.bias
        )
    return F.layer_norm(x_work, [x.shape[-1]], weight, bias).to(orig_dtype)


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
    linear = _safe_linear(x, module.linear_weight, module.linear_bias)
    gate = torch.sigmoid(_safe_linear(x, module.gate_weight, module.gate_bias))
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
    # Triton's tl.dot requires K >= 16 after internal block padding. For
    # D <= 8, the local-attention kernel compiles with BLOCK_D=8 and fails;
    # use the dense fallback directly instead of logging noisy compile errors.
    if _t(x) and D > 8:
        try:
            x_f32 = x.float() if x.dtype != torch.float32 else x
            out = kernels.triton_local_attn(x_f32, W)
            if torch.isfinite(out).all():
                return out.to(x.dtype)
        except Exception as e:
            # Includes triton.compiler.errors.CompilationError, which doesn't
            # inherit from RuntimeError — observed 2026-05-19 at D>=512 where
            # the triton_local_attn kernel's tl.dot rejects K post-pad. The
            # dense fallback below handles all such cases safely.
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
    if _t(x):
        try:
            x_f32 = x.float() if x.dtype != torch.float32 else x
            out = kernels.triton_banded_sliding_window(x_f32, W)
            return out.to(x.dtype)
        except (ImportError, RuntimeError, AttributeError) as e:
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


def _op_role_slot_attention(module, inputs, _):
    """Persistent Role-Slot Attention.

    Tokens (Queries) attend to learned global latent Slots (Keys/Values).
    Provides O(S * num_slots) retrieval.
    """
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, D = x.shape

    q = module.q_proj(x)  # (B, S, D)
    k = module.slot_keys  # (num_slots, D)
    v = module.slot_values  # (num_slots, D)

    # Scaled dot-product attention against slots
    # (B, S, D) @ (D, num_slots) -> (B, S, num_slots)
    # Slots are shared across batch and sequence.
    energy = torch.matmul(q, k.transpose(-2, -1)) * (D**-0.5)
    attn_weights = torch.softmax(energy, dim=-1)

    # (B, S, num_slots) @ (num_slots, D) -> (B, S, D)
    out = torch.matmul(attn_weights, v)

    return module.o_proj(out)


# ── Routing Ops (Phase 1/2) ──────────────────────────────────────────

OP_IMPLS: Dict[str, Callable] = {
    "softmax_attention": _op_softmax_attention,
    "sparsemax_attention": _op_sparsemax_attention,
    "entmax_attention": _op_entmax_attention,
    "learnable_semiring_attention": _op_learnable_semiring_attention,
    "reciprocal_rank_attention": _op_reciprocal_rank_attention,
    "phase_lock_attention": _op_phase_lock_attention,
    "linear_attention": _op_linear_attention,
    "graph_attention": _op_graph_attention,
    "local_window_attn": _op_local_window_attn,
    "sliding_window_mask": _op_sliding_window_mask,
    "role_slot_attention": _op_role_slot_attention,
    "rmsnorm": _op_rmsnorm,
    "qk_norm": _op_qk_norm,
    "logit_softcap": _op_logit_softcap,
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
