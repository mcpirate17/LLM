from __future__ import annotations

import math
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from research.defaults import ROPE_THETA_BASE
from .compiler_op_utils import (
    HAS_ARIA_CORE,
    HAS_KERNELS,
    aria_core,
    kernels,
    _c,
)

def _op_softmax_attention(module, inputs, _):
    """Standard causal multi-head softmax attention."""
    x = inputs[0]
    if not hasattr(module, 'q_proj'):
        return x
    B, S, _ = x.shape
    nh, hd = module.n_heads, module.head_dim
    q = module.q_proj(x).reshape(B, S, nh, hd).transpose(1, 2)
    k = module.k_proj(x).reshape(B, S, nh, hd).transpose(1, 2)
    v = module.v_proj(x).reshape(B, S, nh, hd).transpose(1, 2)
    attn = (q @ k.transpose(-2, -1)) * module.attn_scale
    mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
    attn.masked_fill_(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
    return module.o_proj(out)

def _op_linear_attention(module, inputs, _):
    """Linear attention with ELU kernel (O(S) complexity)."""
    x = inputs[0]
    if not hasattr(module, 'q_proj'):
        return x
    B, S, _ = x.shape
    nh, hd = module.n_heads, module.head_dim
    q = F.elu(module.q_proj(x).reshape(B, S, nh, hd).transpose(1, 2)) + 1
    k = F.elu(module.k_proj(x).reshape(B, S, nh, hd).transpose(1, 2)) + 1
    v = module.v_proj(x).reshape(B, S, nh, hd).transpose(1, 2)
    # Causal linear attention via cumulative sum
    kv = torch.einsum("bhsd,bhse->bhsde", k, v)
    kv_cumsum = kv.cumsum(dim=2)
    k_cumsum = k.cumsum(dim=2)
    out = torch.einsum("bhsd,bhsde->bhse", q, kv_cumsum)
    denom = torch.einsum("bhsd,bhsd->bhs", q, k_cumsum).unsqueeze(-1).clamp(min=1e-6)
    out = out / denom
    return module.o_proj(out.transpose(1, 2).reshape(B, S, -1))

def _op_graph_attention(module, inputs, _):
    """Graph attention with learned edge features + causal softmax attention."""
    x = inputs[0]
    if not hasattr(module, 'q_proj'):
        return x
    B, S, _ = x.shape
    nh, hd = module.n_heads, module.head_dim
    x_e = x + module.edge_proj(x)
    q = module.q_proj(x_e).reshape(B, S, nh, hd).transpose(1, 2)
    k = module.k_proj(x_e).reshape(B, S, nh, hd).transpose(1, 2)
    v = module.v_proj(x_e).reshape(B, S, nh, hd).transpose(1, 2)
    attn = (q @ k.transpose(-2, -1)) * module.attn_scale
    mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
    attn.masked_fill_(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
    return module.o_proj(out)

def _op_rmsnorm(module, inputs, _):
    if not hasattr(module, 'weight'): return inputs[0]
    x = inputs[0]
    if _c(x): return aria_core.rmsnorm_f32(x, module.weight, 1e-6)
    if HAS_KERNELS and x.is_cuda:
        try:
            return kernels.triton_rmsnorm(x, module.weight)
        except Exception:
            pass
    eps = 1e-6
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return (x / rms) * module.weight

def _op_layernorm(module, inputs, _):
    if not hasattr(module, 'weight'): return inputs[0]
    x = inputs[0]
    if _c(x): return aria_core.layernorm_f32(x, module.weight, module.bias, 1e-5)
    return F.layer_norm(x, [x.shape[-1]], module.weight, module.bias)

def _op_gated_linear(module, inputs, _):
    if not hasattr(module, 'linear_weight'): return inputs[0]
    x = inputs[0]
    if _c(x):
        return aria_core.gated_linear_f32(
            x, module.linear_weight, module.linear_bias,
            module.gate_weight, module.gate_bias)
    dt = x.dtype
    linear = F.linear(x, module.linear_weight.to(dt), module.linear_bias.to(dt))
    gate = torch.sigmoid(F.linear(x, module.gate_weight.to(dt), module.gate_bias.to(dt)))
    return linear * gate

def _op_embedding_lookup(module, inputs, _):
    # In compiled model context, input is usually already embedded — pass through.
    if not hasattr(module, 'embed_table'):
        return inputs[0]
    x = inputs[0]
    if (
        HAS_ARIA_CORE
        and hasattr(aria_core, "embedding_lookup_f32")
        and isinstance(x, torch.Tensor)
        and x.device.type == "cpu"
        and x.dtype in (torch.int32, torch.int64)
    ):
        try:
            native_out = aria_core.embedding_lookup_f32(module.embed_table, x)
            if isinstance(native_out, torch.Tensor):
                return native_out
        except Exception:
            pass
    return x

def _op_rope_rotate(_, inputs, __):
    x = inputs[0]
    if _c(x) and hasattr(aria_core, "rope_rotate_f32"):
        return aria_core.rope_rotate_f32(x, ROPE_THETA_BASE)
    B, S, D = x.shape
    pos = torch.arange(S, device=x.device, dtype=x.dtype).unsqueeze(1)
    dim_pairs = D // 2
    freqs = 1.0 / (ROPE_THETA_BASE ** (torch.arange(0, D, 2, device=x.device, dtype=x.dtype) / D))
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
            if native_sim.dim() == 3 and native_sim.shape[-1] == 1 and native_sim.shape[:2] == a.shape[:2]:
                return native_sim
    sim = F.cosine_similarity(a, b, dim=-1)
    return sim.unsqueeze(-1)

def _op_gather_topk(_, inputs, config):
    x, scores = inputs[0], inputs[1]
    k = min(int(config.get("k", 4)), x.shape[1])
    if scores.dim() == 3:
        scores = scores.mean(dim=-1)
    if _c(x) and _c(scores) and scores.dim() == 2 and x.dim() == 3:
        gathered, _ = aria_core.gather_topk_f32(scores, x, k)
        if gathered.shape[1] < x.shape[1]:
            pad = x[:, :x.shape[1] - gathered.shape[1]]
            gathered = torch.cat([gathered, pad], dim=1)
        return gathered
    _, indices = torch.topk(scores, k, dim=-1)
    indices = indices.clamp(0, x.shape[1] - 1)
    gathered = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    if gathered.shape[1] < x.shape[1]:
        pad = x[:, :x.shape[1] - gathered.shape[1]]
        gathered = torch.cat([gathered, pad], dim=1)
    return gathered

def _op_softmax_last(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.softmax_f32(x)
    return F.softmax(x, dim=-1)

def _op_softmax_seq(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.softmax_seq_f32(x)
    return F.softmax(x, dim=1)

def _op_causal_mask(_, inputs, __):
    """Causal integration: every token becomes the average of itself and all previous tokens.
    This is a strictly causal 'mixing' operation that prevents future lookahead.
    """
    x = inputs[0]  # (B, S, D)
    if _c(x) and hasattr(aria_core, "causal_mask_f32"):
        return aria_core.causal_mask_f32(x)
    # Using cumulative sum / counts is O(S) and strictly causal
    return torch.cumsum(x, dim=1) / torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)

def _op_sort_seq(_, inputs, __):
    x = inputs[0]
    if _c(x) and hasattr(aria_core, "sort_seq_f32"):
        native_sorted = aria_core.sort_seq_f32(x)
        if isinstance(native_sorted, tuple):
            native_sorted = native_sorted[0]
        if isinstance(native_sorted, torch.Tensor) and native_sorted.shape == x.shape:
            return native_sorted
    indices = x.mean(dim=-1).argsort(dim=-1)
    indices = indices.clamp(0, x.shape[1] - 1)
    return x.gather(1, indices.unsqueeze(-1).expand_as(x))

def _op_argsort_seq(_, inputs, __):
    x = inputs[0]
    if _c(x) and hasattr(aria_core, "argsort_seq_f32"):
        native_indices = aria_core.argsort_seq_f32(x)
        if isinstance(native_indices, tuple):
            native_indices = native_indices[0]
        if isinstance(native_indices, torch.Tensor):
            if native_indices.dim() == 2:
                return native_indices.unsqueeze(-1).expand_as(x).float()
            if native_indices.shape == x.shape:
                return native_indices.float()
    return x.mean(dim=-1).argsort(dim=-1).unsqueeze(-1).expand_as(x).float()

def _op_local_window_attn(_, inputs, config):
    x = inputs[0]
    B, S, D = x.shape
    W = min(config.get("window_size", 32), S)
    if HAS_KERNELS and x.is_cuda:
        try:
            return kernels.triton_local_attn(x, W)
        except Exception:
            pass
    scores = torch.bmm(x, x.transpose(-2, -1)) / math.sqrt(D)
    row_idx = torch.arange(S, device=x.device).unsqueeze(1)
    col_idx = torch.arange(S, device=x.device).unsqueeze(0)
    mask = (col_idx > row_idx) | (row_idx - col_idx >= W)
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    return torch.bmm(F.softmax(scores, dim=-1), x)

def _op_sliding_window_mask(_, inputs, config):
    x = inputs[0]
    B, S, D = x.shape
    W = int(config.get("window_size", 32))
    
    if _c(x):
        return aria_core.sliding_window_mask_f32(x, W)
        
    # Python Fallback: O(S^2) masking
    W_safe = min(W, S)
    row_idx = torch.arange(S, device=x.device).unsqueeze(1)
    col_idx = torch.arange(S, device=x.device).unsqueeze(0)
    dist = (row_idx - col_idx)
    
    # Causal sliding window: col <= row AND dist < W
    mask = (dist >= 0) & (dist < W_safe)
    decay = torch.exp(-dist.float().clamp(min=0) / max(W_safe / 4, 1.0))
    
    # Normalize per-position to maintain signal scale
    final_mask = (mask.float() * decay)
    final_mask = final_mask / final_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    
    return torch.bmm(final_mask.unsqueeze(0).expand(B, -1, -1), x)

def _op_token_pool_restore(_, inputs, __):
    x = inputs[0]
    if _c(x):
        return aria_core.token_pool_restore_f32(x)
    if x.shape[1] < 2: return x
    S_half = x.shape[1] // 2
    restored = ((x[:, 0::2, :][:, :S_half] + x[:, 1::2, :][:, :S_half]) / 2.0).repeat_interleave(2, dim=1)
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
    "softmax_seq": _op_softmax_seq,
    "sort_seq": _op_sort_seq,
    "argsort_seq": _op_argsort_seq,
    "token_pool_restore": _op_token_pool_restore,
    "causal_mask": _op_causal_mask,
}
