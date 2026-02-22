"""
Computation Graph Compiler

Compiles a ComputationGraph into a live PyTorch nn.Module.
Each OpNode becomes a concrete tensor operation, with learnable
parameters allocated for parameterized ops.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import get_primitive, PrimitiveOp, OpCategory
from .graph import ComputationGraph, OpNode, ShapeInfo

try:
    from . import kernels
    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False

try:
    from . import cpu_ops
    HAS_CPU_OPS = True
except ImportError:
    HAS_CPU_OPS = False


# ── Registry System ───────────────────────────────────────────────────

_OP_DISPATCH: Dict[str, Callable[[nn.Module, Tuple[torch.Tensor, ...], Dict], torch.Tensor]] = {}


def register_op(name: str):
    """Decorator to register an op implementation."""
    def decorator(fn: Callable):
        _OP_DISPATCH[name] = fn
        return fn
    return decorator


def _record_sparse_telemetry(module: nn.Module, op_name: str, density: float,
                             fallback_reason: Optional[str] = None) -> None:
    telemetry = getattr(module, "sparse_telemetry", {})
    stats = telemetry.get(op_name, {
        "calls": 0,
        "fallback_calls": 0,
        "density_sum": 0.0,
        "last_density": 1.0,
        "last_fallback_reason": None,
    })
    stats["calls"] += 1
    stats["density_sum"] += float(density)
    stats["last_density"] = float(density)
    if fallback_reason is not None:
        stats["fallback_calls"] += 1
        stats["last_fallback_reason"] = fallback_reason
    telemetry[op_name] = stats
    setattr(module, "sparse_telemetry", telemetry)


def _record_routing_telemetry(module: nn.Module, n_experts: int, selected_experts: torch.Tensor,
                              logits: Optional[torch.Tensor] = None) -> None:
    """Record MoE routing statistics: entropy, expert utilization, drop rate."""
    telemetry = getattr(module, "routing_telemetry", {
        "tokens_total": 0,
        "tokens_processed": 0,
        "expert_counts": torch.zeros(n_experts, device=selected_experts.device),
        "entropy_sum": 0.0,
        "count": 0,
    })
    
    B, S = selected_experts.shape[:2]
    total_tokens = B * S
    telemetry["tokens_total"] += total_tokens
    telemetry["tokens_processed"] += total_tokens # Assuming all tokens processed for now
    
    # Expert utilization
    counts = torch.histc(selected_experts.float(), bins=n_experts, min=0, max=n_experts-1)
    telemetry["expert_counts"] += counts
    
    # Entropy if logits provided
    if logits is not None:
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
        telemetry["entropy_sum"] += entropy
        telemetry["count"] += 1
        
    setattr(module, "routing_telemetry", telemetry)


def _record_adaptive_telemetry(module: nn.Module, savings_ratio: float, 
                               effective_depth: Optional[float] = None) -> None:
    """Record adaptive compute statistics (MoD/MoR)."""
    telemetry = getattr(module, "adaptive_telemetry", {
        "savings_sum": 0.0,
        "depth_sum": 0.0,
        "count": 0,
    })
    telemetry["savings_sum"] += float(savings_ratio)
    if effective_depth is not None:
        telemetry["depth_sum"] += float(effective_depth)
    telemetry["count"] += 1
    setattr(module, "adaptive_telemetry", telemetry)


def _build_nm_mask(weight: torch.Tensor, n: int, m: int) -> torch.Tensor:
    if n <= 0 or m <= 0 or n > m:
        return torch.ones_like(weight)
    
    if HAS_CPU_OPS and weight.device.type == "cpu" and weight.dtype == torch.float32:
        return cpu_ops.build_nm_mask_cpu(weight, n, m)

    rows, cols = weight.shape
    n_chunks = cols // m
    if n_chunks <= 0:
        return torch.ones_like(weight)

    usable = n_chunks * m
    core = weight[:, :usable].abs().reshape(rows, n_chunks, m)
    keep_idx = core.topk(k=n, dim=-1).indices
    mask_core = torch.zeros_like(core)
    mask_core.scatter_(-1, keep_idx, 1.0)
    mask = torch.ones_like(weight)
    mask[:, :usable] = mask_core.reshape(rows, usable)
    return mask


def _build_block_sparse_mask(weight: torch.Tensor, block_size: int,
                             block_density: float) -> torch.Tensor:
    block_size = max(1, int(block_size))
    block_density = float(max(0.05, min(1.0, block_density)))

    rows, cols = weight.shape
    row_blocks = rows // block_size
    col_blocks = cols // block_size
    if row_blocks <= 0 or col_blocks <= 0:
        return torch.ones_like(weight)

    usable_rows = row_blocks * block_size
    usable_cols = col_blocks * block_size
    core = weight[:usable_rows, :usable_cols]
    blocks = core.view(row_blocks, block_size, col_blocks, block_size).permute(0, 2, 1, 3)
    scores = blocks.abs().mean(dim=(2, 3))

    keep_per_row = max(1, int(round(col_blocks * block_density)))
    keep_idx = scores.topk(k=keep_per_row, dim=1).indices

    block_mask = torch.zeros_like(scores)
    block_mask.scatter_(1, keep_idx, 1.0)
    block_mask = block_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, block_size, block_size)
    block_mask = block_mask.permute(0, 2, 1, 3).reshape(usable_rows, usable_cols)

    mask = torch.ones_like(weight)
    mask[:usable_rows, :usable_cols] = block_mask
    return mask


# ── Op Implementations ──────────────────────────────────────────────

@register_op("neg")
def _op_neg(_, inputs, __): return -inputs[0]

@register_op("abs")
def _op_abs(_, inputs, __): return torch.abs(inputs[0])

@register_op("exp")
def _op_exp(_, inputs, __): return torch.exp(torch.clamp(inputs[0], -20, 20))

@register_op("log")
def _op_log(_, inputs, __): return torch.log(torch.clamp(inputs[0].abs(), min=1e-8))

@register_op("sin")
def _op_sin(_, inputs, __): return torch.sin(inputs[0])

@register_op("cos")
def _op_cos(_, inputs, __): return torch.cos(inputs[0])

@register_op("tanh")
def _op_tanh(_, inputs, __): return torch.tanh(inputs[0])

@register_op("sigmoid")
def _op_sigmoid(_, inputs, __): return torch.sigmoid(inputs[0])

@register_op("relu")
def _op_relu(_, inputs, __): return F.relu(inputs[0])

@register_op("gelu")
def _op_gelu(_, inputs, __): return F.gelu(inputs[0])

@register_op("silu")
def _op_silu(_, inputs, __): return F.silu(inputs[0])

@register_op("sqrt")
def _op_sqrt(_, inputs, __): return torch.sqrt(torch.clamp(inputs[0].abs(), min=1e-8))

@register_op("square")
def _op_square(_, inputs, __): return inputs[0] * inputs[0]

@register_op("sign_ste")
def _op_sign_ste(_, inputs, __):
    signs = torch.sign(inputs[0])
    return inputs[0] + (signs - inputs[0]).detach()

@register_op("reciprocal")
def _op_reciprocal(_, inputs, __):
    return 1.0 / torch.clamp(inputs[0].abs(), min=1e-6) * torch.sign(inputs[0])

@register_op("add")
def _op_add(_, inputs, __): return inputs[0] + inputs[1]

@register_op("mul")
def _op_mul(_, inputs, __): return inputs[0] * inputs[1]

@register_op("sub")
def _op_sub(_, inputs, __): return inputs[0] - inputs[1]

@register_op("div_safe")
def _op_div_safe(_, inputs, __):
    return inputs[0] / torch.clamp(inputs[1].abs(), min=1e-6) * torch.sign(inputs[1])

@register_op("maximum")
def _op_maximum(_, inputs, __): return torch.maximum(inputs[0], inputs[1])

@register_op("minimum")
def _op_minimum(_, inputs, __): return torch.minimum(inputs[0], inputs[1])

@register_op("sum_last")
def _op_sum_last(_, inputs, __): return inputs[0].sum(dim=-1, keepdim=True)

@register_op("mean_last")
def _op_mean_last(_, inputs, __): return inputs[0].mean(dim=-1, keepdim=True)

@register_op("max_last")
def _op_max_last(_, inputs, __): return inputs[0].max(dim=-1, keepdim=True).values

@register_op("norm_last")
def _op_norm_last(_, inputs, __): return inputs[0].norm(dim=-1, keepdim=True)

@register_op("sum_seq")
def _op_sum_seq(_, inputs, __): return inputs[0].sum(dim=1, keepdim=True)

@register_op("mean_seq")
def _op_mean_seq(_, inputs, __): return inputs[0].mean(dim=1, keepdim=True)

@register_op("cumsum")
def _op_cumsum(_, inputs, __): return torch.cumsum(inputs[0], dim=1)

@register_op("cumprod_safe")
def _op_cumprod_safe(_, inputs, __): return torch.cumprod(torch.clamp(inputs[0], -2, 2), dim=1)

@register_op("matmul")
def _op_matmul(_, inputs, __):
    a, b = inputs
    if a.shape[-1] == b.shape[-1]:
        scale = math.sqrt(a.shape[-1])
        scores = torch.bmm(a, b.transpose(-2, -1)) / scale
        return torch.bmm(F.softmax(scores, dim=-1), b)
    return torch.bmm(a, b)

@register_op("outer_product")
def _op_outer_product(_, inputs, __):
    # Elementwise (Hadamard) product (as defined in primitives.py)
    return inputs[0] * inputs[1]

@register_op("transpose_sd")
def _op_transpose_sd(_, inputs, __):
    return inputs[0].transpose(1, 2).contiguous().transpose(1, 2)

@register_op("split2")
def _op_split2(_, inputs, __): return inputs[0][..., :inputs[0].shape[-1] // 2]

@register_op("split3")
def _op_split3(_, inputs, __): return inputs[0][..., :inputs[0].shape[-1] // 3]

@register_op("concat")
def _op_concat(_, inputs, __): return torch.cat([inputs[0], inputs[1]], dim=-1)

@register_op("roll_seq")
def _op_roll_seq(_, inputs, __): return torch.roll(inputs[0], shifts=1, dims=1)

@register_op("roll_neg")
def _op_roll_neg(_, inputs, __): return torch.roll(inputs[0], shifts=-1, dims=1)

@register_op("gather_sorted")
def _op_gather_sorted(_, inputs, __):
    data, indices = inputs
    idx = indices[..., :1].expand_as(data).long().clamp(0, data.shape[1] - 1)
    return data.gather(1, idx)

@register_op("scatter_unsort")
def _op_scatter_unsort(_, inputs, __):
    data, indices = inputs
    idx = indices[..., :1].expand_as(data).long().clamp(0, data.shape[1] - 1)
    return torch.zeros_like(data).scatter_(1, idx, data)

@register_op("multi_head_mix")
def _op_multi_head_mix(_, inputs, config):
    x = inputs[0]
    B, S, D = x.shape
    H = config.get("n_heads", 4)
    if D % H != 0: H = 1
    return F.normalize(x.view(B, S, H, -1), p=2, dim=-1).view(B, S, D)

@register_op("linear_proj")
@register_op("linear_proj_down")
@register_op("linear_proj_up")
def _op_linear_common(module, inputs, _):
    if not hasattr(module, 'weight'): return inputs[0]
    return F.linear(inputs[0], module.weight)

@register_op("fused_linear_gelu")
def _op_fused_linear_gelu(module, inputs, _):
    if not hasattr(module, 'weight'): return inputs[0]
    if HAS_KERNELS and inputs[0].is_cuda:
        bias = getattr(module, 'bias', None)
        return kernels.fused_linear_gelu(inputs[0], module.weight, bias)
    out = F.linear(inputs[0], module.weight)
    if hasattr(module, 'bias'): out = out + module.bias
    return F.gelu(out)

@register_op("learnable_scale")
def _op_learnable_scale(module, inputs, _):
    if not hasattr(module, 'scale'): return inputs[0]
    return inputs[0] * module.scale

@register_op("learnable_bias")
def _op_learnable_bias(module, inputs, _):
    if not hasattr(module, 'bias'): return inputs[0]
    return inputs[0] + module.bias

@register_op("selective_scan")
def _op_selective_scan(module, inputs, _):
    """
    Vectorized Linear Scan.
    Computes h[t] = decay * h[t-1] + B_x[t] * x[t], out[t] = C_x[t] * h[t]
    Since decay is constant in this implementation, this is a linear recurrence
    that can be computed via a parallel scan or cumulative sum in log-space.
    """
    if not hasattr(module, 'A_log'): return inputs[0]
    x = inputs[0]
    B, S, D = x.shape
    
    # h[t] = a h[t-1] + u[t]
    # h[t] = sum_{i=0}^t a^{t-i} u[i]
    A = -torch.exp(module.A_log.clamp(-10, 10))
    dt = F.softplus(module.dt_proj)
    log_a = A * dt  # (D,)
    
    u = torch.sigmoid(x * module.B_proj) * x  # (B, S, D)
    
    # Vectorized computation of h[t]
    # log(h[t]) = log(sum exp(log(a^{t-i}) + log(u[i]))) -- unstable for negative u
    # Use standard linear recurrence vectorized via cumsum if possible.
    # For small S, or constant a, we can use a convolution-like approach.
    
    # h_t = a^t * sum_{i=0}^t a^{-i} u_i
    # We use a 1D convolution with an exponential kernel for numerical stability and memory efficiency.
    # The recurrence h_t = a * h_{t-1} + u_t is equivalent to a causal convolution with kernel [a^{S-1}, ..., a, 1].
    a = torch.exp(log_a)  # (D,)
    indices = torch.arange(S, device=x.device, dtype=x.dtype)
    # kernel: (D, 1, S)
    kernel = torch.pow(a.view(D, 1, 1), indices.flip(0).view(1, 1, S))
    
    u_swapped = u.permute(0, 2, 1) # (B, D, S)
    # Causal convolution via padding
    h_swapped = F.conv1d(F.pad(u_swapped, (S - 1, 0)), kernel, groups=D) # (B, D, S)
    h = h_swapped.permute(0, 2, 1) # (B, S, D)
    
    C_x = torch.sigmoid(x * module.C_proj)
    return C_x * h

@register_op("conv1d_seq")
def _op_conv1d_seq(module, inputs, _):
    if not hasattr(module, 'conv_weight'): return inputs[0]
    B, S, D = inputs[0].shape
    out = F.conv1d(inputs[0].transpose(1, 2), module.conv_weight, padding=1, groups=D)
    return out.transpose(1, 2)

@register_op("topk_gate")
def _op_topk_gate(module, inputs, _):
    if not hasattr(module, 'gate_proj'): return inputs[0]
    x = inputs[0]
    B, S, D = x.shape
    logits = F.linear(x, module.gate_proj)
    gate_weights = F.softmax(logits, dim=-1)
    
    # Record routing telemetry
    _record_routing_telemetry(module, 2, gate_weights.argmax(dim=-1), logits=logits)
    
    half = D // 2
    out = torch.cat([x[..., :half] * gate_weights[..., 0:1], 
                     x[..., half:2*half] * gate_weights[..., 1:2]], dim=-1)
    if D > 2 * half:
        out = torch.cat([out, x[..., 2*half:]], dim=-1)
    return out

@register_op("moe_topk")
def _op_moe_topk(module, inputs, config):
    """Sparse Mixture-of-Experts channel mixer."""
    x = inputs[0]
    B, S, D = x.shape
    
    n_experts = int(config.get("num_experts", 4))
    top_k = int(config.get("top_k", 2))
    
    if not hasattr(module, 'gate_weight'):
        return x
        
    logits = F.linear(x, module.gate_weight)
    weights, indices = logits.topk(top_k, dim=-1)
    weights = F.softmax(weights, dim=-1)
    
    # Record routing telemetry
    _record_routing_telemetry(module, n_experts, indices, logits=logits)
    
    # Simplified routing implementation for compiler
    output = torch.zeros_like(x)
    # Note: CompiledOp for moe_topk needs to manage experts as sub-modules
    # For now we use a linear-based approximation if experts not fully built
    if hasattr(module, 'experts'):
        for i, expert in enumerate(module.experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_input = x[mask]
                # Weight contribution
                expert_weight = weights[indices == i].reshape(-1, 1)
                output[mask] += expert(expert_input) * expert_weight
    else:
        # Fallback to a learned projection if experts sub-modules aren't ready
        output = F.linear(x, module.weight) if hasattr(module, 'weight') else x
        
    return output

@register_op("nm_sparse_linear")
def _op_nm_sparse_linear(module, inputs, config):
    if not hasattr(module, 'weight'): return inputs[0]
    n = int(getattr(module, "sparsity_n", config.get("n", 2)))
    m = int(getattr(module, "sparsity_m", config.get("m", 4)))
    if m <= 0 or n <= 0 or n > m or (module.weight.shape[1] % m != 0):
        _record_sparse_telemetry(module, "nm_sparse_linear", 1.0, "invalid_nm_configuration")
        return F.linear(inputs[0], module.weight)
    mask = _build_nm_mask(module.weight, n=n, m=m)
    _record_sparse_telemetry(module, "nm_sparse_linear", float(mask.mean().item()))
    return F.linear(inputs[0], module.weight * mask)

@register_op("block_sparse_linear")
def _op_block_sparse_linear(module, inputs, config):
    if not hasattr(module, 'weight'): return inputs[0]
    block_size = int(getattr(module, "block_size", config.get("block_size", 16)))
    block_density = float(getattr(module, "block_density", config.get("block_density", 0.25)))
    mask = _build_block_sparse_mask(module.weight, block_size, block_density)
    _record_sparse_telemetry(module, "block_sparse_linear", float(mask.mean().item()))
    
    if HAS_KERNELS and inputs[0].is_cuda:
        # Pass through to Triton kernel optimization
        # Note: kernel implementation handles block-skipping logic
        try:
            return kernels.triton_block_sparse_linear(inputs[0], module.weight, mask, block_size)
        except Exception:
            pass # Fallback if kernel not fully implemented/compiles
            
    return F.linear(inputs[0], module.weight * mask)

@register_op("semi_structured_2_4_linear")
def _op_semi_structured_2_4_linear(module, inputs, config):
    if not hasattr(module, 'weight'): return inputs[0]
    if not getattr(module, "sparse_kernel_ready", False) or not inputs[0].is_cuda:
        _record_sparse_telemetry(module, "semi_structured_2_4_linear", 1.0, "kernel_unavailable")
        return F.linear(inputs[0], module.weight)
    mask = _build_nm_mask(module.weight, n=2, m=4)
    _record_sparse_telemetry(module, "semi_structured_2_4_linear", float(mask.mean().item()))
    return F.linear(inputs[0], module.weight * mask)

@register_op("rmsnorm")
def _op_rmsnorm(module, inputs, _):
    if not hasattr(module, 'weight'): return inputs[0]
    x = inputs[0]
    if HAS_KERNELS and x.is_cuda:
        try:
            return kernels.triton_rmsnorm(x, module.weight)
        except Exception:
            pass
    # PyTorch fallback
    eps = 1e-6
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return (x / rms) * module.weight

@register_op("softmax_last")
def _op_softmax_last(_, inputs, __): return F.softmax(inputs[0], dim=-1)

@register_op("softmax_seq")
def _op_softmax_seq(_, inputs, __): return F.softmax(inputs[0], dim=1)

@register_op("causal_mask")
def _op_causal_mask(_, inputs, __):
    x = inputs[0]
    S = x.shape[1]
    mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
    return x * (~mask).float().unsqueeze(0).unsqueeze(-1)

@register_op("sort_seq")
def _op_sort_seq(_, inputs, __):
    x = inputs[0]
    indices = x.mean(dim=-1).argsort(dim=-1)
    return x.gather(1, indices.unsqueeze(-1).expand_as(x))

@register_op("argsort_seq")
def _op_argsort_seq(_, inputs, __):
    return inputs[0].mean(dim=-1).argsort(dim=-1).unsqueeze(-1).expand_as(inputs[0]).float()

@register_op("local_window_attn")
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

@register_op("sliding_window_mask")
def _op_sliding_window_mask(_, inputs, config):
    x = inputs[0]
    B, S, D = x.shape
    W = min(config.get("window_size", 32), S)
    row_idx = torch.arange(S, device=x.device, dtype=x.dtype).unsqueeze(1)
    col_idx = torch.arange(S, device=x.device, dtype=x.dtype).unsqueeze(0)
    dist = (row_idx - col_idx).abs()
    decay = torch.exp(-dist / max(W / 4, 1.0))
    mask = ((col_idx <= row_idx) & (dist < W)).float() * decay
    mask = mask / mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.bmm(mask.unsqueeze(0).expand(B, -1, -1), x)

@register_op("token_pool_restore")
def _op_token_pool_restore(_, inputs, __):
    x = inputs[0]
    if x.shape[1] < 2: return x
    S_half = x.shape[1] // 2
    restored = ((x[:, 0::2, :][:, :S_half] + x[:, 1::2, :][:, :S_half]) / 2.0).repeat_interleave(2, dim=1)
    if restored.shape[1] < x.shape[1]:
        restored = torch.cat([restored, x[:, -1:, :]], dim=1)
    return restored

@register_op("basis_expansion")
def _op_basis_expansion(module, inputs, _):
    if not hasattr(module, 'weight'): return inputs[0]
    w = module.weight
    expanded = torch.sin(inputs[0] * w[0]) + torch.cos(inputs[0] * w[1]) + \
               torch.sin(inputs[0] * w[2]) + torch.cos(inputs[0] * w[3])
    return expanded * 0.25

@register_op("integral_kernel")
def _op_integral_kernel(module, inputs, config):
    if not hasattr(module, 'weight'): return inputs[0]
    B, S, D = inputs[0].shape
    pos = torch.arange(S, device=inputs[0].device, dtype=inputs[0].dtype).unsqueeze(1)
    kernel = torch.exp(-float(config.get("kernel_scale", 0.25)) * (pos - pos.t()).abs().float())
    kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return F.linear(torch.bmm(kernel.unsqueeze(0).expand(B, -1, -1), inputs[0]), module.weight)

@register_op("fixed_point_iter")
def _op_fixed_point_iter(module, inputs, config):
    """
    Fixed-point iteration vectorized over the sequence and batch dimensions.
    """
    if not hasattr(module, 'weight'): return inputs[0]
    B, S, D = inputs[0].shape
    W = module.weight[:D, :]
    b = module.weight[D, :] if module.weight.shape[0] > D else torch.zeros(D, device=inputs[0].device)
    z = inputs[0]
    n_iters = max(1, int(config.get("n_iters", 3)))
    damping = max(0.0, min(1.0, float(config.get("damping", 0.5))))
    for _ in range(n_iters):
        z = (1.0 - damping) * z + damping * torch.tanh(F.linear(z, W) + b)
    return z

@register_op("rfft_seq")
def _op_rfft_seq(_, inputs, __): return torch.fft.rfft(inputs[0], dim=1).real

@register_op("irfft_seq")
def _op_irfft_seq(_, inputs, __):
    B, S_freq, D = inputs[0].shape
    return torch.fft.irfft(torch.complex(inputs[0], torch.zeros_like(inputs[0])), n=(S_freq - 1) * 2, dim=1)


def _execute_op(module: nn.Module, op_name: str, inputs: Tuple[torch.Tensor, ...],
                config: Dict) -> torch.Tensor:
    """Execute a single primitive operation via the registry."""
    if op_name in _OP_DISPATCH:
        result = _OP_DISPATCH[op_name](module, inputs, config)
        
        # Telemetry for registered math space ops (if any)
        if op_name.startswith("math_"):
            nonfinite = int((~torch.isfinite(result)).sum().item())
            if nonfinite > 0:
                result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
                telemetry = getattr(module, "mathspace_telemetry", {})
                stats = telemetry.get(op_name, {"calls": 0, "nonfinite": 0})
                stats["calls"] += 1
                stats["nonfinite"] += nonfinite
                telemetry[op_name] = stats
                setattr(module, "mathspace_telemetry", telemetry)
        return result

    # Fallback for dynamic math space ops not in _OP_DISPATCH
    from .primitives import PRIMITIVE_REGISTRY
    if op_name in PRIMITIVE_REGISTRY:
        prim = PRIMITIVE_REGISTRY[op_name]
        if hasattr(prim, 'execute_fn') and prim.execute_fn is not None:
            result = prim.execute_fn(module, *inputs)
            # Sanitize non-finite values and record telemetry
            if isinstance(result, torch.Tensor):
                nonfinite = int((~torch.isfinite(result)).sum().item())
                if nonfinite > 0:
                    result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
                    telemetry = getattr(module, "mathspace_telemetry", {})
                    stats = telemetry.get(op_name, {"calls": 0, "nonfinite_elements": 0, "sanitized_calls": 0})
                    stats["calls"] = stats.get("calls", 0) + 1
                    stats["nonfinite_elements"] = stats.get("nonfinite_elements", 0) + nonfinite
                    stats["sanitized_calls"] = stats.get("sanitized_calls", 0) + 1
                    telemetry[op_name] = stats
                    setattr(module, "mathspace_telemetry", telemetry)
                else:
                    telemetry = getattr(module, "mathspace_telemetry", {})
                    stats = telemetry.get(op_name, {"calls": 0, "nonfinite_elements": 0, "sanitized_calls": 0})
                    stats["calls"] = stats.get("calls", 0) + 1
                    telemetry[op_name] = stats
                    setattr(module, "mathspace_telemetry", telemetry)
            return result

    raise ValueError(f"Unknown op: {op_name}")


# ── Module Classes ──────────────────────────────────────────────────

class CompiledOp(nn.Module):
    """A single compiled primitive operation."""

    def __init__(self, op_name: str, config: Dict, input_shape: ShapeInfo,
                 output_shape: ShapeInfo, model_dim: int):
        super().__init__()
        self.op_name = op_name
        self.config = config
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_dim = model_dim

        op = get_primitive(op_name)
        if op.has_params:
            self._init_params(op, config, input_shape)

    def _make_param(self, shape: Tuple[int, ...], std: float = 0.02) -> nn.Parameter:
        """Create a parameter without per-parameter filesystem I/O."""
        return nn.Parameter(torch.empty(shape, dtype=torch.float32).normal_(mean=0.0, std=std))

    def _init_params(self, op: PrimitiveOp, config: Dict, input_shape: ShapeInfo):
        """Initialize learnable parameters for this op."""
        D_in = input_shape.dim
        D_out = config.get("out_dim", D_in)
        std = 1.0 / math.sqrt(D_in)

        if op.name in ("linear_proj", "linear_proj_down", "linear_proj_up"):
            self.weight = self._make_param((D_out, D_in), std=std)
        elif op.name == "fused_linear_gelu":
            self.weight = self._make_param((D_out, D_in), std=std)
            self.bias = nn.Parameter(torch.zeros(D_out))
        elif op.name == "learnable_scale":
            self.scale = nn.Parameter(torch.ones(D_in))
        elif op.name == "learnable_bias":
            self.bias = nn.Parameter(torch.zeros(D_in))
        elif op.name == "selective_scan":
            self.A_log = self._make_param((D_in,), std=0.1)
            self.dt_proj = self._make_param((D_in,), std=0.1)
            self.B_proj = self._make_param((D_in,), std=std)
            self.C_proj = self._make_param((D_in,), std=std)
        elif op.name == "conv1d_seq":
            self.conv_weight = self._make_param((D_in, 1, 3), std=1.0 / math.sqrt(3))
        elif op.name == "topk_gate":
            self.gate_proj = self._make_param((2, D_in), std=std)
        elif op.name == "moe_topk":
            n_experts = int(config.get("num_experts", 4))
            self.gate_weight = self._make_param((n_experts, D_in), std=std)
            # Create a minimal MLP for each expert to keep CompiledOp self-contained
            hidden = int(D_in * config.get("mlp_ratio", 2.0))
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(D_in, hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(hidden, D_in, bias=False)
                ) for _ in range(n_experts)
            ])
        elif op.name == "nm_sparse_linear":
            self.weight = self._make_param((D_out, D_in), std=std)
            self.sparsity_n = int(config.get("n", 2))
            self.sparsity_m = int(config.get("m", 4))
        elif op.name == "block_sparse_linear":
            self.weight = self._make_param((D_out, D_in), std=std)
            self.block_size = max(1, int(config.get("block_size", 16)))
            self.block_density = float(max(0.05, min(1.0, config.get("block_density", 0.25))))
        elif op.name == "rmsnorm":
            self.weight = nn.Parameter(torch.ones(D_in))
        elif op.name == "semi_structured_2_4_linear":
            self.weight = self._make_param((D_out, D_in), std=std)
            self.sparse_kernel_ready = bool(D_in % 4 == 0 and D_out % 4 == 0)
        elif op.name == "basis_expansion":
            self.weight = nn.Parameter(torch.randn(4, D_in) * 0.5)
        elif op.name == "integral_kernel":
            self.weight = nn.Parameter(torch.randn(D_in, D_in) * (1.0 / math.sqrt(D_in)))
        elif op.name == "fixed_point_iter":
            self.weight = nn.Parameter(torch.randn(D_in + 1, D_in) * (1.0 / math.sqrt(D_in)))
        elif op.name == "low_rank_proj":
            rank = max(D_in // 4, 1)
            self.U = nn.Parameter(torch.randn(D_in, rank) * (1.0 / math.sqrt(D_in)))
            self.V = nn.Parameter(torch.randn(rank, D_in) * (1.0 / math.sqrt(rank)))
        elif op.name == "grouped_linear":
            g = 4
            group_dim = max(D_in // g, 1)
            self.weight = nn.Parameter(torch.randn(g, group_dim, group_dim) * (1.0 / math.sqrt(group_dim)))
            self.n_groups = g
        elif op.name == "bottleneck_proj":
            rank = max(D_in // 4, 1)
            self.down = nn.Parameter(torch.randn(rank, D_in) * (1.0 / math.sqrt(D_in)))
            self.up = nn.Parameter(torch.randn(D_in, rank) * (1.0 / math.sqrt(rank)))
        elif op.name == "shared_basis_proj":
            k = 8
            self.basis = nn.Parameter(torch.randn(k, D_in) * (1.0 / math.sqrt(D_in)))
            self.mixing = nn.Parameter(torch.randn(D_in, k) * (1.0 / math.sqrt(k)))
        elif op.name == "tied_proj":
            rank = max(D_in // 4, 1)
            self.tied_weight = nn.Parameter(torch.randn(rank, D_in) * (1.0 / math.sqrt(D_in)))
        else:
            if hasattr(op, 'init_params'):
                op.init_params(self, D_in)
            else:
                self.weight = nn.Parameter(torch.randn(D_in, D_in) * (1.0 / math.sqrt(D_in)))

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Execute this primitive operation."""
        wrapper = getattr(self, '_native_wrapper', None)
        if wrapper is not None:
            result = wrapper.dispatch(self.op_name, *inputs)
            if result is not None:
                return result
        return _execute_op(self, self.op_name, inputs, self.config)


class CompiledLayer(nn.Module):
    """A compiled computation graph as a PyTorch module with memory management."""

    def __init__(self, graph: ComputationGraph):
        super().__init__()
        self.graph = graph
        self.topo_order = graph.topological_order()

        # Track consumer counts for memory reclamation
        self.consumer_counts = {}
        for nid in self.topo_order:
            node = graph.nodes[nid]
            for iid in node.input_ids:
                self.consumer_counts[iid] = self.consumer_counts.get(iid, 0) + 1

        self.ops = nn.ModuleDict()
        for nid in self.topo_order:
            node = graph.nodes[nid]
            if node.is_input: continue
            input_shapes = [graph.nodes[iid].output_shape for iid in node.input_ids]
            self.ops[str(nid)] = CompiledOp(node.op_name, node.config, 
                                            input_shapes[0] if input_shapes else ShapeInfo(),
                                            node.output_shape, graph.model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the computation graph with liveness-based memory management.

        If a ``_subgraph_dispatcher`` is attached (by the native-first
        compile pipeline), tries to execute the entire graph through the
        Rust scheduler in a single call.  Falls back to per-op dispatch
        on failure or when not all ops are native-supported.

        Tensors are deleted as soon as their last consumer finishes, minimizing
        peak VRAM usage.
        """
        # --- Subgraph dispatch fast-path ---
        dispatcher = getattr(self, '_subgraph_dispatcher', None)
        if dispatcher is not None:
            result = dispatcher.try_dispatch(x)
            if result is not None:
                return result

        node_outputs: Dict[int, torch.Tensor] = {}
        counts = self.consumer_counts.copy()
        output_id = self.graph._output_node_id
        if output_id is None:
            raise RuntimeError("Graph has no output node")

        # Track if we need aggressive reclamation (high memory pressure)
        is_cuda = x.is_cuda
        
        for nid in self.topo_order:
            node = self.graph.nodes[nid]
            if node.is_input:
                node_outputs[nid] = x
            else:
                # Build input tuple only for registered consumers
                inputs = tuple(node_outputs[iid] for iid in node.input_ids)
                node_outputs[nid] = self.ops[str(nid)](*inputs)

            # Immediately decrement counts for this node's inputs
            for iid in node.input_ids:
                counts[iid] -= 1
                # If no more consumers and not the final output, reclaim HBM
                if counts[iid] <= 0 and iid != output_id:
                    if iid in node_outputs:
                        # Explicitly clear reference to trigger prompt reclamation
                        out_to_del = node_outputs.pop(iid)
                        if is_cuda:
                            # Manually help reference counting
                            del out_to_del

        out = node_outputs.pop(output_id)
        node_outputs.clear() # Reclaim any lingering intermediates
        return out


class SynthesizedModel(nn.Module):
    """A complete language model built from synthesized layers."""

    def __init__(self, layer_graphs: List[ComputationGraph], vocab_size: int = 32000,
                 model_dim: int = 256, max_seq_len: int = 512):
        super().__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.layers = nn.ModuleList([CompiledLayer(g) for g in layer_graphs])
        self.norm = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._layer_graphs = layer_graphs

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers: x = layer(x)
        return self.lm_head(self.norm(x))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        desc = [f"SynthesizedModel(dim={self.model_dim}, layers={len(self.layers)}, params={self.param_count():,})"]
        for i, g in enumerate(self._layer_graphs):
            desc.append(f"\n  Layer {i}:\n" + "\n".join(f"    {l}" for l in g.describe().split("\n")))
        return "\n".join(desc)


def compile_graph(graph: ComputationGraph, use_ir: bool = True) -> nn.Module:
    """Compile a graph to a PyTorch module.
    
    Args:
        graph: The computation graph to compile.
        use_ir: If True (default), uses the high-performance IRExecutor path.
    """
    if use_ir:
        from .ir_executor import IRExecutor
        return IRExecutor(graph.lower_to_ir())
    return CompiledLayer(graph)


def compile_model(layer_graphs: List[ComputationGraph], vocab_size: int = 32000,
                  max_seq_len: int = 512, use_ir: bool = True) -> SynthesizedModel:
    if not layer_graphs: raise ValueError("Empty layer_graphs list")
    model = SynthesizedModel(layer_graphs, vocab_size, layer_graphs[0].model_dim, max_seq_len)
    if use_ir:
        # Replace standard layers with IR executors
        from .ir_executor import IRExecutor
        model.layers = nn.ModuleList([IRExecutor(g.lower_to_ir()) for g in layer_graphs])
    return model
