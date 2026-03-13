"""
Computation Graph Compiler

Compiles a ComputationGraph into a live PyTorch nn.Module.
Each OpNode becomes a concrete tensor operation, with learnable
parameters allocated for parameterized ops.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import get_primitive, PrimitiveOp
from .graph import ComputationGraph, ShapeInfo

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

from research.env import aria_core, HAS_ARIA_CORE


# ── Registry System ───────────────────────────────────────────────────

_OP_DISPATCH: Dict[str, Callable[[nn.Module, Tuple[torch.Tensor, ...], Dict], torch.Tensor]] = {}


def register_op(name: str):
    """Decorator to register an op implementation."""
    def decorator(fn: Callable):
        _OP_DISPATCH[name] = fn
        return fn
    return decorator


@register_op("identity")
def _op_identity(_, inputs, __):
    """Pass-through op — used by workflow_converter for uniform routing."""
    return inputs[0]


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
    """Record MoE routing statistics: entropy, expert utilization, drop rate.

    Samples every 8th call to reduce overhead while maintaining statistical accuracy.
    """
    telemetry = getattr(module, "routing_telemetry", {
        "tokens_total": 0,
        "tokens_processed": 0,
        "expert_counts": torch.zeros(n_experts, device=selected_experts.device),
        "entropy_sum": 0.0,
        "count": 0,
        "heatmap": None,
        "_call_count": -1,
    })

    telemetry["_call_count"] += 1
    B, S = selected_experts.shape[:2]
    total_tokens = B * S
    telemetry["tokens_total"] += total_tokens
    telemetry["tokens_processed"] += total_tokens

    # Sample every 8th call for expensive histogram + entropy (first call always records)
    if telemetry["_call_count"] & 7 != 0:
        telemetry["count"] += 1
        setattr(module, "routing_telemetry", telemetry)
        return

    # Expert utilization
    counts = torch.histc(selected_experts.float(), bins=n_experts, min=0, max=n_experts-1)
    telemetry["expert_counts"] += counts

    # Entropy if logits provided
    if logits is not None:
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
        telemetry["entropy_sum"] += entropy
        telemetry["count"] += 1

    # Z13: Savings and Depth ratios
    # If selected_experts contains indices of active tokens (e.g. top-k), 
    # we can estimate savings.
    # For now, we'll just track if any tokens were skipped.
    
    # Optional heatmap capture (first batch element only)
    if getattr(module, "_capture_heatmap", False) and telemetry["heatmap"] is None:
        telemetry["heatmap"] = selected_experts[0].detach().cpu().numpy().tolist()
        
    setattr(module, "routing_telemetry", telemetry)


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


def _flatten_for_kernel(x: torch.Tensor):
    """Flatten >=3D tensor to 2D for C kernels that expect (batch, dim).

    Returns (x_2d, orig_shape) so the caller can reshape back via
    out.reshape(*orig_shape[:-1], -1).
    """
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(f"_flatten_for_kernel expected Tensor, got {type(x).__name__}")
    if x.dim() < 1:
        raise RuntimeError(f"_flatten_for_kernel expected >=1D tensor, got {x.dim()}D")
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.contiguous().reshape(-1, orig_shape[-1])
    elif not x.is_contiguous():
        x = x.contiguous()
    return x, orig_shape


def _unflatten_from_kernel(out: torch.Tensor, orig_shape):
    """Reshape 2D kernel output back to match the original input shape."""
    if len(orig_shape) > 2:
        return out.reshape(*orig_shape[:-1], -1)
    return out


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

def _c(x):
    """Check if tensor is eligible for aria_core C kernels.

    C kernels don't support autograd, so skip them when gradients are needed.
    Requires at least 2D tensor with reasonable dimensions.
    """
    return (HAS_ARIA_CORE and x.device.type == "cpu"
            and x.dtype == torch.float32 and not x.requires_grad
            and x.dim() >= 1 and x.numel() > 0)


@register_op("neg")
def _op_neg(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.neg_f32(x)
    return -x

@register_op("abs")
def _op_abs(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.abs_f32(x)
    return torch.abs(x)

@register_op("exp")
def _op_exp(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.exp_f32(torch.clamp(x, -20, 20))
    return torch.exp(torch.clamp(x, -20, 20))

@register_op("log")
def _op_log(_, inputs, __):
    x = inputs[0]
    clamped = torch.clamp(x.abs(), min=1e-8)
    if _c(x): return aria_core.log_f32(clamped)
    return torch.log(clamped)

@register_op("sin")
def _op_sin(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.sin_f32(x)
    return torch.sin(x)

@register_op("cos")
def _op_cos(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.cos_f32(x)
    return torch.cos(x)

@register_op("tanh")
def _op_tanh(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.tanh_f32(x)
    return torch.tanh(x)

@register_op("sigmoid")
def _op_sigmoid(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.sigmoid_f32(x)
    return torch.sigmoid(x)

@register_op("relu")
def _op_relu(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.relu_f32(x)
    return F.relu(x)

@register_op("gelu")
def _op_gelu(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.gelu_f32(x)
    return F.gelu(x)

@register_op("silu")
def _op_silu(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.silu_f32(x)
    return F.silu(x)

@register_op("sqrt")
def _op_sqrt(_, inputs, __):
    x = inputs[0]
    clamped = torch.clamp(x.abs(), min=1e-8)
    if _c(x): return aria_core.sqrt_f32(clamped)
    return torch.sqrt(clamped)

@register_op("square")
def _op_square(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.square_f32(x)
    return x * x

@register_op("sign_ste")
def _op_sign_ste(_, inputs, __):
    x = inputs[0]
    if _c(x):
        signs = aria_core.sign_ste_f32(x)
        return x + (signs - x).detach()
    signs = torch.sign(x)
    return x + (signs - x).detach()

@register_op("reciprocal")
def _op_reciprocal(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.reciprocal_f32(x)
    # Stable reciprocal: push x away from zero by eps in the direction of its sign
    eps = 1e-6
    sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    return 1.0 / (x + eps * sign)

@register_op("add")
def _op_add(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a) and a.numel() == b.numel():
        return aria_core.add_f32(a, b)
    return a + b

@register_op("mul")
def _op_mul(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a): return aria_core.mul_f32(a, b)
    return a * b

@register_op("sub")
def _op_sub(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a): return aria_core.sub_f32(a, b)
    return a - b

@register_op("div_safe")
def _op_div_safe(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a): return aria_core.div_safe_f32(a, b)
    # Stable division: push denominator away from zero by eps in the direction of its sign
    eps = 1e-6
    sign = torch.where(b >= 0, torch.ones_like(b), -torch.ones_like(b))
    return a / (b + eps * sign)

@register_op("maximum")
def _op_maximum(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a): return aria_core.maximum_f32(a, b)
    return torch.maximum(a, b)

@register_op("minimum")
def _op_minimum(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a): return aria_core.minimum_f32(a, b)
    return torch.minimum(a, b)

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
    if a.dim() != 3 or b.dim() != 3:
        return a # Fallback for safety
    
    # CASE 1: Standard Attention Pattern [B, S, D] @ [B, S, D]
    if a.shape[-1] == b.shape[-1]:
        scale = math.sqrt(float(a.shape[-1]))
        # (B, S, D) @ (B, D, S) -> (B, S, S)
        scores = torch.bmm(a, b.transpose(-2, -1)) / scale
        S = a.shape[1]
        if S > 1:
            mask = torch.triu(torch.ones(S, S, device=a.device, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(mask, float('-inf'))
        # (B, S, S) @ (B, S, D) -> (B, S, D)
        return torch.bmm(F.softmax(scores, dim=-1), b)
    
    # CASE 2: Linear Algebra Pattern [B, S, D] @ [B, D, K]
    if a.shape[2] == b.shape[1]:
        if _c(a): return aria_core.matmul_f32(a, b)
        return torch.bmm(a, b)
        
    # CASE 3: Incompatible fallback
    return a

@register_op("outer_product")
def _op_outer_product(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a): return aria_core.mul_f32(a, b)
    return a * b

@register_op("transpose_sd")
def _op_transpose_sd(_, inputs, __):
    return inputs[0].transpose(1, 2).contiguous().transpose(1, 2)

@register_op("split2")
def _op_split2(_, inputs, config):
    part = int(config.get("part", 0))
    x = inputs[0]
    w = x.shape[-1] // 2
    if part == 0: return x[..., :w].contiguous()
    return x[..., w:2*w].contiguous()

@register_op("split3")
def _op_split3(_, inputs, config):
    part = int(config.get("part", 0))
    x = inputs[0]
    w = x.shape[-1] // 3
    if part == 0: return x[..., :w].contiguous()
    if part == 1: return x[..., w:2*w].contiguous()
    return x[..., 2*w:3*w].contiguous()

@register_op("split4")
def _op_split4(_, inputs, config):
    part = int(config.get("part", 0))
    x = inputs[0]
    w = x.shape[-1] // 4
    if part == 0: return x[..., :w].contiguous()
    if part == 1: return x[..., w:2*w].contiguous()
    if part == 2: return x[..., 2*w:3*w].contiguous()
    return x[..., 3*w:4*w].contiguous()

@register_op("concat")
def _op_concat(_, inputs, __):
    return torch.cat(inputs, dim=-1)

@register_op("roll_seq")
def _op_roll_seq(_, inputs, __): return torch.roll(inputs[0], shifts=1, dims=1)

@register_op("roll_neg")
def _op_roll_neg(_, inputs, __): return torch.roll(inputs[0], shifts=-1, dims=1)

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
    x = inputs[0]
    if _c(x):
        xf, orig_shape = _flatten_for_kernel(x)
        bias = getattr(module, 'bias', None)
        out = aria_core.linear_f32(xf, module.weight, bias)
        return _unflatten_from_kernel(out, orig_shape)
    return F.linear(x, module.weight, getattr(module, 'bias', None))

@register_op("fused_linear_gelu")
def _op_fused_linear_gelu(module, inputs, _):
    if not hasattr(module, 'weight'): return inputs[0]
    x = inputs[0]
    if _c(x):
        xf, orig_shape = _flatten_for_kernel(x)
        bias = getattr(module, 'bias', None)
        if bias is None:
            bias = torch.zeros(module.weight.shape[0], device=x.device, dtype=x.dtype)
        out = aria_core.fused_linear_gelu_f32(xf, module.weight, bias)
        return _unflatten_from_kernel(out, orig_shape)
    if HAS_KERNELS and x.is_cuda:
        bias = getattr(module, 'bias', None)
        return kernels.fused_linear_gelu(x, module.weight, bias)
    out = F.linear(x, module.weight)
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
    # Ensure dt matches input dim D
    dt = F.softplus(module.dt_proj[:D])
    # Clamp to [-10, -0.05]: upper bound -0.05 ensures minimum 5% decay
    # per step, bounding kernel sum to ~20 (geometric series) instead of
    # S (~128).  log_a=0 creates a pure integrator (no decay) whose
    # gradient amplification is O(S²) — the primary cause of grad explosion
    # in selective_scan architectures.
    log_a = (A * dt).clamp(-10, -0.05)  # (D,)

    u = torch.sigmoid(module.B_proj(x)) * x  # (B, S, D)

    # Vectorized linear recurrence via causal convolution with exponential kernel.
    # h_t = a * h_{t-1} + u_t, kernel = [a^{S-1}, ..., a, 1].
    # Use log-space arithmetic for numerical stability.
    indices = torch.arange(S, device=x.device, dtype=x.dtype)
    # log_kernel[d, 1, s] = log_a[d] * (S-1-s)
    log_kernel = log_a.view(D, 1, 1) * (S - 1 - indices).view(1, 1, S)
    kernel = torch.exp(log_kernel)  # (D, 1, S) — single exp, stable

    u_swapped = u.permute(0, 2, 1) # (B, D, S)
    # Causal convolution via padding
    h_swapped = F.conv1d(F.pad(u_swapped, (S - 1, 0)), kernel, groups=D) # (B, D, S)
    h = h_swapped.permute(0, 2, 1) # (B, S, D)

    C_x = torch.sigmoid(module.C_proj(x))
    return C_x * h

@register_op("conv1d_seq")
def _op_conv1d_seq(module, inputs, _):
    if not hasattr(module, 'conv_weight'): return inputs[0]
    x = inputs[0]
    if x.ndim == 2:
        x = x.unsqueeze(0)
    B, S, D = x.shape
    if _c(x):
        conv_bias = getattr(module, "conv_bias", None)
        if conv_bias is None:
            conv_bias = torch.zeros(D, device=x.device, dtype=x.dtype)
        return aria_core.conv1d_seq_f32(x, module.conv_weight, conv_bias)
    # Causal padding: pad (kernel_size - 1) on the left
    kernel_size = module.conv_weight.shape[2]
    x_padded = F.pad(x.transpose(1, 2), (kernel_size - 1, 0))
    out = F.conv1d(x_padded, module.conv_weight, groups=D)
    return out.transpose(1, 2)

@register_op("topk_gate")
def _op_topk_gate(module, inputs, _):
    if not hasattr(module, 'gate_proj'): return inputs[0]
    x = inputs[0]
    B, S, D = x.shape
    if (
        _c(x)
        and hasattr(aria_core, "topk_gate_f32")
        and isinstance(module.gate_proj, torch.Tensor)
        and module.gate_proj.dim() == 2
        and module.gate_proj.shape[0] >= 2
        and module.gate_proj.shape[1] == D
    ):
        try:
            native_out = aria_core.topk_gate_f32(x, module.gate_proj, 2)
            if isinstance(native_out, torch.Tensor) and native_out.shape == x.shape:
                return native_out
        except Exception:
            pass
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
    
    # Weighted expert contributions
    if hasattr(module, 'experts'):
        output = torch.zeros_like(x)
        for i, expert in enumerate(module.experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_input = x[mask]
                exp_mask = (indices == i)
                expert_weight = weights[exp_mask].reshape(-1, 1)
                output[mask] = output[mask] + expert(expert_input) * expert_weight
    else:
        output = F.linear(x, module.weight) if hasattr(module, 'weight') else x

    return output


@register_op("moe_2expert")
def _op_moe_2expert(module, inputs, config):
    """Lightweight 2-expert MoE with learned gating."""
    x = inputs[0]
    B, S, D = x.shape

    if not hasattr(module, 'gate_proj'):
        return x

    # Compute gate scores
    logits = F.linear(x, module.gate_proj)  # (B, S, 2)
    weights = F.softmax(logits, dim=-1)     # (B, S, 2)

    # Record routing telemetry
    _record_routing_telemetry(module, 2, weights.argmax(dim=-1), logits=logits)

    # Each expert is a simple linear projection
    e0 = F.linear(x, module.expert_0_weight)  # (B, S, D)
    e1 = F.linear(x, module.expert_1_weight)  # (B, S, D)

    # Weighted combination
    output = weights[..., 0:1] * e0 + weights[..., 1:2] * e1
    return output

@register_op("swiglu_mlp")
def _op_swiglu_mlp(module, inputs, _):
    """SwiGLU MLP channel mixer."""
    x = inputs[0]
    if not hasattr(module, 'gate_proj'):
        return x
    if _c(x) and x.dim() >= 2:
        x2, orig = _flatten_for_kernel(x)
        y = aria_core.swiglu_f32(
            x2, module.gate_proj.weight, module.up_proj.weight, module.down_proj.weight,
            getattr(module.gate_proj, 'bias', None),
            getattr(module.up_proj, 'bias', None),
            getattr(module.down_proj, 'bias', None),
        )
        return _unflatten_from_kernel(y, orig)
    return module.down_proj(F.silu(module.gate_proj(x)) * module.up_proj(x))

@register_op("rwkv_channel")
def _op_rwkv_channel(module, inputs, _):
    """RWKV-style channel mixing with time-shift."""
    x = inputs[0]
    if not hasattr(module, 'mix_k'):
        return x
    if _c(x) and x.ndim == 3:
        return aria_core.rwkv_channel_f32(
            x, module.mix_k.data, module.mix_r.data,
            module.key_proj.weight, module.receptance_proj.weight, module.value_proj.weight,
        )
    # Safe causal time-shift for 3D tensors (B, S, D)
    if x.ndim == 3:
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
    else:
        shifted = x
    xk = x * module.mix_k + shifted * (1 - module.mix_k)
    xr = x * module.mix_r + shifted * (1 - module.mix_r)
    # Receptance-weighted gated linear update
    k = torch.square(torch.relu(module.key_proj(xk)))
    return torch.sigmoid(module.receptance_proj(xr)) * module.value_proj(k)

@register_op("softmax_attention")
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

@register_op("linear_attention")
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

@register_op("graph_attention")
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

@register_op("state_space")
def _op_state_space(module, inputs, _):
    """S4-style state space mixer with parallel scan via causal convolution."""
    if not hasattr(module, 'ssm_A'):
        return inputs[0]
    x = inputs[0]
    B, S, D = x.shape
    N = module.ssm_state_dim
    
    # dt: (B, S, D)
    dt = F.softplus(module.ssm_dt(x))
    # A: (D, N), dt: (B, S, D) -> log_a: (B, S, D, N)
    log_a = module.ssm_A.view(1, 1, D, N) * dt.unsqueeze(-1)
    # Clamp log_a for stability: -10 to 0 (decay factor 4e-5 to 1.0)
    log_a = torch.clamp(log_a, min=-10.0, max=0.0)
    
    # b_x: (B, S, D, N)
    b_x = module.ssm_B(x).view(B, S, D, N)
    
    # Parallel scan approximation via exponential decay convolution.
    # For simplicity and correctness in the synthesis context, we'll use
    # the same vectorized scan as selective_scan but extended to state_dim N.
    # h[t] = sum_{i=0}^t exp(sum_{j=i+1}^t log_a[j]) * b_x[i]
    
    # In state_space, log_a depends on x, so a simple conv1d with constant kernel 
    # only works if log_a is constant over time. If not, we need a true parallel scan.
    # For the synthesis baseline, we'll use the average log_a over the sequence
    # to allow vectorized execution while preserving some input-dependence.
    avg_log_a = log_a.mean(dim=1) # (B, D, N)
    
    indices = torch.arange(S, device=x.device, dtype=x.dtype)
    # kernel: (B, D, N, S)
    log_kernel = avg_log_a.unsqueeze(-1) * (S - 1 - indices).view(1, 1, 1, S)
    kernel = torch.exp(log_kernel)
    
    # Reshape for grouped conv1d: (B*D*N, 1, S)
    kernel_grouped = kernel.view(B * D * N, 1, S)
    u_swapped = b_x.permute(0, 2, 3, 1).reshape(1, B * D * N, S)
    
    h_swapped = F.conv1d(F.pad(u_swapped, (S - 1, 0)), kernel_grouped, groups=B * D * N)
    h = h_swapped.view(B, D, N, S).permute(0, 3, 1, 2) # (B, S, D, N)
    
    y = module.ssm_C(h.reshape(B, S, D * N))
    return y + x * module.ssm_D

@register_op("conv_only")
def _op_conv_only(module, inputs, _):
    """Depthwise causal convolution sequence mixer."""
    x = inputs[0]
    if not hasattr(module, 'conv_dw'):
        return x
    B, S, D = x.shape
    out = module.conv_dw(x.transpose(1, 2))[:, :, :S].transpose(1, 2)
    return module.conv_proj(out)

@register_op("nm_sparse_linear")
def _op_nm_sparse_linear(module, inputs, config):
    if not hasattr(module, 'weight'): return inputs[0]
    n = int(getattr(module, "sparsity_n", config.get("n", 2)))
    m = int(getattr(module, "sparsity_m", config.get("m", 4)))
    if m <= 0 or n <= 0 or n > m or (module.weight.shape[1] % m != 0):
        _record_sparse_telemetry(module, "nm_sparse_linear", 1.0, "invalid_nm_configuration")
        return F.linear(inputs[0], module.weight)
    
    if HAS_ARIA_CORE and inputs[0].device.type == "cpu" and inputs[0].dtype == torch.float32:
        mask = aria_core.nm_sparse_mask_f32(module.weight, n, m)
        _record_sparse_telemetry(module, "nm_sparse_linear", float(mask.float().mean().item()))
        return F.linear(inputs[0], module.weight * mask.float())

    mask = _build_nm_mask(module.weight, n=n, m=m)
    _record_sparse_telemetry(module, "nm_sparse_linear", float(mask.mean().item()))
    return F.linear(inputs[0], module.weight * mask)

@register_op("block_sparse_linear")
def _op_block_sparse_linear(module, inputs, config):
    if not hasattr(module, 'weight'): return inputs[0]
    block_size = int(getattr(module, "block_size", config.get("block_size", 16)))
    block_density = float(getattr(module, "block_density", config.get("block_density", 0.25)))
    
    if HAS_ARIA_CORE and inputs[0].device.type == "cpu" and inputs[0].dtype == torch.float32 and not inputs[0].requires_grad:
        # Generate block mask (coarse)
        mask = _build_block_sparse_mask(module.weight, block_size, block_density)
        # Convert to uint8 for kernel (needs downsampling if we want true block sparsity optimization)
        # For CPU reference, we can just use linear_block_sparse_f32 with uint8 mask
        m_rows = module.weight.shape[0] // block_size
        m_cols = module.weight.shape[1] // block_size
        if m_rows > 0 and m_cols > 0:
            block_mask_uint8 = mask[:m_rows*block_size:block_size, :m_cols*block_size:block_size].to(torch.uint8)
            bias = getattr(module, 'bias', None)
            x, orig_shape = _flatten_for_kernel(inputs[0])
            out = aria_core.linear_block_sparse_f32(x, module.weight, block_mask_uint8, bias, block_size)
            out = _unflatten_from_kernel(out, orig_shape)
            _record_sparse_telemetry(module, "block_sparse_linear", float(mask.mean().item()))
            return out

    mask = _build_block_sparse_mask(module.weight, block_size, block_density)
    _record_sparse_telemetry(module, "block_sparse_linear", float(mask.mean().item()))
    
    if HAS_KERNELS and inputs[0].is_cuda:
        # Pass through to Triton kernel optimization
        try:
            return kernels.triton_block_sparse_linear(inputs[0], module.weight, mask, block_size)
        except Exception:
            pass
            
    return F.linear(inputs[0], module.weight * mask)

@register_op("low_rank_proj")
def _op_low_rank_proj(module, inputs, _):
    if not hasattr(module, 'U') or not hasattr(module, 'V'): return inputs[0]
    if HAS_ARIA_CORE and inputs[0].device.type == "cpu" and inputs[0].dtype == torch.float32:
        bias = getattr(module, 'bias', None)
        x, orig_shape = _flatten_for_kernel(inputs[0])
        # C kernel expects U:[rank, dim_in], V:[dim_out, rank] but Python stores
        # U:[dim_in, rank], V:[rank, dim_out] — transpose both for the kernel
        out = aria_core.linear_low_rank_f32(
            x, module.U.t().contiguous(), module.V.t().contiguous(), bias
        )
        return _unflatten_from_kernel(out, orig_shape)
    # PyTorch fallback
    out = F.linear(F.linear(inputs[0], module.U.t()), module.V.t())
    if hasattr(module, 'bias'): out = out + module.bias
    return out

@register_op("grouped_linear")
def _op_grouped_linear(module, inputs, _):
    if not hasattr(module, 'weight'): return inputs[0]
    if HAS_ARIA_CORE and inputs[0].device.type == "cpu" and inputs[0].dtype == torch.float32:
        bias = getattr(module, 'bias', None)
        x, orig_shape = _flatten_for_kernel(inputs[0])
        out = aria_core.linear_grouped_f32(x, module.weight, bias, module.n_groups)
        return _unflatten_from_kernel(out, orig_shape)
    # PyTorch fallback
    x = inputs[0]
    B, S, D = x.shape
    g = module.n_groups
    group_dim = D // g
    usable = group_dim * g
    x_groups = x[..., :usable].view(B, S, g, group_dim)
    out_groups = torch.einsum('bsgd,gde->bsge', x_groups, module.weight)
    out = out_groups.reshape(B, S, usable)
    if usable < D:
        out = torch.cat([out, x[..., usable:]], dim=-1)
    return out

@register_op("bottleneck_proj")
def _op_bottleneck_proj(module, inputs, _):
    if not hasattr(module, 'down') or not hasattr(module, 'up'): return inputs[0]
    if HAS_ARIA_CORE and inputs[0].device.type == "cpu" and inputs[0].dtype == torch.float32:
        b_down = getattr(module, 'bias_down', None)
        b_up = getattr(module, 'bias_up', None)
        x, orig_shape = _flatten_for_kernel(inputs[0])
        out = aria_core.linear_bottleneck_f32(x, module.down, module.up, b_down, b_up)
        return _unflatten_from_kernel(out, orig_shape)
    # PyTorch fallback
    hidden = F.gelu(F.linear(inputs[0], module.down))
    return F.linear(hidden, module.up)

@register_op("shared_basis_proj")
def _op_shared_basis_proj(module, inputs, _):
    if not hasattr(module, 'mixing') or not hasattr(module, 'basis'): return inputs[0]
    if HAS_ARIA_CORE and inputs[0].device.type == "cpu" and inputs[0].dtype == torch.float32:
        x, orig_shape = _flatten_for_kernel(inputs[0])
        out = aria_core.linear_shared_basis_f32(x, module.mixing, module.basis)
        return _unflatten_from_kernel(out, orig_shape)
    # PyTorch fallback
    return inputs[0] @ module.mixing @ module.basis

@register_op("tied_proj")
def _op_tied_proj(module, inputs, _):
    if not hasattr(module, 'tied_weight'): return inputs[0]
    if HAS_ARIA_CORE and inputs[0].device.type == "cpu" and inputs[0].dtype == torch.float32:
        b_down = getattr(module, 'bias_down', None)
        b_up = getattr(module, 'bias_up', None)
        x, orig_shape = _flatten_for_kernel(inputs[0])
        out = aria_core.linear_tied_f32(x, module.tied_weight, b_down, b_up)
        return _unflatten_from_kernel(out, orig_shape)
    # PyTorch fallback
    hidden = F.gelu(F.linear(inputs[0], module.tied_weight))
    return F.linear(hidden, module.tied_weight.t())

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
    if _c(x): return aria_core.rmsnorm_f32(x, module.weight, 1e-6)
    if HAS_KERNELS and x.is_cuda:
        try:
            return kernels.triton_rmsnorm(x, module.weight)
        except Exception:
            pass
    eps = 1e-6
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return (x / rms) * module.weight

@register_op("layernorm")
def _op_layernorm(module, inputs, _):
    if not hasattr(module, 'weight'): return inputs[0]
    x = inputs[0]
    if _c(x): return aria_core.layernorm_f32(x, module.weight, module.bias, 1e-5)
    return F.layer_norm(x, [x.shape[-1]], module.weight, module.bias)

@register_op("gated_linear")
def _op_gated_linear(module, inputs, _):
    if not hasattr(module, 'linear_weight'): return inputs[0]
    x = inputs[0]
    if _c(x) and x.ndim == 3:
        out = aria_core.gated_linear_f32(
            x, module.linear_weight, module.linear_bias,
            module.gate_weight, module.gate_bias)
        if out.ndim == x.ndim:
            return out
    linear = F.linear(x, module.linear_weight, module.linear_bias)
    gate = torch.sigmoid(F.linear(x, module.gate_weight, module.gate_bias))
    return linear * gate

@register_op("rwkv_time_mixing")
def _op_rwkv_time_mixing(module, inputs, _):
    """RWKV WKV attention optimized with parallel scan semantics."""
    if not hasattr(module, 'W_k'): return inputs[0]
    x = inputs[0]
    if (
        _c(x)
        and hasattr(aria_core, "rwkv_time_mixing_f32")
        and hasattr(module, "w_decay")
        and hasattr(module, "u_bonus")
        and hasattr(module, "W_k")
        and hasattr(module, "W_v")
        and hasattr(module, "W_r")
        and hasattr(module, "W_o")
    ):
        out_native = aria_core.rwkv_time_mixing_f32(
            x,
            module.w_decay,
            module.u_bonus,
            module.W_k,
            module.W_v,
            module.W_r,
        )
        return F.linear(out_native, module.W_o)
    B, S, D = x.shape

    k = F.linear(x, module.W_k)
    v = F.linear(x, module.W_v)
    r_raw = F.linear(x, module.W_r)

    # C kernel fast path: fused WKV scan (handles sigmoid internally)
    if (
        _c(k)
        and hasattr(aria_core, "rwkv_wkv_scan_f32")
        and not k.requires_grad
    ):
        out = aria_core.rwkv_wkv_scan_f32(
            k.contiguous(), v.contiguous(), r_raw.contiguous(),
            module.w_decay, module.u_bonus,
        )
        return F.linear(out, module.W_o)

    r = torch.sigmoid(r_raw)
    w = -torch.exp(module.w_decay)
    u = module.u_bonus

    if S <= 128:
        exp_w = torch.exp(w).unsqueeze(0)
        wkv = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        wkv_denom = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            kt, vt = k[:, t], v[:, t]
            p = torch.exp(u + kt)
            outputs.append(r[:, t] * (wkv + p * vt) / (wkv_denom + p).clamp(min=1e-8))
            wkv = (wkv * exp_w) + torch.exp(kt) * vt
            wkv_denom = (wkv_denom * exp_w) + torch.exp(kt)
        out = torch.stack(outputs, dim=1)
    else:
        out = r * v

    return F.linear(out, module.W_o)

@register_op("padic_residual")
def _op_padic_residual(module, inputs, config):
    """Multi-resolution p-adic residual connection."""
    from ..mathspaces.padic import execute_padic_residual
    return execute_padic_residual(module, inputs[0])

@register_op("embedding_lookup")
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
        # RoPE backward: rotate by -angle = apply RoPE to [-x_odd, x_even] pairs
        # Equivalent to: cos(-a)*g_even - sin(-a)*g_odd = cos(a)*g_even + sin(a)*g_odd
        # But simpler: just negate sin in the rotation
        B, S, D = ctx.shape
        g = grad_output.contiguous()
        half_dim = D // 2
        pos = torch.arange(S, device=g.device, dtype=g.dtype).unsqueeze(1)
        freqs = 1.0 / (ctx.theta_base ** (torch.arange(0, D, 2, device=g.device, dtype=g.dtype) / D))
        angles = pos * freqs.unsqueeze(0)
        cos_a = torch.cos(angles).unsqueeze(0)
        sin_a = torch.sin(angles).unsqueeze(0)
        g1, g2 = g[..., 0::2], g[..., 1::2]
        grad_in = torch.zeros_like(g)
        # Inverse rotation (transpose of rotation matrix)
        grad_in[..., 0::2] = g1 * cos_a + g2 * sin_a
        grad_in[..., 1::2] = -g1 * sin_a + g2 * cos_a
        return grad_in, None

@register_op("rope_rotate")
def _op_rope_rotate(_, inputs, __):
    x = inputs[0]
    if HAS_ARIA_CORE and hasattr(aria_core, "rope_rotate_f32") and x.device.type == "cpu" and x.dtype == torch.float32:
        return _RopeRotateC.apply(x, 10000.0)
    B, S, D = x.shape
    pos = torch.arange(S, device=x.device, dtype=x.dtype).unsqueeze(1)
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=x.device, dtype=x.dtype) / D))
    angles = pos * freqs.unsqueeze(0)
    cos_a = torch.cos(angles).unsqueeze(0)
    sin_a = torch.sin(angles).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.zeros_like(x)
    out[..., 0::2] = x1 * cos_a - x2 * sin_a
    out[..., 1::2] = x1 * sin_a + x2 * cos_a
    return out

@register_op("cosine_similarity")
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

@register_op("gather_topk")
def _op_gather_topk(_, inputs, config):
    x, scores = inputs[0], inputs[1]
    k = min(int(config.get("k", 4)), x.shape[1])
    if scores.dim() == 3:
        scores = scores.squeeze(-1)
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

@register_op("softmax_last")
def _op_softmax_last(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.softmax_f32(x)
    return F.softmax(x, dim=-1)

@register_op("softmax_seq")
def _op_softmax_seq(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.softmax_seq_f32(x)
    return F.softmax(x, dim=1)

@register_op("causal_mask")
def _op_causal_mask(_, inputs, __):
    """Causal integration: every token becomes the average of itself and all previous tokens.
    This is a strictly causal 'mixing' operation that prevents future lookahead.
    """
    x = inputs[0]  # (B, S, D)
    if _c(x) and hasattr(aria_core, "causal_mask_f32"):
        return aria_core.causal_mask_f32(x)
    # Using cumulative sum / counts is O(S) and strictly causal
    return torch.cumsum(x, dim=1) / torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)

@register_op("sort_seq")
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

@register_op("argsort_seq")
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

@register_op("token_pool_restore")
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

@register_op("route_topk")
def _op_route_topk(module, inputs, config):
    """Top-k routing: zero out all but top-k positions along last dim.

    Input:  (B, S, D)
    Output: (B, S, D) with only the top-k values per (B, S) slice kept.
    """
    x = inputs[0]
    k = min(int(config.get("k", 1)), x.shape[-1])
    topk_vals, topk_idx = x.topk(k, dim=-1)  # (B, S, k)
    _record_routing_telemetry(module, x.shape[-1], topk_idx, logits=x)
    # Build sparse mask and scatter top-k values back
    mask = torch.zeros_like(x)
    mask.scatter_(-1, topk_idx, 1.0)
    # STE: forward uses hard mask, backward passes through
    return x * (mask.detach() - x.detach() + x)

@register_op("route_lanes")
def _op_route_lanes(module, inputs, config):
    scores = inputs[0] # (B, S, L)
    if HAS_ARIA_CORE and scores.device.type == "cpu" and scores.dtype == torch.float32:
        lane_indices = aria_core.route_lane_argmax_f32(scores)
        _record_routing_telemetry(module, scores.shape[2], lane_indices, logits=scores)
        return lane_indices
    # Fallback
    lane_indices = scores.argmax(dim=-1)
    _record_routing_telemetry(module, scores.shape[2], lane_indices, logits=scores)
    return lane_indices

@register_op("route_recursion")
def _op_route_recursion(module, inputs, config):
    scores = inputs[0] # (B, S, Dp)
    max_depth = scores.shape[-1]
    if HAS_ARIA_CORE and scores.device.type == "cpu" and scores.dtype == torch.float32:
        depth = aria_core.route_recursion_depth_f32(scores)
    else:
        # Fallback
        depth = scores.argmax(dim=-1) + 1
    _record_routing_telemetry(module, max_depth, depth, logits=scores)
    return depth

@register_op("token_merging")
@register_op("token_merge")
def _op_token_merge(module, inputs, config):
    x = inputs[0]
    n_keep = int(config.get("n_keep", x.shape[1] // 2))
    seq_len = x.shape[1]
    if HAS_ARIA_CORE and x.device.type == "cpu" and x.dtype == torch.float32:
        y, restore_map = aria_core.token_merge_simple_f32(x, n_keep)
    else:
        # Fallback: simple truncation
        y = x[:, :n_keep, :]
        restore_map = torch.arange(seq_len, device=x.device).expand(x.shape[0], -1)
    # Record merge telemetry — tokens_processed = n_keep, tokens_total = seq_len
    merge_telem = getattr(module, "routing_telemetry", {
        "tokens_total": 0, "tokens_processed": 0,
        "merge_kept": 0, "merge_dropped": 0,
        "expert_counts": torch.zeros(1, device=x.device),
        "entropy_sum": 0.0, "count": 0, "heatmap": None,
    })
    B = x.shape[0]
    merge_telem["tokens_total"] += B * seq_len
    merge_telem["tokens_processed"] += B * n_keep
    merge_telem["merge_kept"] = merge_telem.get("merge_kept", 0) + B * n_keep
    merge_telem["merge_dropped"] = merge_telem.get("merge_dropped", 0) + B * (seq_len - n_keep)
    merge_telem["count"] = merge_telem.get("count", 0) + 1
    setattr(module, "routing_telemetry", merge_telem)
    # Restore to original length with causal-safe indexing:
    # Position i can only map to kept positions <= i (never look ahead)
    B_size, S_orig = restore_map.shape
    causal_limit = torch.arange(S_orig, device=restore_map.device).unsqueeze(0).expand(B_size, -1)
    causal_limit = causal_limit.clamp(max=y.shape[1] - 1)
    restore_map = restore_map.clamp(0, y.shape[1] - 1).minimum(causal_limit)
    return y.gather(1, restore_map.unsqueeze(-1).expand(-1, -1, x.shape[2]))


_op_token_merging = _op_token_merge

# ── Routing Control Ops (Phase 2) ────────────────────────────────────

def _routing_scores_from_x(x: torch.Tensor) -> torch.Tensor:
    # Simple, deterministic score: mean over channels
    return x.mean(dim=-1)

@register_op("mod_topk")
def _op_mod_topk(module, inputs, config):
    x = inputs[0]
    B, S, D = x.shape
    capacity = float(config.get("capacity_factor", 0.75))
    scores = _routing_scores_from_x(x)
    # Causal sparsity: deterministic stride-based mask that keeps
    # ~capacity fraction of positions without peeking at future tokens.
    stride = max(1, int(1.0 / max(1.0 - capacity, 0.01)))
    pos = torch.arange(S, device=x.device)
    keep_mask = ((pos % stride) != (stride - 1)).float().unsqueeze(0).expand(B, -1)
    cumsum = scores.cumsum(dim=-1)
    counts = torch.arange(1, S + 1, device=scores.device, dtype=scores.dtype)
    causal_mean = cumsum / counts
    soft_gate = torch.sigmoid(4.0 * (scores - causal_mean))
    gate = soft_gate * keep_mask
    _record_routing_telemetry(module, S, (gate > 0.5).long(), logits=scores)
    return x * gate.unsqueeze(-1)

@register_op("early_exit")
def _op_early_exit(module, inputs, config):
    x = inputs[0]
    threshold = float(config.get("threshold", 0.5))
    scores = _routing_scores_from_x(x)
    gate = torch.sigmoid(scores)
    keep = (gate > threshold).float()
    _record_routing_telemetry(module, 2, keep.long(), logits=gate)
    return x * keep.unsqueeze(-1)

@register_op("cascade")
def _op_cascade(module, inputs, config):
    x = inputs[0]
    threshold = float(config.get("threshold", 0.5))
    scores = _routing_scores_from_x(x)
    gate = torch.sigmoid(scores)
    _record_routing_telemetry(module, 2, (gate > threshold).long(), logits=gate)
    return x * gate.unsqueeze(-1)

@register_op("speculative")
def _op_speculative(module, inputs, config):
    x = inputs[0]
    threshold = float(config.get("threshold", 0.5))
    scores = _routing_scores_from_x(x)
    gate = torch.sigmoid(scores)
    keep = (gate > threshold).float()
    _record_routing_telemetry(module, 2, keep.long(), logits=gate)
    # Mild effect: scale rather than drop
    return x * (0.5 + 0.5 * gate).unsqueeze(-1)

@register_op("adaptive_recursion")
def _op_adaptive_recursion(module, inputs, config):
    x = inputs[0]
    max_depth = int(config.get("max_depth", 3))
    max_depth = max(1, min(6, max_depth))
    scores = _routing_scores_from_x(x)
    depth_scores = torch.stack([scores + (i * 0.1) for i in range(max_depth)], dim=-1)
    if HAS_ARIA_CORE and depth_scores.device.type == "cpu" and depth_scores.dtype == torch.float32:
        depth = aria_core.route_recursion_depth_f32(depth_scores)
    else:
        depth = depth_scores.argmax(dim=-1) + 1
    scale = 1.0 + 0.05 * depth.float()
    return x * scale.unsqueeze(-1)

# ── Exotic Ops (Phase 4) ─────────────────────────────────────────────

@register_op("adaptive_lane_mixer")
def _op_adaptive_lane_mixer(module, inputs, config):
    """Routes tokens to 'fast' vs 'deep' lanes based on learned difficulty."""
    x = inputs[0]
    B, S, D = x.shape
    
    if not hasattr(module, 'gate_proj'): return x
    
    # Compute 3-way gate: [Fast, Medium, Hard]
    logits = F.linear(x, module.gate_proj)
    weights = F.softmax(logits, dim=-1)
    
    _record_routing_telemetry(module, 3, weights.argmax(dim=-1), logits=logits)
    
    # Experts: 0=Identity(Fast), 1=LowRank(Medium), 2=MLP(Hard)
    out = x * weights[..., 0:1] # Fast lane: direct skip
    
    # Medium lane: Low-rank
    if hasattr(module, 'U_mid'):
        mid = F.linear(F.linear(x, module.U_mid), module.V_mid)
        out = out + mid * weights[..., 1:2]
        
    # Hard lane: MLP
    if hasattr(module, 'heavy_mlp'):
        hard = module.heavy_mlp(x)
        out = out + hard * weights[..., 2:3]
        
    return out

@register_op("mixed_recursion_gate")
def _op_mixed_recursion_gate(module, inputs, config):
    """Tokens re-enter block with different parameters each recursion.
    Depth is conditional on input difficulty score (inputs[1]).
    """
    x, scores = inputs[0], inputs[1]
    max_depth = int(config.get("max_depth", 3))
    
    if not hasattr(module, 'step_projs'): return x
    
    # Determine depth per token from scores
    depths = scores.argmax(dim=-1) # [B, S] in range [0, max_depth-1]
    
    out = x
    # Current implementation: sequential application up to max_depth
    # But only tokens whose depth >= current step get the update
    for i in range(max_depth):
        mask = (depths >= i).float().unsqueeze(-1)
        # Apply transformation for this step
        # proj: (D, D) or similar
        step_out = F.linear(out, module.step_projs[i])
        out = (1 - mask) * out + mask * step_out
        
    _record_routing_telemetry(module, max_depth, depths, logits=scores)
    return out

@register_op("latent_attention_compressor")
def _op_latent_attention_compressor(module, inputs, config):
    """MLA-style: compress KV to latent dim, then decompress."""
    x = inputs[0]  # (B, S, D)
    if not hasattr(module, 'kv_compress'):
        return x
    # Compress: (B, S, D) -> (B, S, latent_dim)
    latent = F.linear(x, module.kv_compress)
    # Decompress: (B, S, latent_dim) -> (B, S, D*2) -> split to K, V
    kv = F.linear(latent, module.kv_up)
    D = x.shape[-1]
    k, v = kv[..., :D], kv[..., D:]
    # Simple attention-free compression: gate k against v
    return x + torch.sigmoid(k) * v

@register_op("routing_conditioned_compression")
def _op_routing_conditioned_compression(module, inputs, config):
    """Changes linear layer compression level based on routing signal."""
    x, routing_signal = inputs[0], inputs[1]
    if not hasattr(module, 'weight_full'): return x
    
    # Use routing signal to interpolate between Full and Low-Rank weights
    # routing_signal is expected to be [B, S, 1] or [B, S, 2]
    if routing_signal.shape[-1] > 1:
        s = torch.sigmoid(routing_signal[..., 0:1])
    else:
        s = torch.sigmoid(routing_signal)
        
    full = F.linear(x, module.weight_full)
    
    if hasattr(module, 'U_comp'):
        comp = F.linear(F.linear(x, module.U_comp), module.V_comp)
        return s * full + (1-s) * comp
        
    return full

@register_op("token_type_classifier")
def _op_token_type_classifier(module, inputs, config):
    """Learned classifier: (B,S,D) -> scores -> projected back to (B,S,D)."""
    x = inputs[0]  # (B, S, D)
    if not hasattr(module, 'classifier_weight'):
        return x
    # (B, S, D) -> (B, S, n_classes)
    scores = F.linear(x, module.classifier_weight)
    # Project back to model dim so downstream ops get correct shape
    return F.linear(scores, module.classifier_proj_back)

@register_op("entropy_router")
def _op_entropy_router(module, inputs, config):
    """Produces routing signal [B, S, 1] based on entropy of input scores (B,S,K)."""
    scores = inputs[0]
    probs = F.softmax(scores, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)
    return entropy

@register_op("progressive_compression_gate")
def _op_progressive_compression_gate(module, inputs, config):
    """Learned per-layer compression: interpolates between full and low-rank based on depth."""
    x = inputs[0]
    if not hasattr(module, 'weight_full') or not hasattr(module, 'compress_param'):
        return x
    
    # compress_param is a single scalar per layer
    s = torch.sigmoid(module.compress_param)
    
    full = F.linear(x, module.weight_full)
    if hasattr(module, 'U_comp'):
        comp = F.linear(F.linear(x, module.U_comp), module.V_comp)
        return s * full + (1-s) * comp
    return full

@register_op("compression_mixture_experts")
def _op_compression_mixture_experts(module, inputs, config):
    """Routing assigns tokens to method-specific compression experts."""
    x, routing_signal = inputs[0], inputs[1]
    if not hasattr(module, 'expert_weights'): return x
    
    # 2 experts: 0=LowRank, 1=Bottleneck
    weights = F.softmax(routing_signal, dim=-1) # [B, S, 2]
    
    # Expert 0: Low-Rank
    out0 = F.linear(F.linear(x, module.U_lr), module.V_lr)
    
    # Expert 1: Bottleneck
    hidden1 = F.gelu(F.linear(x, module.W_down))
    out1 = F.linear(hidden1, module.W_up)
    
    return out0 * weights[..., 0:1] + out1 * weights[..., 1:2]

# ── 2026 Frontier Ops ───────────────────────────────────────────────

@register_op("relu_gate_routing")
def _op_relu_gate_routing(module, inputs, config):
    """ReLU-based differentiable MoE gating (ReMoE).
    Unlike Top-K, this learns how many experts to activate per token.
    """
    x = inputs[0]
    if not hasattr(module, 'gate_proj'): return x
    
    # [B, S, n_experts]
    gate_scores = F.relu(F.linear(x, module.gate_proj))
    
    # Record telemetry on 'effective expert count' (sparsity)
    active_count = (gate_scores > 0).float().sum(dim=-1).mean().item()
    _record_routing_telemetry(module, gate_scores.shape[-1], gate_scores.argmax(dim=-1), logits=gate_scores)
    
    # Placeholder: In a real MoE this would dispatch to experts.
    # For micro-eval, we just return the weighted gate signal.
    return gate_scores.sum(dim=-1, keepdim=True).expand_as(x) * x

@register_op("ternary_projection")
def _op_ternary_projection(module, inputs, config):
    """1.58-bit Ternary Weights Simulation (BitNet).
    Weights are restricted to {-1, 0, 1} with a learned scale.
    """
    x = inputs[0]
    if not hasattr(module, 'weight'): return x
    
    # Simulated ternary quantization: W_quant = round(clamp(W / gamma))
    # where gamma is average absolute value
    w = module.weight
    gamma = w.abs().mean().clamp(min=1e-5)
    w_quant = torch.round(torch.clamp(w / gamma, -1, 1))
    
    # STE (Straight-Through Estimator) for training gradients
    w_sim = w + (w_quant * gamma - w).detach()
    
    return F.linear(x, w_sim, getattr(module, 'bias', None))

@register_op("basis_expansion")
def _op_basis_expansion(module, inputs, _):
    if not hasattr(module, 'weight'): return inputs[0]
    x = inputs[0]
    if (
        _c(x)
        and hasattr(aria_core, "basis_expansion_f32")
        and isinstance(module.weight, torch.Tensor)
    ):
        freqs = module.weight
        n_bases = int(freqs.shape[0]) if freqs.dim() > 0 else 1
        try:
            native_out = aria_core.basis_expansion_f32(x, freqs, n_bases)
            if isinstance(native_out, torch.Tensor) and native_out.shape == x.shape:
                return native_out
        except Exception:
            pass
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
    causal_mask = (pos >= pos.t()).float()  # lower-triangular: position i attends only to j <= i
    kernel = kernel * causal_mask
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
    # Ensure real-valued output for downstream ops by making imaginary part zero
    # and ensuring n is correctly set to reconstruct full sequence length
    comp = torch.complex(inputs[0], torch.zeros_like(inputs[0]))
    return torch.fft.irfft(comp, n=(S_freq - 1) * 2, dim=1)

@register_op("tropical_center")
def _op_tropical_center(_, inputs, __):
    """Causal min centering: subtract cumulative minimum to preserve causality."""
    x = inputs[0]
    if _c(x):
        return aria_core.tropical_center_f32(x)
    # torch.cummin(x, dim=1).values is causal
    return x - torch.cummin(x, dim=1).values

@register_op("ultrametric_attention")
def _op_ultrametric_attention(module, inputs, config):
    """Attention using p-adic distances. Dispatched to mathspace implementation."""
    from ..mathspaces.padic import execute_ultrametric_attn
    return execute_ultrametric_attn(module, inputs[0])

# ── Clifford algebra ops ──

@register_op("clifford_attention")
def _op_clifford_attention(module, inputs, config):
    from ..mathspaces.clifford import execute_clifford_attention
    return execute_clifford_attention(module, inputs[0])

@register_op("geometric_product")
def _op_geometric_product(module, inputs, config):
    from ..mathspaces.clifford import execute_geometric_product
    return execute_geometric_product(module, inputs[0], inputs[1] if len(inputs) > 1 else inputs[0])

@register_op("rotor_transform")
def _op_rotor_transform(module, inputs, config):
    from ..mathspaces.clifford import execute_rotor_transform
    return execute_rotor_transform(module, inputs[0])

@register_op("grade_select")
def _op_grade_select(module, inputs, config):
    from ..mathspaces.clifford import execute_grade_select
    return execute_grade_select(module, inputs[0])

@register_op("grade_mix")
def _op_grade_mix(module, inputs, config):
    from ..mathspaces.clifford import execute_grade_mix
    return execute_grade_mix(module, inputs[0])

# ── Hyperbolic ops ──

@register_op("poincare_add")
def _op_poincare_add(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_poincare_add
    return execute_poincare_add(module, inputs[0])

@register_op("exp_map")
def _op_exp_map(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_exp_map
    return execute_exp_map(module, inputs[0])

@register_op("log_map")
def _op_log_map(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_log_map
    return execute_log_map(module, inputs[0])

@register_op("hyp_distance")
def _op_hyp_distance(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_hyp_distance
    return execute_hyp_distance(module, inputs[0], inputs[1] if len(inputs) > 1 else inputs[0])

@register_op("hyp_linear")
def _op_hyp_linear(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_hyp_linear
    return execute_hyp_linear(module, inputs[0])

@register_op("hyp_tangent_nonlinear")
def _op_hyp_tangent_nonlinear(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_hyp_tangent_nonlinear
    return execute_hyp_tangent_nonlinear(module, inputs[0])

@register_op("hyperbolic_norm")
def _op_hyperbolic_norm(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_hyperbolic_norm
    return execute_hyperbolic_norm(module, inputs[0])

# ── Tropical ops ──

@register_op("tropical_add")
def _op_tropical_add(module, inputs, config):
    from ..mathspaces.tropical import execute_tropical_add
    return execute_tropical_add(module, inputs[0], inputs[1] if len(inputs) > 1 else inputs[0])

@register_op("tropical_matmul")
def _op_tropical_matmul(module, inputs, config):
    from ..mathspaces.tropical import execute_tropical_matmul
    return execute_tropical_matmul(module, inputs[0], inputs[1] if len(inputs) > 1 else inputs[0])

@register_op("tropical_attention")
def _op_tropical_attention(module, inputs, config):
    from ..mathspaces.tropical import execute_tropical_attention
    return execute_tropical_attention(module, inputs[0])

@register_op("tropical_gate")
def _op_tropical_gate(module, inputs, config):
    from ..mathspaces.tropical import execute_tropical_gate
    return execute_tropical_gate(module, inputs[0])

# ── p-adic ops ──

@register_op("padic_expand")
def _op_padic_expand(module, inputs, config):
    from ..mathspaces.padic import execute_padic_expand
    return execute_padic_expand(module, inputs[0])

@register_op("padic_gate")
def _op_padic_gate(module, inputs, config):
    from ..mathspaces.padic import execute_padic_gate
    return execute_padic_gate(module, inputs[0])

# ── Spiking ops ──

@register_op("lif_neuron")
def _op_lif_neuron(module, inputs, config):
    from ..mathspaces.spiking import execute_lif
    return execute_lif(module, inputs[0])

@register_op("spike_rate_code")
def _op_spike_rate_code(module, inputs, config):
    from ..mathspaces.spiking import execute_spike_rate_code
    return execute_spike_rate_code(module, inputs[0])

@register_op("stdp_attention")
def _op_stdp_attention(module, inputs, config):
    from ..mathspaces.spiking import execute_stdp_attention
    return execute_stdp_attention(module, inputs[0])

@register_op("sparse_threshold")
def _op_sparse_threshold(module, inputs, config):
    from ..mathspaces.spiking import execute_sparse_threshold
    return execute_sparse_threshold(module, inputs[0])


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
                telemetry = getattr(module, "mathspace_telemetry", {})
                stats = telemetry.get(op_name, {"calls": 0, "nonfinite_elements": 0, "sanitized_calls": 0})
                stats["calls"] = stats.get("calls", 0) + 1
                
                if nonfinite > 0:
                    result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
                    stats["nonfinite_elements"] = stats.get("nonfinite_elements", 0) + nonfinite
                    stats["sanitized_calls"] = stats.get("sanitized_calls", 0) + 1
                    
                telemetry[op_name] = stats
                setattr(module, "mathspace_telemetry", telemetry)

            # Tropical routing telemetry for route-collapse detection
            if op_name in ("tropical_router", "tropical_moe"):
                _tropical_obj = getattr(module, '_tropical_router', None) or getattr(module, '_tropical_moe', None)
                if _tropical_obj is not None:
                    _router = getattr(_tropical_obj, 'router', _tropical_obj)
                    if hasattr(_router, 'centroids'):
                        n_exp = _router.centroids.shape[0]
                        # Re-run router for telemetry (cheap — just reads cached weights)
                        with torch.no_grad():
                            _weights = _router(inputs[0])  # (B, S, n_experts)
                            _top_idx = _weights.argmax(dim=-1).flatten()  # (B*S,)
                            _record_routing_telemetry(module, n_exp, _top_idx.unsqueeze(-1), logits=_weights.reshape(-1, n_exp))

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
        D_in = max(1, input_shape.dim)
        D_out = max(1, config.get("out_dim", D_in))
        # Avoid division by zero for symbolic or unset shapes
        std = 1.0 / math.sqrt(D_in) if D_in > 0 else 0.02

        def _init_attention_stack(op_name: str) -> None:
            n_heads = max(1, D_in // 64)
            head_dim = D_in // n_heads
            self.n_heads = n_heads
            self.head_dim = head_dim
            if op_name in ("softmax_attention", "graph_attention"):
                self.attn_scale = head_dim ** -0.5
            self.q_proj = nn.Linear(D_in, n_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(D_in, n_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(D_in, n_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(n_heads * head_dim, D_in, bias=False)
            self.q_proj.weight.data.normal_(std=0.02)
            self.k_proj.weight.data.normal_(std=0.02)
            self.v_proj.weight.data.normal_(std=0.02)
            self.o_proj.weight.data.normal_(std=0.02)
            if op_name == "graph_attention":
                self.edge_proj = nn.Linear(D_in, D_in, bias=False)
                self.edge_proj.weight.data.normal_(std=0.02)

        def _init_math_space() -> None:
            if op.has_params:
                self.weight = self._make_param((D_out, D_in), std=0.02)
            if op.name in ("padic_expand", "padic_residual"):
                self.weight = self._make_param((D_in, D_in * 2), std=0.02)
            elif op.name == "rotor_transform":
                self.rotor = nn.Parameter(torch.randn(8) * 0.02)
            elif op.name == "poincare_add":
                self.bias = nn.Parameter(torch.zeros(D_in))
            elif op.name == "hyp_linear":
                self.weight = self._make_param((D_in, D_in), std=0.02)
            elif op.name == "tropical_router":
                n_exp = int(config.get("n_experts", 8))
                self.centroids = nn.Parameter(torch.randn(n_exp, D_in) * 0.02)

        dispatch: Dict[str, Callable[[], None]] = {
            "linear_proj": lambda: setattr(self, "weight", self._make_param((D_out, D_in), std=0.02)),
            "linear_proj_down": lambda: setattr(self, "weight", self._make_param((D_out, D_in), std=0.02)),
            "linear_proj_up": lambda: setattr(self, "weight", self._make_param((D_out, D_in), std=0.02)),
            "fused_linear_gelu": lambda: (
                setattr(self, "weight", self._make_param((D_out, D_in), std=0.02)),
                setattr(self, "bias", nn.Parameter(torch.zeros(D_out))),
            ),
            "learnable_scale": lambda: setattr(self, "scale", nn.Parameter(torch.ones(D_in))),
            "learnable_bias": lambda: setattr(self, "bias", nn.Parameter(torch.zeros(D_in))),
            "selective_scan": lambda: (
                setattr(self, "A_log", self._make_param((D_in,), std=0.1)),
                setattr(self, "dt_proj", self._make_param((D_in,), std=0.1)),
                setattr(self, "B_proj", nn.Linear(D_in, D_in, bias=False)),
                setattr(self, "C_proj", nn.Linear(D_in, D_in, bias=False)),
                self.B_proj.weight.data.normal_(std=0.02),
                self.C_proj.weight.data.normal_(std=0.02),
            ),
            "conv1d_seq": lambda: setattr(self, "conv_weight", self._make_param((D_in, 1, 3), std=1.0 / math.sqrt(3))),
            "topk_gate": lambda: setattr(self, "gate_proj", self._make_param((2, D_in), std=0.02)),
            "moe_topk": lambda: self._init_moe_topk(config, D_in),
            "moe_2expert": lambda: (
                setattr(self, "gate_proj", self._make_param((2, D_in), std=0.02)),
                setattr(self, "expert_0_weight", self._make_param((D_in, D_in), std=0.02)),
                setattr(self, "expert_1_weight", self._make_param((D_in, D_in), std=0.02)),
            ),
            "nm_sparse_linear": lambda: (
                setattr(self, "weight", self._make_param((D_out, D_in), std=0.02)),
                setattr(self, "sparsity_n", int(config.get("n", 2))),
                setattr(self, "sparsity_m", int(config.get("m", 4))),
            ),
            "block_sparse_linear": lambda: (
                setattr(self, "weight", self._make_param((D_out, D_in), std=0.02)),
                setattr(self, "block_size", max(1, int(config.get("block_size", 16)))),
                setattr(self, "block_density", float(max(0.05, min(1.0, config.get("block_density", 0.25))))),
            ),
            "rmsnorm": lambda: setattr(self, "weight", nn.Parameter(torch.ones(D_in))),
            "layernorm": lambda: (
                setattr(self, "weight", nn.Parameter(torch.ones(D_in))),
                setattr(self, "bias", nn.Parameter(torch.zeros(D_in))),
            ),
            "gated_linear": lambda: (
                setattr(self, "linear_weight", self._make_param((D_out, D_in), std=0.02)),
                setattr(self, "gate_weight", self._make_param((D_out, D_in), std=0.02)),
                setattr(self, "linear_bias", nn.Parameter(torch.zeros(D_out))),
                setattr(self, "gate_bias", nn.Parameter(torch.zeros(D_out))),
            ),
            "rwkv_time_mixing": lambda: (
                setattr(self, "w_decay", nn.Parameter(torch.ones(D_in) * -0.5)),
                setattr(self, "u_bonus", nn.Parameter(torch.zeros(D_in))),
                setattr(self, "W_k", self._make_param((D_in, D_in), std=0.02)),
                setattr(self, "W_v", self._make_param((D_in, D_in), std=0.02)),
                setattr(self, "W_r", self._make_param((D_in, D_in), std=0.02)),
                setattr(self, "W_o", self._make_param((D_in, D_in), std=0.02)),
            ),
            "embedding_lookup": lambda: setattr(self, "embed_table", nn.Embedding(int(config.get("vocab_size", 32000)), D_in)),
            "rope_rotate": lambda: None,
            "cosine_similarity": lambda: None,
            "gather_topk": lambda: None,
            "semi_structured_2_4_linear": lambda: (
                setattr(self, "weight", self._make_param((D_out, D_in), std=0.02)),
                setattr(self, "sparse_kernel_ready", bool(D_in % 4 == 0 and D_out % 4 == 0)),
            ),
            "basis_expansion": lambda: setattr(self, "weight", nn.Parameter(torch.randn(4, D_in) * 0.5)),
            "integral_kernel": lambda: setattr(self, "weight", nn.Parameter(torch.randn(D_in, D_in) * 0.02)),
            "fixed_point_iter": lambda: setattr(self, "weight", nn.Parameter(torch.randn(D_in + 1, D_in) * 0.02)),
            "low_rank_proj": lambda: self._init_low_rank_proj(D_in),
            "grouped_linear": lambda: self._init_grouped_linear(D_in),
            "bottleneck_proj": lambda: self._init_bottleneck_proj(D_in),
            "shared_basis_proj": lambda: self._init_shared_basis_proj(D_in),
            "tied_proj": lambda: self._init_tied_proj(D_in),
            "swiglu_mlp": lambda: self._init_swiglu_mlp(config, D_in),
            "rwkv_channel": lambda: self._init_rwkv_channel(config, D_in),
            "softmax_attention": lambda: _init_attention_stack("softmax_attention"),
            "linear_attention": lambda: _init_attention_stack("linear_attention"),
            "graph_attention": lambda: _init_attention_stack("graph_attention"),
            "state_space": lambda: self._init_state_space(D_in),
            "conv_only": lambda: self._init_conv_only(D_in),
            "stdp_attention": lambda: setattr(self, "log_tau", nn.Parameter(torch.tensor(0.0))),
            "adaptive_lane_mixer": lambda: self._init_adaptive_lane_mixer(D_in),
            "mixed_recursion_gate": lambda: self._init_mixed_recursion_gate(config, D_in),
            "token_type_classifier": lambda: self._init_token_type_classifier(config, D_in),
            "progressive_compression_gate": lambda: self._init_progressive_compression_gate(D_in, D_out),
            "compression_mixture_experts": lambda: self._init_compression_mixture_experts(D_in, D_out),
            "relu_gate_routing": lambda: setattr(self, "gate_proj", self._make_param((int(config.get("n_experts", 8)), D_in), std=0.02)),
            "ternary_projection": lambda: self._init_ternary_projection(config, D_in, D_out),
            "latent_attention_compressor": lambda: self._init_latent_attention_compressor(D_in),
            "routing_conditioned_compression": lambda: self._init_routing_conditioned_compression(D_in),
        }

        handler = dispatch.get(op.name)
        if handler is not None:
            handler()
            return

        if op.category.value == "math_space":
            _init_math_space()
            return

        if hasattr(op, 'init_params'):
            op.init_params(self, D_in)
            return
        self.weight = nn.Parameter(torch.randn(D_in, D_in) * std)

    def _init_moe_topk(self, config: Dict, d_in: int) -> None:
        n_experts = int(config.get("num_experts", 4))
        self.gate_weight = self._make_param((n_experts, d_in), std=0.02)
        hidden = int(d_in * float(config.get("mlp_ratio", 2.0)))
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, d_in, bias=False),
            ) for _ in range(n_experts)
        ])
        for expert in self.experts:
            expert[0].weight.data.normal_(mean=0.0, std=0.02)
            expert[2].weight.data.normal_(mean=0.0, std=1.0 / math.sqrt(hidden if hidden > 0 else 1))

    def _init_low_rank_proj(self, d_in: int) -> None:
        rank = max(d_in // 4, 1)
        self.U = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(rank, d_in) * 0.02)

    def _init_grouped_linear(self, d_in: int) -> None:
        g = 4
        group_dim = max(d_in // g, 1)
        self.weight = nn.Parameter(torch.randn(g, group_dim, group_dim) * 0.02)
        self.n_groups = g

    def _init_bottleneck_proj(self, d_in: int) -> None:
        rank = max(d_in // 4, 1)
        self.down = nn.Parameter(torch.randn(rank, d_in) * 0.02)
        self.up = nn.Parameter(torch.randn(d_in, rank) * 0.02)

    def _init_shared_basis_proj(self, d_in: int) -> None:
        k = 8
        self.basis = nn.Parameter(torch.randn(k, d_in) * 0.02)
        self.mixing = nn.Parameter(torch.randn(d_in, k) * 0.02)

    def _init_tied_proj(self, d_in: int) -> None:
        rank = max(d_in // 4, 1)
        self.tied_weight = nn.Parameter(torch.randn(rank, d_in) * 0.02)

    def _init_swiglu_mlp(self, config: Dict, d_in: int) -> None:
        hidden = int(d_in * float(config.get("mlp_ratio", 3.0)))
        self.gate_proj = nn.Linear(d_in, hidden, bias=False)
        self.up_proj = nn.Linear(d_in, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_in, bias=False)
        self.gate_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.down_proj.weight.data.normal_(mean=0.0, std=1.0 / math.sqrt(hidden if hidden > 0 else 1))

    def _init_rwkv_channel(self, config: Dict, d_in: int) -> None:
        hidden = int(d_in * float(config.get("mlp_ratio", 3.0)))
        self.mix_k = nn.Parameter(torch.ones(d_in) * 0.5)
        self.mix_r = nn.Parameter(torch.ones(d_in) * 0.5)
        self.key_proj = nn.Linear(d_in, hidden, bias=False)
        self.receptance_proj = nn.Linear(d_in, d_in, bias=False)
        self.value_proj = nn.Linear(hidden, d_in, bias=False)
        self.key_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.receptance_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.value_proj.weight.data.normal_(mean=0.0, std=1.0 / math.sqrt(hidden if hidden > 0 else 1))

    def _init_state_space(self, d_in: int) -> None:
        state_dim = 16
        self.ssm_state_dim = state_dim
        self.ssm_A = nn.Parameter(torch.randn(d_in, state_dim) * 0.01)
        self.ssm_B = nn.Linear(d_in, d_in * state_dim, bias=False)
        self.ssm_C = nn.Linear(d_in * state_dim, d_in, bias=False)
        self.ssm_D = nn.Parameter(torch.ones(d_in))
        self.ssm_dt = nn.Linear(d_in, d_in)
        self.ssm_B.weight.data.normal_(std=0.02)
        self.ssm_C.weight.data.normal_(std=0.02)
        self.ssm_dt.weight.data.normal_(std=0.02)

    def _init_conv_only(self, d_in: int) -> None:
        self.conv_dw = nn.Conv1d(d_in, d_in, 3, padding=2, groups=d_in)
        self.conv_proj = nn.Linear(d_in, d_in, bias=False)
        self.conv_proj.weight.data.normal_(std=0.02)

    def _init_adaptive_lane_mixer(self, d_in: int) -> None:
        self.gate_proj = self._make_param((3, d_in), std=0.02)
        rank = max(d_in // 4, 1)
        self.U_mid = self._make_param((rank, d_in), std=0.02)
        self.V_mid = self._make_param((d_in, rank), std=0.02)
        hidden = d_in * 2
        self.heavy_mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_in),
        )
        self.heavy_mlp[0].weight.data.normal_(std=0.02)
        self.heavy_mlp[2].weight.data.normal_(std=0.02)

    def _init_mixed_recursion_gate(self, config: Dict, d_in: int) -> None:
        max_depth = int(config.get("max_depth", 3))
        projs = []
        for _ in range(max_depth):
            p = nn.Parameter(torch.empty(d_in, d_in))
            p.data.normal_(std=0.02)
            projs.append(p)
        self.step_projs = nn.ParameterList(projs)

    def _init_token_type_classifier(self, config: Dict, d_in: int) -> None:
        n_classes = int(config.get("n_classes", 2))
        self.classifier_weight = self._make_param((n_classes, d_in), std=0.02)
        self.classifier_proj_back = self._make_param((d_in, n_classes), std=0.02)

    def _init_progressive_compression_gate(self, d_in: int, d_out: int) -> None:
        self.weight_full = self._make_param((d_out, d_in), std=0.02)
        self.compress_param = nn.Parameter(torch.zeros(1))
        rank = max(d_in // 8, 1)
        self.U_comp = self._make_param((rank, d_in), std=0.02)
        self.V_comp = self._make_param((d_out, rank), std=0.02)

    def _init_compression_mixture_experts(self, d_in: int, d_out: int) -> None:
        self.expert_weights = nn.Parameter(torch.ones(2))
        rank = max(d_in // 8, 1)
        self.U_lr = self._make_param((rank, d_in), std=0.02)
        self.V_lr = self._make_param((d_out, rank), std=0.02)
        rank_bn = max(d_in // 4, 1)
        self.W_down = self._make_param((rank_bn, d_in), std=0.02)
        self.W_up = self._make_param((d_out, rank_bn), std=0.02)

    def _init_ternary_projection(self, config: Dict, d_in: int, d_out: int) -> None:
        self.weight = self._make_param((d_out, d_in), std=0.02)
        if config.get("bias"):
            self.bias = nn.Parameter(torch.zeros(d_out))

    def _init_latent_attention_compressor(self, d_in: int) -> None:
        latent_dim = max(d_in // 4, 16)
        self.kv_compress = self._make_param((latent_dim, d_in), std=0.02)
        self.kv_up = self._make_param((d_in * 2, latent_dim), std=0.02)

    def _init_routing_conditioned_compression(self, d_in: int) -> None:
        self.weight_full = self._make_param((d_in, d_in), std=0.02)
        rank = max(d_in // 8, 1)
        self.U_comp = self._make_param((rank, d_in), std=0.02)
        self.V_comp = self._make_param((d_in, rank), std=0.02)

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

    def set_capture_heatmap(self, enabled: bool = True) -> None:
        """Enable or disable heatmap capture for all ops in this layer."""
        for op in self.ops.values():
            op._capture_heatmap = enabled


class SynthesizedModel(nn.Module):
    """A complete language model built from synthesized layers."""

    def __init__(self, layer_graphs: List[ComputationGraph], vocab_size: int = 32000,
                 model_dim: int = 256, max_seq_len: int = 512):
        super().__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim)
        # NOTE: Keep PyTorch default N(0,1) init for embeddings. With weight-
        # tied lm_head, std=0.02 makes initial logits too flat (std≈0.3),
        # causing loss_ratio≈1.0 in 500-step micro-training. Default std=1.0
        # gives initial logits std≈16, providing strong gradient signal.
        self.layers = nn.ModuleList([CompiledLayer(g) for g in layer_graphs])
        self.norm = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        
        self._layer_graphs = layer_graphs
        # Pre-calculate which layers need an external residual connection
        # If a graph has NO internal residual, we MUST add one between layers.
        self.layer_needs_residual = [not g.has_residual_path() for g in layer_graphs]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            if self.layer_needs_residual[i]:
                # Standard inter-layer residual for "flat" blocks
                out = layer(x)
                if out.shape == x.shape:
                    x = x + out
                else:
                    x = out
            else:
                # References usually have their own residuals internally
                x = layer(x)
        return self.lm_head(self.norm(x))

    def set_capture_heatmap(self, enabled: bool = True) -> None:
        """Enable or disable heatmap capture for all layers."""
        for layer in self.layers:
            if hasattr(layer, "set_capture_heatmap"):
                layer.set_capture_heatmap(enabled)

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
