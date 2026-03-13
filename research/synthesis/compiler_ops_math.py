from __future__ import annotations

import math
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from .compiler_op_utils import (
    HAS_KERNELS,
    aria_core,
    kernels,
    _c,
    _flatten_for_kernel,
    _unflatten_from_kernel,
)

def _op_identity(_, inputs, __):
    """Pass-through op — used by workflow_converter for uniform routing."""
    return inputs[0]

def _op_neg(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.neg_f32(x)
    return -x

def _op_abs(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.abs_f32(x)
    return torch.abs(x)

def _op_exp(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.exp_f32(torch.clamp(x, -20, 20))
    return torch.exp(torch.clamp(x, -20, 20))

def _op_log(_, inputs, __):
    x = inputs[0]
    clamped = torch.clamp(x.abs(), min=1e-8)
    if _c(x): return aria_core.log_f32(clamped)
    return torch.log(clamped)

def _op_sin(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.sin_f32(x)
    return torch.sin(x)

def _op_cos(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.cos_f32(x)
    return torch.cos(x)

def _op_tanh(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.tanh_f32(x)
    return torch.tanh(x)

def _op_sigmoid(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.sigmoid_f32(x)
    return torch.sigmoid(x)

def _op_relu(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.relu_f32(x)
    return F.relu(x)

def _op_gelu(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.gelu_f32(x)
    return F.gelu(x)

def _op_silu(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.silu_f32(x)
    return F.silu(x)

def _op_sqrt(_, inputs, __):
    x = inputs[0]
    clamped = torch.clamp(x.abs(), min=1e-8)
    if _c(x): return aria_core.sqrt_f32(clamped)
    return torch.sqrt(clamped)

def _op_square(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.square_f32(x)
    return x * x

def _op_sign_ste(_, inputs, __):
    x = inputs[0]
    if _c(x):
        signs = aria_core.sign_ste_f32(x)
        return x + (signs - x).detach()
    signs = torch.sign(x)
    return x + (signs - x).detach()

def _op_reciprocal(_, inputs, __):
    x = inputs[0]
    if _c(x): return aria_core.reciprocal_f32(x)
    # Stable reciprocal: push x away from zero by eps in the direction of its sign
    eps = 1e-6
    sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    return 1.0 / (x + eps * sign)

def _op_add(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a): return aria_core.add_f32(a, b)
    return a + b

def _op_mul(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a): return aria_core.mul_f32(a, b)
    return a * b

def _op_sub(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a): return aria_core.sub_f32(a, b)
    return a - b

def _op_div_safe(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a): return aria_core.div_safe_f32(a, b)
    # Stable division: push denominator away from zero by eps in the direction of its sign
    eps = 1e-6
    sign = torch.where(b >= 0, torch.ones_like(b), -torch.ones_like(b))
    return a / (b + eps * sign)

def _op_maximum(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a): return aria_core.maximum_f32(a, b)
    return torch.maximum(a, b)

def _op_minimum(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a): return aria_core.minimum_f32(a, b)
    return torch.minimum(a, b)

def _op_sum_last(_, inputs, __): return inputs[0].sum(dim=-1, keepdim=True)

def _op_mean_last(_, inputs, __): return inputs[0].mean(dim=-1, keepdim=True)

def _op_max_last(_, inputs, __): return inputs[0].max(dim=-1, keepdim=True).values

def _op_norm_last(_, inputs, __): return inputs[0].norm(dim=-1, keepdim=True)

def _op_sum_seq(_, inputs, __): return inputs[0].sum(dim=1, keepdim=True)

def _op_mean_seq(_, inputs, __): return inputs[0].mean(dim=1, keepdim=True)

def _op_cumsum(_, inputs, __): return torch.cumsum(inputs[0], dim=1)

def _op_cumprod_safe(_, inputs, __): return torch.cumprod(torch.clamp(inputs[0], -2, 2), dim=1)

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

def _op_outer_product(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a): return aria_core.mul_f32(a, b)
    return a * b

def _op_transpose_sd(_, inputs, __):
    return inputs[0].transpose(1, 2).contiguous().transpose(1, 2)

def _op_split2(_, inputs, config):
    part = int(config.get("part", 0))
    x = inputs[0]
    w = x.shape[-1] // 2
    if part == 0: return x[..., :w]
    return x[..., w:2*w]

def _op_split3(_, inputs, config):
    part = int(config.get("part", 0))
    x = inputs[0]
    w = x.shape[-1] // 3
    if part == 0: return x[..., :w]
    if part == 1: return x[..., w:2*w]
    return x[..., 2*w:3*w]

def _op_split4(_, inputs, config):
    part = int(config.get("part", 0))
    x = inputs[0]
    w = x.shape[-1] // 4
    if part == 0: return x[..., :w]
    if part == 1: return x[..., w:2*w]
    if part == 2: return x[..., 2*w:3*w]
    return x[..., 3*w:4*w]

def _op_concat(_, inputs, __):
    return torch.cat(inputs, dim=-1)

def _op_roll_seq(_, inputs, __): return torch.roll(inputs[0], shifts=1, dims=1)

def _op_roll_neg(_, inputs, __): return torch.roll(inputs[0], shifts=-1, dims=1)

def _op_multi_head_mix(_, inputs, config):
    x = inputs[0]
    B, S, D = x.shape
    H = config.get("n_heads", 4)
    if D % H != 0: H = 1
    return F.normalize(x.view(B, S, H, -1), p=2, dim=-1).view(B, S, D)

def _op_linear_common(module, inputs, _):
    if not hasattr(module, 'weight'): return inputs[0]
    x = inputs[0]
    if _c(x):
        xf, orig_shape = _flatten_for_kernel(x)
        bias = getattr(module, 'bias', None)
        out = aria_core.linear_f32(xf, module.weight, bias)
        return _unflatten_from_kernel(out, orig_shape)
    return F.linear(x, module.weight, getattr(module, 'bias', None))

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

def _op_learnable_scale(module, inputs, _):
    if not hasattr(module, 'scale'): return inputs[0]
    return inputs[0] * module.scale

def _op_learnable_bias(module, inputs, _):
    if not hasattr(module, 'bias'): return inputs[0]
    return inputs[0] + module.bias

OP_IMPLS: Dict[str, Callable] = {
    "identity": _op_identity,
    "neg": _op_neg,
    "abs": _op_abs,
    "exp": _op_exp,
    "log": _op_log,
    "sin": _op_sin,
    "cos": _op_cos,
    "tanh": _op_tanh,
    "sigmoid": _op_sigmoid,
    "relu": _op_relu,
    "gelu": _op_gelu,
    "silu": _op_silu,
    "sqrt": _op_sqrt,
    "square": _op_square,
    "sign_ste": _op_sign_ste,
    "reciprocal": _op_reciprocal,
    "add": _op_add,
    "mul": _op_mul,
    "sub": _op_sub,
    "div_safe": _op_div_safe,
    "maximum": _op_maximum,
    "minimum": _op_minimum,
    "sum_last": _op_sum_last,
    "mean_last": _op_mean_last,
    "max_last": _op_max_last,
    "norm_last": _op_norm_last,
    "sum_seq": _op_sum_seq,
    "mean_seq": _op_mean_seq,
    "cumsum": _op_cumsum,
    "cumprod_safe": _op_cumprod_safe,
    "matmul": _op_matmul,
    "outer_product": _op_outer_product,
    "transpose_sd": _op_transpose_sd,
    "split2": _op_split2,
    "split3": _op_split3,
    "split4": _op_split4,
    "concat": _op_concat,
    "roll_seq": _op_roll_seq,
    "roll_neg": _op_roll_neg,
    "multi_head_mix": _op_multi_head_mix,
    "linear_proj": _op_linear_common,
    "linear_proj_down": _op_linear_common,
    "linear_proj_up": _op_linear_common,
    "fused_linear_gelu": _op_fused_linear_gelu,
    "learnable_scale": _op_learnable_scale,
    "learnable_bias": _op_learnable_bias,
}
