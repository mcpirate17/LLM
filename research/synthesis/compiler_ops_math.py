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
    _c16,
    _flatten_for_kernel,
    _safe_linear,
    _unflatten_from_kernel,
)


# ── Table-driven dispatch for simple unary/binary ops ────────────────
#
# Ops that follow a uniform pattern are generated from tables rather than
# hand-written per-op functions.  Custom ops that need preprocessing
# (clamp, softplus, STE, etc.) remain as explicit functions below.


def _make_unary_op(torch_fn, native_name, native_f16_name=None):
    """Generate a unary op with f32 and optional f16 C kernel dispatch."""
    native_fn = None
    native_f16_fn = None

    def op(_, inputs, __):
        nonlocal native_fn, native_f16_fn
        x = inputs[0]
        if _c(x):
            if native_fn is None:
                native_fn = getattr(aria_core, native_name)
            return native_fn(x)
        if native_f16_name and _c16(x):
            if native_f16_fn is None:
                native_f16_fn = getattr(aria_core, native_f16_name, None)
            if native_f16_fn is not None:
                return native_f16_fn(x)
        return torch_fn(x)

    return op


def _make_binary_op(torch_fn, native_name, native_f16_name=None):
    """Generate a binary op with f32 and optional f16 C kernel dispatch."""
    native_fn = None
    native_f16_fn = None

    def op(_, inputs, __):
        nonlocal native_fn, native_f16_fn
        a, b = inputs[0], inputs[1]
        if _c(a):
            if native_fn is None:
                native_fn = getattr(aria_core, native_name)
            return native_fn(a, b)
        if native_f16_name and _c16(a):
            if native_f16_fn is None:
                native_f16_fn = getattr(aria_core, native_f16_name, None)
            if native_f16_fn is not None:
                return native_f16_fn(a, b)
        return torch_fn(a, b)

    return op


# name -> (torch_fn, aria_core f32 name, optional f16 name)
_SIMPLE_UNARY_OPS = {
    "abs": (torch.abs, "abs_f32", None),
    "sin": (torch.sin, "sin_f32", None),
    "cos": (torch.cos, "cos_f32", None),
    "tanh": (torch.tanh, "tanh_f32", None),
    "sigmoid": (torch.sigmoid, "sigmoid_f32", "sigmoid_f16"),
    "relu": (F.relu, "relu_f32", "relu_f16"),
    "gelu": (F.gelu, "gelu_f32", "gelu_f16"),
    "silu": (F.silu, "silu_f32", "silu_f16"),
}

_SIMPLE_BINARY_OPS = {
    "sub": (lambda a, b: a - b, "sub_f32", None),
}

# Generate op functions from tables
_TABLE_OPS: Dict[str, Callable] = {}
for _name, (_tfn, _nname, _nname16) in _SIMPLE_UNARY_OPS.items():
    _TABLE_OPS[_name] = _make_unary_op(_tfn, _nname, _nname16)
for _name, (_tfn, _nname, _nname16) in _SIMPLE_BINARY_OPS.items():
    _TABLE_OPS[_name] = _make_binary_op(_tfn, _nname, _nname16)


# ── Custom ops that need special logic ───────────────────────────────


def _op_identity(_, inputs, __):
    """Pass-through op — used by workflow_converter for uniform routing."""
    return inputs[0]


def _op_neg(_, inputs, __):
    x = inputs[0]
    if _c(x):
        return aria_core.neg_f32(x)
    return -x


def _op_minimum(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a) and not a.requires_grad:
        native_fn = getattr(aria_core, "minimum_f32", None)
        if native_fn is not None:
            return native_fn(a, b)
    # Smooth min: -tau * log(exp(-a/tau) + exp(-b/tau)), tau=1.0
    # Gradient flows through both inputs (unlike hard torch.minimum)
    tau = 1.0
    return -tau * torch.logaddexp(-a / tau, -b / tau)


def _op_maximum(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a) and not a.requires_grad:
        native_fn = getattr(aria_core, "maximum_f32", None)
        if native_fn is not None:
            return native_fn(a, b)
    # Smooth max: tau * log(exp(a/tau) + exp(b/tau)), tau=1.0
    tau = 1.0
    return tau * torch.logaddexp(a / tau, b / tau)


def _op_exp(_, inputs, __):
    x = inputs[0]
    if _c(x):
        return aria_core.exp_f32(torch.clamp(x, -20, 20))
    return torch.exp(torch.clamp(x, -20, 20))


def _op_log(_, inputs, __):
    x = inputs[0]
    # softplus ensures always-positive input.
    # Clamp softplus output to >= 0.01 to bound log gradient (d/dx = 1/x):
    # at 0.01 grad is 100, at 1e-6 grad is 1e6.
    soft = F.softplus(x, beta=1.0, threshold=20).clamp(min=0.01)
    if _c(x) and not x.requires_grad:
        return aria_core.log_f32(soft)
    return torch.log(soft)


def _op_sqrt(_, inputs, __):
    x = inputs[0]
    # Clamp to 1e-4 (not 1e-8): grad of sqrt at 1e-8 is 5623, at 1e-4 is 50
    clamped = torch.clamp(x.abs(), min=1e-4)
    if _c(x) and not x.requires_grad:
        return aria_core.sqrt_f32(clamped)
    return torch.sqrt(clamped)


def _op_square(_, inputs, __):
    x = inputs[0]
    if _c(x):
        return aria_core.square_f32(x)
    return x * x


def _op_sign_ste(_, inputs, __):
    x = inputs[0]
    if _c(x):
        signs = aria_core.sign_ste_f32(x)
    else:
        signs = torch.sign(x)
    # STE: forward uses hard sign, backward uses identity (gradient passes through)
    return signs + (x - signs).detach()


def _op_reciprocal(_, inputs, __):
    x = inputs[0]
    if _c(x) and not x.requires_grad:
        return aria_core.reciprocal_f32(x)
    # Sigmoid-based: 1/(1+sigmoid(x)), range [0.5, 1.0], bounded gradient.
    # Always use this path during training — C kernel uses raw 1/x which
    # has unbounded gradient near zero.
    # Clamp input to prevent extreme sigmoid saturation from amplifying
    # gradients when composed with mul(x, reciprocal(x)).
    x_clamped = torch.clamp(x, min=-10.0, max=10.0)
    return 1.0 / (1.0 + torch.sigmoid(x_clamped))


def _op_add(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if a.numel() == b.numel():
        if _c(a):
            return aria_core.add_f32(a, b)
        if _c16(a):
            return aria_core.add_f16(a, b)
    return a + b


def _op_mul(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if a.numel() == b.numel():
        if _c(a):
            return aria_core.mul_f32(a, b)
        if _c16(a):
            return aria_core.mul_f16(a, b)
    return a * b


def _op_div_safe(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a):
        return aria_core.div_safe_f32(a, b)
    # Sigmoid denominator: range [1, 2], always positive.
    # Clamp numerator to prevent gradient explosion through b:
    # d/db = -a * sigmoid'(b) / (1+sigmoid(b))^2, unbounded if a is large.
    a_clamped = torch.clamp(a, min=-10.0, max=10.0)
    return a_clamped / (1.0 + torch.sigmoid(b))


def _op_sum_last(_, inputs, __):
    return inputs[0].sum(dim=-1, keepdim=True)


def _op_mean_last(_, inputs, __):
    return inputs[0].mean(dim=-1, keepdim=True)


def _op_max_last(_, inputs, __):
    return inputs[0].max(dim=-1, keepdim=True).values


def _op_norm_last(_, inputs, __):
    return inputs[0].norm(dim=-1, keepdim=True)


def _op_cumsum(_, inputs, __):
    x = inputs[0]
    if _c(x):
        return aria_core.cumsum_f32(x)
    return torch.cumsum(x, dim=1)


def _op_cumprod_safe(_, inputs, __):
    x = inputs[0]
    if _c(x):
        return aria_core.cumprod_safe_f32(x, -2.0, 2.0)
    return torch.cumprod(torch.clamp(x, -2, 2), dim=1)


def _op_matmul(_, inputs, __):
    a, b = inputs
    if a.dim() != 3 or b.dim() != 3:
        return a  # Fallback for safety

    # CASE 1: Standard Attention Pattern [B, S, D] @ [B, S, D]
    if a.shape[-1] == b.shape[-1]:
        scale = math.sqrt(float(a.shape[-1]))
        # (B, S, D) @ (B, D, S) -> (B, S, S)
        scores = torch.bmm(a, b.transpose(-2, -1)) / scale
        S = a.shape[1]
        if S > 1:
            mask = torch.triu(
                torch.ones(S, S, device=a.device, dtype=torch.bool), diagonal=1
            )
            scores.masked_fill_(mask, float("-inf"))
        # (B, S, S) @ (B, S, D) -> (B, S, D)
        return torch.bmm(F.softmax(scores, dim=-1), b)

    # CASE 2: Linear Algebra Pattern [B, S, D] @ [B, D, K]
    if a.shape[2] == b.shape[1]:
        if _c(a):
            return aria_core.matmul_f32(a, b)
        return torch.bmm(a, b)

    # CASE 3: Incompatible fallback
    return a


def _op_outer_product(_, inputs, __):
    a, b = inputs[0], inputs[1]
    if _c(a):
        return aria_core.mul_f32(a, b)
    return a * b


def _op_transpose_sd(_, inputs, __):
    x = inputs[0]
    *lead, D = x.shape
    if D % 2 != 0:
        return x
    if _c(x) and x.dim() == 3:
        return aria_core.transpose_sd_f32(x.float()).to(x.dtype)
    return x.view(*lead, D // 2, 2).transpose(-1, -2).contiguous().view(*lead, D)


def _op_split2(_, inputs, config):
    part = int(config.get("part", 0))
    x = inputs[0]
    w = x.shape[-1] // 2
    if part == 0:
        return x[..., :w].contiguous()
    return x[..., w : 2 * w].contiguous()


def _op_split3(_, inputs, config):
    part = int(config.get("part", 0))
    x = inputs[0]
    w = x.shape[-1] // 3
    if part == 0:
        return x[..., :w].contiguous()
    if part == 1:
        return x[..., w : 2 * w].contiguous()
    return x[..., 2 * w : 3 * w].contiguous()


def _op_split4(_, inputs, config):
    part = int(config.get("part", 0))
    x = inputs[0]
    w = x.shape[-1] // 4
    if part == 0:
        return x[..., :w].contiguous()
    if part == 1:
        return x[..., w : 2 * w].contiguous()
    if part == 2:
        return x[..., 2 * w : 3 * w].contiguous()
    return x[..., 3 * w : 4 * w].contiguous()


def _op_concat(_, inputs, __):
    return torch.cat(inputs, dim=-1)


def _op_roll_seq(_, inputs, __):
    return torch.roll(inputs[0], shifts=1, dims=1)


def _op_roll_neg(_, inputs, __):
    return torch.roll(inputs[0], shifts=-1, dims=1)


def _op_multi_head_mix(_, inputs, config):
    x = inputs[0]
    B, S, D = x.shape
    H = config.get("n_heads", 4)
    if D % H != 0:
        H = 1
    return F.normalize(x.view(B, S, H, -1), p=2, dim=-1).view(B, S, D)


def _op_linear_common(module, inputs, _):
    if not hasattr(module, "weight"):
        return inputs[0]
    x = inputs[0]
    if x.shape[-1] != module.weight.shape[1]:
        return x
    return _safe_linear(x, module.weight, getattr(module, "bias", None))


def _op_fused_linear_gelu(module, inputs, _):
    if not hasattr(module, "weight"):
        return inputs[0]
    x = inputs[0]
    if _c(x):
        xf, orig_shape = _flatten_for_kernel(x)
        bias = getattr(module, "bias", None)
        if bias is None:
            bias = torch.zeros(module.weight.shape[0], device=x.device, dtype=x.dtype)
        out = aria_core.fused_linear_gelu_f32(xf, module.weight, bias)
        return _unflatten_from_kernel(out, orig_shape)
    if HAS_KERNELS and x.is_cuda:
        bias = getattr(module, "bias", None)
        return kernels.fused_linear_gelu(x, module.weight, bias)
    out = _safe_linear(x, module.weight, getattr(module, "bias", None))
    return F.gelu(out)


def _op_learnable_scale(module, inputs, _):
    if not hasattr(module, "scale"):
        return inputs[0]
    return inputs[0] * module.scale


def _op_learnable_bias(module, inputs, _):
    if not hasattr(module, "bias"):
        return inputs[0]
    return inputs[0] + module.bias


OP_IMPLS: Dict[str, Callable] = {
    # Table-generated simple ops (unary + binary)
    **_TABLE_OPS,
    # Custom ops with special logic
    "identity": _op_identity,
    "neg": _op_neg,
    "exp": _op_exp,
    "log": _op_log,
    "sqrt": _op_sqrt,
    "square": _op_square,
    "sign_ste": _op_sign_ste,
    "reciprocal": _op_reciprocal,
    "add": _op_add,
    "mul": _op_mul,
    "div_safe": _op_div_safe,
    "sum_last": _op_sum_last,
    "mean_last": _op_mean_last,
    "max_last": _op_max_last,
    "norm_last": _op_norm_last,
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
    "minimum": _op_minimum,
    "maximum": _op_maximum,
}
