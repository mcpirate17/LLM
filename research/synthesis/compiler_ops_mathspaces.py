from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F

from .compiler_op_utils import (
    aria_core,
    _c,
    _flatten_for_kernel,
    _unflatten_from_kernel,
    record_kernel_fallback,
)


def _op_padic_residual(module, inputs, config):
    """Multi-resolution p-adic residual connection."""
    from ..mathspaces.padic import execute_padic_residual

    return execute_padic_residual(module, inputs[0])


def _op_basis_expansion(module, inputs, _):
    if not hasattr(module, "weight"):
        return inputs[0]
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
        except (ImportError, RuntimeError, AttributeError) as e:
            record_kernel_fallback("basis_expansion_f32", e)
    w = module.weight
    expanded = (
        torch.sin(inputs[0] * w[0])
        + torch.cos(inputs[0] * w[1])
        + torch.sin(inputs[0] * w[2])
        + torch.cos(inputs[0] * w[3])
    )
    return expanded * 0.25


def _op_integral_kernel(module, inputs, config):
    if not hasattr(module, "weight"):
        return inputs[0]
    B, S, D = inputs[0].shape
    pos = torch.arange(S, device=inputs[0].device, dtype=inputs[0].dtype).unsqueeze(1)
    kernel = torch.exp(
        -float(config.get("kernel_scale", 0.25)) * (pos - pos.t()).abs().float()
    )
    causal_mask = (
        pos >= pos.t()
    ).float()  # lower-triangular: position i attends only to j <= i
    kernel = kernel * causal_mask
    kernel = kernel / kernel.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return F.linear(
        torch.bmm(kernel.unsqueeze(0).expand(B, -1, -1), inputs[0]), module.weight
    )


def _op_fixed_point_iter(module, inputs, config):
    """
    Fixed-point iteration vectorized over the sequence and batch dimensions.
    """
    if not hasattr(module, "weight"):
        return inputs[0]
    B, S, D = inputs[0].shape
    W = module.weight[:D, :]
    b = (
        module.weight[D, :]
        if module.weight.shape[0] > D
        else torch.zeros(D, device=inputs[0].device)
    )
    z = inputs[0]
    n_iters = max(1, int(config.get("n_iters", 3)))
    damping = max(0.0, min(1.0, float(config.get("damping", 0.5))))
    for _ in range(n_iters):
        z = (1.0 - damping) * z + damping * torch.tanh(F.linear(z, W) + b)
    return z


def _op_tropical_center(module, inputs, __):
    """Causal min centering via smooth cummin for gradient flow."""
    from ..mathspaces.tropical import execute_tropical_center

    return execute_tropical_center(module, inputs[0])


def _op_ultrametric_attention(module, inputs, config):
    """Attention using p-adic distances. Dispatched to mathspace implementation."""
    from ..mathspaces.padic import execute_ultrametric_attn

    return execute_ultrametric_attn(module, inputs[0])


# ── Clifford algebra ops ──


def _op_clifford_attention(module, inputs, config):
    from ..mathspaces.clifford import execute_clifford_attention

    return execute_clifford_attention(module, inputs[0])


def _op_geometric_product(module, inputs, config):
    from ..mathspaces.clifford import execute_geometric_product

    return execute_geometric_product(
        module, inputs[0], inputs[1] if len(inputs) > 1 else inputs[0]
    )


def _op_rotor_transform(module, inputs, config):
    from ..mathspaces.clifford import execute_rotor_transform

    return execute_rotor_transform(module, inputs[0])


def _op_grade_select(module, inputs, config):
    from ..mathspaces.clifford import execute_grade_select

    return execute_grade_select(module, inputs[0])


def _op_grade_mix(module, inputs, config):
    from ..mathspaces.clifford import execute_grade_mix

    return execute_grade_mix(module, inputs[0])


def _op_clifford_inverse(module, inputs, config):
    from ..mathspaces.clifford import execute_clifford_inverse

    return execute_clifford_inverse(module, inputs[0])


def _op_versor_apply(module, inputs, config):
    """Versor sandwich product: versor · multivector · versor⁻¹.

    Two-input op. Falls back to single-input self-versor when only one tensor
    is provided so the picker can wire it into existing slot machinery.
    """
    from ..mathspaces.clifford import execute_versor_apply

    versor = inputs[0]
    mv = inputs[1] if len(inputs) > 1 else inputs[0]
    return execute_versor_apply(module, versor, mv)


# ── Hyperbolic ops ──


def _op_poincare_add(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_poincare_add

    return execute_poincare_add(module, inputs[0])


def _op_exp_map(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_exp_map

    return execute_exp_map(module, inputs[0])


def _op_log_map(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_log_map

    return execute_log_map(module, inputs[0])


def _op_hyp_distance(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_hyp_distance

    return execute_hyp_distance(
        module, inputs[0], inputs[1] if len(inputs) > 1 else inputs[0]
    )


def _op_hyp_linear(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_hyp_linear

    return execute_hyp_linear(module, inputs[0])


def _op_hyp_tangent_nonlinear(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_hyp_tangent_nonlinear

    return execute_hyp_tangent_nonlinear(module, inputs[0])


def _op_hyperbolic_norm(module, inputs, config):
    from ..mathspaces.hyperbolic import execute_hyperbolic_norm

    return execute_hyperbolic_norm(module, inputs[0])


# ── Tropical ops ──


def _op_tropical_add(module, inputs, config):
    from ..mathspaces.tropical import execute_tropical_add

    return execute_tropical_add(
        module, inputs[0], inputs[1] if len(inputs) > 1 else inputs[0]
    )


def _op_tropical_matmul(module, inputs, config):
    from ..mathspaces.tropical import execute_tropical_matmul

    return execute_tropical_matmul(
        module, inputs[0], inputs[1] if len(inputs) > 1 else inputs[0]
    )


def _op_tropical_attention(module, inputs, config):
    from ..mathspaces.tropical import execute_tropical_attention

    return execute_tropical_attention(module, inputs[0])


def _op_tropical_gate(module, inputs, config):
    from ..mathspaces.tropical import execute_tropical_gate

    return execute_tropical_gate(module, inputs[0])


def _op_tropical_softmax(module, inputs, config):
    from ..mathspaces.tropical import execute_tropical_softmax

    return execute_tropical_softmax(module, inputs[0])


def _op_tree_mix(module, inputs, config):
    from ..mathspaces.tree_mix import execute_tree_mix

    return execute_tree_mix(
        module, inputs[0], inputs[1] if len(inputs) > 1 else inputs[0]
    )


def _op_mla_attention(module, inputs, config):
    from ..mathspaces.mla import execute_mla_attention

    return execute_mla_attention(
        module, inputs[0], inputs[1] if len(inputs) > 1 else inputs[0]
    )


# ── p-adic ops ──


def _op_padic_expand(module, inputs, config):
    from ..mathspaces.padic import execute_padic_expand

    return execute_padic_expand(module, inputs[0])


def _op_padic_gate(module, inputs, config):
    from ..mathspaces.padic import execute_padic_gate

    return execute_padic_gate(module, inputs[0])


# ── Spiking ops ──


def _op_lif_neuron(module, inputs, config):
    from ..mathspaces.spiking import execute_lif

    return execute_lif(module, inputs[0])


def _op_spike_rate_code(module, inputs, config):
    from ..mathspaces.spiking import execute_spike_rate_code

    return execute_spike_rate_code(module, inputs[0])


def _op_stdp_attention(module, inputs, config):
    from ..mathspaces.spiking import execute_stdp_attention

    return execute_stdp_attention(module, inputs[0])


def _op_sparse_threshold(module, inputs, config):
    from ..mathspaces.spiking import execute_sparse_threshold

    return execute_sparse_threshold(module, inputs[0])


def _op_spectral_filter(module, inputs, config):
    """Learnable spectral filter over feature dim — causal (per-position, no sequence interaction)."""
    x = inputs[0]
    if not hasattr(module, "freq_mask"):
        return x
    orig_dtype = x.dtype
    # torch.fft does not support bf16 on the current CUDA path.
    # Compute the FFT in fp32, then restore the original dtype.
    x_fft = x.float() if x.dtype in (torch.bfloat16, torch.float16) else x
    X_f = torch.fft.rfft(x_fft, dim=-1)
    # Clamp mask to [-2, 2] to prevent spectral blow-up during training.
    # Values beyond ±2 cause FFT output explosion (48% forward_error rate).
    mask = module.freq_mask.clamp(-2.0, 2.0).to(X_f.dtype)
    X_f = X_f * mask
    out = torch.fft.irfft(X_f, n=x.shape[-1], dim=-1)
    return out.to(orig_dtype) if out.dtype != orig_dtype else out


# ── Compression ops ──


def _op_low_rank_proj(module, inputs, _):
    from ..mathspaces.compression import execute_low_rank_proj

    if not hasattr(module, "U") or not hasattr(module, "V"):
        return inputs[0]
    if _c(inputs[0]):
        bias = getattr(module, "bias", None)
        x, orig_shape = _flatten_for_kernel(inputs[0])
        out = aria_core.linear_low_rank_f32(
            x, module.U.t().contiguous(), module.V.t().contiguous(), bias
        )
        return _unflatten_from_kernel(out, orig_shape)
    return execute_low_rank_proj(module, inputs[0])


def _op_grouped_linear(module, inputs, _):
    from ..mathspaces.compression import execute_grouped_linear

    if not hasattr(module, "weight"):
        return inputs[0]
    if _c(inputs[0]):
        bias = getattr(module, "bias", None)
        x, orig_shape = _flatten_for_kernel(inputs[0])
        out = aria_core.linear_grouped_f32(x, module.weight, bias, module.n_groups)
        return _unflatten_from_kernel(out, orig_shape)
    return execute_grouped_linear(module, inputs[0])


def _op_bottleneck_proj(module, inputs, _):
    from ..mathspaces.compression import execute_bottleneck_proj

    if not hasattr(module, "down") or not hasattr(module, "up"):
        return inputs[0]
    if _c(inputs[0]):
        b_down = getattr(module, "bias_down", None)
        b_up = getattr(module, "bias_up", None)
        x, orig_shape = _flatten_for_kernel(inputs[0])
        out = aria_core.linear_bottleneck_f32(x, module.down, module.up, b_down, b_up)
        return _unflatten_from_kernel(out, orig_shape)
    return execute_bottleneck_proj(module, inputs[0])


def _op_shared_basis_proj(module, inputs, _):
    from ..mathspaces.compression import execute_shared_basis_proj

    if not hasattr(module, "mixing") or not hasattr(module, "basis"):
        return inputs[0]
    if _c(inputs[0]):
        x, orig_shape = _flatten_for_kernel(inputs[0])
        out = aria_core.linear_shared_basis_f32(
            x, module.mixing.T.contiguous(), module.basis
        )
        return _unflatten_from_kernel(out, orig_shape)
    return execute_shared_basis_proj(module, inputs[0])


def _op_tied_proj(module, inputs, _):
    from ..mathspaces.compression import execute_tied_proj

    if not hasattr(module, "tied_weight"):
        return inputs[0]
    if _c(inputs[0]):
        b_down = getattr(module, "bias_down", None)
        b_up = getattr(module, "bias_up", None)
        x, orig_shape = _flatten_for_kernel(inputs[0])
        out = aria_core.linear_tied_f32(x, module.tied_weight, b_down, b_up)
        return _unflatten_from_kernel(out, orig_shape)
    return execute_tied_proj(module, inputs[0])


# ── Tropical routing ops ──


def _op_tropical_router(module, inputs, config):
    from ..mathspaces.tropical_routing import execute_tropical_router

    return execute_tropical_router(module, inputs[0])


def _op_tropical_moe(module, inputs, config):
    from ..mathspaces.tropical_routing import execute_tropical_moe

    return execute_tropical_moe(module, inputs[0])


OP_IMPLS: Dict[str, Callable] = {
    "poincare_add": _op_poincare_add,
    "exp_map": _op_exp_map,
    "log_map": _op_log_map,
    "hyp_distance": _op_hyp_distance,
    "hyp_linear": _op_hyp_linear,
    "hyp_tangent_nonlinear": _op_hyp_tangent_nonlinear,
    "hyperbolic_norm": _op_hyperbolic_norm,
    "tropical_add": _op_tropical_add,
    "tropical_matmul": _op_tropical_matmul,
    "tropical_attention": _op_tropical_attention,
    "tropical_gate": _op_tropical_gate,
    "tropical_softmax": _op_tropical_softmax,
    "tropical_center": _op_tropical_center,
    "clifford_attention": _op_clifford_attention,
    "geometric_product": _op_geometric_product,
    "rotor_transform": _op_rotor_transform,
    "grade_select": _op_grade_select,
    "grade_mix": _op_grade_mix,
    # Phase 5 V2 (2026-05-04) — Clifford companion ops per
    # research/reports/novel_math_ops_proposal_20260504.md §3
    "clifford_inverse": _op_clifford_inverse,
    "versor_apply": _op_versor_apply,
    "padic_expand": _op_padic_expand,
    "padic_gate": _op_padic_gate,
    "padic_residual": _op_padic_residual,
    "lif_neuron": _op_lif_neuron,
    "spike_rate_code": _op_spike_rate_code,
    "stdp_attention": _op_stdp_attention,
    "sparse_threshold": _op_sparse_threshold,
    "spectral_filter": _op_spectral_filter,
    "basis_expansion": _op_basis_expansion,
    "integral_kernel": _op_integral_kernel,
    "fixed_point_iter": _op_fixed_point_iter,
    "ultrametric_attention": _op_ultrametric_attention,
    "low_rank_proj": _op_low_rank_proj,
    "grouped_linear": _op_grouped_linear,
    "bottleneck_proj": _op_bottleneck_proj,
    "shared_basis_proj": _op_shared_basis_proj,
    "tied_proj": _op_tied_proj,
    "tropical_router": _op_tropical_router,
    "tropical_moe": _op_tropical_moe,
    "tree_mix": _op_tree_mix,
    "mla_attention": _op_mla_attention,
}
