from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F

from .compiler_op_utils import (
    aria_core,
    _c,
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
        except Exception:
            pass
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


def _op_rfft_seq(_, inputs, __):
    return torch.fft.rfft(inputs[0], dim=1).real


def _op_irfft_seq(_, inputs, __):
    B, S_freq, D = inputs[0].shape
    # Ensure real-valued output for downstream ops by making imaginary part zero
    # and ensuring n is correctly set to reconstruct full sequence length
    comp = torch.complex(inputs[0], torch.zeros_like(inputs[0]))
    return torch.fft.irfft(comp, n=(S_freq - 1) * 2, dim=1)


def _op_tropical_center(_, inputs, __):
    """Causal min centering via smooth cummin for gradient flow."""
    from ..mathspaces.tropical import execute_tropical_center

    class _FakeModule:
        pass

    return execute_tropical_center(_FakeModule(), inputs[0])


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
    "tropical_center": _op_tropical_center,
    "clifford_attention": _op_clifford_attention,
    "geometric_product": _op_geometric_product,
    "rotor_transform": _op_rotor_transform,
    "grade_select": _op_grade_select,
    "grade_mix": _op_grade_mix,
    "padic_expand": _op_padic_expand,
    "padic_gate": _op_padic_gate,
    "padic_residual": _op_padic_residual,
    "lif_neuron": _op_lif_neuron,
    "spike_rate_code": _op_spike_rate_code,
    "stdp_attention": _op_stdp_attention,
    "sparse_threshold": _op_sparse_threshold,
    "basis_expansion": _op_basis_expansion,
    "integral_kernel": _op_integral_kernel,
    "fixed_point_iter": _op_fixed_point_iter,
    "rfft_seq": _op_rfft_seq,
    "irfft_seq": _op_irfft_seq,
    "ultrametric_attention": _op_ultrametric_attention,
}
