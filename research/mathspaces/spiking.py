"""
Spiking / Event-Driven Primitives

A fundamentally different compute paradigm: sparse, binary-spike-based
operations with surrogate gradients for backprop compatibility.

Operations:
- Leaky Integrate-and-Fire neuron (temporal membrane dynamics)
- Spike rate coding (continuous ↔ spike ↔ continuous)
- STDP-inspired causal attention (temporal decay kernel)
- Adaptive sparse threshold gate (median-based sparsity)

All ops are parameter-free, preserve (B, S, D) shape, and use
straight-through estimators (STE) for gradient flow.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from research.env import aria_core, HAS_ARIA_CORE as _HAS_ARIA_CORE


# ── Surrogate gradient helpers ──


class _SigmoidSTE(torch.autograd.Function):
    """Straight-through estimator using sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Sigmoid surrogate: σ(x) * (1 - σ(x))
        sig = torch.sigmoid(x)
        return grad_output * sig * (1 - sig)


# ── Spiking primitives ──


def _gradient_scale(module: nn.Module | None) -> float:
    for attr in ("n_layers", "num_layers", "layer_count", "depth", "max_depth"):
        value = getattr(module, attr, None)
        if value is not None:
            depth = max(1, int(value))
            return float(depth**-0.5)
    return 1.0


def _apply_grad_scale(tensor: torch.Tensor, module: nn.Module | None) -> torch.Tensor:
    scale = _gradient_scale(module)
    if tensor.requires_grad and scale < 0.999:
        tensor.register_hook(lambda grad: grad * scale)
    return tensor


@torch.jit.script
def _lif_membrane_loop(
    x: torch.Tensor, threshold: torch.Tensor, decay: float
) -> torch.Tensor:
    """JIT-compiled LIF membrane dynamics with STE surrogate gradient.

    Runs the sequential recurrence (membrane accumulation, spike, reset)
    without Python-loop overhead.  The straight-through estimator uses
    sigmoid(5·(membrane − threshold)) as surrogate gradient.

    Args:
        x: (B, S, D) input current.
        threshold: (B, S, D) per-position adaptive threshold.
        decay: membrane leak factor (0 < decay < 1).

    Returns:
        (B, S, D) spike tensor with STE gradient.
    """
    B, S, D = x.shape
    membrane = torch.zeros(B, D, device=x.device, dtype=x.dtype)
    output = torch.empty_like(x)

    for t in range(S):
        xt = x[:, t, :]
        thr = threshold[:, t, :]
        membrane = decay * membrane + xt
        spike_surr = torch.sigmoid(5.0 * (membrane - thr))
        spike_hard = (membrane >= thr).float()
        # STE: hard spike forward, sigmoid surrogate backward
        output[:, t, :] = spike_hard + spike_surr - spike_surr.detach()
        membrane = membrane * (1.0 - spike_hard)

    return output


def execute_lif(module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    """Leaky Integrate-and-Fire neuron.

    Treats the sequence dimension as time steps. Accumulates membrane
    potential with exponential decay, fires binary spikes at threshold,
    and resets on fire. Threshold adapts to input magnitude so spike
    rate stays in the 10-50% range regardless of input scale.

    Args:
        module: The CompiledOp instance
        inputs: Variadic input tensors (expects inputs[0] as (B, S, D))

    Returns:
        Binary spike tensor of shape (B, S, D), values in {0, 1}
    """
    x = inputs[0]  # (B, S, D)
    decay = 0.9

    if (
        _HAS_ARIA_CORE
        and x.is_contiguous()
        and x.ndim == 3
        and x.device.type == "cpu"
        and not x.requires_grad
    ):
        # C kernel uses fixed threshold derived from first-position std
        input_std = x[:, 0, :].detach().std().clamp(min=1e-6).item()
        threshold = input_std / (1.0 - decay**2) ** 0.5 * 1.5
        return aria_core.lif_neuron_f32(x, decay, threshold)

    B, S, D = x.shape

    # ── Vectorised threshold computation ──
    # Running variance via cumsum (no Python loop, fully parallel).
    x_det = x.detach()  # threshold stats don't need gradients
    cumsum = x_det.cumsum(dim=1)  # (B, S, D)
    cumsq = (x_det * x_det).cumsum(dim=1)  # (B, S, D)
    counts = torch.arange(1, S + 1, device=x.device, dtype=x.dtype).view(1, S, 1)
    running_var = (cumsq / counts - (cumsum / counts) ** 2).clamp(min=1e-12)
    causal_std = running_var.sqrt()  # (B, S, D)
    threshold = causal_std / (1.0 - decay**2) ** 0.5 * 1.5  # (B, S, D)

    # ── Sequential membrane dynamics (JIT-scripted) ──
    output = _lif_membrane_loop(x, threshold, decay)
    return _apply_grad_scale(output, module)


def execute_spike_rate_code(module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    """Continuous → spike → continuous rate coding.

    Encodes continuous activations as stochastic spike trains over
    multiple timesteps, then decodes by averaging spike rates. The
    output is in [0, 1] and genuinely sparse (low-magnitude inputs
    produce near-zero spike rates).

    Args:
        module: The CompiledOp instance
        inputs: Variadic input tensors (expects inputs[0] as (B, S, D))

    Returns:
        Spike-rate-coded tensor of shape (B, S, D), values in [0, 1]
    """
    x = inputs[0]  # (B, S, D)
    if (
        _HAS_ARIA_CORE
        and x.is_contiguous()
        and x.ndim == 3
        and x.device.type == "cpu"
        and not x.requires_grad
    ):
        return aria_core.spike_rate_code_f32(x)

    # Deterministic rate coding: sigmoid gives the expected spike rate
    # directly, without stochastic Bernoulli sampling. The old stochastic
    # version was non-causal because torch.bernoulli's global RNG state
    # at position t depended on how many positions exist after t.
    spike_rate = torch.sigmoid(x)  # (B, S, D), in [0, 1]

    return _apply_grad_scale(spike_rate, module)


def execute_stdp_attention(module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    """STDP-inspired causal attention with spike gating.

    Uses an exponential temporal decay kernel to create causal attention
    weights, then applies a spike gate so only high-activation tokens
    produce non-zero output (achieving genuine sparsity).

    Args:
        module: The CompiledOp instance
        inputs: Variadic input tensors (expects inputs[0] as (B, S, D))

    Returns:
        Spike-gated attended tensor of shape (B, S, D)
    """
    x = inputs[0]  # (B, S, D)
    B, S, D = x.shape

    # Learnable tau: use module.log_tau if available (set by CompiledOp._init_params)
    # Keep as tensor (no .item()) so gradient flows to log_tau parameter.
    if hasattr(module, "log_tau"):
        tau = torch.exp(module.log_tau).clamp(min=1.0)
    else:
        tau = max(S / 8.0, 1.0)

    # Compute attention via C kernel or Python fallback
    if (
        _HAS_ARIA_CORE
        and x.is_contiguous()
        and x.ndim == 3
        and x.device.type == "cpu"
        and not x.requires_grad
    ):
        tau_scalar = (
            float(tau.detach()) if isinstance(tau, torch.Tensor) else float(tau)
        )
        attended = aria_core.stdp_attention_f32(x, tau_scalar, 0.0)
    else:
        # Build causal exponential decay kernel: weight[i,j] = exp(-(i-j)/tau) for j<=i
        positions = torch.arange(S, device=x.device, dtype=x.dtype)
        dt = positions.unsqueeze(1) - positions.unsqueeze(0)
        # Clamp dt to non-negative BEFORE exp to avoid exp(+inf) for future
        # tokens. Without this, exp(-(-127)/tau) = exp(127) = inf, and
        # inf * causal_mask(0) = NaN when gradients are enabled.
        dt_causal = dt.clamp(min=0)
        causal_mask = (dt >= 0).float()
        if hasattr(module, "log_tau"):
            tau_t = torch.exp(module.log_tau).clamp(min=1.0)
            weights = torch.exp(-dt_causal.float() / tau_t) * causal_mask
        else:
            weights = torch.exp(-dt_causal.float() / tau) * causal_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attended = torch.matmul(weights.unsqueeze(0), x)  # (B, S, D)

    # Causal spike gate: use cumulative mean of abs values as running
    # threshold so we never peek at future tokens.
    abs_att = attended.abs()  # (B, S, D)
    # Cumulative mean over the sequence dimension (causal)
    cumsum = abs_att.cumsum(dim=1)
    counts = torch.arange(1, S + 1, device=x.device, dtype=x.dtype).view(1, S, 1)
    causal_mean = cumsum / counts  # (B, S, D)
    gate_input = 2.0 * (abs_att - causal_mean)
    spike_gate = _SigmoidSTE.apply(gate_input)

    return _apply_grad_scale(attended * spike_gate, module)


def execute_sparse_threshold(module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    """Adaptive median-based threshold gate.

    Zeros activations below the median absolute value, targeting ~50%
    sparsity. Uses a sigmoid STE for gradient flow through the
    thresholding operation.

    Args:
        module: The CompiledOp instance
        inputs: Variadic input tensors (expects inputs[0] as (B, S, D))

    Returns:
        Sparsified tensor of shape (B, S, D)
    """
    x = inputs[0]  # (B, S, D)
    if (
        _HAS_ARIA_CORE
        and x.is_contiguous()
        and x.ndim == 3
        and x.device.type == "cpu"
        and x.dtype == torch.float32
        and not x.requires_grad
    ):
        return aria_core.sparse_threshold_f32(x)
    abs_x = x.abs()
    # Causal running-mean threshold: at each position t, threshold is the
    # mean of abs values at positions 0..t (never peeks at future tokens).
    # This replaces the old global median which was non-causal.
    cumsum = abs_x.cumsum(dim=1)  # (B, S, D)
    counts = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(
        1, -1, 1
    )
    running_mean = cumsum / counts  # (B, S, D)

    # Sigmoid STE gate: hard threshold forward, sigmoid surrogate backward.
    # Scale factor of 5.0 keeps the sigmoid soft enough for gradient flow
    # Softer scale for better gradient flow through the STE gate.
    # 5.0 saturated sigmoid → gradient ≈ 0; 2.0 keeps gradient alive.
    scale = 2.0
    gate_input = scale * (abs_x - running_mean)
    gate = _SigmoidSTE.apply(gate_input)

    return _apply_grad_scale(x * gate, module)
