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


class _BernoulliSTE(torch.autograd.Function):
    """Bernoulli sampling with straight-through estimator."""

    @staticmethod
    def forward(ctx, probs):
        ctx.save_for_backward(probs)
        return torch.bernoulli(probs)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient straight through
        return grad_output


# ── Spiking primitives ──

def _gradient_scale(module: nn.Module | None) -> float:
    for attr in ("n_layers", "num_layers", "layer_count", "depth", "max_depth"):
        value = getattr(module, attr, None)
        if value is not None:
            depth = max(1, int(value))
            return float(depth ** -0.5)
    return 1.0


def _apply_grad_scale(tensor: torch.Tensor, module: nn.Module | None) -> torch.Tensor:
    scale = _gradient_scale(module)
    if tensor.requires_grad and scale < 0.999:
        tensor.register_hook(lambda grad: grad * scale)
    return tensor

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

    # Adaptive threshold: scale with input magnitude.
    # Steady-state membrane std ≈ input_std / sqrt(1 - decay²).
    # Setting threshold at ~1.5× that gives ~20-40% spike rate.
    input_std = x.detach().std().clamp(min=1e-6).item()
    steady_state_std = input_std / (1.0 - decay**2) ** 0.5
    threshold = steady_state_std * 1.5

    if _HAS_ARIA_CORE and x.is_contiguous() and x.ndim == 3 and x.device.type == "cpu" and not x.requires_grad:
        return aria_core.lif_neuron_f32(x, decay, threshold)

    B, S, D = x.shape

    membrane = torch.zeros(B, D, device=x.device, dtype=x.dtype)
    spike_list = []

    for t in range(S):
        membrane = decay * membrane + x[:, t, :]
        # Surrogate spike: sigmoid of shifted membrane (differentiable)
        spike_surr = torch.sigmoid(5.0 * (membrane - threshold))
        # Hard spike for forward, surrogate for backward (STE)
        spike_hard = (membrane >= threshold).float()
        spike = spike_hard + spike_surr - spike_surr.detach()  # straight-through
        spike_list.append(spike)
        membrane = membrane * (1.0 - spike_hard)  # reset on fire

    output = torch.stack(spike_list, dim=1)
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
    if _HAS_ARIA_CORE and x.is_contiguous() and x.ndim == 3 and x.device.type == "cpu" and not x.requires_grad:
        return aria_core.spike_rate_code_f32(x)

    n_steps = 8
    # Firing probability from continuous activation
    probs = torch.sigmoid(x)  # (B, S, D), in [0, 1]

    # Multi-step stochastic rate coding with Bernoulli STE
    spike_sum = torch.zeros_like(x)
    for _ in range(n_steps):
        spikes = _BernoulliSTE.apply(probs)  # binary {0, 1} with STE
        spike_sum = spike_sum + spikes

    # Decode: average spike rate across timesteps
    spike_rate = spike_sum / n_steps  # in [0, 1]

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
    if hasattr(module, 'log_tau'):
        tau = torch.exp(module.log_tau).clamp(min=1.0).item()
    else:
        tau = max(S / 8.0, 1.0)

    # Compute attention via C kernel or Python fallback
    if _HAS_ARIA_CORE and x.is_contiguous() and x.ndim == 3 and x.device.type == "cpu" and not x.requires_grad:
        attended = aria_core.stdp_attention_f32(x, tau, 0.0)
    else:
        # Build causal exponential decay kernel: weight[i,j] = exp(-(i-j)/tau) for j<=i
        positions = torch.arange(S, device=x.device, dtype=x.dtype)
        dt = positions.unsqueeze(1) - positions.unsqueeze(0)
        causal_mask = (dt >= 0).float()
        # Use differentiable tau when learnable
        if hasattr(module, 'log_tau'):
            tau_t = torch.exp(module.log_tau).clamp(min=1.0)
            weights = torch.exp(-dt.float() / tau_t) * causal_mask
        else:
            weights = torch.exp(-dt.float() / tau) * causal_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attended = torch.matmul(weights.unsqueeze(0), x)  # (B, S, D)

    # Causal spike gate: use cumulative mean of abs values as running
    # threshold so we never peek at future tokens.
    abs_att = attended.abs()  # (B, S, D)
    # Cumulative mean over the sequence dimension (causal)
    cumsum = abs_att.cumsum(dim=1)
    counts = torch.arange(1, S + 1, device=x.device, dtype=x.dtype).view(1, S, 1)
    causal_mean = cumsum / counts  # (B, S, D)
    gate_input = 5.0 * (abs_att - causal_mean)
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
    if (_HAS_ARIA_CORE and x.is_contiguous() and x.ndim == 3
            and x.device.type == "cpu" and x.dtype == torch.float32
            and not x.requires_grad):
        return aria_core.sparse_threshold_f32(x)
    abs_x = x.abs()
    # Per-sample median across all positions (flatten S*D)
    median_val = abs_x.reshape(x.shape[0], -1).median(dim=-1).values  # (B,)
    median_val = median_val.view(-1, 1, 1)  # (B, 1, 1)

    # Sigmoid STE gate: hard threshold forward, sigmoid surrogate backward.
    # Scale factor of 5.0 keeps the sigmoid soft enough for gradient flow
    # (scale=10.0 causes extreme saturation → STE_ratio ≈ 0).
    scale = 5.0
    gate_input = scale * (abs_x - median_val)
    gate = _SigmoidSTE.apply(gate_input)

    return _apply_grad_scale(x * gate, module)
