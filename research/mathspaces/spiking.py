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
import torch.nn.functional as F


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

def execute_lif(inputs: list, params: dict, dim: int) -> torch.Tensor:
    """Leaky Integrate-and-Fire neuron.

    Treats the sequence dimension as time steps. Accumulates membrane
    potential with exponential decay, fires binary spikes at threshold,
    and resets on fire.

    Args:
        inputs: Single tensor of shape (B, S, D)
        params: Unused (parameter-free)
        dim: Model dimension (unused)

    Returns:
        Binary spike tensor of shape (B, S, D), values in {0, 1}
    """
    x = inputs[0]  # (B, S, D)
    B, S, D = x.shape
    decay = 0.9
    threshold = 1.0

    spikes = []
    membrane = torch.zeros(B, D, device=x.device, dtype=x.dtype)

    for t in range(S):
        # Leaky integration
        membrane = decay * membrane + x[:, t, :]
        # Fire: use surrogate gradient for the threshold comparison
        spike = _SigmoidSTE.apply(membrane - threshold)
        spikes.append(spike)
        # Reset: subtract threshold on fire (soft reset)
        membrane = membrane - spike * threshold

    return torch.stack(spikes, dim=1)  # (B, S, D)


def execute_spike_rate_code(inputs: list, params: dict, dim: int) -> torch.Tensor:
    """Continuous → spike → continuous rate coding.

    Maps continuous activations to firing probabilities via sigmoid,
    samples binary spikes (with STE), then scales by original magnitude
    to preserve information content.

    Args:
        inputs: Single tensor of shape (B, S, D)
        params: Unused (parameter-free)
        dim: Model dimension (unused)

    Returns:
        Spike-coded tensor of shape (B, S, D)
    """
    x = inputs[0]  # (B, S, D)
    # Firing probability from continuous activation
    probs = torch.sigmoid(x)
    # Binary spikes with STE
    spikes = _BernoulliSTE.apply(probs)
    # Scale by original magnitude to preserve information
    magnitude = x.abs()
    return spikes * magnitude


def execute_stdp_attention(inputs: list, params: dict, dim: int) -> torch.Tensor:
    """STDP-inspired causal attention.

    Uses an exponential temporal decay kernel to create causal attention
    weights. Tokens attend more strongly to recent predecessors,
    mimicking spike-timing-dependent plasticity.

    No learnable parameters — purely temporal structure.

    Args:
        inputs: Single tensor of shape (B, S, D)
        params: Unused (parameter-free)
        dim: Model dimension (unused)

    Returns:
        Attended tensor of shape (B, S, D)
    """
    x = inputs[0]  # (B, S, D)
    B, S, D = x.shape

    # Temporal decay constant: tau = S/8, minimum 1
    tau = max(S / 8.0, 1.0)

    # Build causal exponential decay kernel: weight[i,j] = exp(-(i-j)/tau) for j<=i
    positions = torch.arange(S, device=x.device, dtype=x.dtype)
    # (S, S) matrix of time differences: dt[i,j] = i - j
    dt = positions.unsqueeze(1) - positions.unsqueeze(0)  # (S, S): dt[i,j] = i - j
    # Causal mask: only attend to past and current (j <= i means dt >= 0)
    causal_mask = (dt >= 0).float()
    # Exponential decay based on time gap
    weights = torch.exp(-dt.float() / tau) * causal_mask  # (S, S)
    # Normalize rows to sum to 1
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Apply attention: (B, S, D) via matrix multiply on seq dim
    # weights: (S, S), x: (B, S, D) -> (B, S, D)
    return torch.matmul(weights.unsqueeze(0), x)


def execute_sparse_threshold(inputs: list, params: dict, dim: int) -> torch.Tensor:
    """Adaptive median-based threshold gate.

    Zeros activations below the median absolute value, targeting ~50%
    sparsity. Uses a sigmoid STE for gradient flow through the
    thresholding operation.

    Args:
        inputs: Single tensor of shape (B, S, D)
        params: Unused (parameter-free)
        dim: Model dimension (unused)

    Returns:
        Sparsified tensor of shape (B, S, D)
    """
    x = inputs[0]  # (B, S, D)
    abs_x = x.abs()
    # Per-sample median across all positions (flatten S*D)
    median_val = abs_x.reshape(x.shape[0], -1).median(dim=-1).values  # (B,)
    median_val = median_val.view(-1, 1, 1)  # (B, 1, 1)

    # Sigmoid STE gate: smooth approximation for gradient, hard threshold forward
    # Scale factor makes sigmoid sharper around the threshold
    scale = 10.0
    gate_input = scale * (abs_x - median_val)
    gate = _SigmoidSTE.apply(gate_input)

    return x * gate
