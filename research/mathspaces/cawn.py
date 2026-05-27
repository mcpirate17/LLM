"""
Continuous Acoustic Wave Network (CAWN) Operations

CAWN represents a 2026 innovation mapping sequence mixing to
complex-domain wave propagation. It models tokens as continuous
acoustic waves (phasors) undergoing phase accumulation.

Operations:
- phase_accumulation (O(L) causal complex wave mixing)
- cawn_attention (Phase-resonant sequence mixer)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def phase_accumulation(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """O(L) causal mixing of complex phasors.

    amplitude: (B, S, D) - real
    phase: (B, S, D) - real (represents radians)

    Returns: (B, S, D) mixed real features.
    """
    B, S, D = amplitude.shape

    # Convert to complex
    z = torch.polar(amplitude, phase)  # (B, S, D)

    # Accumulate causal state (simple discrete approximation of continuous wave)
    # y_t = y_{t-1} * exp(i * phase_t) + z_t
    state = torch.zeros((B, D), dtype=torch.complex64, device=amplitude.device)
    outputs = []

    for t in range(S):
        z_t = z[:, t, :]
        # Causal mix: rotate previous state by current phase and add new signal
        rotator = torch.polar(torch.ones_like(phase[:, t, :]), phase[:, t, :])
        state = state * rotator + z_t
        outputs.append(state)

    out_z = torch.stack(outputs, dim=1)  # (B, S, D)

    # Return the real part (or amplitude) as the mixed feature
    return out_z.real


def cawn_mixer(x: torch.Tensor) -> torch.Tensor:
    """CAWN continuous wave mixing block."""
    D = x.shape[-1]
    half_D = D // 2

    # Split into amplitude and phase
    amplitude = x[..., :half_D]
    phase = x[..., half_D:] * math.pi  # scale to [-pi, pi] approximately

    # Accumulate
    mixed = phase_accumulation(amplitude, phase)

    # Pad back to D
    return torch.cat([mixed, mixed], dim=-1)


# ── Primitive execution functions ─────────────────────────────────────


def execute_cawn_mixer(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Phase-resonant continuous wave sequence mixer."""
    orig_dtype = x.dtype
    # CAWN uses complex math, cast to float32
    mixed = cawn_mixer(x.to(torch.float32))
    return mixed.to(orig_dtype)
