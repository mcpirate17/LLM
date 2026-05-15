"""Jacobian-based Effective Receptive Field (ERF) density probe.

Adapted from ``research/eval/jacobian_erf.py``. Measures how much
information from each input position influences the **last** output
position. A dense, uniform ERF means the architecture preserves global
dependency; a steeply-decaying or near-zero ERF means information is
bottlenecked or lost.

One forward + one backward pass — no training. Cheap gate that catches
information-bottleneck architectures (extreme local-only, pathological
attention collapse) before any training-based probe runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ERFResult:
    density: float  # mean(|grad|) / max(|grad|) — 1.0 = perfectly uniform
    density_entropy: float  # entropy(pp/sum(pp))/log(seq_len), 0..1, less peak-biased
    variance: float  # var(|grad|) across positions
    decay_slope: float  # linear-regression slope of |grad| vs distance-from-last
    last_position_norm: float
    first_position_norm: float
    passed: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def measure_erf(
    lane_block: nn.Module,
    *,
    seq_len: int = 32,
    dim: int = 32,
    batch_size: int = 2,
    density_threshold: float = 0.05,
    density_entropy_threshold: float = 0.10,
    max_decay_slope: float = 0.5,
    seed: int = 0,
) -> ERFResult:
    """Compute ERF density + decay for ``lane_block`` at one position pair.

    Pass conditions (density OR entropy, AND slope):
    - density >= ``density_threshold`` (mean/max ratio — penalizes peakiness)
      OR density_entropy >= ``density_entropy_threshold`` (entropy of normalized
      per-position response, less biased by the residual peak)
    - |decay_slope| <= ``max_decay_slope`` (information doesn't trivially fade)

    Reference values at default init (dim=32, seq_len=32):
      per-position nn.Linear + residual:  density=0.031, entropy=0.000
      LinearStateSpaceLane (diagonal SSM): density=0.041, entropy=0.248
    The OR-path lets the SSM pass; the per-position lane still fails.
    """
    torch.manual_seed(seed)
    lane_block.eval()
    try:
        x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        y = lane_block(x)
        target = y[:, -1, :].sum()
        grads = torch.autograd.grad(target, x, retain_graph=False)[0]
        per_position = grads.abs().mean(dim=(0, 2))  # [seq_len]
    except Exception as exc:  # noqa: BLE001
        return ERFResult(
            density=0.0,
            density_entropy=0.0,
            variance=0.0,
            decay_slope=0.0,
            last_position_norm=0.0,
            first_position_norm=0.0,
            passed=False,
            notes=(f"{type(exc).__name__}: {exc}",),
        )

    pp = per_position.detach().cpu()
    max_val = float(pp.max().item())
    if max_val <= 0.0:
        return ERFResult(
            density=0.0,
            density_entropy=0.0,
            variance=0.0,
            decay_slope=0.0,
            last_position_norm=0.0,
            first_position_norm=0.0,
            passed=False,
            notes=("zero gradient — disconnected from input",),
        )
    density = float(pp.mean().item()) / max_val
    pp_sum = float(pp.sum().item())
    if pp_sum > 0.0 and seq_len > 1:
        pp_norm = pp / pp_sum
        entropy = float(-(pp_norm * pp_norm.clamp_min(1e-12).log()).sum().item())
        density_entropy = entropy / math.log(seq_len)
    else:
        density_entropy = 0.0
    variance = float(pp.var().item())
    distances = torch.arange(seq_len - 1, -1, -1, dtype=pp.dtype)
    centered_d = distances - distances.mean()
    centered_p = pp - pp.mean()
    decay_slope = float(
        (centered_d * centered_p).sum() / centered_d.pow(2).sum().clamp_min(1e-12)
    )
    last_norm = float(pp[-1].item())
    first_norm = float(pp[0].item())
    density_path = density >= density_threshold
    entropy_path = density_entropy >= density_entropy_threshold
    passes = (density_path or entropy_path) and abs(decay_slope) <= max_decay_slope
    return ERFResult(
        density=density,
        density_entropy=density_entropy,
        variance=variance,
        decay_slope=decay_slope,
        last_position_norm=last_norm,
        first_position_norm=first_norm,
        passed=passes,
    )
