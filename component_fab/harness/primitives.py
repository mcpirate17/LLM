"""Shared numeric primitives for harness blocks and probes.

Single home for the small nn.Modules / functions that were duplicated
across ``standard_block``, ``top_ar_block``, ``tiny_lm``, ``probe_tasks``
and ``probe_block``:

- ``RMSNorm`` — the ``top_ar_block`` formulation (``x / (rms + eps)``).
  NOTE for former ``standard_block._RMSNorm`` users: that variant clamped
  the mean-square by ``eps`` *before* the sqrt; the surviving formulation
  adds ``eps`` *after* the sqrt. Numerically near-identical except for
  near-zero inputs, where the survivor is the one the generator stack
  (``block_templates``, ``routing_primitives``) was already trained
  against — grading comparability is now uniform.
- ``swiglu`` / ``SwiGLU`` — gated FFN. ``tiny_lm._MLP`` keeps its own
  module attribute names (``fc1/fc2/fc3``, biased, 2/3-width) for
  checkpoint compatibility but routes through :func:`swiglu`.
- ``CausalDepthwiseConv1d`` — causal depthwise conv along the sequence
  axis (state-dict attribute ``conv``, matching the former
  ``top_ar_block.CausalConv1dSeq`` / ``tiny_lm.CausalConv1dLane.conv``).
- ``causal_running_mean`` — the canonical running-mean target.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Standard RMSNorm (no bias) — matches `_op_rmsnorm` semantics."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * self.weight


def swiglu(
    x: torch.Tensor, gate: nn.Linear, value: nn.Linear, out: nn.Linear
) -> torch.Tensor:
    """Functional SwiGLU: ``out( SiLU(gate(x)) * value(x) )``."""
    return out(F.silu(gate(x)) * value(x))


class SwiGLU(nn.Module):
    """SwiGLU FFN: ``W3(silu(W1 x) * (W2 x))``.

    Hidden width defaults to ``round(dim * mlp_ratio)`` (the historical
    ``top_ar_block`` formula); pass ``hidden`` to override (e.g. the
    param-matched ``2/3``-width formula used by ``tiny_lm._MLP``).
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        *,
        hidden: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        width = int(round(dim * float(mlp_ratio))) if hidden is None else int(hidden)
        self.w1 = nn.Linear(dim, width, bias=bias)
        self.w2 = nn.Linear(dim, width, bias=bias)
        self.w3 = nn.Linear(width, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu(x, self.w1, self.w2, self.w3)


class CausalDepthwiseConv1d(nn.Module):
    """Causal depthwise 1D convolution along the sequence axis.

    Mirrors the ``conv1d_seq`` compiler op: depthwise causal mixing with a
    short kernel. ``[B, S, D] -> [B, S, D]``; left-pads so position ``i``
    only sees ``<= i``.
    """

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.pad = self.kernel_size - 1
        self.conv = nn.Conv1d(dim, dim, self.kernel_size, groups=dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = F.pad(h, (self.pad, 0))
        h = self.conv(h)
        return h.transpose(1, 2)


def causal_running_mean(x: torch.Tensor) -> torch.Tensor:
    """``target[i] = mean(x[0:i+1])`` along the sequence axis of ``[B, L, D]``."""
    seq_len = x.shape[1]
    weights = torch.arange(1, seq_len + 1, dtype=x.dtype, device=x.device).view(
        1, -1, 1
    )
    return x.cumsum(dim=1) / weights
