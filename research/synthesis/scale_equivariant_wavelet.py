"""NM-F6 - Scale-equivariant wavelet stack.

A causal ``[B, L, D] -> [B, L, D]`` sequence mixer built from one learned mother
filter reused at dyadic dilations:

    y_j[t] = sum_m psi[m] * x[t - m * 2**j]

This is the dilation-group tie NM-F6 is meant to test. Long-range filters are not
separate parameters; they are the same short filter resampled onto a wider grid.
The scale mixer is a small non-probability linear combination over scales, not a
softmax over positions, and the channel readout is a 1x1 map. The result is a
non-QKV, non-attention mechanism whose receptive field grows exponentially with
``n_scales`` while parameters grow only linearly in ``n_scales`` and once in the
filter length.

This module is deliberately self-contained and torch-only, matching the other
NM-F reference operators. The grouped conv1d path is the highest-performance
portable PyTorch option for the probe implementation; a native fused scan is
the later production path if this lane graduates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def scale_wavelet_param_count(dim: int, kernel_size: int, n_scales: int) -> int:
    """Trainable params: shared mother filter + scale mix + D x D readout + ReZero."""
    _validate(dim, kernel_size, n_scales)
    return kernel_size + n_scales + dim * dim + 1


def _validate(dim: int, kernel_size: int, n_scales: int) -> None:
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if kernel_size < 2:
        raise ValueError(f"kernel_size must be >= 2, got {kernel_size}")
    if n_scales < 1:
        raise ValueError(f"n_scales must be >= 1, got {n_scales}")


def causal_atrous_kernel(
    mother_filter: torch.Tensor,
    *,
    dilation: int,
) -> torch.Tensor:
    """Expand lag-indexed taps into a dense causal à trous kernel.

    ``mother_filter[m]`` multiplies lag ``m * dilation``. The returned tensor is
    ordered for ``torch.nn.functional.conv1d`` with left padding, so its last
    element is the current-token tap.
    """
    if mother_filter.ndim != 1:
        raise ValueError(f"mother_filter must be 1D, got {tuple(mother_filter.shape)}")
    if dilation < 1:
        raise ValueError(f"dilation must be >= 1, got {dilation}")
    kernel_size = int(mother_filter.numel())
    dense = mother_filter.new_zeros((kernel_size - 1) * dilation + 1)
    for lag in range(kernel_size):
        dense[-1 - lag * dilation] = mother_filter[lag]
    return dense


def causal_atrous_conv1d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv with one shared kernel for every channel."""
    if x.ndim != 3:
        raise ValueError(f"x must be (B,L,D), got {tuple(x.shape)}")
    if kernel.ndim != 1:
        raise ValueError(f"kernel must be 1D, got {tuple(kernel.shape)}")
    channels = x.shape[-1]
    x_t = x.transpose(1, 2)
    padded = F.pad(x_t, (kernel.numel() - 1, 0))
    weight = kernel.view(1, 1, -1).expand(channels, 1, -1)
    return F.conv1d(padded, weight, groups=channels).transpose(1, 2)


class ScaleEquivariantWaveletStack(nn.Module):
    """Shared-mother-filter dyadic à trous wavelet mixer."""

    def __init__(
        self,
        dim: int,
        *,
        kernel_size: int = 8,
        n_scales: int = 5,
    ) -> None:
        super().__init__()
        _validate(dim, kernel_size, n_scales)
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.n_scales = int(n_scales)

        self.mother_filter = nn.Parameter(self._init_mother_filter(kernel_size))
        self.scale_mix = nn.Parameter(torch.full((n_scales,), 1.0 / n_scales))
        self.out_lift = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.out_lift.weight.copy_(torch.eye(dim))
        self.residual_scale = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _init_mother_filter(kernel_size: int) -> torch.Tensor:
        """All-tap zero-mean compact wavelet seed, normalized to unit energy."""
        n_pos = (kernel_size + 1) // 2
        n_neg = kernel_size - n_pos
        taps = torch.empty(kernel_size, dtype=torch.float32)
        taps[:n_pos] = 1.0 / n_pos
        taps[n_pos:] = -1.0 / n_neg
        norm = taps.norm().clamp_min(1e-6)
        return taps / norm

    @property
    def num_parameters(self) -> int:
        return scale_wavelet_param_count(self.dim, self.kernel_size, self.n_scales)

    @property
    def max_receptive_field(self) -> int:
        """Largest causal span covered by the stack."""
        return 1 + (self.kernel_size - 1) * self.dilation_for_scale(self.n_scales - 1)

    def dilation_for_scale(self, scale: int) -> int:
        """Dyadic dilation for scale index ``scale``."""
        if scale < 0 or scale >= self.n_scales:
            raise ValueError(f"scale must be in [0, {self.n_scales}), got {scale}")
        return 1 << scale

    def expanded_kernel(self, scale: int) -> torch.Tensor:
        """Dense causal kernel for a scale, with the same mother taps inserted."""
        return causal_atrous_kernel(
            self.mother_filter,
            dilation=self.dilation_for_scale(scale),
        )

    def wavelet_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-scale responses ``(B, L, S, D)``."""
        if x.ndim != 3:
            raise ValueError(f"x must be (B,L,D), got {tuple(x.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"last dim must be {self.dim}, got {x.shape[-1]}")
        responses = [
            causal_atrous_conv1d(x, self.expanded_kernel(scale).to(dtype=x.dtype))
            for scale in range(self.n_scales)
        ]
        return torch.stack(responses, dim=2)

    def scale_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Mean squared response per dyadic scale, useful for octave diagnostics."""
        return self.wavelet_features(x).square().mean(dim=(0, 1, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual plus shared-filter wavelet stack readout."""
        features = self.wavelet_features(x)
        mixed = torch.einsum("s,blsd->bld", self.scale_mix.to(x.dtype), features)
        return x + self.residual_scale * self.out_lift(mixed)
