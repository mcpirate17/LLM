"""Compression-quality metric for paired (compress, restore) modules.

Measures intrinsic properties of a compressor:
- ``reconstruction_mse_per_param``: average squared error of
  ``restore(compress(x)) - x`` divided by the parameter count.
- ``effective_rank_ratio``: rank of compressed activations measured by
  participation ratio of SVD singular values, divided by the declared
  latent dim. A value near 1.0 means the compressor genuinely uses
  its declared latent budget.
- ``flops_per_token_reduction``: 1 - (compressed_params / dense_params),
  approximated from parameter counts since the compress+restore path
  is the only thing flowing through the bottleneck.

Decoupled from any specific op registry — caller passes torch modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class CompressionScorecard:
    reconstruction_mse: float
    reconstruction_mse_per_param: float
    effective_rank: float
    effective_rank_ratio: float
    latent_dim_declared: int
    input_dim: int
    n_compress_params: int
    n_restore_params: int
    flops_per_token_reduction: float


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _participation_rank(latent: torch.Tensor) -> float:
    flat = latent.reshape(-1, latent.shape[-1])
    if flat.shape[0] == 0 or flat.shape[1] == 0:
        return 0.0
    centered = flat - flat.mean(dim=0, keepdim=True)
    if centered.numel() == 0:
        return 0.0
    try:
        s = torch.linalg.svdvals(centered.float())
    except RuntimeError:
        return 0.0
    s2 = s.pow(2)
    total = s2.sum()
    if total.item() <= 0.0:
        return 0.0
    return float(total.pow(2).item() / s2.pow(2).sum().item())


def measure_compression_quality(
    compress_fn: Callable[[torch.Tensor], torch.Tensor],
    restore_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    input_dim: int,
    latent_dim_declared: int,
    n_compress_params: int | None = None,
    n_restore_params: int | None = None,
    seq_len: int = 64,
    batch_size: int = 8,
    n_trials: int = 4,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> CompressionScorecard:
    """Probe a (compress, restore) pair for reconstruction + rank quality."""
    if isinstance(compress_fn, nn.Module) and n_compress_params is None:
        n_compress_params = _count_params(compress_fn)
    if isinstance(restore_fn, nn.Module) and n_restore_params is None:
        n_restore_params = _count_params(restore_fn)
    n_compress_params = int(n_compress_params or 0)
    n_restore_params = int(n_restore_params or 0)
    total_params = n_compress_params + n_restore_params

    generator = torch.Generator(device=device).manual_seed(seed)
    mse_total = 0.0
    rank_total = 0.0
    for _ in range(n_trials):
        x = torch.randn(
            batch_size,
            seq_len,
            input_dim,
            generator=generator,
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            latent = compress_fn(x)
            x_hat = restore_fn(latent)
        if x_hat.shape != x.shape:
            raise ValueError(
                f"restore(compress(x)) must match input shape; got {tuple(x_hat.shape)}"
            )
        mse_total += float((x - x_hat).pow(2).mean().item())
        rank_total += _participation_rank(latent.detach())

    reconstruction_mse = mse_total / n_trials
    effective_rank = rank_total / n_trials
    effective_rank_ratio = (
        effective_rank / latent_dim_declared if latent_dim_declared > 0 else 0.0
    )
    mse_per_param = (
        reconstruction_mse / total_params if total_params > 0 else float("inf")
    )
    dense_params = max(1, input_dim * input_dim)
    flops_reduction = max(0.0, 1.0 - (total_params / dense_params))

    return CompressionScorecard(
        reconstruction_mse=reconstruction_mse,
        reconstruction_mse_per_param=mse_per_param,
        effective_rank=effective_rank,
        effective_rank_ratio=effective_rank_ratio,
        latent_dim_declared=latent_dim_declared,
        input_dim=input_dim,
        n_compress_params=n_compress_params,
        n_restore_params=n_restore_params,
        flops_per_token_reduction=flops_reduction,
    )
