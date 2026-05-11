"""
Product-Quantized Embedding

Per external_research_2026-05-10.md §2.3 — "sparse connected tables /
product-quantization embedding". Splits the feature dim D into M equal
subspaces of size D/M, each with its own learned codebook of K centroids.
Each token's slice in subspace m is replaced by a softmax-weighted
combination of the K centroids in that subspace; M slices are
concatenated back to D.

Compression: an arbitrary D-vector is now reconstructed from M·log2(K)
bits of indices into the codebook (or M·K soft weights). For the
typical M=4, K=16 setting on D=128, the per-token state is M·log2(K) =
16 bits vs the original 128*32 = 4096 bits — a 256× compression of the
representation if used for KV caching. As a *layer*, it acts as an
expressive nonlinear quantization-style projection that the grammar
can compose with attention or routing.

Forward (soft assignment, training mode):

    x ∈ (B, S, D)
    reshape: (B, S, M, D/M)                  # M subspaces
    distances[m] = -||x[:,:,m,:] - C[m]||²    # (B, S, M, K)
    weights = softmax(distances / tau)        # (B, S, M, K)
    quant[m] = weights[m] @ C[m]              # (B, S, M, D/M)
    out = reshape(quant, (B, S, D))

Hot path: pure torch — reshape, cdist (or manual squared-distance),
softmax, matmul. All C++/CUDA backed. The squared-distance is the only
non-trivial op; at M=4, K=16, D/M=32 the per-subspace cost is small.
No custom kernel warranted at this scale.

Gradient health: the soft assignment + matmul against codebooks is
fully differentiable. The codebooks themselves are nn.Parameters
trained jointly. Straight-through-estimator hardening (argmax in the
forward, soft in the backward) would be a future tightening; the soft
form is more grammar-friendly because every codebook entry receives
gradient.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_DEFAULT_M = 4  # number of subspaces
_DEFAULT_K = 16  # codebook centroids per subspace
_DEFAULT_TAU = 1.0  # softmax temperature for assignment


def execute_pq_embedding(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Soft product-quantized embedding.

    Falls back to identity if module has no codebook param.
    """
    if not hasattr(module, "codebooks"):
        return x

    *batch_dims, D = x.shape
    codebooks = module.codebooks  # (M, K, sub_dim)
    M = codebooks.shape[0]
    sub_dim = codebooks.shape[2]
    if M * sub_dim != D:
        # Mismatched config (e.g. d_in changed since init). Fail-soft.
        return x

    dtype = x.dtype
    cb = codebooks.to(dtype)
    tau = float(getattr(module, "pq_tau", _DEFAULT_TAU))
    # Reshape: (..., D) → (..., M, sub_dim)
    x_split = x.reshape(*batch_dims, M, sub_dim)
    # Squared distance to each codebook entry, per subspace.
    # x_split is (..., M, sub_dim) → unsqueeze to (..., M, 1, sub_dim)
    # cb is (M, K, sub_dim) → broadcast as (..., M, K, sub_dim)
    diff = x_split.unsqueeze(-2) - cb
    sq = (diff * diff).sum(-1)  # (..., M, K)
    # Softmax over -squared-distance (closer = higher weight)
    weights = torch.softmax(-sq / max(tau, 1e-4), dim=-1)
    # Weighted sum over centroids: (..., M, K) × (M, K, sub_dim) → (..., M, sub_dim)
    quant = torch.einsum("...mk,mks->...ms", weights, cb)
    return quant.reshape(*batch_dims, D)
