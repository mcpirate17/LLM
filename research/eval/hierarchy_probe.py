"""
Hierarchy Probe — Gromov Delta-Hyperbolicity Detection

Detects when learned representations exhibit tree-like structure,
indicating that hyperbolic geometry would be a better fit than Euclidean.

Uses Gromov's 4-point delta-hyperbolicity: for any 4 points in a metric
space, the delta measures how tree-like the space is. Low delta = tree-like.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)

try:
    import aria_core

    _HAS_GROMOV_C = hasattr(aria_core, "gromov_delta_f32")
except ImportError:
    _HAS_GROMOV_C = False

import numpy as np


# Maximum number of point samples for delta computation (O(n^4) complexity)
_MAX_SAMPLE_POINTS = 30


def gromov_delta(distance_matrix: np.ndarray) -> float:
    """Estimate Gromov's 4-point delta-hyperbolicity from a distance matrix.

    For any 4 points (x, y, z, w), compute the three pairwise distance sums:
        S1 = d(x,y) + d(z,w)
        S2 = d(x,z) + d(y,w)
        S3 = d(x,w) + d(y,z)

    Sort so S1 <= S2 <= S3. Then delta = (S3 - S2) / 2.
    The space is delta-hyperbolic where delta is the max over all 4-tuples.

    Args:
        distance_matrix: Square symmetric distance matrix (n, n).

    Returns:
        Estimated delta (>= 0). Low values indicate tree-like structure.
    """
    n = distance_matrix.shape[0]
    if n < 4:
        return 0.0

    # Subsample if too many points
    if n > _MAX_SAMPLE_POINTS:
        idx = np.random.choice(n, _MAX_SAMPLE_POINTS, replace=False)
        distance_matrix = distance_matrix[np.ix_(idx, idx)]
        n = _MAX_SAMPLE_POINTS

    d = distance_matrix
    indices = np.arange(n)

    # C kernel fast path: ~10-20x faster for n>=15
    if _HAS_GROMOV_C:
        d_contig = np.ascontiguousarray(d, dtype=np.float32)
        d_tensor = torch.from_numpy(d_contig)
        idx_tensor = torch.from_numpy(indices.astype(np.int32))
        return float(aria_core.gromov_delta_f32(d_tensor, idx_tensor))

    idx = np.array(indices)
    n_idx = len(idx)
    i0, i1, i2, i3 = np.meshgrid(
        np.arange(n_idx),
        np.arange(n_idx),
        np.arange(n_idx),
        np.arange(n_idx),
        indexing="ij",
    )
    mask = (i0 < i1) & (i1 < i2) & (i2 < i3)
    x = idx[i0[mask]]
    y = idx[i1[mask]]
    z = idx[i2[mask]]
    w = idx[i3[mask]]
    s1 = d[x, y] + d[z, w]
    s2 = d[x, z] + d[y, w]
    s3 = d[x, w] + d[y, z]
    sums = np.stack([s1, s2, s3], axis=1)
    sums.sort(axis=1)
    deltas = (sums[:, 2] - sums[:, 1]) / 2.0
    return float(deltas.max()) if len(deltas) > 0 else 0.0


def hierarchy_fitness(
    representations: torch.Tensor,
    max_tokens: int = 100,
) -> Dict[str, float]:
    """Compute hierarchy fitness score from model representations.

    Returns 0-1 score where high = tree-like structure detected.

    Args:
        representations: Tensor of shape (B, S, D) — model hidden states.
        max_tokens: Maximum tokens to sample for analysis.

    Returns:
        Dict with keys:
            hierarchy_fitness: float in [0, 1] (1 = very tree-like)
            gromov_delta: float (raw delta value)
            n_tokens_sampled: int
    """
    if representations.ndim != 3:
        return {
            "hierarchy_fitness": 0.0,
            "gromov_delta": float("inf"),
            "n_tokens_sampled": 0,
        }

    B, S, D = representations.shape

    flat = representations.detach().reshape(-1, D).float().cpu()
    n = flat.shape[0]

    if n < 4:
        return {
            "hierarchy_fitness": 0.0,
            "gromov_delta": float("inf"),
            "n_tokens_sampled": n,
        }

    if n > max_tokens:
        idx = torch.randperm(n)[:max_tokens]
        flat = flat.index_select(0, idx)
        n = max_tokens

    dist_matrix_t = torch.cdist(flat, flat)
    dist_matrix = dist_matrix_t.numpy()

    delta = gromov_delta(dist_matrix)

    nonzero = dist_matrix_t[dist_matrix_t > 0]
    median_dist = float(nonzero.median().item()) if nonzero.numel() > 0 else 1.0
    if median_dist < 1e-10:
        median_dist = 1.0

    normalized_delta = delta / median_dist
    fitness = float(torch.exp(torch.tensor(-normalized_delta)).item())

    return {
        "hierarchy_fitness": max(0.0, min(1.0, fitness)),
        "gromov_delta": delta,
        "n_tokens_sampled": n,
    }
