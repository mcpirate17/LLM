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
import numpy as np

logger = logging.getLogger(__name__)

try:
    import aria_core

    _HAS_GROMOV_C = hasattr(aria_core, "gromov_delta_f32")
except ImportError:
    _HAS_GROMOV_C = False

# Maximum number of point samples for delta computation (O(n^4) complexity)
_MAX_SAMPLE_POINTS = 50


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

    # Sample 4-tuples
    n_sample = min(n, 30)
    indices = (
        np.random.choice(n, n_sample, replace=False) if n > n_sample else np.arange(n)
    )

    # C kernel fast path: ~10-20x faster for n>=15
    if _HAS_GROMOV_C:
        d_contig = np.ascontiguousarray(d, dtype=np.float32)
        d_tensor = torch.from_numpy(d_contig)
        idx_tensor = torch.from_numpy(indices.astype(np.int32))
        return float(aria_core.gromov_delta_f32(d_tensor, idx_tensor))

    max_delta = 0.0
    for i_idx in range(len(indices)):
        for j_idx in range(i_idx + 1, len(indices)):
            for k_idx in range(j_idx + 1, len(indices)):
                for l_idx in range(k_idx + 1, len(indices)):
                    x, y, z, w = (
                        indices[i_idx],
                        indices[j_idx],
                        indices[k_idx],
                        indices[l_idx],
                    )
                    s1 = d[x, y] + d[z, w]
                    s2 = d[x, z] + d[y, w]
                    s3 = d[x, w] + d[y, z]
                    sums = sorted([s1, s2, s3])
                    delta_val = (sums[2] - sums[1]) / 2.0
                    if delta_val > max_delta:
                        max_delta = delta_val

    return float(max_delta)


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

    # Flatten batch and sequence dims, sample tokens
    flat = representations.detach().reshape(-1, D).float().cpu().numpy()
    n = flat.shape[0]

    if n < 4:
        return {
            "hierarchy_fitness": 0.0,
            "gromov_delta": float("inf"),
            "n_tokens_sampled": n,
        }

    if n > max_tokens:
        idx = np.random.choice(n, max_tokens, replace=False)
        flat = flat[idx]
        n = max_tokens

    # Compute pairwise Euclidean distances
    try:
        from scipy.spatial.distance import pdist, squareform

        dist_condensed = pdist(flat, metric="euclidean")
        dist_matrix = squareform(dist_condensed)
    except ImportError:
        # Fallback without scipy
        diff = flat[:, None, :] - flat[None, :, :]
        dist_matrix = np.sqrt((diff**2).sum(axis=-1) + 1e-10)

    # Compute Gromov delta
    delta = gromov_delta(dist_matrix)

    # Normalize delta to [0, 1] fitness score
    # Use median distance as scale factor
    median_dist = (
        float(np.median(dist_matrix[dist_matrix > 0]))
        if (dist_matrix > 0).any()
        else 1.0
    )
    if median_dist < 1e-10:
        median_dist = 1.0

    # Normalized delta: delta / median_dist
    # Low normalized delta = tree-like = high fitness
    normalized_delta = delta / median_dist
    # Map to [0, 1]: fitness = exp(-normalized_delta)
    fitness = float(np.exp(-normalized_delta))

    return {
        "hierarchy_fitness": max(0.0, min(1.0, fitness)),
        "gromov_delta": delta,
        "n_tokens_sampled": n,
    }
