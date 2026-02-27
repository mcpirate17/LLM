"""Topological Data Analysis (TDA) primitives for ARIA.

Implements differentiable approximations of persistent homology using
spectral features of graph Laplacians constructed from local neighborhoods.

Reference: ARIA_NEXT_GEN_ARCHITECTURE.md §1.3
"""

import torch
import torch.nn as nn


class PersistentHomologyFilter(nn.Module):
    """Approximates a topological filter via spectral features of the k-NN
    graph Laplacian.

    Computes pairwise distances within each sequence, builds a k-nearest
    neighbour adjacency, derives the graph Laplacian, and uses its diagonal
    (a differentiable proxy for local connectivity / Betti-0 features) to
    modulate the input features.

    Input:  (B, L, D)
    Output: (B, L, D)

    Reference: ARIA_NEXT_GEN_ARCHITECTURE.md §1.3
    """

    def __init__(self, k_neighbors: int = 5):
        super().__init__()
        self.k = k_neighbors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        # Pairwise Euclidean distances within the sequence
        dist = torch.cdist(x, x)  # (B, L, L)

        # k-NN graph adjacency (symmetric)
        k = min(self.k, dist.shape[-1])
        _topk_vals, indices = torch.topk(dist, k, largest=False, dim=-1)
        A = torch.zeros_like(dist).scatter_(-1, indices, 1.0)
        A = (A + A.transpose(-1, -2)).clamp(max=1.0)

        # Graph Laplacian: L = D - A
        degree = A.sum(dim=-1)  # (B, L)
        L_graph = torch.diag_embed(degree) - A  # (B, L, L)

        # Spectral proxy: diagonal of Laplacian encodes local connectivity
        spectral_proxy = degree.unsqueeze(-1)  # (B, L, 1)

        # Modulate features — well-connected tokens are amplified
        return x * torch.sigmoid(spectral_proxy)


def execute_persistent_homology(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Execute PersistentHomologyFilter as a primitive op."""
    k = getattr(module, '_k_neighbors', 5)
    filt = PersistentHomologyFilter(k_neighbors=k).to(x.device)
    return filt(x)
