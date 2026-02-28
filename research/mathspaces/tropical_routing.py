
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .tropical import tropical_matmul, tropical_softmax

class TropicalRouter(nn.Module):
    """
    High-Order Tropical Routing for MoE/MoD.
    (Project Hephaestus Phase 4)

    Uses (max, +) algebra to compute routing scores. In the tropical semiring,
    matrix multiplication computes shortest-path distances. This allows for
    extremely efficient routing to a large number of "micro-experts" (1,000+).

    Instead of Softmax(x @ W), it uses:
    1. x_tropical = x - min(x) (Normalization)
    2. scores = tropical_matmul(x_tropical, expert_centroids)
    3. routing = tropical_softmax(-scores)

    This captures topological proximity in the embedding space with lower
    computational "tax" than standard dot-product attention.
    """
    def __init__(self, dim: int, n_experts: int = 128, temperature: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.temperature = temperature
        # Expert centroids in the embedding space
        self.centroids = nn.Parameter(torch.randn(n_experts, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, S, D)
        Output: (B, S, n_experts) - Routing probabilities
        """
        B, S, D = x.shape
        
        # 1. Tropical Normalization (Min-centering)
        # x_min: (B, S, 1)
        x_min = torch.min(x, dim=-1, keepdim=True).values
        x_norm = x - x_min
        
        # 2. Tropical Distance to Centroids
        # centroids: (n_experts, D)
        # We want (B, S, n_experts) scores where score[b,s,e] = min_d(x_norm[b,s,d] + centroids[e,d])
        # This is tropical_matmul(x_norm, centroids.T)
        
        # x_norm: (B, S, D), centroids_t: (D, n_experts)
        # pairwise: (B, S, n_experts, D)
        expanded_x = x_norm.unsqueeze(2) # (B, S, 1, D)
        expanded_c = self.centroids.unsqueeze(0).unsqueeze(0) # (1, 1, n_experts, D)
        
        # Tropical multiply: addition
        pairwise = expanded_x + expanded_c # (B, S, n_experts, D)
        
        # Tropical add: minimum over D
        # scores[b,s,e] is the "shortest path" from token to expert
        scores = torch.min(pairwise, dim=-1).values # (B, S, n_experts)
        
        # 3. Routing Weights
        # Higher similarity (smaller distance) -> Higher probability
        routing_weights = tropical_softmax(scores, dim=-1, temperature=self.temperature)
        
        return routing_weights

class TropicalMoE(nn.Module):
    """
    Mixture-of-Experts using Tropical Routing.
    Optimized for high expert counts (Micro-Experts).
    """
    def __init__(self, dim: int, n_experts: int = 1024, top_k: int = 2):
        super().__init__()
        self.router = TropicalRouter(dim, n_experts)
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(n_experts)
        ]) if n_experts <= 32 else None # Standard MoE only for small expert counts
        
        # For 1,000+ experts, we use a vectorized expert bank (ExpertPool)
        if n_experts > 32:
            self.expert_weights = nn.Parameter(torch.randn(n_experts, dim, dim) / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        routing_weights = self.router(x) # (B, S, n_experts)
        
        # Select top-k experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Combine expert outputs
        if self.experts is not None:
            # Small expert count (<=32): iterate experts, mask by topk_indices
            output = torch.zeros_like(x)
            for k in range(self.top_k):
                expert_idx = topk_indices[:, :, k]  # (B, S)
                weight = topk_weights[:, :, k].unsqueeze(-1)  # (B, S, 1)
                for e_idx in range(len(self.experts)):
                    mask = (expert_idx == e_idx).unsqueeze(-1)  # (B, S, 1)
                    if mask.any():
                        expert_out = self.experts[e_idx](x)
                        output = output + mask.float() * weight * expert_out
        else:
            # Large expert count (>32): batched matmul via expert_weights
            # Gather selected expert weight matrices and combine
            output = torch.zeros_like(x)
            for k in range(self.top_k):
                idx = topk_indices[:, :, k]  # (B, S)
                weight = topk_weights[:, :, k].unsqueeze(-1)  # (B, S, 1)
                # Gather expert weights: (B, S, D, D)
                W = self.expert_weights[idx.reshape(-1)]  # (B*S, D, D)
                W = W.view(B, S, D, D)
                expert_out = torch.einsum('bsd,bsde->bse', x, W)
                output = output + weight * expert_out

        return output


# ── Primitive execution functions ─────────────────────────────────────

def execute_tropical_router(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Tropical routing as a gating signal: (B,S,D) -> (B,S,D)."""
    B, S, D = x.shape
    if not hasattr(module, '_tropical_router'):
        n_experts = max(4, min(64, D // 4))
        module._tropical_router = TropicalRouter(D, n_experts).to(x.device)
    router = module._tropical_router
    # Use routing weights to gate input: weighted sum of routing scores
    weights = router(x)  # (B, S, n_experts)
    # Project routing weights back to D via centroids transpose
    gated = torch.matmul(weights, router.centroids)  # (B, S, D)
    return x * torch.sigmoid(gated)


def execute_tropical_moe(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Full tropical MoE: (B,S,D) -> (B,S,D)."""
    B, S, D = x.shape
    if not hasattr(module, '_tropical_moe'):
        n_experts = max(4, min(32, D // 8))
        module._tropical_moe = TropicalMoE(D, n_experts=n_experts, top_k=2).to(x.device)
    return module._tropical_moe(x)
