
import math

import torch
import torch.nn as nn
from .tropical import tropical_softmax, _adaptive_temperature, _SMOOTH_TAU


class _TropicalRouterFn(torch.autograd.Function):
    """Memory-efficient tropical distance routing.

    Computes min_d(x_norm[b,s,d] + centroids[e,d]) per chunk of S
    without retaining the full (B, S, n_experts, D) broadcast tensor.
    """

    @staticmethod
    def forward(ctx, x_norm: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        B, S, D = x_norm.shape
        n_experts = centroids.shape[0]
        chunk = 32

        adaptive_tau = _adaptive_temperature(_SMOOTH_TAU, D)
        inv_tau = 1.0 / adaptive_tau

        scores = torch.empty((B, S, n_experts), device=x_norm.device, dtype=x_norm.dtype)
        c_exp = centroids.unsqueeze(0).unsqueeze(0)  # (1, 1, E, D)

        for i in range(0, S, chunk):
            end = min(i + chunk, S)
            x_chunk = x_norm[:, i:end, :].unsqueeze(2)  # (B, c, 1, D)
            with torch.no_grad():
                pairwise = x_chunk + c_exp  # (B, c, E, D)
                scores[:, i:end, :] = -adaptive_tau * torch.logsumexp(
                    -pairwise * inv_tau, dim=-1)

        ctx.save_for_backward(x_norm, centroids)
        return scores

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_norm, centroids = ctx.saved_tensors
        B, S, D = x_norm.shape
        chunk = 16

        adaptive_tau = _adaptive_temperature(_SMOOTH_TAU, D)
        inv_tau = 1.0 / adaptive_tau

        grad_x = torch.zeros_like(x_norm)
        grad_c = torch.zeros_like(centroids)
        c_exp = centroids.unsqueeze(0).unsqueeze(0)

        for i in range(0, S, chunk):
            end = min(i + chunk, S)
            x_chunk = x_norm[:, i:end, :].unsqueeze(2)
            pairwise = x_chunk + c_exp
            neg_pw = -pairwise * inv_tau
            lse = torch.logsumexp(neg_pw, dim=-1, keepdim=True)
            sm_weights = torch.exp(neg_pw - lse)  # (B, c, E, D)

            g_out = grad_output[:, i:end, :].unsqueeze(-1)  # (B, c, E, 1)
            g_pw = g_out * sm_weights  # (B, c, E, D)

            grad_x[:, i:end, :] += g_pw.sum(dim=2)  # sum over experts
            grad_c += g_pw.sum(dim=(0, 1))  # sum over batch and seq chunk

            del pairwise, sm_weights, g_pw

        return grad_x, grad_c


class TropicalRouter(nn.Module):
    """High-Order Tropical Routing for MoE/MoD.

    Uses (max, +) algebra to compute routing scores. In the tropical semiring,
    matrix multiplication computes shortest-path distances.
    """
    def __init__(self, dim: int, n_experts: int = 128, temperature: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.temperature = temperature
        self.centroids = nn.Parameter(torch.randn(n_experts, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_min = torch.min(x, dim=-1, keepdim=True).values
        x_norm = x - x_min

        scores = _TropicalRouterFn.apply(x_norm, self.centroids)
        return tropical_softmax(scores, dim=-1, temperature=self.temperature)


class TropicalMoE(nn.Module):
    """Mixture-of-Experts using Tropical Routing."""
    def __init__(self, dim: int, n_experts: int = 1024, top_k: int = 2):
        super().__init__()
        self.router = TropicalRouter(dim, n_experts)
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(n_experts)
        ]) if n_experts <= 32 else None

        if n_experts > 32:
            self.expert_weights = nn.Parameter(torch.randn(n_experts, dim, dim) / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        routing_weights = self.router(x)

        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        if self.experts is not None:
            output = torch.zeros_like(x)
            for k in range(self.top_k):
                expert_idx = topk_indices[:, :, k]
                weight = topk_weights[:, :, k].unsqueeze(-1)
                for e_idx in range(len(self.experts)):
                    mask = (expert_idx == e_idx).unsqueeze(-1)
                    if mask.any():
                        expert_out = self.experts[e_idx](x)
                        output = output + mask.float() * weight * expert_out
        else:
            output = torch.zeros_like(x)
            for k in range(self.top_k):
                idx = topk_indices[:, :, k]
                weight = topk_weights[:, :, k].unsqueeze(-1)
                W = self.expert_weights[idx.reshape(-1)]
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
    weights = router(x)
    gated = torch.matmul(weights, router.centroids)
    return x * torch.sigmoid(gated)


def execute_tropical_moe(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Full tropical MoE: (B,S,D) -> (B,S,D)."""
    B, S, D = x.shape
    if not hasattr(module, '_tropical_moe'):
        n_experts = max(4, min(32, D // 8))
        module._tropical_moe = TropicalMoE(D, n_experts=n_experts, top_k=2).to(x.device)
    return module._tropical_moe(x)
