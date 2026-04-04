"""Python fallback kernel for n_way_sparse_router."""

import torch
import torch.nn.functional as F
from aria_designer.components._weight_cache import cached_randn


class ComponentHandler:
    """N-way sparse router: N bottleneck experts, top-k activation.

    Fully vectorized: all expert weights are stacked into (N, H, D) and (N, D, H)
    tensors, a single batched einsum computes all expert outputs, then torch.gather
    selects the top-k outputs. No Python loops over experts or top-k slots.
    """

    __slots__ = ()

    def validate_config(self, config):
        errors = []
        n = config.get("n_ways", 4)
        k = config.get("top_k", 2)
        if not isinstance(n, int) or n < 2:
            errors.append("n_ways must be int >= 2")
        if not isinstance(k, int) or k < 1:
            errors.append("top_k must be int >= 1")
        if isinstance(n, int) and isinstance(k, int) and k > n:
            errors.append("top_k must be <= n_ways")
        return errors

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        B, S, D = x.shape
        n_ways = max(2, min(config.get("n_ways", 4), 16))
        top_k = max(1, min(config.get("top_k", 2), n_ways))
        hidden = max(1, D // n_ways)

        # Gate — cached weight
        W_gate = cached_randn(
            n_ways, D, seed=D * 65537, device=x.device, dtype=x.dtype, scale=D**-0.5
        )
        gate_logits = F.linear(x, W_gate)  # (B, S, N)

        # Top-k selection
        topk_vals, topk_idx = gate_logits.topk(top_k, dim=-1)  # (B, S, k)
        gate_weights = F.softmax(topk_vals, dim=-1)  # (B, S, k)

        # Stack all expert weights into single tensors for batched matmul.
        # For n_ways <= 16, computing all experts then gathering is faster than
        # Python loops — the BLAS parallelism on the batched einsum dominates.
        W_downs = torch.stack(
            [
                cached_randn(
                    hidden,
                    D,
                    seed=100 * n_ways + i,
                    device=x.device,
                    dtype=x.dtype,
                    scale=D**-0.5,
                )
                for i in range(n_ways)
            ]
        )  # (N, hidden, D)
        W_ups = torch.stack(
            [
                cached_randn(
                    D,
                    hidden,
                    seed=200 * n_ways + i,
                    device=x.device,
                    dtype=x.dtype,
                    scale=hidden**-0.5,
                )
                for i in range(n_ways)
            ]
        )  # (N, D, hidden)

        # All experts forward in one batched einsum each:
        # x: (B,S,D) @ W_downs^T: (N,D,hidden) -> (B,S,N,hidden)
        expert_hidden = torch.einsum("bsd,nhd->bsnh", x, W_downs)
        expert_hidden = F.gelu(expert_hidden)
        # (B,S,N,hidden) @ W_ups^T: (N,hidden,D) -> (B,S,N,D)
        expert_out = torch.einsum("bsnh,ndh->bsnd", expert_hidden, W_ups)

        # Gather top-k expert outputs: (B,S,N,D) -> (B,S,k,D)
        idx = topk_idx.unsqueeze(-1).expand(B, S, top_k, D)  # (B, S, k, D)
        selected = torch.gather(expert_out, dim=2, index=idx)  # (B, S, k, D)

        # Weighted sum over top-k dimension
        output = (selected * gate_weights.unsqueeze(-1)).sum(dim=2)  # (B, S, D)

        return {"y": output}
