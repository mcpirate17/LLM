"""Python fallback kernel for pq_embedding_moe_block."""

import torch
import torch.nn.functional as F
from aria_designer.components._weight_cache import cached_randn
from research.mathspaces.pq_embedding import execute_pq_embedding


class _StubModule:
    def __init__(self, codebooks, pq_tau):
        self.codebooks = codebooks
        self.pq_tau = pq_tau


class ComponentHandler:
    """Fallback handler for pq_embedding_moe_block."""

    __slots__ = ()

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        if x.dim() != 3:
            raise ValueError("pq_embedding_moe_block expects x with shape [B, S, D]")
        D = x.shape[-1]
        M = int(config.get("M", 4))
        K = int(config.get("K", 16))
        sub_dim = D // M

        # Proj in
        W_in = cached_randn(
            D, D, seed=D * 31, device=x.device, dtype=x.dtype, scale=D**-0.5
        )
        h = F.linear(x, W_in)

        # PQ
        codebooks = cached_randn(
            M, K, sub_dim, seed=D * 37, device=x.device, dtype=x.dtype, scale=0.02
        )
        stub = _StubModule(codebooks, float(config.get("tau", 1.0)))
        h = execute_pq_embedding(stub, h)

        # Proj out
        W_out = cached_randn(
            D, D, seed=D * 41, device=x.device, dtype=x.dtype, scale=D**-0.5
        )
        h = F.linear(h, W_out)

        # Simple MoE Gate (simplified like moe_topk fallback)
        n_experts = int(config.get("n_experts", 4))
        gate_w = cached_randn(
            n_experts, D, seed=D * 43, device=x.device, dtype=x.dtype, scale=D**-0.5
        )
        logits = F.linear(h, gate_w)
        weights = F.softmax(logits, dim=-1)

        # Simplified expert: just identity scaled by gate weights sum
        return {"y": h * weights.sum(dim=-1, keepdim=True)}
