"""Python fallback kernel for moe_topk."""

import torch.nn.functional as F

from aria_designer.components._weight_cache import cached_randn


class ComponentHandler:
    """Fallback handler for moe_topk: simplified 2-expert MoE with top-1 gating."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        D = x.shape[-1]
        n_experts = config.get("n_experts", 2)
        gate_w = cached_randn(
            n_experts,
            D,
            seed=D * 65537 + n_experts,
            device=x.device,
            dtype=x.dtype,
            scale=D**-0.5,
        )
        logits = F.linear(x, gate_w)  # (B, S, n_experts)
        weights = F.softmax(logits, dim=-1)
        return {"y": x * weights.sum(dim=-1, keepdim=True)}
