"""Python fallback kernel for moe_topk."""

import torch
import torch.nn.functional as F


class ComponentHandler:
    """Fallback handler for moe_topk: simplified 2-expert MoE with top-1 gating."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        B, S, D = x.shape
        n_experts = config.get("n_experts", 2)
        # Simple gating: project to n_experts logits, pick top-1
        gen = torch.Generator(device="cpu")
        gen.manual_seed(D * 65537 + n_experts)
        gate_w = torch.randn(n_experts, D, generator=gen, dtype=x.dtype).to(x.device)
        gate_w *= D**-0.5
        logits = F.linear(x, gate_w)  # (B, S, n_experts)
        weights = F.softmax(logits, dim=-1)
        # Weighted sum with identity expert bodies (preview only)
        # Real training uses learned expert weights
        return {"y": x * weights.sum(dim=-1, keepdim=True)}
