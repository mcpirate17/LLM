"""Python fallback kernel for cascade."""

import torch


class ComponentHandler:
    """Fallback handler for cascade: progressive difficulty gating."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        # Soft gating by token difficulty (mean activation as proxy)
        scores = x.mean(dim=-1, keepdim=True)
        gate = torch.sigmoid(scores)
        return {"y": x * gate}
