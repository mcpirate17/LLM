"""Python fallback kernel for speculative."""

import torch


class ComponentHandler:
    """Fallback handler for speculative: cheap path + learned blending."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        # Cheap path: identity. Gate: sigmoid blend.
        gate = torch.sigmoid(x.mean(dim=-1, keepdim=True))
        return {"y": x * gate + x * (1 - gate)}
