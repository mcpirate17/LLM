"""Python fallback kernel for concat."""

import torch


class ComponentHandler:
    """Fallback handler for concat."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        a = inputs["a"]
        b = inputs["b"]
        return {"y": torch.cat([a, b], dim=-1)}
