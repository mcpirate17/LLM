"""Auto-generated Python fallback kernel for maximum."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for maximum."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        a = inputs["a"]
        b = inputs["b"]
        # TODO: implement maximum
        return {"y": a}
