"""Auto-generated Python fallback kernel for minimum."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for minimum."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        a = inputs["a"]
        b = inputs["b"]
        # TODO: implement minimum
        return {"y": a}
