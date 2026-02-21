"""Auto-generated Python fallback kernel for mul."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for mul."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        a = inputs["a"]
        b = inputs["b"]
        # TODO: implement mul
        return {"y": a}
