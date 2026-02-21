"""Auto-generated Python fallback kernel for reciprocal."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for reciprocal."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement reciprocal
        return {"y": x}
