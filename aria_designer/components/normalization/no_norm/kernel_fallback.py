"""Auto-generated Python fallback kernel for no_norm."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for no_norm."""

    def validate_config(self, config):
        return []

    def build(self, config):
        # TODO: implement parameterized module
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement no_norm
        return {"y": x}
