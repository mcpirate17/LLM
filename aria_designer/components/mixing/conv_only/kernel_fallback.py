"""Auto-generated Python fallback kernel for conv_only."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for conv_only."""

    def validate_config(self, config):
        return []

    def build(self, config):
        # TODO: implement parameterized module
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement conv_only
        return {"y": x}
