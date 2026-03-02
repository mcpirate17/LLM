"""Auto-generated Python fallback kernel for sequential."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for sequential."""

    def validate_config(self, config):
        return []

    def build(self, config):
        # TODO: implement parameterized module
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement sequential
        return {"y": x}
