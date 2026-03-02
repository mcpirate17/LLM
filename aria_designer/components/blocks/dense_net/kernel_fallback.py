"""Auto-generated Python fallback kernel for dense_net."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for dense_net."""

    def validate_config(self, config):
        return []

    def build(self, config):
        # TODO: implement parameterized module
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement dense_net
        return {"y": x}
