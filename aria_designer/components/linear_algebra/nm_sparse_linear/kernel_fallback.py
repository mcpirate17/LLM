"""Auto-generated Python fallback kernel for nm_sparse_linear."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for nm_sparse_linear."""

    def validate_config(self, config):
        return []

    def build(self, config):
        # TODO: implement parameterized module
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement nm_sparse_linear
        return {"y": x}
