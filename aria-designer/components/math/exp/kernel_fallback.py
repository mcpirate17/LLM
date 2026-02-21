"""Auto-generated Python fallback kernel for exp."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for exp."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement exp
        return {"y": x}
