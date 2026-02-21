"""Auto-generated Python fallback kernel for max_last."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for max_last."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement max_last
        return {"y": x}
