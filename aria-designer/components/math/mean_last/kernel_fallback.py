"""Auto-generated Python fallback kernel for mean_last."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for mean_last."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement mean_last
        return {"y": x}
