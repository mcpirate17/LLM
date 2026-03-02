"""Auto-generated Python fallback kernel for sum_last."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for sum_last."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement sum_last
        return {"y": x}
