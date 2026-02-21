"""Auto-generated Python fallback kernel for softmax_last."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for softmax_last."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement softmax_last
        return {"y": x}
