"""Auto-generated Python fallback kernel for gather_sorted."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for gather_sorted."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        a = inputs["x"]
        b = inputs["idx"]
        # TODO: implement gather_sorted
        return {"y": a}
