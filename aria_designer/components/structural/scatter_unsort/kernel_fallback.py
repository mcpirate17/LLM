"""Auto-generated Python fallback kernel for scatter_unsort."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for scatter_unsort."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        a = inputs["x"]
        b = inputs["idx"]
        # TODO: implement scatter_unsort
        return {"y": a}
