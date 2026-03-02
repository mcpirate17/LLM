"""Auto-generated Python fallback kernel for sort_seq."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for sort_seq."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement sort_seq
        return {"y": x, "idx": torch.zeros(x.shape[0], x.shape[1], dtype=torch.long)}
