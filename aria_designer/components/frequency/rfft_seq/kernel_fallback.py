"""Auto-generated Python fallback kernel for rfft_seq."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for rfft_seq."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement rfft_seq
        return {"y": x}
