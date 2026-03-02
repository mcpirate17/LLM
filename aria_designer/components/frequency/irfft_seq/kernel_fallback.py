"""Auto-generated Python fallback kernel for irfft_seq."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for irfft_seq."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement irfft_seq
        return {"y": x}
