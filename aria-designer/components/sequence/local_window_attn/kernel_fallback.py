"""Auto-generated Python fallback kernel for local_window_attn."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for local_window_attn."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        # TODO: implement local_window_attn
        return {"y": x}
