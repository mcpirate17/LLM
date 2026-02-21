"""Auto-generated Python fallback kernel for relu."""
import torch
import torch.nn as nn


class ComponentHandler:
    """Fallback handler for relu."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.ReLU()

    def forward(self, inputs, config):
        x = inputs["x"]
        return {"y": torch.relu(x)}
