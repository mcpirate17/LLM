"""Python fallback kernel for minimum (element-wise min)."""
import torch
import torch.nn as nn


class MinimumModule(nn.Module):
    def forward(self, a, b):
        return torch.minimum(a, b)


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return MinimumModule()

    def forward(self, inputs, config):
        a = inputs["a"]
        b = inputs["b"]
        return {"y": torch.minimum(a, b)}
