"""Python fallback kernel for mul (element-wise multiplication)."""
import torch
import torch.nn as nn


class MulModule(nn.Module):
    def forward(self, a, b):
        return a * b


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return MulModule()

    def forward(self, inputs, config):
        a = inputs["a"]
        b = inputs["b"]
        return {"y": a * b}
