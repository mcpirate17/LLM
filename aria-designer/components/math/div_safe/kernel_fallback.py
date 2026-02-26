"""Python fallback kernel for div_safe (element-wise safe division)."""
import torch
import torch.nn as nn


class DivSafeModule(nn.Module):
    def forward(self, a, b):
        return a / (b + 1e-8)


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return DivSafeModule()

    def forward(self, inputs, config):
        a = inputs["a"]
        b = inputs["b"]
        return {"y": a / (b + 1e-8)}
