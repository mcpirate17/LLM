"""Python fallback kernel for maximum."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaximumModule(nn.Module):
    def forward(self, a, b):
        return torch.maximum(a, b)

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return MaximumModule()

    def forward(self, inputs, config):
        a = inputs['a']
        b = inputs['b']
        return {'y': torch.maximum(a, b)}
