"""Python fallback kernel for minimum."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimumModule(nn.Module):
    def forward(self, a, b):
        return torch.minimum(a, b)

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return MinimumModule()

    def forward(self, inputs, config):
        a = inputs['a']
        b = inputs['b']
        return {'y': torch.minimum(a, b)}
