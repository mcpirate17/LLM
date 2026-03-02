"""Python fallback kernel for div_safe."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Div_safeModule(nn.Module):
    def forward(self, a, b):
        return a / (b + 1e-6 * torch.where(b >= 0, 1.0, -1.0))

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return Div_safeModule()

    def forward(self, inputs, config):
        a = inputs['a']
        b = inputs['b']
        return {'y': a / (b + 1e-6 * torch.where(b >= 0, 1.0, -1.0))}
