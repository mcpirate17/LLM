"""Python fallback kernel for sqrt."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SqrtModule(nn.Module):
    def forward(self, x):
        return torch.sqrt(torch.clamp(x.abs(), min=1e-8))

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return SqrtModule()

    def forward(self, inputs, config):
        x = inputs['x']
        return {'y': torch.sqrt(torch.clamp(x.abs(), min=1e-8))}
