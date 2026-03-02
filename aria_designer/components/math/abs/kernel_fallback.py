"""Python fallback kernel for abs."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AbsModule(nn.Module):
    def forward(self, x):
        return torch.abs(x)

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return AbsModule()

    def forward(self, inputs, config):
        x = inputs['x']
        return {'y': torch.abs(x)}
