"""Python fallback kernel for exp."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpModule(nn.Module):
    def forward(self, x):
        return torch.exp(torch.clamp(x, -20, 20))

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return ExpModule()

    def forward(self, inputs, config):
        x = inputs['x']
        return {'y': torch.exp(torch.clamp(x, -20, 20))}
