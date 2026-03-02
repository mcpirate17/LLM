"""Python fallback kernel for log."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogModule(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x.abs(), min=1e-8))

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return LogModule()

    def forward(self, inputs, config):
        x = inputs['x']
        return {'y': torch.log(torch.clamp(x.abs(), min=1e-8))}
