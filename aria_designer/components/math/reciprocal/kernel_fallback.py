"""Python fallback kernel for reciprocal."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReciprocalModule(nn.Module):
    def forward(self, x):
        return 1.0 / torch.clamp(x, min=1e-8) if x.mean() > 0 else 1.0 / torch.clamp(x, max=-1e-8)

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return ReciprocalModule()

    def forward(self, inputs, config):
        x = inputs['x']
        return {'y': 1.0 / torch.clamp(x, min=1e-8) if x.mean() > 0 else 1.0 / torch.clamp(x, max=-1e-8)}
