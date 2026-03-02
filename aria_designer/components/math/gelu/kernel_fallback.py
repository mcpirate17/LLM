"""Python fallback kernel for gelu."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeluModule(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return GeluModule()

    def forward(self, inputs, config):
        x = inputs['x']
        return {'y': F.gelu(x)}
