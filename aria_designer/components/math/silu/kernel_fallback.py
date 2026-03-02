"""Python fallback kernel for silu."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiluModule(nn.Module):
    def forward(self, x):
        return F.silu(x)

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return SiluModule()

    def forward(self, inputs, config):
        x = inputs['x']
        return {'y': F.silu(x)}
