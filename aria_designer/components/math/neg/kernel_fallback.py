"""Python fallback kernel for neg."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class NegModule(nn.Module):
    def forward(self, x):
        return -x

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return NegModule()

    def forward(self, inputs, config):
        x = inputs['x']
        return {'y': -x}
