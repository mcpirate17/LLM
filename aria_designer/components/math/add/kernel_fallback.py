"""Python fallback kernel for add."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AddModule(nn.Module):
    def forward(self, a, b):
        return a + b

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return AddModule()

    def forward(self, inputs, config):
        a = inputs['a']
        b = inputs['b']
        return {'y': a + b}
