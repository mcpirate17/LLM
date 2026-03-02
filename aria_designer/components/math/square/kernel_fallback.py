"""Python fallback kernel for square."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SquareModule(nn.Module):
    def forward(self, x):
        return x * x

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return SquareModule()

    def forward(self, inputs, config):
        x = inputs['x']
        return {'y': x * x}
