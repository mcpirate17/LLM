"""Python fallback kernel for relu."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReluModule(nn.Module):
    def forward(self, x):
        return F.relu(x)

class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return ReluModule()

    def forward(self, inputs, config):
        x = inputs['x']
        return {'y': F.relu(x)}
