"""Kernel handler for tropical_add — dispatches to aria_core.tropical_add_f32."""
import torch
import torch.nn as nn
from components.base import SimpleBinaryOpHandler

class TropicalAddModule(nn.Module):
    def forward(self, a, b):
        return torch.minimum(a, b)

class ComponentHandler(SimpleBinaryOpHandler):
    def __init__(self):
        super().__init__(TropicalAddModule, torch.minimum, native_op_name="tropical_add")
