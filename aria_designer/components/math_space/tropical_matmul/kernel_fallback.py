"""Kernel handler for tropical_matmul — dispatches to aria_core.tropical_matmul_f32."""
import torch
import torch.nn as nn
from components.base import SimpleBinaryOpHandler

class TropicalMatmulModule(nn.Module):
    def forward(self, a, b):
        return a @ b  # Approximate; real tropical matmul in aria_core

class ComponentHandler(SimpleBinaryOpHandler):
    def __init__(self):
        super().__init__(TropicalMatmulModule, lambda a, b: a @ b, native_op_name="tropical_matmul")
