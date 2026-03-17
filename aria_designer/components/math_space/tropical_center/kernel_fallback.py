"""Kernel handler for tropical_center — dispatches to aria_core.tropical_center_f32."""

import torch.nn as nn
from components.base import SimpleUnaryOpHandler


class TropicalCenterModule(nn.Module):
    def forward(self, x):
        baseline = x.min(dim=-2, keepdim=True).values
        return x - baseline


class ComponentHandler(SimpleUnaryOpHandler):
    def __init__(self):
        super().__init__(
            TropicalCenterModule,
            lambda x: x - x.min(dim=-2, keepdim=True).values,
            native_op_name="tropical_center",
        )
