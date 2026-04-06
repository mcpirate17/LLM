"""Python fallback kernel for lane_conditioned_block."""

import torch
import torch.nn.functional as F


class ComponentHandler:
    def __init__(self):
        self._weight = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        d = x.shape[-1]
        if self._weight is None or self._weight.shape != (d, d):
            self._weight = torch.randn(d, d, device=x.device, dtype=x.dtype) * (d**-0.5)
        return {"y": F.gelu(x @ self._weight)}
