"""Python fallback kernel for rmsnorm."""

import torch
import torch.nn as nn


class _RMSNorm(nn.Module):
    __slots__ = ("eps",)

    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = 1e-6

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class ComponentHandler:
    """Fallback handler for rmsnorm."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        d = x.shape[-1]
        weight = torch.ones(d, device=x.device, dtype=x.dtype)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return {"y": (x / rms) * weight}
