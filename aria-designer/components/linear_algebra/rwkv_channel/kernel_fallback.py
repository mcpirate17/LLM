"""Python fallback kernel for rwkv_channel (RWKV channel mixing)."""
import torch
import torch.nn as nn


class RWKVChannelModule(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        k = self.key(x)
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))
        return r * v * torch.relu(k) ** 2


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        dim = config.get("dim", 64)
        return RWKVChannelModule(dim)

    def forward(self, inputs, config):
        x = inputs["x"]
        dim = config.get("dim", x.shape[-1] if hasattr(x, "shape") else 64)
        mod = RWKVChannelModule(dim)
        return {"y": mod(x)}
