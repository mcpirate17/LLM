"""Python fallback kernel for rwkv_channel (RWKV channel mixing)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKVChannelModule(nn.Module):
    def __init__(self, dim=64, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 3
        self.mix_k = nn.Parameter(torch.ones(dim) * 0.5)
        self.mix_r = nn.Parameter(torch.ones(dim) * 0.5)
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # Causal time-shift
        if x.ndim == 3:
            shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        else:
            shifted = x
        xk = x * self.mix_k + shifted * (1 - self.mix_k)
        xr = x * self.mix_r + shifted * (1 - self.mix_r)
        # Receptance-weighted gated linear update
        k = torch.square(torch.relu(self.key(xk)))
        return torch.sigmoid(self.receptance(xr)) * self.value(k)


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        dim = config.get("dim", 64)
        return RWKVChannelModule(dim)

    def forward(self, inputs, config):
        x = inputs["x"]
        in_dim = x.shape[-1] if hasattr(x, "shape") else 64
        dim = config.get("dim", in_dim)
        if dim != in_dim:
            dim = in_dim
        mod = RWKVChannelModule(dim)
        return {"y": mod(x)}
