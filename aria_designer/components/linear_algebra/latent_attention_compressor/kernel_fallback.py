"""Kernel handler for latent_attention_compressor — MLA-style KV cache compression."""

import torch
import torch.nn as nn


class ComponentHandler:
    def __init__(self):
        self._compress = None
        self._decompress = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        D = x.shape[-1]
        latent_dim = max(1, D // 4)

        if self._compress is None or self._compress.in_features != D:
            self._compress = nn.Linear(D, latent_dim, bias=False)
            self._decompress = nn.Linear(latent_dim, D * 2, bias=False)
            nn.init.normal_(self._compress.weight, std=0.02)
            nn.init.normal_(self._decompress.weight, std=0.02)
            self._compress.to(device=x.device, dtype=x.dtype)
            self._decompress.to(device=x.device, dtype=x.dtype)

        latent = self._compress(x)  # (B, S, latent_dim)
        kv = self._decompress(latent)  # (B, S, D*2)
        k, v = kv.chunk(2, dim=-1)  # each (B, S, D)
        gate = torch.sigmoid(k)
        y = x + gate * v
        return {"y": y}
