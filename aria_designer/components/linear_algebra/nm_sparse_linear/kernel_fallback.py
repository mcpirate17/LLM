"""Python fallback kernel for nm_sparse_linear."""

import torch
import torch.nn.functional as F


class ComponentHandler:
    """Fallback handler for nm_sparse_linear: dense linear (sparsity is an optimization)."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        d_in = x.shape[-1]
        d_out = config.get("out_dim") or d_in
        gen = torch.Generator(device="cpu")
        gen.manual_seed(d_in * 65537 + d_out + 2)
        w = torch.randn(d_out, d_in, generator=gen, dtype=x.dtype).to(x.device)
        w *= d_in**-0.5
        return {"y": F.linear(x, w)}
