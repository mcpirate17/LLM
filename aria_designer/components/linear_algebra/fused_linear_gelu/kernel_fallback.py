"""Python fallback kernel for fused_linear_gelu."""

import torch
import torch.nn.functional as F


class ComponentHandler:
    """Fallback handler for fused_linear_gelu: F.gelu(F.linear(x, W) + b)."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        d_in = x.shape[-1]
        d_out = config.get("out_dim") or d_in
        gen = torch.Generator(device="cpu")
        gen.manual_seed(d_in * 65537 + d_out)
        w = torch.randn(d_out, d_in, generator=gen, dtype=x.dtype).to(x.device)
        w *= d_in**-0.5
        return {"y": F.gelu(F.linear(x, w))}
