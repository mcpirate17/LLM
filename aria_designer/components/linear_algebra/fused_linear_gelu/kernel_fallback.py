"""Python fallback kernel for fused_linear_gelu."""

import torch.nn.functional as F

from aria_designer.components._weight_cache import cached_randn


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
        w = cached_randn(
            d_out,
            d_in,
            seed=d_in * 65537 + d_out,
            device=x.device,
            dtype=x.dtype,
            scale=d_in**-0.5,
        )
        return {"y": F.gelu(F.linear(x, w))}
