"""Python fallback kernel for linear_proj_down."""

import torch.nn.functional as F

from aria_designer.components._weight_cache import cached_randn


class ComponentHandler:
    """Fallback handler for linear_proj_down: project to smaller dim (default /4)."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        d_in = x.shape[-1]
        d_out = config.get("out_dim") or max(1, d_in // 4)
        w = cached_randn(
            d_out,
            d_in,
            seed=d_in * 65537 + d_out,
            device=x.device,
            dtype=x.dtype,
            scale=d_in**-0.5,
        )
        return {"y": F.linear(x, w)}
