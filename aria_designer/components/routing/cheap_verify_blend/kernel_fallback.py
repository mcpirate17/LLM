"""Python fallback kernel for speculative verification blend."""

import torch
from components._weight_cache import cached_randn


class ComponentHandler:
    """Fallback handler for speculative: cheap draft path + learned blending."""

    __slots__ = ()

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        D = x.shape[-1]
        # Cheap draft path: small linear projection as a fast approximation
        W_draft = cached_randn(
            D, D, seed=D * 7919, device=x.device, dtype=x.dtype, scale=D**-0.5
        )
        draft = torch.nn.functional.linear(x, W_draft)
        # Learned gate blends draft with original
        gate = torch.sigmoid(x.mean(dim=-1, keepdim=True))
        return {"y": x * (1 - gate) + draft * gate}
