"""Python fallback kernel for output_head."""

import torch.nn.functional as F

from aria_designer.components._weight_cache import cached_randn


class ComponentHandler:
    """Fallback handler for output_head: linear projection to vocab size."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"] if "x" in inputs else next(iter(inputs.values()))  # (B, S, D)
        vocab_size = config.get("vocab_size", 32000)
        d_in = x.shape[-1]
        w = cached_randn(
            vocab_size,
            d_in,
            seed=d_in * 65537 + vocab_size,
            device=x.device,
            dtype=x.dtype,
            scale=d_in**-0.5,
        )
        return {"logits": F.linear(x, w)}
