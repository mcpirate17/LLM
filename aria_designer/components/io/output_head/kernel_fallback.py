"""Python fallback kernel for output_head."""

import torch
import torch.nn.functional as F


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
        gen = torch.Generator(device="cpu")
        gen.manual_seed(d_in * 65537 + vocab_size)
        w = torch.randn(vocab_size, d_in, generator=gen, dtype=x.dtype).to(x.device)
        w *= d_in**-0.5
        return {"logits": F.linear(x, w)}
