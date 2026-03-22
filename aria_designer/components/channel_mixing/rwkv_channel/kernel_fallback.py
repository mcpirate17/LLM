"""Python fallback kernel for rwkv_channel."""

import torch


class ComponentHandler:
    """Fallback handler for rwkv_channel: simplified RWKV channel mixing with time-shift."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        # Causal time-shift: shift by 1 along sequence dim
        shifted = torch.nn.functional.pad(x[:, :-1, :], (0, 0, 1, 0))
        # Simple channel mixing: lerp between x and shifted
        mix = 0.5
        mixed = x * mix + shifted * (1 - mix)
        # Square-ReLU channel gate
        gated = torch.relu(mixed).square()
        return {"y": torch.sigmoid(x) * gated}
