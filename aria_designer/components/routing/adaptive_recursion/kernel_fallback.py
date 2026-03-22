"""Python fallback kernel for adaptive_recursion."""

import torch


class ComponentHandler:
    """Fallback handler for adaptive_recursion: variable-depth processing via tanh iterations."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        n_steps = config.get("max_depth", 3)
        # Simplified: apply tanh repeatedly (mimics depth without learnable weights)
        z = x
        for _ in range(n_steps):
            z = torch.tanh(z)
        # Blend original and processed
        gate = torch.sigmoid(x.mean(dim=-1, keepdim=True))
        return {"y": x * (1 - gate) + z * gate}
