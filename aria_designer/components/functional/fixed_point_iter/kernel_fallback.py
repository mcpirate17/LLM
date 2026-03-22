"""Python fallback kernel for fixed_point_iter."""

import torch


class ComponentHandler:
    """Fallback handler for fixed_point_iter: damped fixed-point iteration with tanh."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        n_iters = config.get("n_iters", 3)
        damping = config.get("damping", 0.5)
        z = x
        for _ in range(n_iters):
            z = (1 - damping) * z + damping * torch.tanh(z)
        return {"y": z}
