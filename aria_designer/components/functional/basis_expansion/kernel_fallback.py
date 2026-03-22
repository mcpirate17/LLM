"""Python fallback kernel for basis_expansion."""

import torch


class ComponentHandler:
    """Fallback handler for basis_expansion: Fourier basis features."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        # Fourier basis: average of sin/cos at different frequencies
        return {
            "y": 0.25
            * (torch.sin(x) + torch.cos(x) + torch.sin(2 * x) + torch.cos(2 * x))
        }
