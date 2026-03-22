"""Python fallback kernel for selective_scan."""

import torch


class ComponentHandler:
    """Fallback handler for selective_scan: simplified Mamba-style linear scan."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        B, S, D = x.shape
        # Simplified scan: exponential moving average along sequence
        alpha = 0.9
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        out = torch.empty_like(x)
        for t in range(S):
            h = alpha * h + (1 - alpha) * x[:, t, :]
            out[:, t, :] = h
        return {"y": out}
