"""Kernel handler for difficulty_scorer — 2-layer MLP scoring per-token difficulty."""
import torch
import torch.nn as nn


class ComponentHandler:
    def __init__(self):
        self._w1 = None
        self._b1 = None
        self._w2 = None
        self._b2 = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        mode = config.get("mode", "active")
        if mode == "passthrough":
            B, S, D = x.shape
            return {"y": x, "scores": torch.zeros(B, S, 1, device=x.device, dtype=x.dtype)}

        D = x.shape[-1]
        hidden = max(1, D // 8)

        # Lazy weight init
        if self._w1 is None or self._w1.shape != (hidden, D):
            self._w1 = torch.randn(hidden, D, device=x.device, dtype=x.dtype) * 0.02
            self._b1 = torch.zeros(hidden, device=x.device, dtype=x.dtype)
            self._w2 = torch.randn(1, hidden, device=x.device, dtype=x.dtype) * 0.02
            self._b2 = torch.zeros(1, device=x.device, dtype=x.dtype)

        h = torch.relu(x @ self._w1.T + self._b1)
        scores = torch.sigmoid(h @ self._w2.T + self._b2)
        return {"y": x, "scores": scores}
