"""Kernel handler for difficulty_scorer — 2-layer MLP scoring per-token difficulty."""

import torch.nn as nn


class ComponentHandler:
    def __init__(self):
        self._module = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        D = x.shape[-1]
        hidden = max(1, D // 8)

        # Lazy module init with proper nn.Parameters for gradient flow
        if self._module is None or self._module.fc1.in_features != D:
            self._module = nn.Sequential(
                nn.Linear(D, hidden, bias=True),
                nn.ReLU(),
                nn.Linear(hidden, 1, bias=True),
                nn.Sigmoid(),
            )
            self._module.to(device=x.device, dtype=x.dtype)
            for p in self._module.parameters():
                if p.dim() >= 2:
                    nn.init.normal_(p, std=0.02)

        scores = self._module(x)  # (B, S, 1)
        return {"y": x, "scores": scores}
