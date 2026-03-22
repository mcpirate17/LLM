"""Kernel handler for ternary_projection — 1.58-bit ternary simulated projection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComponentHandler:
    def __init__(self):
        self._weight = None
        self._threshold = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        D = x.shape[-1]

        if self._weight is None or self._weight.shape[0] != D:
            self._weight = nn.Parameter(
                torch.randn(D, D, device=x.device, dtype=x.dtype) * 0.02
            )
            self._threshold = nn.Parameter(
                torch.tensor(0.5, device=x.device, dtype=x.dtype)
            )

        # Ternary quantization: sign(w) * (|w| > threshold) with STE
        abs_w = self._weight.abs()
        mask = (abs_w > self._threshold).to(x.dtype)
        ternary_w = self._weight.sign() * mask
        # Straight-through estimator: use ternary forward, full-precision backward
        w_ste = self._weight + (ternary_w - self._weight).detach()
        y = F.linear(x, w_ste)
        return {"y": y}
