"""Kernel handler for moe_2expert — lightweight 2-expert MoE with learned gating."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComponentHandler:
    def __init__(self):
        self._gate = None
        self._expert0 = None
        self._expert1 = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        D = x.shape[-1]

        if self._gate is None or self._gate.in_features != D:
            self._gate = nn.Linear(D, 2, bias=False)
            self._expert0 = nn.Linear(D, D, bias=False)
            self._expert1 = nn.Linear(D, D, bias=False)
            for m in (self._gate, self._expert0, self._expert1):
                nn.init.normal_(m.weight, std=0.02)
                m.to(device=x.device, dtype=x.dtype)

        weights = F.softmax(self._gate(x), dim=-1)  # (B, S, 2)
        e0 = self._expert0(x)  # (B, S, D)
        e1 = self._expert1(x)  # (B, S, D)
        y = weights[..., 0:1] * e0 + weights[..., 1:2] * e1
        return {"y": y}
