"""Kernel handler for tropical_moe — MoE with tropical (min-plus) routing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComponentHandler:
    def __init__(self):
        self._trop_weight = None
        self._expert0 = None
        self._expert1 = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        D = x.shape[-1]

        if self._trop_weight is None or self._trop_weight.shape[0] != D:
            self._trop_weight = nn.Parameter(
                torch.randn(D, 2, device=x.device, dtype=x.dtype) * 0.02
            )
            self._expert0 = nn.Linear(D, D, bias=False)
            self._expert1 = nn.Linear(D, D, bias=False)
            for m in (self._expert0, self._expert1):
                nn.init.normal_(m.weight, std=0.02)
                m.to(device=x.device, dtype=x.dtype)

        # Tropical distance: min over pairwise (x_i + w_ij) for each expert
        # x: (B, S, D), trop_weight: (D, 2)
        trop_scores = (
            (x.unsqueeze(-1) + self._trop_weight).min(dim=-2).values
        )  # (B, S, 2)
        gate_weights = F.softmax(
            -trop_scores, dim=-1
        )  # negate: shorter distance = higher weight

        e0 = self._expert0(x)
        e1 = self._expert1(x)
        y = gate_weights[..., 0:1] * e0 + gate_weights[..., 1:2] * e1
        return {"y": y}
