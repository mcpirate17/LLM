"""Kernel handler for tropical_router — tropical (shortest-path) routing signal."""

import torch
import torch.nn as nn


class ComponentHandler:
    def __init__(self):
        self._weight = None
        self._scale = None

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
            self._scale = nn.Parameter(torch.ones(1, device=x.device, dtype=x.dtype))

        # Tropical distance: min over j of (x_i + w_ij) for each output dim
        # x: (B, S, D) → expand for pairwise: (B, S, D, 1) + (D, D) → min over input dim
        trop_out = (x.unsqueeze(-1) + self._weight).min(dim=-2).values  # (B, S, D)
        # Collapse to routing signal (B, S, 1) then broadcast-scale input
        route_signal = torch.sigmoid(trop_out.mean(dim=-1, keepdim=True) * self._scale)
        y = x * route_signal
        return {"y": y}
