"""Kernel handler for mod_topk — Mixture-of-Depths top-k token routing."""

import torch
import torch.nn as nn


class ComponentHandler:
    def __init__(self):
        self._router_weight = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        B, S, D = x.shape
        capacity_factor = float(config.get("capacity_factor", 0.5))
        k = max(1, int(S * capacity_factor))

        if self._router_weight is None or self._router_weight.shape[-1] != D:
            self._router_weight = nn.Parameter(
                torch.randn(1, D, device=x.device, dtype=x.dtype) * 0.02
            )

        scores = (x * self._router_weight).sum(dim=-1)  # (B, S)
        _, topk_idx = scores.topk(k, dim=-1)  # (B, k)

        # Build binary mask with STE for gradient flow
        mask = torch.zeros(B, S, device=x.device, dtype=x.dtype)
        mask.scatter_(1, topk_idx, 1.0)
        mask_ste = scores.sigmoid() + (mask - scores.sigmoid()).detach()

        y = x * mask_ste.unsqueeze(-1)
        return {"y": y}
