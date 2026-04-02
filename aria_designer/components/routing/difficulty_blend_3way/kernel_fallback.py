"""Python fallback kernel for difficulty_blend_3way."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComponentHandler:
    """Three-lane routing fallback: skip, low-rank, and dense paths."""

    def __init__(self):
        self._u = None
        self._v = None
        self._dense = None
        self._router = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def _ensure_weights(self, x):
        d_model = x.shape[-1]
        rank = max(1, d_model // 4)
        device = x.device
        dtype = x.dtype
        if self._u is None or self._u.shape != (d_model, rank):
            self._u = nn.Parameter(
                torch.randn(d_model, rank, device=device, dtype=dtype) * 0.02
            )
            self._v = nn.Parameter(
                torch.randn(rank, d_model, device=device, dtype=dtype) * 0.02
            )
            self._dense = nn.Parameter(
                torch.randn(d_model, d_model, device=device, dtype=dtype)
                * (d_model**-0.5)
            )
            self._router = nn.Parameter(
                torch.randn(d_model, 3, device=device, dtype=dtype) * 0.02
            )

    def forward(self, inputs, config):
        x = inputs["x"]
        self._ensure_weights(x)
        lane_logits = x @ self._router
        lane_weights = F.softmax(lane_logits, dim=-1)
        skip_lane = x
        low_rank_lane = x @ self._u @ self._v
        dense_lane = F.gelu(x @ self._dense)
        y = (
            skip_lane * lane_weights[..., 0:1]
            + low_rank_lane * lane_weights[..., 1:2]
            + dense_lane * lane_weights[..., 2:3]
        )
        return {"y": y}
