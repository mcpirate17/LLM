"""Python fallback kernel for adaptive_rank_gate."""

import torch
import torch.nn as nn


class ComponentHandler:
    """Adaptive low-rank gating fallback with cached weights."""

    def __init__(self):
        self._u = None
        self._v = None
        self._dense = None

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

    def forward(self, inputs, config):
        x = inputs["x"]
        self._ensure_weights(x)
        low_rank = x @ self._u @ self._v
        dense = x @ self._dense
        gate = torch.sigmoid(x.pow(2).mean(dim=-1, keepdim=True))
        return {"y": low_rank * (1.0 - gate) + dense * gate}
