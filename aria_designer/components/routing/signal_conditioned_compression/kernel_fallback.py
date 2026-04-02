"""Python fallback kernel for signal_conditioned_compression."""

import torch
import torch.nn as nn


class ComponentHandler:
    """Blend dense and low-rank projections from an external routing signal."""

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
        routing_signal = inputs.get("routing_signal")
        if routing_signal is None:
            routing_signal = x.mean(dim=-1, keepdim=True)
        self._ensure_weights(x)
        low_rank = x @ self._u @ self._v
        dense = x @ self._dense
        gate = torch.sigmoid(routing_signal.to(dtype=x.dtype))
        return {"y": low_rank * (1.0 - gate) + dense * gate}
