"""Python fallback kernel for compression_mixture_experts."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComponentHandler:
    """2-expert MoE: low-rank expert + bottleneck expert, routed by input signal."""

    def __init__(self):
        self._U_lr = None
        self._V_lr = None
        self._W_down = None
        self._W_up = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        routing_signal = inputs.get("routing_signal")
        D = x.shape[-1]
        rank = max(D // 8, 1)
        rank_bn = max(D // 4, 1)

        # Lazy init with proper parameters
        if self._U_lr is None or self._U_lr.shape != (rank, D):
            self._U_lr = nn.Parameter(torch.randn(rank, D, device=x.device, dtype=x.dtype) * 0.02)
            self._V_lr = nn.Parameter(torch.randn(D, rank, device=x.device, dtype=x.dtype) * 0.02)
            self._W_down = nn.Parameter(torch.randn(rank_bn, D, device=x.device, dtype=x.dtype) * 0.02)
            self._W_up = nn.Parameter(torch.randn(D, rank_bn, device=x.device, dtype=x.dtype) * 0.02)

        # Expert 0: Low-rank factored linear
        out0 = F.linear(F.linear(x, self._U_lr), self._V_lr)

        # Expert 1: Bottleneck with GELU
        out1 = F.linear(F.gelu(F.linear(x, self._W_down)), self._W_up)

        # Route by softmax over routing signal (or equal mix if no signal)
        if routing_signal is not None and routing_signal.shape[-1] >= 2:
            weights = F.softmax(routing_signal[..., :2], dim=-1)
        else:
            weights = torch.full((*x.shape[:-1], 2), 0.5, device=x.device, dtype=x.dtype)

        return {"y": out0 * weights[..., 0:1] + out1 * weights[..., 1:2]}
