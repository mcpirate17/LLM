"""Kernel handler for state_space — Mamba-style selective scan mixer."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComponentHandler:
    def __init__(self):
        self._A_log = None
        self._B_proj = None
        self._C_proj = None
        self._D = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        B, S, D = x.shape

        if self._A_log is None or self._A_log.shape[0] != D:
            self._A_log = nn.Parameter(torch.randn(D, device=x.device, dtype=x.dtype) * 0.1)
            self._B_proj = nn.Parameter(torch.randn(D, device=x.device, dtype=x.dtype) * 0.02)
            self._C_proj = nn.Parameter(torch.randn(D, device=x.device, dtype=x.dtype) * 0.02)
            self._D = nn.Parameter(torch.ones(D, device=x.device, dtype=x.dtype))

        # Selective scan: exponential decay kernel
        A = -torch.exp(self._A_log)  # (D,) negative for stability
        B_gate = torch.sigmoid(x * self._B_proj)  # (B, S, D) input gate
        dt = F.softplus(x.mean(dim=-1, keepdim=True))  # (B, S, 1) step size

        # Causal convolution via cumulative sum of log-space decay
        decay = (A.unsqueeze(0).unsqueeze(0) * dt).exp()  # (B, S, D)
        gated_input = B_gate * x  # (B, S, D)

        # Sequential scan approximation via cumsum in log-space
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(S):
            h = decay[:, t] * h + gated_input[:, t]
            outputs.append(h)
        scan_out = torch.stack(outputs, dim=1)  # (B, S, D)

        y = scan_out * self._C_proj + x * self._D  # output gate + skip
        return {"y": y}
