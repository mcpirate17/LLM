"""Kernel handler for state_space — Mamba-style selective scan mixer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Block size for numerically stable log-space cumulative products.
# Within 256 steps, cumulative decay products stay representable in float32.
_BLOCK = 256


class ComponentHandler:
    """Selective state space with variable per-position decay.

    Recurrence: h[t] = decay[t] * h[t-1] + input[t], h[-1] = 0.

    Vectorized via blocked log-space cumulative products — within each block:
        A_cum[t] = exp(cumsum(log(decay)))[t]   (cumulative product of decay)
        h_block[t] = A_cum[t] * cumsum(input / A_cum)[t]
    Between blocks, the carry state is propagated analytically:
        h[t] = h_block[t] + carry * A_cum[t]
    Zero Python-level per-timestep iterations.
    """

    def __init__(self):
        self._A_log = None
        self._B_proj = None
        self._C_proj = None
        self._D = None
        self._dim = None

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        B, S, D = x.shape

        if self._A_log is None or self._dim != D:
            self._A_log = nn.Parameter(
                torch.randn(D, device=x.device, dtype=x.dtype) * 0.1
            )
            self._B_proj = nn.Parameter(
                torch.randn(D, device=x.device, dtype=x.dtype) * 0.02
            )
            self._C_proj = nn.Parameter(
                torch.randn(D, device=x.device, dtype=x.dtype) * 0.02
            )
            self._D = nn.Parameter(torch.ones(D, device=x.device, dtype=x.dtype))
            self._dim = D

        # Selective scan: exponential decay kernel
        A = -torch.exp(self._A_log)  # (D,) negative for stability
        B_gate = torch.sigmoid(x * self._B_proj)  # (B, S, D) input gate
        dt = F.softplus(x.mean(dim=-1, keepdim=True))  # (B, S, 1) step size

        decay = (A.unsqueeze(0).unsqueeze(0) * dt).exp()  # (B, S, D)
        gated_input = B_gate * x  # (B, S, D)

        # Blocked vectorized scan
        scan_out = torch.empty_like(x)
        carry = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        for start in range(0, S, _BLOCK):
            end = min(start + _BLOCK, S)
            decay_blk = decay[:, start:end, :]  # (B, blen, D)
            input_blk = gated_input[:, start:end, :]  # (B, blen, D)

            # Cumulative product of decay within block (log-space for stability)
            log_decay_blk = torch.log(decay_blk.clamp(min=1e-8))
            A_cum = torch.exp(torch.cumsum(log_decay_blk, dim=1))  # (B, blen, D)

            # Within-block scan (zero initial state)
            scaled = input_blk / A_cum.clamp(min=1e-8)
            block_scan = A_cum * torch.cumsum(scaled, dim=1)

            # Propagate carry: at local position t, carry decays by A_cum[t]
            # because A_cum[t] = prod_{k=0}^{t} decay[k] in this block
            block_out = block_scan + carry.unsqueeze(1) * A_cum

            scan_out[:, start:end, :] = block_out
            carry = block_out[:, -1, :]

        y = scan_out * self._C_proj + x * self._D  # output gate + skip
        return {"y": y}
