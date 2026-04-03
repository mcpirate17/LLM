"""Python fallback kernel for selective_scan."""

import torch

# Block size chosen so alpha^(-BLOCK) stays within float32 range.
# For alpha=0.9: (1/0.9)^256 ≈ 3.3e11, well under float32 max (~3.4e38).
_BLOCK = 256


class ComponentHandler:
    """Fallback handler for selective_scan: simplified Mamba-style linear scan.

    Uses IIR filter formulation: y[t] = alpha * y[t-1] + (1 - alpha) * x[t].

    Vectorized via blocked cumsum trick — within each block of 256 steps:
        Let b[t] = (1 - alpha) * x[t], powers[t] = alpha^t.
        Then y_block[t] = powers[t] * cumsum(b / powers)[t].
    Between blocks, the carry state is propagated analytically.
    Zero Python-level per-timestep iterations.
    """

    __slots__ = ()

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        alpha = 0.9
        B, S, D = x.shape

        b = (1.0 - alpha) * x  # (B, S, D)
        out = torch.empty_like(x)

        # Pre-compute powers for one full block: alpha^0 .. alpha^(BLOCK-1)
        block = min(_BLOCK, S)
        powers = alpha ** torch.arange(
            block, device=x.device, dtype=x.dtype
        )  # (block,)
        powers_col = powers.unsqueeze(0).unsqueeze(-1)  # (1, block, 1)
        inv_powers_col = 1.0 / powers_col

        carry = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        for start in range(0, S, _BLOCK):
            end = min(start + _BLOCK, S)
            blen = end - start
            b_blk = b[:, start:end, :]  # (B, blen, D)

            p = powers_col[:, :blen, :]
            ip = inv_powers_col[:, :blen, :]

            # Within-block vectorized scan (zero initial state)
            block_scan = p * torch.cumsum(b_blk * ip, dim=1)

            # Propagate carry: at local t, carry decays by alpha^(t+1)
            block_out = block_scan + carry.unsqueeze(1) * (alpha * p)

            out[:, start:end, :] = block_out
            carry = block_out[:, -1, :]

        return {"y": out}
