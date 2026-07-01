"""Numerically-stable chunked causal power-law decay scan (speed + inference VRAM).

Several novel lanes need the causal decayed context

    c[b, t, c] = Σ_{s ≤ t} ρ_c**(t - s) · x[b, s, c]

which is the linear recurrence ``c_t = ρ ⊙ c_{t-1} + x_t``. Materializing the full
``[C, L, L]`` decay matrix is O(L²) time AND memory — fine at nano/probe scale but
the wrong asymptotics for long sequences. This helper computes the SAME result in
chunks of length ``K``: an exact intra-chunk ``[C, K, K]`` matmul plus a carried
state across chunks, so it is O(L·K) time and O(K²) memory. Because the largest
decay power inside a chunk is ``K``, it is also numerically stable (no
``ρ**(-s)`` cumsum trick that overflows for small ρ).

Shared by ``reversible_primitives._CausalDecayMLP`` and
``novel_math_primitives.OctonionicMixerLane`` (per-group decay via
``repeat_interleave``), replacing their duplicated dense implementations.
"""

from __future__ import annotations

import torch
from torch import Tensor


def causal_decay_context(x: Tensor, decay: Tensor, *, chunk: int = 64) -> Tensor:
    """Causal power-law decayed context, chunked and stable.

    Args:
        x: ``[B, L, C]`` input.
        decay: ``[C]`` per-channel decay ``ρ ∈ (0, 1)``.
        chunk: chunk length ``K`` bounding the intra-chunk matrix and the max
            decay power (so it stays numerically stable).

    Returns:
        ``[B, L, C]`` with ``c[b, t, c] = Σ_{s ≤ t} decay[c]**(t-s) x[b, s, c]``.
    """
    if x.dim() != 3:
        raise ValueError(f"expected [B, L, C]; got {tuple(x.shape)}")
    if decay.shape != (x.shape[-1],):
        raise ValueError(f"decay must be [{x.shape[-1]}]; got {tuple(decay.shape)}")
    if chunk <= 0:
        raise ValueError("chunk must be positive")

    length = x.shape[1]
    log_decay = torch.log(decay)  # [C]
    out = torch.empty_like(x)
    # state = c at the last position of the previous chunk (zeros before the first)
    state = x.new_zeros(x.shape[0], x.shape[-1])
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        xc = x[:, start:end, :]  # [B, n, C]
        n = end - start
        idx = torch.arange(n, device=x.device, dtype=x.dtype)
        exps = (idx[:, None] - idx[None, :]).clamp(min=0)  # [n, n]
        causal = (idx[:, None] >= idx[None, :]).to(x.dtype)  # lower-tri
        powmat = torch.exp(exps[None] * log_decay[:, None, None]) * causal[None]
        intra = torch.einsum("cts,bsc->btc", powmat, xc)  # within-chunk
        # carry from earlier chunks: c_{start+j} += decay**(j+1) * state
        carry_factor = torch.exp((idx + 1.0)[:, None] * log_decay[None, :])  # [n, C]
        c = intra + carry_factor[None] * state[:, None, :]
        out[:, start:end, :] = c
        state = c[:, -1, :]
    return out
