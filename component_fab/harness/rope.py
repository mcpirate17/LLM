"""Rotary Positional Embedding (RoPE).

Replaces ``TinyLM.pos_embed`` (absolute learned embedding capped at the
trained ``max_seq_len``). RoPE rotates ``Q`` and ``K`` per position inside
each attention lane; the rotation matrix is content-independent and the
cos/sin cache extrapolates trivially.

Usage:
    rope = RotaryEmbedding(dim=320, max_seq_len=1024)
    cos, sin = rope(seq_len=l, device=x.device, dtype=x.dtype)
    q_rot = apply_rope(q, cos, sin)
    k_rot = apply_rope(k, cos, sin)

The half-split variant from LLaMA: ``x`` of shape ``(..., D)`` is split
into ``(x1, x2)`` along the last dim, and rotated as
``(x1*cos - x2*sin, x1*sin + x2*cos)``. Requires ``D`` even.
"""

from __future__ import annotations

import torch
from torch import nn


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` by the per-position cos/sin tables.

    ``x``:   ``(..., L, D)`` — last two dims are sequence and feature.
    ``cos``, ``sin``: ``(L, D/2)`` — typically broadcast across the batch
    and head dims by simply not adding them.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RotaryEmbedding(nn.Module):
    """Cached cos/sin tables for RoPE.

    Stores ``(max_seq_len, dim//2)`` cos and sin buffers, sliced per call.
    Buffers are non-persistent — checkpoints stay small and the table is
    rebuilt to match the live ``dim`` / ``max_seq_len`` on construction.
    """

    def __init__(
        self, dim: int, max_seq_len: int = 1024, base: float = 10_000.0
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even dim, got {dim}")
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (L, dim/2)
        self.register_buffer("_cos", freqs.cos(), persistent=False)
        self.register_buffer("_sin", freqs.sin(), persistent=False)

    def forward(
        self, seq_len: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"RoPE seq_len={seq_len} exceeds cached max_seq_len={self.max_seq_len}; "
                f"reconstruct RotaryEmbedding with a larger max_seq_len."
            )
        cos = self._cos[:seq_len].to(device=device, dtype=dtype)
        sin = self._sin[:seq_len].to(device=device, dtype=dtype)
        return cos, sin
