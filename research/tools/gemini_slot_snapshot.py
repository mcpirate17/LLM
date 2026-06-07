"""Snapshot of gemini's slot-memory lane for the matched-budget re-grade.

Vectorized version of ContentRoutedMasterLane to avoid slow Python loops.
"""

from __future__ import annotations

import torch
from torch import nn


class GeminiSlotMemoryLane(nn.Module):
    """Slotted latched memory with content-aware routing (gemini snapshot, vectorized)."""

    def __init__(
        self, dim: int, n_slots: int = 16, memory_dim: int = 16, latch_len: int = 3
    ) -> None:
        super().__init__()
        self.q = nn.Linear(dim, memory_dim, bias=False)
        self.k = nn.Linear(dim, memory_dim, bias=False)
        self.v = nn.Linear(dim, memory_dim, bias=False)
        self.write_route = nn.Linear(memory_dim, n_slots)
        self.latch_mix = nn.Linear(memory_dim * latch_len, memory_dim)
        self.out = nn.Linear(memory_dim, dim, bias=False)
        self.n_slots = n_slots
        self.memory_dim = memory_dim
        self.latch_len = latch_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, dim = x.shape
        device = x.device
        dtype = x.dtype
        
        # 1. Projections
        kt = torch.tanh(self.k(x))
        vt = self.v(x)
        qt = torch.tanh(self.q(x))
        
        # 2. Latching (Vectorized Shifted Window)
        kt_padded = torch.cat([torch.zeros(b, self.latch_len-1, self.memory_dim, device=device, dtype=dtype), kt], dim=1)
        l_keys = kt_padded.unfold(1, self.latch_len, 1) # [B, L, MemDim, LatchLen]
        l_keys = l_keys.reshape(b, seq_len, -1) # [B, L, MemDim * LatchLen]
        latched_context = self.latch_mix(l_keys) # [B, L, MemDim]
        
        # 3. Slotted Routing
        w_route = torch.softmax(self.write_route(latched_context), dim=-1)
        w_idx = w_route.argmax(dim=-1)
        mask = torch.nn.functional.one_hot(w_idx, num_classes=self.n_slots).to(dtype)
        
        # 4. Slotted Writing (Parallel cumsum)
        writes = mask.unsqueeze(-1) * vt.unsqueeze(2) # [B, L, Slots, MemDim]
        slot_vals_over_time = writes.cumsum(dim=1)
        
        # 5. Read
        read = torch.einsum("bld,blsd->bld", qt, slot_vals_over_time)
        return self.out(read)
