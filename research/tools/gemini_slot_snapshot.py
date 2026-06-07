"""Snapshot of gemini's slot-memory lane for the matched-budget re-grade.

Verbatim copy of `ContentRoutedMasterLane` from gemini's volatile scratch
`research/tools/test_fix_hypotheses.py` (the lane gemini reported solving
distractor at 0.61, 2026-06-07). Snapshotted here so the fairness re-grade is
reproducible and not coupled to gemini's actively-renamed file. Attribution:
gemini (architecture-repair lane). The mechanism: hard-routed slot memory with a
softmax READ over slots (= query-time selection over separable keys — why it
handles interference where additive linear memories can't).
"""

from __future__ import annotations

import torch
from torch import nn


class GeminiSlotMemoryLane(nn.Module):
    """Slotted latched memory with content-aware routing (gemini snapshot)."""

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
        b, seq_len, _ = x.shape
        slot_keys = torch.zeros(
            b, self.n_slots, self.memory_dim, device=x.device, dtype=x.dtype
        )
        slot_vals = torch.zeros(
            b, self.n_slots, self.memory_dim, device=x.device, dtype=x.dtype
        )
        key_latch = [
            torch.zeros(b, self.memory_dim, device=x.device, dtype=x.dtype)
            for _ in range(self.latch_len)
        ]
        outputs = []
        for t in range(seq_len):
            token = x[:, t]
            kt = torch.tanh(self.k(token))
            vt = self.v(token)
            qt = torch.tanh(self.q(token))
            latched_context = self.latch_mix(torch.cat(key_latch, dim=-1))
            w_route = torch.softmax(self.write_route(latched_context), dim=-1)
            w_idx = w_route.argmax(dim=-1)
            mask = (
                torch.nn.functional.one_hot(w_idx, num_classes=self.n_slots)
                .unsqueeze(-1)
                .to(x.dtype)
            )
            slot_keys = slot_keys * (1.0 - mask) + mask * latched_context.unsqueeze(1)
            slot_vals = slot_vals * (1.0 - mask) + mask * vt.unsqueeze(1)
            read_weights = torch.softmax(
                torch.einsum("bd,bsd->bs", qt, slot_keys), dim=-1
            )
            read = torch.einsum("bs,bsd->bd", read_weights, slot_vals)
            outputs.append(self.out(read))
            key_latch = key_latch[1:] + [kt]
        return torch.stack(outputs, dim=1)
