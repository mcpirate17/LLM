"""Shared TinyLM and assertions for probe smoke tests.

Used by the per-probe test suites (ar_intermediate, binding_multislot,
language_control, etc.) to avoid copy-pasting the same minimal model and
state-preservation check across files.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyLM(nn.Module):
    """Embedding -> Linear LM. Sufficient for probe smoke tests on CPU."""

    def __init__(self, vocab_size: int = 128, dim: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(input_ids))


def snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def assert_state_preserved(model: nn.Module, before: dict[str, torch.Tensor]) -> None:
    after = model.state_dict()
    assert before.keys() == after.keys()
    for key, expected in before.items():
        assert torch.allclose(after[key], expected), key
