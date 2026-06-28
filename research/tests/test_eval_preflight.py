"""Pre-flight contract check for the post-hoc eval battery (Gemini plan P2.3)."""

from __future__ import annotations

import torch
import torch.nn as nn

from research.defaults import VOCAB_SIZE
from research.tools.eval_trained_checkpoint import _preflight_check


class _GoodLM(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, dim: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.emb(ids))


def test_preflight_passes_on_valid_model() -> None:
    res = _preflight_check(_GoodLM(), "cpu")
    assert res["ok"] is True
    assert res["logits_shape"][-1] == VOCAB_SIZE
    assert res["n_params_with_grad"] > 0


def test_preflight_flags_wrong_vocab_dim() -> None:
    # Embedding accepts the ids (full vocab) but the head emits the wrong vocab
    # size -> shape mismatch caught before the battery, not a forward index error.
    class _WrongHead(_GoodLM):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(32, VOCAB_SIZE - 7)

    res = _preflight_check(_WrongHead(), "cpu")
    assert res["ok"] is False
    assert res["stage"] == "shape"


def test_preflight_flags_nan_forward() -> None:
    class _NaNLM(_GoodLM):
        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return super().forward(ids) * float("nan")

    res = _preflight_check(_NaNLM(), "cpu")
    assert res["ok"] is False
    assert res["stage"] == "finite_forward"


def test_preflight_flags_frozen_model() -> None:
    # No trainable params -> backward cannot flow gradient. Either backward raises
    # (no grad_fn) or no finite grad accumulates; both are loud preflight failures.
    model = _GoodLM()
    for p in model.parameters():
        p.requires_grad_(False)
    res = _preflight_check(model, "cpu")
    assert res["ok"] is False
    assert res["stage"] in {"backward", "no_gradient"}


def test_preflight_tuple_output_unwrapped() -> None:
    class _TupleLM(_GoodLM):
        def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
            return super().forward(ids), {"aux": 1}

    res = _preflight_check(_TupleLM(), "cpu")
    assert res["ok"] is True
