from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from research.eval import blimp_eval


pytestmark = pytest.mark.unit


class _NextTokenLM(nn.Module):
    def __init__(self, vocab_size: int, preferred: dict[int, int]):
        super().__init__()
        self.vocab_size = vocab_size
        self.preferred = preferred

    def forward(self, x):
        batch, seq_len = x.shape
        logits = torch.full(
            (batch, seq_len, self.vocab_size),
            -9.0,
            dtype=torch.float32,
            device=x.device,
        )
        for src_token, dst_token in self.preferred.items():
            logits[x == src_token, dst_token] = 9.0
        return logits


def test_get_tokenized_subtask_examples_reuses_cache(monkeypatch):
    blimp_eval._tokenized_subtask_cache.clear()
    calls = {"count": 0}

    monkeypatch.setattr(
        blimp_eval,
        "_download_blimp",
        lambda: {
            "task": [
                {"good": "a good sentence", "bad": "a bad sentence"},
                {"good": "another good sentence", "bad": "another bad sentence"},
            ]
        },
    )

    def fake_tokenize(text: str, vocab_size: int):
        del vocab_size
        calls["count"] += 1
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)

    monkeypatch.setattr(blimp_eval, "tokenize_string", fake_tokenize)

    first = blimp_eval._get_tokenized_subtask_examples(
        2,
        vocab_size=256,
        max_seq_len=128,
    )
    second = blimp_eval._get_tokenized_subtask_examples(
        2,
        vocab_size=256,
        max_seq_len=128,
    )

    assert first is second
    assert calls["count"] == 4


def test_score_token_pairs_batched_scores_tokenized_pairs():
    model = _NextTokenLM(vocab_size=32, preferred={1: 2, 2: 3, 4: 5})
    pairs = [
        {"good": [1, 2, 3], "bad": [1, 4, 6]},
        {"good": [4, 5], "bad": [4, 6]},
        {"good": [9], "bad": [9]},
    ]

    result = blimp_eval._score_token_pairs_batched(
        model,
        pairs,
        vocab_size=32,
        device="cpu",
    )

    assert result == 2


def test_evaluate_blimp_batches_across_subtasks(monkeypatch):
    model = _NextTokenLM(vocab_size=32, preferred={1: 2, 2: 3, 4: 5})
    subtasks = {
        "alpha": [
            {"good": [1, 2, 3], "bad": [1, 4, 6]},
            {"good": [4, 5], "bad": [4, 6]},
        ],
        "beta": [
            {"good": [1, 4, 6], "bad": [1, 2, 3]},
        ],
    }

    monkeypatch.setattr(
        blimp_eval,
        "_get_tokenized_subtask_examples",
        lambda *args, **kwargs: subtasks,
    )

    result = blimp_eval.evaluate_blimp(
        model,
        vocab_size=32,
        device="cpu",
        n_per_subtask=2,
        max_seq_len=16,
    )

    assert result.status == "ok"
    assert result.n_subtasks == 2
    assert result.n_examples == 3
    assert result.subtask_accuracies == {"alpha": 1.0, "beta": 0.0}
    assert result.overall_accuracy == 0.6667
