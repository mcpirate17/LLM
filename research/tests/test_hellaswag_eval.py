from __future__ import annotations

import numpy as np
import pytest

from research.eval import hellaswag_eval


pytestmark = pytest.mark.unit


def test_get_tokenized_examples_reuses_cache(monkeypatch):
    hellaswag_eval._tokenized_examples_cache.clear()

    calls = {"count": 0}

    def fake_download():
        return [
            {
                "ctx": "ctx",
                "endings": ["a", "b", "c", "d"],
                "label": 1,
            }
        ]

    def fake_tokenize(text: str, vocab_size: int):
        del vocab_size
        calls["count"] += 1
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(
            np.int64, copy=False
        )

    monkeypatch.setattr(hellaswag_eval, "_download_hellaswag", fake_download)
    monkeypatch.setattr(hellaswag_eval, "tokenize_string", fake_tokenize)

    first = hellaswag_eval._get_tokenized_examples(256)
    second = hellaswag_eval._get_tokenized_examples(256)

    assert first is second
    assert calls["count"] == 5


def test_score_example_batch_accepts_tokenized_examples(monkeypatch):
    examples = [
        {
            "ctx_tokens": np.array([1, 2, 3], dtype=np.int64),
            "ending_tokens": (
                np.array([4], dtype=np.int64),
                np.array([5], dtype=np.int64),
                np.array([6], dtype=np.int64),
                np.array([7], dtype=np.int64),
            ),
            "label": 2,
        }
    ]

    def fake_grouped_choice_scores(*args, **kwargs):
        del args, kwargs
        return [[0.1, 0.2, 0.9, 0.0]]

    monkeypatch.setattr(
        hellaswag_eval,
        "grouped_choice_scores",
        fake_grouped_choice_scores,
    )

    correct, total = hellaswag_eval._score_example_batch(
        object(),
        examples,
        vocab_size=256,
        device="cpu",
        max_seq_len=16,
    )

    assert (correct, total) == (1, 1)
