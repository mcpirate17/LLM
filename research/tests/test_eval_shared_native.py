from __future__ import annotations

import numpy as np
import pytest
import torch

from research.eval.choice_scoring import grouped_choice_scores
from research.eval.utils import batched_span_mean_log_probs
from pathlib import Path
from research.tests._tokenize_helpers import bytes_to_int64_tokens

from research.eval.corpus_pipeline import (
    _batch_cache,
    _token_cache,
    _trim_text_chunks,
    prepare_text_corpus_split_batches,
    prepare_text_split_batches,
    split_token_array,
)


pytestmark = pytest.mark.unit


def test_trim_text_chunks_truncates_without_overrun():
    text = _trim_text_chunks(["abc", "", "defgh", "zzz"], 6)
    assert text == "abcdef"


def test_split_token_array_respects_fraction():
    tokens = np.arange(10, dtype=np.int64)
    train, val = split_token_array(tokens, train_fraction=0.6)
    assert train.tolist() == [0, 1, 2, 3, 4, 5]
    assert val.tolist() == [6, 7, 8, 9]


def test_grouped_choice_scores_regroups_flat_scores(monkeypatch):
    captured = {}

    def fake_batched_span_mean_log_probs(
        model, sequences, starts, *, vocab_size, device
    ):
        del model, vocab_size, device
        captured["sequences"] = [list(seq) for seq in sequences]
        captured["starts"] = list(starts)
        return torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    monkeypatch.setattr(
        "research.eval.choice_scoring.batched_span_mean_log_probs",
        fake_batched_span_mean_log_probs,
    )

    scores = grouped_choice_scores(
        object(),
        grouped_sequences=(
            ([1, 2], [3]),
            ([4, 5, 6],),
        ),
        grouped_start_positions=(
            (0, 1),
            (2,),
        ),
        vocab_size=256,
        device="cpu",
    )

    assert captured["sequences"] == [[1, 2], [3], [4, 5, 6]]
    assert captured["starts"] == [0, 1, 2]
    assert scores == [[1.0, 2.0], [3.0]]


def test_batched_span_mean_log_probs_matches_manual_reference():
    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(32, 16)
            self.head = torch.nn.Linear(16, 32)

        def forward(self, x):
            return self.head(self.embed(x))

    model = TinyLM()
    sequences = [
        np.array([1, 2, 3, 4, 5], dtype=np.int64),
        np.array([2, 2, 2], dtype=np.int64),
        np.array([9], dtype=np.int64),
    ]
    start_positions = [1, 0, 0]

    observed = batched_span_mean_log_probs(
        model,
        sequences,
        start_positions,
        vocab_size=32,
        device="cpu",
    )

    padded = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 2, 0, 0],
        ],
        dtype=torch.long,
    )
    logits = model(padded)
    log_probs = torch.nn.functional.log_softmax(logits[:, :-1], dim=-1)
    targets = padded[:, 1:]
    token_lps = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)

    expected0 = token_lps[0, 1:4].mean()
    expected1 = token_lps[1, 0:2].mean()

    assert torch.allclose(
        observed[:2],
        torch.stack((expected0, expected1)).to(torch.float32),
        atol=1e-6,
    )
    assert torch.isneginf(observed[2])


def test_prepare_text_split_batches_reuses_token_cache(tmp_path, monkeypatch):
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    text = "hello world test data " * 100
    train_path.write_text(text, encoding="utf-8")
    val_path.write_text(text, encoding="utf-8")

    calls = {"count": 0}

    def fake_tokenize(path: Path, vocab_size: int):
        del vocab_size
        calls["count"] += 1
        return bytes_to_int64_tokens(path)

    monkeypatch.setattr("research.eval.corpus_pipeline.tokenize_file", fake_tokenize)
    _batch_cache.clear()
    _token_cache.clear()

    kwargs = dict(
        namespace="cache-test",
        train_path=train_path,
        val_path=val_path,
        vocab_size=256,
        seq_len=8,
        train_batch_size=2,
        eval_batch_size=2,
        n_train_batches=4,
        n_eval_batches=2,
        device="cpu",
    )

    prepare_text_split_batches(**kwargs)
    _batch_cache.clear()
    prepare_text_split_batches(**kwargs)

    assert calls["count"] == 2


def test_prepare_text_corpus_split_batches_reuses_token_cache(tmp_path, monkeypatch):
    path = tmp_path / "corpus.txt"
    text = "hello world test data " * 100
    path.write_text(text, encoding="utf-8")

    calls = {"count": 0}

    def fake_tokenize(corpus_path: Path, vocab_size: int):
        del vocab_size
        calls["count"] += 1
        return bytes_to_int64_tokens(corpus_path)

    monkeypatch.setattr("research.eval.corpus_pipeline.tokenize_file", fake_tokenize)
    _batch_cache.clear()
    _token_cache.clear()

    kwargs = dict(
        path=path,
        namespace="fractional-cache-test",
        vocab_size=256,
        seq_len=8,
        train_batch_size=2,
        eval_batch_size=2,
        n_train_batches=4,
        n_eval_batches=2,
        device="cpu",
    )

    prepare_text_corpus_split_batches(**kwargs)
    _batch_cache.clear()
    prepare_text_corpus_split_batches(**kwargs)

    assert calls["count"] == 1
