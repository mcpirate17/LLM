from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from research.training.data_pipeline import CorpusConfig, CorpusTokenBatcher
from research.training.loss_ops import clip_grad_norm_, next_token_cross_entropy


pytestmark = pytest.mark.unit


def test_loss_ops_routes_to_canonical_native_wrappers(monkeypatch):
    seen: dict[str, object] = {}

    def _fake_loss(logits, targets, vocab_size):
        seen["loss_vocab"] = vocab_size
        return torch.tensor(1.5)

    def _fake_clip(params, max_norm):
        params = list(params)
        seen["clip_count"] = len(params)
        seen["clip_max_norm"] = max_norm
        return torch.tensor(0.75)

    monkeypatch.setattr("research.training.loss_ops._language_model_loss", _fake_loss)
    monkeypatch.setattr("research.training.loss_ops._clip_grad_norm", _fake_clip)

    logits = torch.randn(2, 4, 11)
    targets = torch.randint(0, 11, (2, 4))
    loss = next_token_cross_entropy(logits, targets, 11)

    model = torch.nn.Linear(4, 3)
    model.weight.grad = torch.ones_like(model.weight)
    total_norm = clip_grad_norm_(model, 1.25)

    assert float(loss.item()) == 1.5
    assert float(total_norm.item()) == 0.75
    assert seen == {"loss_vocab": 11, "clip_count": 2, "clip_max_norm": 1.25}


def test_npy_in_range_tokens_stay_zero_copy(tmp_path: Path):
    path = tmp_path / "tokens.npy"
    arr = np.arange(128, dtype=np.int64)
    np.save(path, arr)

    batcher = CorpusTokenBatcher(CorpusConfig(path=str(path)), vocab_size=256)

    tokens = batcher._tokens
    np_view = tokens.numpy()
    assert np_view.base is not None
    assert np.shares_memory(np_view, arr) is False  # disk-backed memmap, not original array
    assert int(tokens[0].item()) == 0
    assert int(tokens[-1].item()) == 127


def test_byte_text_loader_uses_native_file_prefix_tokenizer(monkeypatch, tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("abcdef" * 10, encoding="utf-8")

    called: dict[str, object] = {}

    class _Native:
        def byte_tokenize_file_prefix_utf8(self, file_path, vocab_size, max_chars):
            called["file_path"] = file_path
            called["vocab_size"] = vocab_size
            called["max_chars"] = max_chars
            return torch.arange(max_chars, dtype=torch.long)

        def gather_token_batch(self, tokens, starts, seq_len):
            raise AssertionError("not expected in this test")

    monkeypatch.setattr("research.training.data_pipeline.load_data_native", lambda: _Native())

    batcher = CorpusTokenBatcher(
        CorpusConfig(path=str(path), fmt="txt", tokenizer="byte", max_chars=12),
        vocab_size=256,
    )

    assert batcher.ready
    assert int(batcher._tokens.numel()) == 12
    assert called == {
        "file_path": str(path),
        "vocab_size": 256,
        "max_chars": 12,
    }


def test_jsonl_byte_loader_uses_native_jsonl_path(monkeypatch):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as handle:
        handle.write('{"text":"alpha"}\n')
        handle.write('{"text":"beta"}\n')
        path = Path(handle.name)

    called: dict[str, object] = {}

    class _Native:
        def jsonl_byte_tokenize_file(self, file_path, text_key, vocab_size, max_chars):
            called["file_path"] = file_path
            called["text_key"] = text_key
            called["vocab_size"] = vocab_size
            called["max_chars"] = max_chars
            return torch.arange(9, dtype=torch.long)

        def gather_token_batch(self, tokens, starts, seq_len):
            raise AssertionError("not expected in this test")

    try:
        monkeypatch.setattr("research.training.data_pipeline.load_data_native", lambda: _Native())
        batcher = CorpusTokenBatcher(
            CorpusConfig(
                path=str(path),
                fmt="jsonl",
                text_key="text",
                tokenizer="byte",
                max_chars=64,
            ),
            vocab_size=256,
        )

        assert batcher.ready
        assert called == {
            "file_path": str(path),
            "text_key": "text",
            "vocab_size": 256,
            "max_chars": 64,
        }
    finally:
        path.unlink(missing_ok=True)

