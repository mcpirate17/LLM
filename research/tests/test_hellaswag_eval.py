from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from research.eval import hellaswag_eval


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


def test_get_tokenized_examples_reuses_cache(monkeypatch):
    hellaswag_eval._tokenized_examples_cache.clear()
    hellaswag_eval._tokenized_subset_cache.clear()

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


def test_get_tokenized_subset_reuses_cache(monkeypatch):
    hellaswag_eval._tokenized_examples_cache.clear()
    hellaswag_eval._tokenized_subset_cache.clear()

    examples = [
        {
            "ctx_tokens": np.array([i], dtype=np.int64),
            "ending_tokens": tuple(np.array([j], dtype=np.int64) for j in range(4)),
            "label": i % 4,
        }
        for i in range(8)
    ]
    calls = {"count": 0}

    def fake_get_tokenized_examples(vocab_size: int):
        del vocab_size
        calls["count"] += 1
        return examples

    monkeypatch.setattr(
        hellaswag_eval,
        "_get_tokenized_examples",
        fake_get_tokenized_examples,
    )

    first = hellaswag_eval._get_tokenized_subset(4, vocab_size=256, seed=7)
    second = hellaswag_eval._get_tokenized_subset(4, vocab_size=256, seed=7)

    assert first is second
    assert calls["count"] == 2


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

    captured = {}

    class _FakeEvalNative:
        def hellaswag_score_batch_native(
            self,
            model,
            ctx_tokens,
            ending_tokens,
            labels,
            vocab_size,
            device,
            max_seq_len,
        ):
            del model, vocab_size, device, max_seq_len
            captured["ctx_tokens"] = ctx_tokens
            captured["ending_tokens"] = ending_tokens
            captured["labels"] = labels
            return 1, 1

    monkeypatch.setattr(
        "research.eval._eval_native.load_eval_native",
        lambda: _FakeEvalNative(),
    )

    correct, total = hellaswag_eval._score_example_batch(
        object(),
        examples,
        vocab_size=256,
        device="cpu",
        max_seq_len=16,
    )

    assert (correct, total) == (1, 1)
    assert captured["ctx_tokens"] == [[1, 2, 3]]
    assert captured["ending_tokens"] == [[[4], [5], [6], [7]]]
    assert captured["labels"] == [2]


def test_score_example_batch_native_matches_python_for_empty_endings():
    model = _NextTokenLM(vocab_size=32, preferred={1: 4})
    examples = [
        {
            "ctx_tokens": np.array([1], dtype=np.int64),
            "ending_tokens": (
                np.array([], dtype=np.int64),
                np.array([4], dtype=np.int64),
                np.array([5], dtype=np.int64),
                np.array([6], dtype=np.int64),
            ),
            "label": 1,
        }
    ]

    python_result = hellaswag_eval._score_example_batch_python(
        model,
        examples,
        vocab_size=32,
        device="cpu",
        max_seq_len=16,
    )
    ctx_tokens = [ex["ctx_tokens"].tolist() for ex in examples]
    ending_tokens = [[t.tolist() for t in ex["ending_tokens"]] for ex in examples]
    labels = [int(ex["label"]) for ex in examples]
    native_result = hellaswag_eval._score_example_batch_native(
        model,
        ctx_tokens,
        ending_tokens,
        labels,
        vocab_size=32,
        device="cpu",
        max_seq_len=16,
    )

    assert python_result == (1, 1)
    assert native_result == python_result


def test_score_example_batch_falls_back_to_python(monkeypatch):
    model = _NextTokenLM(vocab_size=32, preferred={1: 4})
    examples = [
        {
            "ctx_tokens": np.array([1], dtype=np.int64),
            "ending_tokens": (
                np.array([4], dtype=np.int64),
                np.array([5], dtype=np.int64),
                np.array([6], dtype=np.int64),
                np.array([7], dtype=np.int64),
            ),
            "label": 0,
        }
    ]
    expected = hellaswag_eval._score_example_batch_python(
        model,
        examples,
        vocab_size=32,
        device="cpu",
        max_seq_len=16,
    )

    def _raise_native(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("native scorer blew up")

    monkeypatch.setattr(hellaswag_eval, "_score_example_batch_native", _raise_native)

    assert (
        hellaswag_eval._score_example_batch(
            model,
            examples,
            vocab_size=32,
            device="cpu",
            max_seq_len=16,
        )
        == expected
    )


def test_recommended_batch_examples_caps_cpu_batches():
    assert (
        hellaswag_eval._recommended_batch_examples(
            requested=16,
            vocab_size=256,
            max_seq_len=512,
            device="cpu",
            model_dim=48,
        )
        == 2
    )
    assert (
        hellaswag_eval._recommended_batch_examples(
            requested=4,
            vocab_size=256,
            max_seq_len=512,
            device="cpu",
            model_dim=48,
        )
        == 2
    )
    assert (
        hellaswag_eval._recommended_batch_examples(
            requested=16,
            vocab_size=256,
            max_seq_len=512,
            device="cpu",
            model_dim=96,
        )
        == 2
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_screening_hellaswag_bypasses_native_dispatch(monkeypatch, device):
    calls = {"enter": 0, "exit": 0, "device": None}

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(256, 8)
            self.head = nn.Linear(8, 256)

        def forward(self, x):
            return self.head(self.embed(x))

    class _ProbeCtx:
        def __enter__(self):
            calls["enter"] += 1

        def __exit__(self, exc_type, exc, tb):
            calls["exit"] += 1
            return False

    monkeypatch.setattr(
        hellaswag_eval,
        "_get_tokenized_subset",
        lambda n_examples, *, vocab_size, seed=42: [
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
        ],
    )
    monkeypatch.setattr(
        hellaswag_eval,
        "_get_native_subset_payload",
        lambda n_examples, *, vocab_size, seed=42: (
            [[1, 2, 3]],
            [[[4], [5], [6], [7]]],
            [2],
        ),
    )
    monkeypatch.setattr(
        hellaswag_eval,
        "_score_example_batch_native",
        lambda *args, **kwargs: (1, 1),
    )

    def _fake_disable(model, *, device):
        del model
        calls["device"] = device
        return _ProbeCtx()

    monkeypatch.setattr(
        hellaswag_eval,
        "disable_native_probe_dispatch",
        _fake_disable,
    )

    result = hellaswag_eval.screening_hellaswag_eval(
        TinyLM(),
        vocab_size=256,
        device=device,
        n_examples=1,
    )

    assert result["hellaswag_status"] == "ok"
    assert calls["device"] == device
    assert calls["enter"] == 1
    assert calls["exit"] == 1
