from __future__ import annotations

import torch

from research.tools import backfill


def test_sample_micro_train_batch_prefers_corpus(monkeypatch):
    expected = torch.arange(12, dtype=torch.long).view(3, 4)

    class _Batcher:
        def sample_batch(self, **kwargs):
            assert kwargs["batch_size"] == 3
            assert kwargs["seq_len"] == 4
            return expected

    monkeypatch.setattr(
        backfill, "_get_backfill_batcher", lambda vocab_size: _Batcher()
    )
    generator = torch.Generator(device="cpu")
    batch = backfill._sample_micro_train_batch(
        256,
        batch_size=3,
        seq_len=4,
        device="cpu",
        generator=generator,
    )
    assert torch.equal(batch, expected)


def test_sample_micro_train_batch_falls_back_to_random(monkeypatch):
    monkeypatch.setattr(backfill, "_get_backfill_batcher", lambda vocab_size: None)
    monkeypatch.setattr(backfill, "_BACKFILL_CORPUS_WARNED", True)
    generator = torch.Generator(device="cpu")
    batch = backfill._sample_micro_train_batch(
        17,
        batch_size=2,
        seq_len=5,
        device="cpu",
        generator=generator,
    )
    assert batch.shape == (2, 5)
    assert batch.dtype == torch.long
    assert int(batch.min().item()) >= 0
    assert int(batch.max().item()) < 17


def test_next_token_cross_entropy_prefers_native(monkeypatch):
    class _Native:
        def next_token_cross_entropy(self, logits, targets, vocab_size, reduction):
            assert vocab_size == 9
            assert reduction == "mean"
            return torch.tensor(3.25)

    monkeypatch.setattr(
        backfill,
        "next_token_cross_entropy",
        lambda logits, targets, vocab_size: _Native().next_token_cross_entropy(
            logits, targets, vocab_size, "mean"
        ),
    )
    logits = torch.randn(2, 4, 9)
    targets = torch.randint(0, 9, (2, 4))

    loss = backfill.next_token_cross_entropy(logits, targets, 9)

    assert float(loss.item()) == 3.25


def test_clip_grad_norm_prefers_native(monkeypatch):
    class _Native:
        def clip_grad_norm_(self, grads, max_norm, eps):
            assert len(grads) == 1
            assert max_norm == 1.0
            assert eps == 1e-6
            return torch.tensor(0.5)

    model = torch.nn.Linear(4, 4)
    model.weight.grad = torch.ones_like(model.weight)
    monkeypatch.setattr(
        backfill,
        "clip_grad_norm_",
        lambda model_or_params, max_norm=1.0: _Native().clip_grad_norm_(
            [model.weight.grad], max_norm, 1e-6
        ),
    )

    total_norm = backfill.clip_grad_norm_(model, 1.0)

    assert float(total_norm.item()) == 0.5
