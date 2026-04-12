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
