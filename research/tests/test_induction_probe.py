from __future__ import annotations

import torch

from research.eval.induction_probe import _generate_induction_batch


def test_generate_induction_batch_excludes_repeated_token_from_noise():
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)

    batch, targets = _generate_induction_batch(
        batch_size=64,
        gap=16,
        device="cpu",
        generator=gen,
    )

    repeated = batch[:, 0]
    noise = batch[:, 2:-1]

    assert batch.shape == (64, 19)
    assert targets.shape == (64,)
    assert torch.equal(batch[:, -1], repeated)
    assert torch.equal(targets, batch[:, 1])
    assert not torch.any(noise == repeated.unsqueeze(1))
    assert int(batch.min()) >= 1
    assert int(batch.max()) < 256


def test_generate_induction_batch_handles_gap_one():
    batch, targets = _generate_induction_batch(
        batch_size=8,
        gap=1,
        device="cpu",
    )

    assert batch.shape == (8, 4)
    assert torch.equal(batch[:, -1], batch[:, 0])
    assert torch.equal(targets, batch[:, 1])
    assert not torch.any(batch[:, 2:3] == batch[:, 0:1])
