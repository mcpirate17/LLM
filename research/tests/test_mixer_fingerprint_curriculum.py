import torch

from research.tools.mixer_fingerprint import _scheduled_seq_len
from research.tools.scaling_blimp_study import _RandomWindowBatcher


def test_scheduled_seq_len_fixed_ignores_initial_and_warmup():
    assert (
        _scheduled_seq_len(
            schedule="fixed",
            step=1,
            max_seq_len=256,
            initial_seq_len=16,
            warmup_steps=2000,
        )
        == 256
    )


def test_scheduled_seq_len_growing_reaches_max_at_warmup():
    assert (
        _scheduled_seq_len(
            schedule="growing",
            step=1,
            max_seq_len=256,
            initial_seq_len=16,
            warmup_steps=2000,
        )
        == 16
    )
    assert (
        _scheduled_seq_len(
            schedule="growing",
            step=1000,
            max_seq_len=256,
            initial_seq_len=16,
            warmup_steps=2000,
        )
        == 135
    )
    assert (
        _scheduled_seq_len(
            schedule="growing",
            step=2000,
            max_seq_len=256,
            initial_seq_len=16,
            warmup_steps=2000,
        )
        == 256
    )


def test_random_window_batcher_supports_shorter_curriculum_windows():
    tokens = torch.arange(100, dtype=torch.long)
    batcher = _RandomWindowBatcher(
        tokens,
        batch_size=3,
        seq_len=16,
        device="cpu",
        seed=7,
    )

    short = batcher.next(4)
    full = batcher.next()

    assert short.shape == (3, 4)
    assert full.shape == (3, 16)
    assert torch.all(short[:, 1:] - short[:, :-1] == 1)
    assert torch.all(full[:, 1:] - full[:, :-1] == 1)
