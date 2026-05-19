from types import SimpleNamespace
import threading

import pytest
import torch

from research.scientist.runner.selection import _SelectionMixin


pytestmark = pytest.mark.unit


class _TinyNextTokenModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 11, dim: int = 8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, dim)
        self.proj = torch.nn.Linear(dim, vocab_size)

    def forward(self, input_ids):
        return self.proj(self.embedding(input_ids))


class _SelectionProbeHarness(_SelectionMixin):
    def __init__(self):
        self._stop_event = threading.Event()

    def _sample_training_input_ids(
        self,
        *,
        config,
        dev,
        batch_size,
        seq_len,
        seed,
    ):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randint(
            0,
            config.vocab_size,
            (batch_size, seq_len),
            generator=generator,
            device=dev,
        )


def test_candidate_probe_training_returns_loss_fields():
    runner = _SelectionProbeHarness()
    config = SimpleNamespace(vocab_size=11, max_seq_len=6, stage1_batch_size=2)
    model = _TinyNextTokenModel(vocab_size=config.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    result = runner._run_candidate_probe_training(
        model=model,
        optimizer=optimizer,
        config=config,
        dev=torch.device("cpu"),
        n_steps=3,
        seed=7,
    )

    assert result["initial_loss"] is not None
    assert result["final_loss"] is not None
    assert result["loss_ratio"] is not None


def test_candidate_probe_training_honors_stop_event():
    runner = _SelectionProbeHarness()
    runner._stop_event.set()
    config = SimpleNamespace(vocab_size=11, max_seq_len=6, stage1_batch_size=2)
    model = _TinyNextTokenModel(vocab_size=config.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    result = runner._run_candidate_probe_training(
        model=model,
        optimizer=optimizer,
        config=config,
        dev=torch.device("cpu"),
        n_steps=3,
        seed=7,
    )

    assert result == {"initial_loss": None, "final_loss": None, "loss_ratio": None}
