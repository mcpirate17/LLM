from __future__ import annotations

import torch
import pytest

from research.eval.retrieval_eval_utils import (
    eval_restricted_last_token_accuracy,
    run_retrieval_probe_config,
)


class _StubModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch = input_ids.shape[0]
        return self._logits[:batch]


class _TrainableProbe(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(input_ids))


def test_eval_restricted_last_token_accuracy_respects_vocab_slice() -> None:
    eval_ids = torch.zeros((3, 4), dtype=torch.long)
    eval_targets = torch.tensor([100, 101, 100], dtype=torch.long)
    logits = torch.zeros((3, 4, 200), dtype=torch.float32)

    logits[0, -1, 100] = 5.0
    logits[1, -1, 101] = 4.0
    logits[2, -1, 102] = 6.0

    model = _StubModel(logits)
    acc = eval_restricted_last_token_accuracy(
        model,
        eval_ids,
        eval_targets,
        batch_size=8,
        vocab_lo=100,
        vocab_hi=103,
    )

    assert acc == pytest.approx(2.0 / 3.0)


def test_eval_restricted_last_token_accuracy_honors_query_pos_override() -> None:
    eval_ids = torch.zeros((2, 5), dtype=torch.long)
    eval_targets = torch.tensor([10, 11], dtype=torch.long)
    logits = torch.zeros((2, 5, 32), dtype=torch.float32)

    logits[0, 2, 10] = 3.0
    logits[1, 2, 11] = 3.0
    logits[0, -1, 11] = 9.0
    logits[1, -1, 10] = 9.0

    model = _StubModel(logits)
    acc = eval_restricted_last_token_accuracy(
        model,
        eval_ids,
        eval_targets,
        batch_size=8,
        vocab_lo=10,
        vocab_hi=12,
        query_pos=2,
    )

    assert acc == pytest.approx(1.0)


def test_run_retrieval_probe_config_returns_timeout_without_training() -> None:
    model = _TrainableProbe()
    eval_ids = torch.zeros((2, 4), dtype=torch.long)
    eval_targets = torch.tensor([10, 11], dtype=torch.long)

    def _batch(_batch_size: int, _device: str) -> tuple[torch.Tensor, torch.Tensor]:
        return eval_ids.clone(), eval_targets.clone()

    acc, timed_out = run_retrieval_probe_config(
        model,
        n_train_steps=5,
        eval_ids=eval_ids,
        eval_targets=eval_targets,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        deadline=0.0,
        make_train_batch=_batch,
        query_pos=3,
        vocab_lo=10,
        vocab_hi=12,
    )

    assert timed_out is True
    assert 0.0 <= acc <= 1.0


def test_passkey_and_multi_hop_use_shared_probe_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.eval import multi_hop_retrieval as mh
    from research.eval import passkey_retrieval as pk

    calls: list[dict[str, int]] = []

    def _fake_runner(model, **kwargs):
        del model
        calls.append(
            {
                "kind": kwargs.get("kind", -1),
                "query_pos": int(kwargs["query_pos"]),
                "vocab_lo": int(kwargs["vocab_lo"]),
                "vocab_hi": int(kwargs["vocab_hi"]),
                "n_train_steps": int(kwargs["n_train_steps"]),
            }
        )
        return 0.25, False

    monkeypatch.setattr(pk, "run_retrieval_probe_config", _fake_runner)
    monkeypatch.setattr(mh, "run_retrieval_probe_config", _fake_runner)

    model = _TrainableProbe(vocab_size=512)
    passkey_acc, passkey_timeout = pk._train_passkey_at_length(
        model,
        seq_len=16,
        n_train_steps=7,
        n_eval=4,
        lr=1e-3,
        batch_size=2,
        device="cpu",
        deadline=1e9,
    )
    multi_hop_acc, multi_hop_timeout = mh._train_multi_hop_at_config(
        model,
        seq_len=20,
        n_hops=2,
        n_train_steps=9,
        n_eval=4,
        lr=1e-3,
        batch_size=2,
        device="cpu",
        deadline=1e9,
    )

    assert (passkey_acc, passkey_timeout) == (0.25, False)
    assert (multi_hop_acc, multi_hop_timeout) == (0.25, False)
    assert calls == [
        {
            "kind": -1,
            "query_pos": 15,
            "vocab_lo": 100,
            "vocab_hi": 356,
            "n_train_steps": 7,
        },
        {
            "kind": -1,
            "query_pos": 19,
            "vocab_lo": 100,
            "vocab_hi": 356,
            "n_train_steps": 9,
        },
    ]


def test_associative_recall_uses_shared_learning_curve_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.eval import associative_recall as ar

    def _fake_runner(model, **kwargs):
        del model
        assert kwargs["query_pos"] == 18
        assert kwargs["vocab_lo"] == 100
        assert kwargs["vocab_hi"] == 356
        assert kwargs["n_train_steps"] == 11
        assert kwargs["eval_every"] == 5
        return [(0, 0.0), (5, 0.2), (10, 0.4)], 10, False, "ok"

    monkeypatch.setattr(ar, "run_retrieval_probe_learning_curve", _fake_runner)

    model = _TrainableProbe(vocab_size=1024)
    result = ar.associative_recall_score(
        model,
        n_pairs=5,
        n_train_steps=11,
        n_eval=4,
        eval_every=5,
        lr=1e-3,
        batch_size=2,
        device="cpu",
        timeout_s=60.0,
    )

    assert result.learning_curve == [(0, 0.0), (5, 0.2), (10, 0.4)]
    assert result.steps_trained == 10
    assert result.status == "ok"
    assert result.final_acc == 0.4
