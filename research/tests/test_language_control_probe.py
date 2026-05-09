from __future__ import annotations

import torch

from research.eval.language_control_probe import language_control_probe
from research.tests._probe_test_support import (
    TinyLM,
    assert_state_preserved,
    snapshot_state,
)


def test_language_control_probe_records_checkpoints_and_restores_state() -> None:
    model = TinyLM()
    model.eval()
    before = snapshot_state(model)

    result = language_control_probe(
        model,
        active_vocab_size=32,
        n_train_steps=2,
        checkpoint_steps=(1, 2),
        eval_repeats=1,
        batch_size=4,
        device="cpu",
        seed=123,
    )

    payload = result.to_dict()
    assert payload["language_control_metric_version"] == "language_control_v2"
    assert payload["language_control_train_steps"] == 2
    assert [cp["steps"] for cp in payload["language_control_checkpoints"]] == [1, 2]
    assert all(
        "nano_blimp_score" in cp for cp in payload["language_control_checkpoints"]
    )
    assert not model.training
    assert_state_preserved(model, before)


def test_language_control_probe_can_skip_state_restore_for_disposable_model() -> None:
    model = TinyLM()
    before = snapshot_state(model)

    result = language_control_probe(
        model,
        active_vocab_size=32,
        n_train_steps=2,
        checkpoint_steps=(2,),
        eval_repeats=1,
        batch_size=4,
        device="cpu",
        seed=123,
        preserve_state=False,
    )

    assert result.status == "ok"
    after = model.state_dict()
    assert any(not torch.allclose(after[key], before[key]) for key in before)
