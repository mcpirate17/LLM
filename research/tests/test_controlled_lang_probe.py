from __future__ import annotations

import torch
import torch.nn as nn

from research.eval.controlled_lang_probe import controlled_lang_probe


class _TinyLM(nn.Module):
    vocab_size = 128

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(self.vocab_size, 16)
        self.proj = nn.Linear(16, self.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.emb(input_ids))


def test_controlled_lang_probe_records_checkpoints_and_restores_state() -> None:
    model = _TinyLM()
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    result = controlled_lang_probe(
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
    assert payload["controlled_lang_metric_version"] == "controlled_lang_v2"
    assert payload["controlled_lang_train_steps"] == 2
    assert [cp["steps"] for cp in payload["controlled_lang_checkpoints"]] == [1, 2]
    assert all(
        "nano_blimp_score" in cp for cp in payload["controlled_lang_checkpoints"]
    )
    assert not model.training
    after = model.state_dict()
    assert before.keys() == after.keys()
    for key, expected in before.items():
        assert torch.allclose(after[key], expected), key


def test_controlled_lang_probe_can_skip_state_restore_for_disposable_model() -> None:
    model = _TinyLM()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    result = controlled_lang_probe(
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
