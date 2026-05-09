from __future__ import annotations

import torch

from research.scientist.runner import ExperimentRunner, RunConfig


class _TinyLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, dim: int = 32) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.proj = torch.nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(input_ids))


class _StopEvent:
    def is_set(self) -> bool:
        return False


def test_micro_train_skips_screening_probes_after_failed_gate(monkeypatch):
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner._stop_event = _StopEvent()
    runner._corpus_batcher = None
    runner._corpus_signature = None
    runner._corpus_warned_unavailable = False
    runner._hydra_loader = None

    cfg = RunConfig(
        device="cpu",
        data_mode="random",
        vocab_size=64,
        stage1_steps=2,
        stage1_batch_size=2,
        max_seq_len=16,
        collect_training_curve=False,
    )

    def _fail_gate(**_kwargs):
        return False, "forced_failure"

    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError(
            "post-S1 screening probe should not run for failed candidates"
        )

    monkeypatch.setattr(
        "research.scientist.runner.execution_training.stage1_learning_gate",
        _fail_gate,
    )
    monkeypatch.setattr(
        "research.eval.wikitext_eval.screening_wikitext_eval",
        _unexpected_call,
    )
    monkeypatch.setattr(
        "research.eval.hellaswag_eval.screening_hellaswag_eval",
        _unexpected_call,
    )
    monkeypatch.setattr(
        "research.eval.binding_range.binding_range_profile",
        _unexpected_call,
    )
    monkeypatch.setattr(
        "research.eval.native_induction.induction_score_gold",
        _unexpected_call,
    )

    result = runner._micro_train(_TinyLM(), cfg, torch.device("cpu"), seed=123)

    assert result["passed"] is False
    assert result["gate_reason"] == "forced_failure"
    assert "wikitext_perplexity" not in result
    assert "hellaswag_acc" not in result
    assert "binding_screening_auc" not in result
