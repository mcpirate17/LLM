"""Tests for component_fab.harness.lm_eval — wikitext train + BLiMP wrap.

Wikitext download + BLiMP eval are mocked out; we verify pipeline shape
and TinyLM-with-FFN structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from component_fab.harness import lm_eval as lm_eval_mod
from component_fab.harness.lm_eval import (
    LMEvalResult,
    WikitextTrainTrace,
    _build_lm,
    _causal_lm_loss,
    _eval_ppl,
    evaluate_lm,
    train_on_wikitext,
)
from component_fab.harness.tiny_lm import (
    SoftmaxCausalAttention,
    TinyLMConfig,
)


# ---------- _build_lm + FFN ----------


def test_build_lm_includes_ffn_in_blocks() -> None:
    """A pre-norm Transformer block must have an FFN — that's the proven
    template. Without it, the lane handles all the heavy lifting alone.
    """
    model = _build_lm(
        SoftmaxCausalAttention,
        dim=16,
        n_blocks=2,
        vocab_size=64,
        max_seq_len=32,
    )
    for block in model.blocks:
        assert block.mlp is not None
        assert block.norm2 is not None


def test_tiny_lm_use_ffn_false_skips_mlp() -> None:
    """Discrete-binding pathway sets use_ffn=False — verify mlp is None then."""
    from component_fab.harness.tiny_lm import TinyLM

    cfg = TinyLMConfig(vocab_size=32, dim=8, n_blocks=1, use_ffn=False, max_seq_len=8)
    model = TinyLM(SoftmaxCausalAttention, cfg)
    assert model.blocks[0].mlp is None


def test_causal_lm_loss_finite() -> None:
    logits = torch.randn(2, 8, 16)
    ids = torch.randint(0, 16, (2, 8))
    loss = _causal_lm_loss(logits, ids)
    assert torch.isfinite(loss)


def test_eval_ppl_returns_finite_for_small_model() -> None:
    model = _build_lm(
        SoftmaxCausalAttention, dim=16, n_blocks=1, vocab_size=32, max_seq_len=16
    )
    batches = [torch.randint(0, 32, (2, 16))]
    ppl = _eval_ppl(model, batches)
    assert torch.isfinite(torch.tensor(ppl))


def test_eval_ppl_handles_empty_batches() -> None:
    model = _build_lm(
        SoftmaxCausalAttention, dim=16, n_blocks=1, vocab_size=32, max_seq_len=16
    )
    import math

    assert math.isnan(_eval_ppl(model, []))


# ---------- train_on_wikitext with mocked data path ----------


def _mock_prepare_batches(
    variant: str,
    vocab_size: int,
    seq_len: int,
    train_batch_size: int,
    eval_batch_size: int,
    n_train_batches: int,
    n_eval_batches: int,
    max_chars_train: int,
    max_chars_val: int,
    device: str,
) -> tuple:
    """Replace ``_prepare_batches`` with tiny in-memory batches so the
    train+eval loop runs without touching the wikitext cache."""
    gen = torch.Generator().manual_seed(0)
    train = [
        torch.randint(0, vocab_size, (train_batch_size, seq_len), generator=gen)
        for _ in range(n_train_batches)
    ]
    val = [
        torch.randint(0, vocab_size, (eval_batch_size, seq_len), generator=gen)
        for _ in range(n_eval_batches)
    ]
    n_tok_tr = sum(b.numel() for b in train)
    n_tok_val = sum(b.numel() for b in val)
    return train, val, n_tok_tr, n_tok_val


def test_train_on_wikitext_runs_with_mocked_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lm_eval_mod, "_prepare_batches", _mock_prepare_batches)
    model = _build_lm(
        SoftmaxCausalAttention, dim=16, n_blocks=1, vocab_size=64, max_seq_len=16
    )
    trace = train_on_wikitext(
        model,
        n_steps=3,
        seq_len=16,
        batch_size=2,
        n_train_batches=2,
        n_eval_batches=1,
    )
    assert trace.converged
    assert trace.n_steps == 3
    assert torch.isfinite(torch.tensor(trace.pre_train_ppl))
    assert torch.isfinite(torch.tensor(trace.post_train_ppl))


def test_train_on_wikitext_handles_no_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    def empty_prep(*args: Any, **kwargs: Any) -> tuple:
        return [], [], 0, 0

    monkeypatch.setattr(lm_eval_mod, "_prepare_batches", empty_prep)
    model = _build_lm(
        SoftmaxCausalAttention, dim=16, n_blocks=1, vocab_size=64, max_seq_len=16
    )
    trace = train_on_wikitext(model, n_steps=3, n_train_batches=0, n_eval_batches=0)
    assert not trace.converged
    assert trace.n_steps == 0


# ---------- evaluate_lm with full pipeline mocked ----------


@dataclass
class _MockBLiMPResult:
    overall_accuracy: float = 0.5
    subtask_accuracies: dict[str, float] = field(default_factory=dict)
    status: str = "ok"


def test_evaluate_lm_shape_with_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lm_eval_mod, "_prepare_batches", _mock_prepare_batches)
    monkeypatch.setattr(
        lm_eval_mod,
        "evaluate_blimp",
        lambda model, vocab_size, device, n_per_subtask, max_seq_len: _MockBLiMPResult(
            overall_accuracy=0.67,
            subtask_accuracies={"foo": 0.7, "bar": 0.65},
            status="ok",
        ),
    )
    result = evaluate_lm(
        SoftmaxCausalAttention,
        mixer_label="softmax_test",
        dim=16,
        n_blocks=1,
        vocab_size=64,
        max_seq_len=16,
        n_train_steps=3,
        blimp_n_per_subtask=2,
        blimp_max_seq_len=32,
    )
    assert isinstance(result, LMEvalResult)
    assert result.mixer_label == "softmax_test"
    assert isinstance(result.wikitext, WikitextTrainTrace)
    assert result.blimp_overall_accuracy == 0.67
    assert result.blimp_by_subtask == {"foo": 0.7, "bar": 0.65}
    assert result.blimp_status == "ok"
    assert result.n_params > 0
