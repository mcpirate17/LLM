"""Tests for the Workstream E staged-regime A/B runner."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from research.synthesis.training_regime_grammar import (
    TrainStageSpec,
    TrainingRegimeSpec,
    implemented_training_regimes,
)
from research.tools.training_regime_ab import (
    _rescale_stage_steps,
    _summarize,
    _train_curve_staged,
)


class _TinyNextToken(nn.Module):
    def __init__(self, vocab_size: int = 16, dim: int = 8) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.body = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.body(self.token_embedding(x)))


def test_rescale_stage_steps_preserves_budget_and_proportions() -> None:
    regime = implemented_training_regimes()["embed_warm_then_all"]

    scaled = _rescale_stage_steps(regime, 10)

    assert [stage.steps for stage in scaled.stages] == [2, 8]
    assert scaled.total_steps == 10
    assert scaled.stages[0].target == "embedding_lm_head"
    assert scaled.stages[1].target == "all"


def test_summarize_uses_all_train_final_loss_as_threshold() -> None:
    results = [
        {
            "condition": "all_train",
            "final": {"val_loss": 2.0, "top1_acc": 0.10},
            "curve": [{"step": 0, "val_loss": 3.0}, {"step": 4, "val_loss": 2.0}],
        },
        {
            "condition": "embed_warm_then_all",
            "final": {"val_loss": 1.8, "top1_acc": 0.20},
            "curve": [{"step": 0, "val_loss": 2.2}, {"step": 2, "val_loss": 1.9}],
        },
    ]

    summary = _summarize(results)

    assert summary["baseline_all_train_val_loss"] == 2.0
    assert summary["by_condition"]["all_train"]["mean_steps_to_all_train_final"] == 4.0
    assert (
        summary["by_condition"]["embed_warm_then_all"][
            "mean_steps_to_all_train_final"
        ]
        == 2.0
    )
    assert summary["by_condition"]["embed_warm_then_all"]["mean_final_top1"] == 0.2


def test_train_curve_staged_applies_stage_masks_on_cpu() -> None:
    torch.manual_seed(0)
    model = _TinyNextToken()
    tokens = np.arange(128, dtype=np.int64) % 16
    regime = TrainingRegimeSpec(
        name="tiny_embed_then_all",
        stages=(
            TrainStageSpec("embedding_lm_head", 2),
            TrainStageSpec("all", 2),
        ),
    )

    curve = _train_curve_staged(
        model,
        tokens,
        tokens,
        regime,
        seq=4,
        batch=2,
        steps=4,
        lr=1e-3,
        device="cpu",
        eval_every=2,
        eval_batches=1,
    )

    assert curve[0]["step"] == 0
    assert curve[-1]["step"] == 4
    assert {pt["stage"] for pt in curve} == {"embedding_lm_head", "all"}
    embed_stage = next(pt for pt in curve if pt["stage"] == "embedding_lm_head")
    all_stage = next(pt for pt in curve if pt["stage"] == "all")
    assert all_stage["trainable_params"] > embed_stage["trainable_params"]
    assert "val_loss" in curve[-1]
    assert "top1_acc" in curve[-1]
