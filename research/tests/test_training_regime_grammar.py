"""Unit tests for Workstream E training-regime genotype + stage masks."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from research.synthesis.training_regime_grammar import (
    AXIS_FREEZE_SCHEDULE,
    AXIS_TRAIN_REGIME,
    AXIS_TRAIN_STAGES,
    TRAIN_TARGETS,
    TrainStageSpec,
    TrainingRegimeSpec,
    implemented_training_regimes,
    parse_train_stage,
    parse_train_stages,
    sample_training_regime_spec,
    serialize_train_stage,
    training_regime_from_axes,
    training_regime_to_axes,
)
from research.training.staged_training import (
    apply_stage_sequence,
    apply_train_stage,
    matched_parameter_names,
    trainable_parameters,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mixer = nn.Linear(8, 8)
        self.ffn = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8))
        self.router = nn.Linear(8, 1)
        self.norm = nn.LayerNorm(8)


class _PairedTiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(32, 8)
        self.position_embedding = nn.Embedding(16, 8)
        self.blocks = nn.ModuleList([_Block()])
        self.lane_partner = nn.Linear(8, 8)
        self.lane_loss = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 32, bias=False)


def _names(report) -> set[str]:  # noqa: ANN001 - tiny test helper
    return set(report.trainable_names)


def test_stage_spec_validates_target_steps_and_lr() -> None:
    TrainStageSpec("all", 1)
    with pytest.raises(ValueError, match="unknown train target"):
        TrainStageSpec("unknown", 1)
    with pytest.raises(ValueError, match="positive"):
        TrainStageSpec("all", 0)
    with pytest.raises(ValueError, match="lr_scale"):
        TrainStageSpec("all", 1, lr_scale=0.0)


def test_training_regime_axes_round_trip() -> None:
    spec = TrainingRegimeSpec(
        name="embed_then_carrier",
        stages=(
            TrainStageSpec("embedding_lm_head", 100, lr_scale=1.5),
            TrainStageSpec("carrier", 200, lr_scale=1.25),
            TrainStageSpec("all", 700, freeze_others=True),
        ),
        optimizer="muon",
        base_lr=2e-4,
        weight_decay=0.02,
        scheduler="cosine",
        warmup_fraction=0.1,
        max_grad_norm=0.75,
    )

    axes = training_regime_to_axes(spec)

    assert axes[AXIS_TRAIN_REGIME] == "embed_then_carrier"
    assert axes[AXIS_TRAIN_STAGES] == (
        "embedding_lm_head:100:freeze:1.5:reset|"
        "carrier:200:freeze:1.25:reset|all:700:freeze:1:reset"
    )
    assert axes[AXIS_FREEZE_SCHEDULE] == (
        "embedding_lm_head=freeze_others>"
        "carrier=freeze_others>all=freeze_others"
    )
    assert training_regime_from_axes(axes) == spec


def test_stage_serialization_round_trip() -> None:
    stage = TrainStageSpec(
        "router", 33, lr_scale=2.0, freeze_others=False, reset_optimizer=False
    )

    assert serialize_train_stage(stage) == "router:33:add:2:keep"
    assert parse_train_stage("router:33:add:2:keep") == stage
    assert parse_train_stages("router:33:add:2:keep|all:67:freeze:1:reset") == (
        stage,
        TrainStageSpec("all", 67),
    )


def test_sampler_emits_implemented_regimes_deterministically() -> None:
    gen_a = torch.Generator().manual_seed(11)
    gen_b = torch.Generator().manual_seed(11)

    samples_a = [sample_training_regime_spec(gen_a) for _ in range(8)]
    samples_b = [sample_training_regime_spec(gen_b) for _ in range(8)]

    assert samples_a == samples_b
    assert {stage.target for spec in samples_a for stage in spec.stages} <= set(
        TRAIN_TARGETS
    )
    assert all(spec.total_steps == 1000 for spec in samples_a)


def test_implemented_regime_roster_includes_body_warmup() -> None:
    regimes = implemented_training_regimes()

    assert set(regimes) == {
        "all_train",
        "embed_warm_then_all",
        "body_warm_then_all",
        "carrier_warm_then_all",
        "router_warm_then_all",
    }
    assert regimes["body_warm_then_all"].stages[0].target == "non_embedding"
    assert regimes["body_warm_then_all"].stages[-1].target == "all"


def test_embedding_lm_head_stage_exclusively_trains_tables_and_head() -> None:
    model = _PairedTiny()

    report = apply_train_stage(model, TrainStageSpec("embedding_lm_head", 100))

    trainable = _names(report)
    assert "token_embedding.weight" in trainable
    assert "position_embedding.weight" in trainable
    assert "lm_head.weight" in trainable
    assert not any(name.startswith("blocks.") for name in trainable)
    assert report.trainable_param_count < report.total_param_count
    assert trainable_parameters(model)


def test_additive_unfreeze_keeps_existing_trainable_parameters() -> None:
    model = _PairedTiny()
    first = apply_train_stage(model, TrainStageSpec("embeddings", 50))
    second = apply_train_stage(
        model,
        TrainStageSpec("router", 50, freeze_others=False),
    )

    trainable = _names(second)
    assert set(first.trainable_names) <= trainable
    assert "blocks.0.router.weight" in trainable
    assert "blocks.0.router.bias" in trainable
    assert "lm_head.weight" not in trainable


def test_carrier_and_loss_lane_targets_are_distinct() -> None:
    model = _PairedTiny()

    carrier = matched_parameter_names(model, "carrier")
    loss_lane = matched_parameter_names(model, "loss_lane")

    assert set(carrier) == {"lane_partner.weight", "lane_partner.bias"}
    assert set(loss_lane) == {"lane_loss.weight", "lane_loss.bias"}


def test_stage_sequence_reports_intermediate_masks() -> None:
    model = _PairedTiny()
    reports = apply_stage_sequence(
        model,
        (
            TrainStageSpec("carrier", 10),
            TrainStageSpec("all", 20),
        ),
    )

    assert len(reports) == 2
    assert set(reports[0].trainable_names) == {
        "lane_partner.weight",
        "lane_partner.bias",
    }
    assert len(reports[1].frozen_names) == 0


def test_unknown_or_empty_stage_target_fails_loud() -> None:
    model = nn.Linear(4, 4)

    with pytest.raises(ValueError, match="matched no parameters"):
        apply_train_stage(model, TrainStageSpec("carrier", 1))
