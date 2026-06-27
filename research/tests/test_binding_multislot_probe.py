from __future__ import annotations

import json

import pytest
import torch

from research.tests._probe_test_support import (
    TinyLM,
    assert_state_preserved,
    snapshot_state,
)

pytestmark = pytest.mark.unit


def _tiny_cfg():
    from research.eval.binding_multislot_probe import BindingMultislotConfig

    return BindingMultislotConfig(
        seed=5,
        vocab_lo=8,
        n_entities=12,
        n_held_entities=4,
        n_color_values=8,
        n_object_values=8,
        bindings_per_example=3,
        query_slots=3,
        train_steps=3,
        eval_every=1,
        batch_size=2,
        n_eval=4,
        lr=1e-3,
        timeout_s=20.0,
    )


def test_multi_blank_layout_and_batch_have_multiple_query_slots():
    from research.eval.binding_multislot_probe import (
        build_multi_blank_layout,
        make_multi_blank_batch,
    )

    cfg = _tiny_cfg()
    layout = build_multi_blank_layout(cfg)
    assert set(layout.train_entities.tolist()).isdisjoint(
        set(layout.held_entities.tolist())
    )

    gen = torch.Generator(device="cpu").manual_seed(17)
    ids, targets, classes, ans_positions = make_multi_blank_batch(
        layout,
        split="held_entity",
        batch_size=4,
        bindings_per_example=cfg.bindings_per_example,
        query_slots=cfg.query_slots,
        device=torch.device("cpu"),
        generator=gen,
    )

    expected_len = cfg.bindings_per_example * 3 + 1 + cfg.query_slots * 4
    assert ids.shape == (4, expected_len)
    assert targets.shape == (4, cfg.query_slots)
    assert classes.shape == (4, cfg.query_slots)
    assert ans_positions.tolist() == [13, 17, 21]
    assert targets.min().item() >= layout.value_lo
    assert targets.max().item() < layout.value_hi


def test_multi_blank_defaults_match_hardened_intermediate_setting():
    from research.eval.binding_multislot_probe import BindingMultislotConfig

    cfg = BindingMultislotConfig()

    assert cfg.n_entities == 96
    assert cfg.n_held_entities == 24
    assert cfg.n_color_values == 32
    assert cfg.n_object_values == 32
    assert cfg.bindings_per_example == 5
    assert cfg.query_slots == 3
    assert cfg.train_steps == 1000
    assert cfg.eval_every == 125
    assert cfg.n_eval == 256
    assert cfg.threshold == pytest.approx(0.08)


def test_multi_blank_score_does_not_reward_class_only_behavior_heavily():
    from research.eval import binding_multislot_probe as probe

    class_only = probe._score(
        held_slot_lift=0.0,
        held_class_lift=1.0,
        two_plus_slots_lift=0.0,
        all_slots_lift=0.0,
        mixed_query_lift=0.0,
        mixed_two_plus_slots_lift=0.0,
        mixed_all_slots_lift=0.0,
        auc_lift=0.0,
    )
    exact_binding = probe._score(
        held_slot_lift=0.18,
        held_class_lift=1.0,
        two_plus_slots_lift=0.16,
        all_slots_lift=0.12,
        mixed_query_lift=0.2,
        mixed_two_plus_slots_lift=0.14,
        mixed_all_slots_lift=0.08,
        auc_lift=0.1,
    )

    assert class_only == pytest.approx(0.5)
    assert exact_binding > class_only


def test_multi_blank_tiny_cpu_completes_populates_schema_and_preserves_state():
    from research.eval.binding_multislot_probe import (
        BINDING_MULTISLOT_METRIC_VERSION,
        binding_multislot_probe,
    )

    model = TinyLM()
    model.eval()
    before = snapshot_state(model)

    result = binding_multislot_probe(model, cfg=_tiny_cfg(), device="cpu")
    data = result.to_dict()

    assert result.status == "ok"
    assert not model.training
    assert_state_preserved(model, before)
    assert result.metric_version == BINDING_MULTISLOT_METRIC_VERSION
    assert result.steps_trained == 3
    assert len(result.learning_curve) == 3
    assert data["binding_multislot_diagnostic_score"] >= 0.0
    assert data["binding_multislot_slot_chance_acc"] == pytest.approx(1 / 16)
    assert data["binding_multislot_class_chance_acc"] == pytest.approx(0.5)
    assert data["binding_multislot_two_plus_slots_chance_acc"] == pytest.approx(
        1 - (15 / 16) ** 3 - 3 * (1 / 16) * (15 / 16) ** 2,
        abs=1e-6,
    )
    assert data["binding_multislot_all_slots_chance_acc"] == pytest.approx(
        (1 / 16) ** 3,
        abs=1e-6,
    )
    assert "binding_multislot_held_slot_lift" in data
    assert "binding_multislot_two_plus_slots_lift" in data
    assert "binding_multislot_mixed_two_plus_slots_lift" in data
    assert "binding_multislot_mixed_all_slots_lift" in data
    assert "binding_multislot_auc_lift" in data
    assert result.best_slot_acc >= min(result.early_slot_acc, result.final_slot_acc)
    curve = json.loads(data["binding_multislot_learning_curve_json"])
    assert curve
    assert {
        "step",
        "loss",
        "held_entity_slot_acc",
        "held_entity_class_acc",
        "two_plus_slots_acc",
        "all_slots_acc",
        "mixed_two_plus_slots_acc",
        "mixed_all_slots_acc",
        "held_slot_lift",
        "two_plus_slots_lift",
        "all_slots_lift",
    }.issubset(curve[0])
    assert all(row["loss"] > 0.0 for row in curve)


def test_multi_blank_reports_vocab_too_small_without_curve_placeholders():
    from research.eval.binding_multislot_probe import binding_multislot_probe

    result = binding_multislot_probe(
        TinyLM(vocab_size=24),
        cfg=_tiny_cfg(),
        device="cpu",
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.startswith("model_vocab_too_small")
    assert result.learning_curve == []
