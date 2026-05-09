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
    from research.eval.ar_intermediate_probe import ARIntermediateConfig

    return ARIntermediateConfig(
        seed=3,
        vocab_lo=8,
        n_key_tokens=40,
        n_value_tokens=16,
        n_value_classes=4,
        n_train_pairs=8,
        n_held_pairs=4,
        pairs_per_example=3,
        train_steps=3,
        eval_every=1,
        batch_size=2,
        n_eval=4,
        lr=1e-3,
        timeout_s=20.0,
    )


def test_ar_intermediate_pair_table_and_batch_have_disjoint_held_split():
    from research.eval.ar_intermediate_probe import (
        build_ar_intermediate_pair_table,
        make_ar_intermediate_batch,
    )

    cfg = _tiny_cfg()
    table = build_ar_intermediate_pair_table(cfg)
    train_keys = {tuple(row.tolist()) for row in table.train_keys}
    held_keys = {tuple(row.tolist()) for row in table.held_keys}
    assert train_keys
    assert held_keys
    assert train_keys.isdisjoint(held_keys)

    gen = torch.Generator(device="cpu").manual_seed(11)
    ids, targets, classes = make_ar_intermediate_batch(
        table,
        split="held",
        batch_size=5,
        pairs_per_example=cfg.pairs_per_example,
        sep_token=126,
        ans_token=127,
        device=torch.device("cpu"),
        generator=gen,
        episodic_values=False,
    )

    assert ids.shape == (5, 3 * cfg.pairs_per_example + 4)
    assert targets.shape == (5,)
    assert classes.shape == (5,)
    assert set(targets.tolist()).issubset(set(table.held_values.tolist()))


def test_ar_intermediate_defaults_match_calibrated_intermediate_setting():
    from research.eval.ar_intermediate_probe import ARIntermediateConfig

    cfg = ARIntermediateConfig()

    assert cfg.n_key_tokens == 256
    assert cfg.n_value_tokens == 48
    assert cfg.n_value_classes == 12
    assert cfg.n_train_pairs == 96
    assert cfg.n_held_pairs == 32
    assert cfg.pairs_per_example == 5
    assert cfg.train_steps == 1500
    assert cfg.n_eval == 256
    assert cfg.timeout_s == pytest.approx(300.0)
    assert cfg.threshold == pytest.approx(0.12)


def test_ar_intermediate_diagnostic_score_prioritizes_exact_and_learning_speed():
    from research.eval import ar_intermediate_probe as probe

    class_only = probe._score(
        held_pair_lift=0.0,
        held_class_lift=1.0,
        auc_lift=0.0,
        improvement_lift=0.0,
        steps_to_threshold=None,
        train_steps=1200,
    )
    exact_with_slope = probe._score(
        held_pair_lift=0.35,
        held_class_lift=0.15,
        auc_lift=0.25,
        improvement_lift=0.2,
        steps_to_threshold=300,
        train_steps=1200,
    )

    assert class_only == pytest.approx(2.0)
    assert exact_with_slope > class_only


def test_ar_intermediate_tiny_cpu_completes_and_populates_schema():
    from research.eval.ar_intermediate_probe import (
        AR_INTERMEDIATE_METRIC_VERSION,
        ar_intermediate_probe,
    )

    result = ar_intermediate_probe(TinyLM(), cfg=_tiny_cfg(), device="cpu")
    data = result.to_dict()

    assert result.status == "ok"
    assert result.metric_version == AR_INTERMEDIATE_METRIC_VERSION
    assert result.steps_trained == 3
    assert len(result.learning_curve) == 3
    assert data["ar_intermediate_diagnostic_score"] >= 0.0
    assert data["ar_intermediate_pair_chance_acc"] == pytest.approx(1 / 16)
    assert data["ar_intermediate_class_chance_acc"] == pytest.approx(1 / 4)
    assert "ar_intermediate_held_pair_lift" in data
    assert "ar_intermediate_auc_lift" in data
    assert result.best_held_pair_acc >= min(
        result.early_held_pair_acc,
        result.final_held_pair_acc,
    )
    curve = json.loads(data["ar_intermediate_learning_curve_json"])
    assert curve
    assert {
        "step",
        "loss",
        "held_pair_acc",
        "held_class_acc",
        "held_pair_lift",
        "held_class_lift",
    }.issubset(curve[0])
    assert all(row["loss"] > 0.0 for row in curve)


def test_ar_intermediate_preserves_model_state_with_copy_model():
    from research.eval.ar_intermediate_probe import ar_intermediate_probe

    model = TinyLM()
    model.eval()
    before = snapshot_state(model)

    result = ar_intermediate_probe(model, cfg=_tiny_cfg(), device="cpu")

    assert result.status == "ok"
    assert not model.training
    assert_state_preserved(model, before)


def test_ar_intermediate_reports_vocab_too_small_without_curve_placeholders():
    from research.eval.ar_intermediate_probe import ar_intermediate_probe

    result = ar_intermediate_probe(TinyLM(vocab_size=32), cfg=_tiny_cfg(), device="cpu")

    assert result.status == "error"
    assert result.error is not None
    assert result.error.startswith("model_vocab_too_small")
    assert result.learning_curve == []
