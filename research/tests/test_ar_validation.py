from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from research.tests._probe_test_support import TinyLM

pytestmark = pytest.mark.unit


def test_ar_validation_pair_table_uses_large_default_vocab_and_disjoint_splits():
    from research.eval.ar_validation import (
        DEFAULT_HELD_PAIRS,
        DEFAULT_KEY_TOKENS,
        DEFAULT_PAIRS_PER_EXAMPLE,
        DEFAULT_STORY_BINDINGS,
        DEFAULT_STORY_NOISE_SENTENCES,
        DEFAULT_TRAIN_PAIRS,
        DEFAULT_VALUE_CLASSES,
        DEFAULT_VALUE_TOKENS,
        DEFAULT_AR_VALIDATION_PROTOCOL,
        ARValidationConfig,
        build_ar_validation_pair_table,
    )

    cfg = ARValidationConfig()
    table = build_ar_validation_pair_table(cfg)

    assert cfg.episodic_values is True
    assert cfg.protocol == DEFAULT_AR_VALIDATION_PROTOCOL == "integer_v2"
    assert cfg.story_bindings_per_example == DEFAULT_STORY_BINDINGS == 4
    assert cfg.story_noise_sentences_per_example == DEFAULT_STORY_NOISE_SENTENCES == 0
    assert cfg.n_key_tokens == DEFAULT_KEY_TOKENS == 1024
    assert cfg.n_value_tokens == DEFAULT_VALUE_TOKENS == 96
    assert cfg.n_value_classes == DEFAULT_VALUE_CLASSES == 12
    assert cfg.n_train_pairs == DEFAULT_TRAIN_PAIRS == 256
    assert cfg.n_held_pairs == DEFAULT_HELD_PAIRS == 64
    assert cfg.pairs_per_example == DEFAULT_PAIRS_PER_EXAMPLE == 9
    assert table.total_token_span > 1000
    train_keys = {tuple(row.tolist()) for row in table.train_keys}
    held_keys = {tuple(row.tolist()) for row in table.held_keys}
    assert train_keys
    assert held_keys
    assert train_keys.isdisjoint(held_keys)


def test_ar_validation_batch_shape_and_held_targets_are_from_held_split():
    from research.eval.ar_validation import (
        ARValidationConfig,
        build_ar_validation_pair_table,
        make_ar_validation_batch,
    )

    cfg = ARValidationConfig(
        vocab_lo=100,
        n_key_tokens=64,
        n_value_tokens=32,
        n_value_classes=8,
        n_train_pairs=20,
        n_held_pairs=8,
        pairs_per_example=4,
    )
    table = build_ar_validation_pair_table(cfg)
    gen = torch.Generator(device="cpu").manual_seed(1)
    ids, targets, classes = make_ar_validation_batch(
        table,
        split="held",
        batch_size=6,
        pairs_per_example=cfg.pairs_per_example,
        sep_token=510,
        ans_token=511,
        device=torch.device("cpu"),
        generator=gen,
        episodic_values=False,
    )

    assert ids.shape == (6, 3 * cfg.pairs_per_example + 4)
    assert targets.shape == (6,)
    assert classes.shape == (6,)
    assert set(targets.tolist()).issubset(set(table.held_values.tolist()))


def test_ar_validation_default_uses_episodic_values_to_block_key_memorization():
    from research.eval.ar_validation import (
        ARValidationConfig,
        build_ar_validation_pair_table,
        make_ar_validation_batch,
    )

    cfg = ARValidationConfig(
        vocab_lo=100,
        n_key_tokens=32,
        n_value_tokens=16,
        n_value_classes=4,
        n_train_pairs=8,
        n_held_pairs=1,
        pairs_per_example=4,
    )
    table = build_ar_validation_pair_table(cfg)
    gen = torch.Generator(device="cpu").manual_seed(7)
    ids, targets, _classes = make_ar_validation_batch(
        table,
        split="held",
        batch_size=32,
        pairs_per_example=cfg.pairs_per_example,
        sep_token=510,
        ans_token=511,
        device=torch.device("cpu"),
        generator=gen,
    )

    assert ids.shape == (32, 3 * cfg.pairs_per_example + 4)
    assert targets.min().item() >= table.value_lo
    assert targets.max().item() < table.value_hi
    assert len(set(targets.tolist())) > 1


def test_ar_validation_episode_bank_is_seed_stable_and_seed_sensitive():
    from research.eval._kv_pair import build_kv_episode_bank
    from research.eval.ar_validation import (
        ARValidationConfig,
        build_ar_validation_pair_table,
    )

    cfg = ARValidationConfig(
        vocab_lo=100,
        n_key_tokens=64,
        n_value_tokens=32,
        n_value_classes=8,
        n_train_pairs=20,
        n_held_pairs=8,
        pairs_per_example=4,
    )
    table = build_ar_validation_pair_table(cfg)
    kwargs = {
        "table": table,
        "split": "held",
        "n_examples": 16,
        "batch_size": 4,
        "pairs_per_example": cfg.pairs_per_example,
        "sep_token": 510,
        "ans_token": 511,
        "device": torch.device("cpu"),
    }

    bank_a = build_kv_episode_bank(seed=11, **kwargs)
    bank_b = build_kv_episode_bank(seed=11, **kwargs)
    bank_c = build_kv_episode_bank(seed=12, **kwargs)

    assert torch.equal(bank_a.ids, bank_b.ids)
    assert torch.equal(bank_a.targets, bank_b.targets)
    assert not torch.equal(bank_a.ids, bank_c.ids)
    assert set(bank_a.targets.tolist()).issubset(
        set(range(table.value_lo, table.value_hi))
    )


def test_ar_validation_size_budget_boundaries():
    from research.eval.ar_validation import (
        ar_validation_budget_for_param_count,
        ar_validation_size_bucket,
    )

    assert ar_validation_size_bucket(14_999_999) == "10m"
    assert ar_validation_size_bucket(15_000_000) == "20m"
    assert ar_validation_size_bucket(25_000_000) == "30m"
    assert ar_validation_size_bucket(45_000_000) == "60m"
    assert ar_validation_size_bucket(80_000_000) == "100m_plus"

    budget = ar_validation_budget_for_param_count(100_000_000)
    assert budget.size_bucket == "100m_plus"
    assert budget.train_steps == 25_000
    assert budget.n_eval == 2048
    assert budget.seed_count == 3


def test_ar_validation_result_serializes_expected_fields():
    from research.eval.ar_validation import (
        INTEGER_AR_VALIDATION_METRIC_VERSION,
        ARValidationResult,
    )

    result = ARValidationResult(
        final_acc=0.25,
        held_pair_acc=0.125,
        held_class_acc=0.5,
        learning_curve=[{"step": 10, "held_pair_acc": 0.125}],
        steps_to_floor=10,
        score=2.75,
        status="ok",
        elapsed_ms=12.3,
    )
    data = result.to_dict()

    assert data["ar_validation_metric_version"] == INTEGER_AR_VALIDATION_METRIC_VERSION
    assert data["ar_validation_rank_score"] == pytest.approx(2.75)
    assert json.loads(data["ar_validation_learning_curve_json"]) == [
        {"held_pair_acc": 0.125, "step": 10}
    ]


def test_stable_ar_validation_result_serializes_aggregate_metadata():
    from research.eval.ar_validation import (
        STABLE_AR_VALIDATION_METRIC_VERSION,
        ARValidationResult,
    )

    result = ARValidationResult(
        metric_version=STABLE_AR_VALIDATION_METRIC_VERSION,
        final_acc=0.25,
        held_pair_acc=0.125,
        held_class_acc=0.5,
        score=2.75,
        size_bucket="20m",
        param_count=20_000_000,
        seed_count=3,
        seed_scores=[{"seed": 0, "score": 2.0}, {"seed": 1, "score": 3.5}],
        rank_score_mean=2.75,
        rank_score_std=0.75,
        rank_score_stable=2.0,
        held_pair_acc_mean=0.125,
        held_pair_acc_std=0.025,
        held_class_acc_mean=0.5,
        held_class_acc_std=0.1,
        budget={"train_steps": 7500, "size_bucket": "20m"},
        checkpoint_path="/tmp/stage.pt",
        stage_status="ok",
        stage_elapsed_ms=55.5,
    )

    data = result.to_dict()

    assert data["ar_validation_metric_version"] == STABLE_AR_VALIDATION_METRIC_VERSION
    assert data["ar_validation_seed_count"] == 3
    assert data["ar_validation_rank_score_mean"] == pytest.approx(2.75)
    assert data["ar_validation_rank_score_stable"] == pytest.approx(2.0)
    assert json.loads(data["ar_validation_seed_scores_json"]) == [
        {"score": 2.0, "seed": 0},
        {"score": 3.5, "seed": 1},
    ]
    assert json.loads(data["ar_validation_budget_json"]) == {
        "size_bucket": "20m",
        "train_steps": 7500,
    }
    assert data["ar_validation_checkpoint_path"] == "/tmp/stage.pt"


def test_ar_validation_probe_refuses_cpu():
    from research.eval.ar_validation import (
        INTEGER_AR_VALIDATION_METRIC_VERSION,
        ARValidationConfig,
        run_ar_validation,
    )

    cfg = ARValidationConfig(
        seed=3,
        vocab_lo=100,
        n_key_tokens=64,
        n_value_tokens=32,
        n_value_classes=8,
        n_train_pairs=20,
        n_held_pairs=8,
        pairs_per_example=4,
        train_steps=2,
        eval_every=1,
        batch_size=2,
        n_eval=4,
        timeout_s=20.0,
    )
    result = run_ar_validation(TinyLM(vocab_size=512), cfg=cfg, device="cpu")

    assert result.metric_version == INTEGER_AR_VALIDATION_METRIC_VERSION
    assert result.status == "missing_accelerator"
    assert result.error == "ar_validation_requires_cuda"
    assert not result.learning_curve


def test_ar_validation_story_micro_probe_refuses_cpu():
    from research.eval.ar_validation import (
        AR_VALIDATION_METRIC_VERSION,
        ARValidationConfig,
        run_ar_validation,
    )

    cfg = ARValidationConfig(
        protocol="story_micro",
        seed=4,
        vocab_lo=100,
        n_key_tokens=64,
        n_value_tokens=32,
        n_value_classes=8,
        n_train_pairs=20,
        n_held_pairs=8,
        pairs_per_example=4,
        train_steps=2,
        eval_every=1,
        batch_size=2,
        n_eval=4,
        timeout_s=20.0,
    )
    result = run_ar_validation(TinyLM(vocab_size=512), cfg=cfg, device="cpu")

    assert result.metric_version == AR_VALIDATION_METRIC_VERSION
    assert result.status == "missing_accelerator"
    assert result.error == "ar_validation_requires_cuda"
    assert not result.learning_curve


def test_ar_validation_hotpath_benchmark_smoke():
    from research.tools.bench_ar_validation_hotpath import parse_args, run_benchmark

    args = parse_args(
        [
            "--device",
            "cpu",
            "--batches",
            "2",
            "--warmup-batches",
            "1",
            "--batch-size",
            "2",
            "--pairs-per-example",
            "4",
            "--key-tokens",
            "64",
            "--value-tokens",
            "32",
            "--train-pairs",
            "20",
            "--held-pairs",
            "8",
        ]
    )
    payload = run_benchmark(args)

    assert payload["benchmark"] == "ar_validation_batch_hotpath"
    assert payload["examples"] == 4
    assert payload["tokens_per_s"] > 0
    assert payload["target_checksum"] >= 0


def test_investigation_probe_helper_wires_ar_validation_fields(monkeypatch):
    from research.scientist.runner._helpers_benchmark import (
        _run_investigation_v2_probes,
    )

    induction_intermediate_result = SimpleNamespace(
        auc=0.12,
        max_gap_acc=0.34,
        gap_accuracies={4: 0.2},
        steps_trained=500,
        status="ok",
        elapsed_ms=123.0,
        protocol_version="induction_intermediate_test",
    )
    binding_result = SimpleNamespace(
        auc=0.56,
        max_distance_acc=0.78,
        distance_accuracies={4: 0.7},
        train_steps=2400,
        status="ok",
        elapsed_ms=789.0,
        protocol_version="binding-test",
    )
    ar_validation_result = SimpleNamespace(
        metric_version="ar_validation_test",
        final_acc=0.7,
        held_pair_acc=0.6,
        held_class_acc=0.8,
        learning_curve=[{"step": 1, "held_pair_acc": 0.6}],
        steps_to_floor=1,
        score=6.8,
        status="ok",
        elapsed_ms=99.0,
    )

    monkeypatch.setitem(
        sys.modules,
        "research.eval.induction_intermediate_probe",
        SimpleNamespace(
            run_induction_intermediate=lambda model, device: (
                induction_intermediate_result
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_intermediate_probe",
        SimpleNamespace(run_binding_intermediate=lambda model, device: binding_result),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.ar_validation",
        SimpleNamespace(run_ar_validation=lambda model, device: ar_validation_result),
    )

    updates = _run_investigation_v2_probes(
        object(),
        "cpu",
        run_ar_validation_probe=True,
    )

    assert updates["ar_validation_metric_version"] == "ar_validation_test"
    assert updates["ar_validation_final_acc"] == pytest.approx(0.7)
    assert updates["ar_validation_held_pair_acc"] == pytest.approx(0.6)
    assert updates["ar_validation_rank_score"] == pytest.approx(6.8)
    assert json.loads(updates["ar_validation_learning_curve_json"]) == [
        {"held_pair_acc": 0.6, "step": 1}
    ]


def test_ar_validation_calibration_selects_attention_over_no_context():
    from research.tools.ar_validation_calibration import (
        SELECTED_CONFIG_NAME,
        select_calibrated_setting,
        selected_ar_validation_config,
    )

    cfg = selected_ar_validation_config(train_steps=5000)
    chance = 1.0 / cfg.n_value_tokens
    rows = [
        {
            "config_name": SELECTED_CONFIG_NAME,
            "model_family": "attention",
            "status": "ok",
            "held_pair_acc": 0.078,
            "score": 1.3,
            "value_token_chance": chance,
            "config": {"pairs_per_example": cfg.pairs_per_example},
        },
        {
            "config_name": SELECTED_CONFIG_NAME,
            "model_family": "no_context",
            "status": "ok",
            "held_pair_acc": 0.012,
            "score": 0.2,
            "value_token_chance": chance,
            "config": {"pairs_per_example": cfg.pairs_per_example},
        },
    ]

    selected = select_calibrated_setting(rows)

    assert selected is not None
    assert selected["config_name"] == SELECTED_CONFIG_NAME
    assert selected["attention_held_pair_acc"] > chance * 5.0
    assert selected["no_context_held_pair_acc"] <= chance * 3.0
