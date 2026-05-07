from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 512, dim: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, input_ids):
        return self.proj(self.embed(input_ids))


def test_small_ar_pair_table_uses_large_default_vocab_and_disjoint_splits():
    from research.eval.small_ar_champion import (
        DEFAULT_HELD_PAIRS,
        DEFAULT_KEY_TOKENS,
        DEFAULT_PAIRS_PER_EXAMPLE,
        DEFAULT_STORY_BINDINGS,
        DEFAULT_STORY_NOISE_SENTENCES,
        DEFAULT_TRAIN_PAIRS,
        DEFAULT_VALUE_CLASSES,
        DEFAULT_VALUE_TOKENS,
        SmallARChampionConfig,
        build_small_ar_pair_table,
    )

    cfg = SmallARChampionConfig()
    table = build_small_ar_pair_table(cfg)

    assert cfg.episodic_values is True
    assert cfg.protocol == "story_micro"
    assert cfg.story_bindings_per_example == DEFAULT_STORY_BINDINGS == 4
    assert cfg.story_noise_sentences_per_example == DEFAULT_STORY_NOISE_SENTENCES == 0
    assert cfg.n_key_tokens == DEFAULT_KEY_TOKENS == 1024
    assert cfg.n_value_tokens == DEFAULT_VALUE_TOKENS == 96
    assert cfg.n_value_classes == DEFAULT_VALUE_CLASSES == 12
    assert cfg.n_train_pairs == DEFAULT_TRAIN_PAIRS == 256
    assert cfg.n_held_pairs == DEFAULT_HELD_PAIRS == 64
    assert cfg.pairs_per_example == DEFAULT_PAIRS_PER_EXAMPLE == 12
    assert table.total_token_span > 1000
    train_keys = {tuple(row.tolist()) for row in table.train_keys}
    held_keys = {tuple(row.tolist()) for row in table.held_keys}
    assert train_keys
    assert held_keys
    assert train_keys.isdisjoint(held_keys)


def test_small_ar_batch_shape_and_held_targets_are_from_held_split():
    from research.eval.small_ar_champion import (
        SmallARChampionConfig,
        build_small_ar_pair_table,
        make_small_ar_batch,
    )

    cfg = SmallARChampionConfig(
        vocab_lo=100,
        n_key_tokens=64,
        n_value_tokens=32,
        n_value_classes=8,
        n_train_pairs=20,
        n_held_pairs=8,
        pairs_per_example=4,
    )
    table = build_small_ar_pair_table(cfg)
    gen = torch.Generator(device="cpu").manual_seed(1)
    ids, targets, classes = make_small_ar_batch(
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


def test_small_ar_default_uses_episodic_values_to_block_key_memorization():
    from research.eval.small_ar_champion import (
        SmallARChampionConfig,
        build_small_ar_pair_table,
        make_small_ar_batch,
    )

    cfg = SmallARChampionConfig(
        vocab_lo=100,
        n_key_tokens=32,
        n_value_tokens=16,
        n_value_classes=4,
        n_train_pairs=8,
        n_held_pairs=1,
        pairs_per_example=4,
    )
    table = build_small_ar_pair_table(cfg)
    gen = torch.Generator(device="cpu").manual_seed(7)
    ids, targets, _classes = make_small_ar_batch(
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


def test_small_ar_result_serializes_expected_fields():
    from research.eval.small_ar_champion import (
        SMALL_AR_CHAMPION_METRIC_VERSION,
        SmallARChampionResult,
    )

    result = SmallARChampionResult(
        final_acc=0.25,
        held_pair_match_acc=0.125,
        held_class_acc=0.5,
        learning_curve=[{"step": 10, "held_pair_match_acc": 0.125}],
        steps_to_floor=10,
        score=2.75,
        status="ok",
        elapsed_ms=12.3,
    )
    data = result.to_dict()

    assert data["small_ar_champion_metric_version"] == SMALL_AR_CHAMPION_METRIC_VERSION
    assert data["small_ar_champion_score"] == pytest.approx(2.75)
    assert json.loads(data["small_ar_champion_learning_curve_json"]) == [
        {"held_pair_match_acc": 0.125, "step": 10}
    ]


def test_small_ar_probe_smoke_cpu():
    from research.eval.small_ar_champion import (
        SmallARChampionConfig,
        run_small_ar_champion,
    )

    cfg = SmallARChampionConfig(
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
    result = run_small_ar_champion(TinyLM(), cfg=cfg, device="cpu")

    assert result.metric_version == "small_ar_champion_story_micro_v1"
    assert result.status in {"ok", "timeout"}
    assert 0.0 <= result.final_acc <= 1.0
    assert 0.0 <= result.held_pair_match_acc <= 1.0
    assert 0.0 <= result.held_class_acc <= 1.0
    assert result.learning_curve
    assert "counterfactual_acc" in result.learning_curve[-1]


def test_small_ar_integer_v2_probe_remains_available_cpu():
    from research.eval.small_ar_champion import (
        INTEGER_SMALL_AR_CHAMPION_METRIC_VERSION,
        SmallARChampionConfig,
        run_small_ar_champion,
    )

    cfg = SmallARChampionConfig(
        protocol="integer_v2",
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
    result = run_small_ar_champion(TinyLM(), cfg=cfg, device="cpu")

    assert result.metric_version == INTEGER_SMALL_AR_CHAMPION_METRIC_VERSION
    assert result.status in {"ok", "timeout"}
    assert result.learning_curve


def test_investigation_probe_helper_wires_small_ar_fields(monkeypatch):
    from research.scientist.runner._helpers_benchmark import (
        _run_investigation_v2_probes,
    )

    induction_v2_result = SimpleNamespace(
        auc=0.12,
        max_gap_acc=0.34,
        gap_accuracies={4: 0.2},
        steps_trained=500,
        status="ok",
        elapsed_ms=123.0,
        protocol_version="induction_v2_test",
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
    small_ar_result = SimpleNamespace(
        metric_version="small_ar_champion_test",
        final_acc=0.7,
        held_pair_match_acc=0.6,
        held_class_acc=0.8,
        learning_curve=[{"step": 1, "held_pair_match_acc": 0.6}],
        steps_to_floor=1,
        score=6.8,
        status="ok",
        elapsed_ms=99.0,
    )

    monkeypatch.setitem(
        sys.modules,
        "research.eval.induction_probe_v2_investigation",
        SimpleNamespace(
            run_induction_v2_investigation=lambda model, device: induction_v2_result
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.binding_probe_v2_investigation",
        SimpleNamespace(
            run_binding_v2_investigation=lambda model, device: binding_result
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.small_ar_champion",
        SimpleNamespace(run_small_ar_champion=lambda model, device: small_ar_result),
    )

    updates = _run_investigation_v2_probes(
        object(),
        "cpu",
        run_small_ar_champion_probe=True,
    )

    assert updates["small_ar_champion_metric_version"] == "small_ar_champion_test"
    assert updates["small_ar_champion_final_acc"] == pytest.approx(0.7)
    assert updates["small_ar_champion_held_pair_match_acc"] == pytest.approx(0.6)
    assert updates["small_ar_champion_score"] == pytest.approx(6.8)
    assert json.loads(updates["small_ar_champion_learning_curve_json"]) == [
        {"held_pair_match_acc": 0.6, "step": 1}
    ]


def test_small_ar_calibration_selects_attention_over_no_context():
    from research.tools.small_ar_champion_calibration import (
        SELECTED_CONFIG_NAME,
        select_calibrated_setting,
        selected_small_ar_config,
    )

    cfg = selected_small_ar_config(train_steps=5000)
    chance = 1.0 / cfg.n_value_tokens
    rows = [
        {
            "config_name": SELECTED_CONFIG_NAME,
            "model_family": "attention",
            "status": "ok",
            "held_pair_match_acc": 0.078,
            "score": 1.3,
            "value_token_chance": chance,
            "config": {"pairs_per_example": cfg.pairs_per_example},
        },
        {
            "config_name": SELECTED_CONFIG_NAME,
            "model_family": "no_context",
            "status": "ok",
            "held_pair_match_acc": 0.012,
            "score": 0.2,
            "value_token_chance": chance,
            "config": {"pairs_per_example": cfg.pairs_per_example},
        },
    ]

    selected = select_calibrated_setting(rows)

    assert selected is not None
    assert selected["config_name"] == SELECTED_CONFIG_NAME
    assert selected["attention_held_pair_match_acc"] > chance * 5.0
    assert selected["no_context_held_pair_match_acc"] <= chance * 3.0
