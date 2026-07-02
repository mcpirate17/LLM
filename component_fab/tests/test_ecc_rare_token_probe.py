"""MiniMax-M3-align M3X-C1 learned rare-token/JL control benchmark."""

from __future__ import annotations

import pytest

from component_fab.harness.ecc_rare_token_probe import (
    RareTokenProbeConfig,
    rare_token_ids,
    run_rare_token_embedding_probe,
)


def test_rare_token_ids_pick_modulo_collision_group() -> None:
    cfg = RareTokenProbeConfig(field_size=17, rare_bucket=3, rare_group_size=8)

    ids = rare_token_ids(cfg)

    assert ids.tolist() == [3, 20, 37, 54, 71, 88, 105, 122]
    assert ids.remainder(cfg.field_size).unique().tolist() == [cfg.rare_bucket]


def test_rare_token_config_rejects_unavailable_collision_group() -> None:
    cfg = RareTokenProbeConfig(vocab_size=32, field_size=17, rare_group_size=8)

    with pytest.raises(ValueError, match="rare_group_size exceeds available tokens"):
        rare_token_ids(cfg)


def test_ecc_beats_equal_budget_hash_and_jl_rare_token_controls() -> None:
    cfg = RareTokenProbeConfig(
        vocab_size=128,
        dim=64,
        code_length=8,
        field_size=17,
        rare_group_size=8,
        train_steps=10,
        rare_per_batch=2,
        learning_rate=5e-2,
        seed=123,
        data_seed=9,
        jl_seed=5,
    )

    results = {row.name: row for row in run_rare_token_embedding_probe(cfg)}
    ecc = results["ecc_codeword"]
    modulo_hash = results["modulo_hash"]
    jl = results["jl_low_rank"]

    assert ecc.trainable_params == modulo_hash.trainable_params == jl.trainable_params
    assert ecc.trainable_params == cfg.field_size * cfg.dim

    assert ecc.rare_accuracy == 1.0
    assert modulo_hash.rare_accuracy <= 1.0 / cfg.rare_group_size + 1e-6
    assert ecc.rare_accuracy >= jl.rare_accuracy

    assert ecc.rare_margin > jl.rare_margin + 2.0
    assert ecc.rare_margin > modulo_hash.rare_margin + 5.0
    assert ecc.final_loss < jl.final_loss
