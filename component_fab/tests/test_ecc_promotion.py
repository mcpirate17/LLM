"""MiniMax-M3-align M3X-C1 scale-promotion wiring tests."""

from __future__ import annotations

import pytest

from component_fab.harness.ecc_promotion import (
    compact_embedding_param_count,
    run_m3x_c1_embedding_promotion_probe,
)


def test_compact_embedding_param_count_tracks_materialized_head_free_budget() -> None:
    kwargs = {
        "vocab_size": 128,
        "dim": 64,
        "ecc_code_length": 8,
        "ecc_field_size": 17,
        "hash_n_buckets": None,
        "jl_rank": None,
    }

    assert compact_embedding_param_count("dense", **kwargs) == 128 * 64
    assert compact_embedding_param_count("ecc_codeword", **kwargs) == 17 * 64
    assert compact_embedding_param_count("modulo_hash", **kwargs) == 17 * 64
    assert compact_embedding_param_count("jl_low_rank", **kwargs) == 17 * 64

    with pytest.raises(ValueError, match="unknown M3X-C1 embedding kind"):
        compact_embedding_param_count("unknown_control", **kwargs)


def test_m3x_c1_promotion_probe_runs_harder_binding_with_fixed_mixer() -> None:
    rows = run_m3x_c1_embedding_promotion_probe(
        embedding_kinds=("dense", "ecc_codeword", "modulo_hash", "jl_low_rank"),
        dim=16,
        n_blocks=1,
        n_train_steps=2,
        batch_size=4,
        ecc_code_length=4,
        ecc_field_size=17,
        seed=0,
    )
    by_kind = {row.embedding_kind: row for row in rows}

    assert set(by_kind) == {"dense", "ecc_codeword", "modulo_hash", "jl_low_rank"}
    assert all(row.task_name == "multi_query_kv_recall" for row in rows)
    assert all(row.mixer_label.startswith("causal_conv:") for row in rows)
    assert all(row.converged for row in rows)

    assert by_kind["dense"].embedding_params == 20 * 16
    assert by_kind["ecc_codeword"].embedding_params == 17 * 16
    assert by_kind["modulo_hash"].embedding_params == 17 * 16
    assert by_kind["jl_low_rank"].embedding_params == 17 * 16
    assert by_kind["ecc_codeword"].n_params < by_kind["dense"].n_params
