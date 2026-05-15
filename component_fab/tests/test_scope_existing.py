"""Smoke + classification tests for component_fab.intake.scope_existing."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from component_fab.intake.scope_existing import (
    CATEGORY_COMPRESSION,
    CATEGORY_LANE,
    CATEGORY_ROUTING,
    DEFAULT_META_DB,
    classify_op_row,
    classify_template_row,
    scope_all,
    select_underperforming_novel,
)


def _row(**kwargs: object) -> dict[str, object]:
    base = {
        "op_name": "x",
        "op_category": "",
        "op_n_inputs": 1,
        "op_activation_sparsity_pattern": "",
        "op_algebraic_space": "euclidean",
        "op_geometric_receptive_field": "",
        "eval_count": 0,
        "s1_pass_count": 0,
    }
    base.update(kwargs)
    return base


def test_classify_op_routing_by_name() -> None:
    assert classify_op_row(_row(op_name="n_way_sparse_router")) == CATEGORY_ROUTING
    assert classify_op_row(_row(op_name="moe_topk")) == CATEGORY_ROUTING
    assert classify_op_row(_row(op_name="route_lanes")) == CATEGORY_ROUTING


def test_classify_op_compression_by_name() -> None:
    assert (
        classify_op_row(_row(op_name="latent_attention_compressor"))
        == CATEGORY_COMPRESSION
    )
    assert classify_op_row(_row(op_name="bottleneck_proj")) == CATEGORY_COMPRESSION


def test_classify_op_lane_for_single_input_compute() -> None:
    assert (
        classify_op_row(_row(op_name="swiglu_mlp", op_category="parameterized"))
        == CATEGORY_LANE
    )
    assert (
        classify_op_row(_row(op_name="gelu", op_category="elementwise_unary"))
        == CATEGORY_LANE
    )
    assert (
        classify_op_row(_row(op_name="tropical_attention", op_category="mixing"))
        == CATEGORY_LANE
    )


def test_classify_op_unclassified_for_wiring() -> None:
    assert (
        classify_op_row(
            _row(op_name="add", op_category="elementwise_binary", op_n_inputs=2)
        )
        is None
    )
    assert (
        classify_op_row(_row(op_name="concat", op_category="structural", op_n_inputs=2))
        is None
    )
    assert (
        classify_op_row(_row(op_name="cumsum", op_category="reduction", op_n_inputs=1))
        is None
    )


def test_classify_template_routing_for_multilane() -> None:
    row = {
        "template_name": "intelligent_multilane_router",
        "template_family": "routing",
        "template_has_routing": 1,
        "template_has_parallel_paths": 1,
        "template_est_parallel_paths": 4,
        "template_routing_intensity": 0.75,
    }
    assert classify_template_row(row) == CATEGORY_ROUTING


def test_classify_template_lane_for_chain() -> None:
    row = {
        "template_name": "0_champion_baseline",
        "template_family": "generic",
        "template_has_routing": 0,
        "template_has_parallel_paths": 0,
        "template_routing_intensity": 0.0,
        "template_compression_intensity": 0.0,
    }
    assert classify_template_row(row) == CATEGORY_LANE


def test_select_underperforming_novel_filters_correctly() -> None:
    rows = [
        _row(
            op_name="tropical_attention",
            op_algebraic_space="tropical",
            op_category="mixing",
            eval_count=300,
            s1_pass_count=60,
        ),
        _row(
            op_name="softmax_attention",
            op_algebraic_space="euclidean",
            op_category="mixing",
            eval_count=500,
            s1_pass_count=200,
        ),
        _row(
            op_name="tropical_one_shot",
            op_algebraic_space="tropical",
            op_category="mixing",
            eval_count=5,
            s1_pass_count=1,
        ),
        _row(
            op_name="clifford_winner",
            op_algebraic_space="clifford",
            op_category="mixing",
            eval_count=200,
            s1_pass_count=180,
        ),
    ]
    from component_fab.intake.scope_existing import _op_record

    records = [_op_record(r) for r in rows]
    out = select_underperforming_novel(records, min_evals=30, pass_rate_ceiling=0.35)
    names = [r.name for r in out]
    assert "tropical_attention" in names
    assert "softmax_attention" not in names
    assert "tropical_one_shot" not in names
    assert "clifford_winner" not in names


def test_scope_all_against_real_db_well_formed() -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    report = scope_all()
    assert report["totals"]["ops"] > 0
    assert report["totals"]["templates"] > 0
    assert report["multilane_routing_templates"]
    assert report["underperforming_novel_ops"]
    names = {r["name"] for r in report["multilane_routing_templates"]}
    assert "intelligent_multilane_router" in names
    assert "three_way_split" in names
    novel_names = {r["name"] for r in report["underperforming_novel_ops"]}
    assert any("tropical" in n for n in novel_names)


def test_scope_all_missing_db_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scope_all(db_path=tmp_path / "absent.db")


def test_real_db_schema_has_expected_columns() -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    conn = sqlite3.connect(f"file:{DEFAULT_META_DB}?mode=ro", uri=True)
    try:
        cur = conn.execute("PRAGMA table_info(op_property_catalog)")
        columns = {row[1] for row in cur.fetchall()}
    finally:
        conn.close()
    for required in (
        "op_name",
        "op_category",
        "op_algebraic_space",
        "op_dynamical_has_state",
        "eval_count",
        "s1_pass_count",
    ):
        assert required in columns, f"missing required column: {required}"
