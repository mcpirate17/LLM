"""Tests for the priors.py capability target extension.

Verifies the new ar_gate / ar_curriculum / ar_retention / binding_intermediate
target keys are recognized as VALID_TARGETS, that _global_stats and
_aggregate_table fetch the right capability column, and that _score_row
weights capability lift correctly.

Per the AR framing rule: AR AUC and retention stay separate axes; this
module never blends them into a single capability scalar.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from research.meta_analysis.priors import (
    VALID_TARGETS,
    _CAPABILITY_LIFT_FLOOR,
    _CAPABILITY_METRIC_COLUMN,
    _aggregate_table,
    _global_stats,
    _score_row,
)


def _build_op_observations(path: Path, rows: list[dict]) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE op_observations (
            result_id TEXT,
            op_name TEXT,
            stage1_passed REAL,
            composite_score REAL,
            induction_intermediate_auc REAL,
            ar_gate_score REAL,
            ar_curriculum_auc_pair_final REAL,
            ar_curriculum_s0_retention REAL,
            binding_intermediate_auc REAL
        )
        """
    )
    for r in rows:
        conn.execute(
            """
            INSERT INTO op_observations
            (result_id, op_name, stage1_passed, composite_score,
             induction_intermediate_auc,
             ar_gate_score, ar_curriculum_auc_pair_final,
             ar_curriculum_s0_retention, binding_intermediate_auc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["result_id"],
                r["op_name"],
                r.get("stage1_passed"),
                r.get("composite_score"),
                r.get("induction_intermediate_auc"),
                r.get("ar_gate_score"),
                r.get("ar_curriculum_auc_pair_final"),
                r.get("ar_curriculum_s0_retention"),
                r.get("binding_intermediate_auc"),
            ),
        )
    conn.commit()
    return conn


def test_capability_targets_in_valid_targets():
    assert "ar_gate" in VALID_TARGETS
    assert "ar_curriculum" in VALID_TARGETS
    assert "ar_retention" in VALID_TARGETS
    assert "binding_intermediate" in VALID_TARGETS


def test_capability_metric_map_covers_all_capability_targets():
    cap_targets = {"ar_gate", "ar_curriculum", "ar_retention", "binding_intermediate"}
    assert set(_CAPABILITY_METRIC_COLUMN.keys()) == cap_targets
    assert set(_CAPABILITY_LIFT_FLOOR.keys()) == cap_targets


def test_global_stats_fetches_ar_gate_mean(tmp_path: Path):
    conn = _build_op_observations(
        tmp_path / "meta.db",
        [
            {"result_id": "r1", "op_name": "x", "ar_gate_score": 0.4},
            {"result_id": "r2", "op_name": "y", "ar_gate_score": 0.6},
        ],
    )
    try:
        stats = _global_stats(conn, "ar_gate")
        assert stats["mean_capability"] == 0.5
    finally:
        conn.close()


def test_global_stats_returns_none_capability_for_legacy_target(tmp_path: Path):
    conn = _build_op_observations(
        tmp_path / "meta.db",
        [{"result_id": "r1", "op_name": "x", "ar_gate_score": 0.5}],
    )
    try:
        stats = _global_stats(conn, "balanced")
        assert stats["mean_capability"] is None
    finally:
        conn.close()


def test_aggregate_table_exposes_mean_capability_per_op(tmp_path: Path):
    conn = _build_op_observations(
        tmp_path / "meta.db",
        [
            {"result_id": "r1", "op_name": "alpha", "binding_intermediate_auc": 0.8},
            {"result_id": "r2", "op_name": "alpha", "binding_intermediate_auc": 0.6},
            {"result_id": "r3", "op_name": "beta", "binding_intermediate_auc": 0.3},
        ],
    )
    try:
        rows = _aggregate_table(
            conn, "op_observations", "op_name", target="binding_intermediate"
        )
    finally:
        conn.close()
    by_key = {row["key"]: row for row in rows}
    assert abs(by_key["alpha"]["mean_capability"] - 0.7) < 1e-6
    assert abs(by_key["beta"]["mean_capability"] - 0.3) < 1e-6
    # capability_support reflects rows with non-null capability metric
    assert by_key["alpha"]["support"] == 2
    assert by_key["beta"]["support"] == 1


def test_score_row_uses_capability_lift_for_capability_target():
    row_strong = {"mean_capability": 0.6, "s1_rate": 0.5}
    row_weak = {"mean_capability": 0.1, "s1_rate": 0.5}
    stats = {
        "mean_induction": 0.0,
        "mean_composite": 0.0,
        "s1_rate": 0.5,
        "mean_capability": 0.3,
    }
    s_strong = _score_row(row_strong, stats, "ar_gate")
    s_weak = _score_row(row_weak, stats, "ar_gate")
    assert s_strong > s_weak  # higher capability → higher score


def test_score_row_legacy_target_unchanged_by_capability_extension():
    """Existing 'balanced' target must score identically to pre-extension code."""
    row = {"mean_induction": 0.05, "mean_composite": 5.0, "s1_rate": 0.4}
    stats = {
        "mean_induction": 0.04,
        "mean_composite": 4.0,
        "s1_rate": 0.3,
        "mean_capability": None,
    }
    score = _score_row(row, stats, "balanced")
    # 0.50 * ind_lift + 0.35 * comp_lift + 0.15 * s1_lift
    # ind_lift = (0.05-0.04)/max(|0.04|,0.02) = 0.25
    # comp_lift = (5.0-4.0)/max(|4.0|,1.0) = 0.25
    # s1_lift = (0.4-0.3)/max(|0.3|,0.05) = 0.333...
    expected = 0.50 * 0.25 + 0.35 * 0.25 + 0.15 * (0.1 / 0.3)
    assert abs(score - expected) < 1e-6


def test_ar_retention_uses_lower_s1_weight():
    """ar_retention deliberately weights s1 less than the AUC capability targets."""
    row = {"mean_capability": 0.5, "s1_rate": 0.8}
    stats = {
        "mean_induction": 0.0,
        "mean_composite": 0.0,
        "s1_rate": 0.4,
        "mean_capability": 0.4,
    }
    s_retention = _score_row(row, stats, "ar_retention")
    s_curriculum = _score_row(row, stats, "ar_curriculum")
    # Same cap_lift, larger s1_lift coefficient on ar_curriculum → curriculum > retention.
    assert s_curriculum > s_retention
