"""Tests for the triplet instability blocklist."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from research.meta_analysis.triplet_blocklist import (
    blocked_triplet_set,
    derive_triplet_blocklist,
)


def _build_meta(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE op_triplet_profile_catalog (
            op_a TEXT, op_b TEXT, op_c TEXT,
            output_std REAL, output_has_nan INTEGER,
            grad_norm REAL, grad_has_nan INTEGER,
            grad_vanishing INTEGER, grad_exploding INTEGER,
            lipschitz_estimate REAL, forward_time_us REAL,
            pair_ab_predicted_stable INTEGER, pair_bc_predicted_stable INTEGER,
            triplet_stable INTEGER, diverges_from_pair_prediction INTEGER,
            error TEXT, profiled_at REAL,
            profile_source_db_path TEXT, profile_source_mtime REAL,
            PRIMARY KEY (op_a, op_b, op_c)
        )
        """
    )
    for r in rows:
        conn.execute(
            """
            INSERT INTO op_triplet_profile_catalog
            (op_a, op_b, op_c, output_std, output_has_nan,
             grad_norm, grad_has_nan, grad_vanishing, grad_exploding,
             pair_ab_predicted_stable, pair_bc_predicted_stable,
             triplet_stable)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["op_a"],
                r["op_b"],
                r["op_c"],
                r.get("output_std", 0.1),
                r.get("output_has_nan", 0),
                r.get("grad_norm", 1.0),
                r.get("grad_has_nan", 0),
                r.get("grad_vanishing", 0),
                r.get("grad_exploding", 0),
                r.get("pair_ab_predicted_stable", 1),
                r.get("pair_bc_predicted_stable", 1),
                r.get("triplet_stable", 1),
            ),
        )
    conn.commit()
    conn.close()


def test_emergent_filter_keeps_only_pair_stable_triplets(tmp_path: Path):
    db = tmp_path / "meta.db"
    _build_meta(
        db,
        [
            # emergent unstable: both pairs stable, triplet collapses
            {
                "op_a": "a",
                "op_b": "b",
                "op_c": "c",
                "triplet_stable": 0,
                "grad_vanishing": 1,
                "pair_ab_predicted_stable": 1,
                "pair_bc_predicted_stable": 1,
            },
            # known-bad pair: pair_ab predicted unstable, triplet collapses
            {
                "op_a": "x",
                "op_b": "y",
                "op_c": "z",
                "triplet_stable": 0,
                "grad_vanishing": 1,
                "pair_ab_predicted_stable": 0,
                "pair_bc_predicted_stable": 1,
            },
            # stable triplet: should never appear in blocklist
            {"op_a": "p", "op_b": "q", "op_c": "r", "triplet_stable": 1},
        ],
    )
    blocklist = derive_triplet_blocklist(db)
    sigs = [row["signature"] for row in blocklist]
    assert sigs == ["a->b->c"]
    assert blocklist[0]["emergent"] is True
    assert blocklist[0]["reason"] == "grad_vanishing"


def test_include_non_emergent_returns_all_unstable(tmp_path: Path):
    db = tmp_path / "meta.db"
    _build_meta(
        db,
        [
            {
                "op_a": "a",
                "op_b": "b",
                "op_c": "c",
                "triplet_stable": 0,
                "grad_vanishing": 1,
                "pair_ab_predicted_stable": 1,
                "pair_bc_predicted_stable": 1,
            },
            {
                "op_a": "x",
                "op_b": "y",
                "op_c": "z",
                "triplet_stable": 0,
                "grad_vanishing": 1,
                "pair_ab_predicted_stable": 0,
                "pair_bc_predicted_stable": 1,
            },
        ],
    )
    blocklist = derive_triplet_blocklist(db, require_pair_predicted_stable=False)
    sigs = sorted(row["signature"] for row in blocklist)
    assert sigs == ["a->b->c", "x->y->z"]
    emergence = {r["signature"]: r["emergent"] for r in blocklist}
    assert emergence == {"a->b->c": True, "x->y->z": False}


def test_classifies_failure_modes(tmp_path: Path):
    db = tmp_path / "meta.db"
    _build_meta(
        db,
        [
            {
                "op_a": "a",
                "op_b": "b",
                "op_c": "c1",
                "triplet_stable": 0,
                "output_has_nan": 1,
            },
            {
                "op_a": "a",
                "op_b": "b",
                "op_c": "c2",
                "triplet_stable": 0,
                "grad_exploding": 1,
            },
            {
                "op_a": "a",
                "op_b": "b",
                "op_c": "c3",
                "triplet_stable": 0,
                "grad_vanishing": 1,
            },
            {
                "op_a": "a",
                "op_b": "b",
                "op_c": "c4",
                "triplet_stable": 0,
                "output_std": 0.0,
            },
        ],
    )
    blocklist = derive_triplet_blocklist(db)
    by_sig = {row["signature"]: row["reason"] for row in blocklist}
    assert by_sig["a->b->c1"] == "output_has_nan"
    assert by_sig["a->b->c2"] == "grad_exploding"
    assert by_sig["a->b->c3"] == "grad_vanishing"
    assert by_sig["a->b->c4"] == "output_collapsed"


def test_blocked_triplet_set_lookup(tmp_path: Path):
    db = tmp_path / "meta.db"
    _build_meta(
        db,
        [
            {
                "op_a": "a",
                "op_b": "b",
                "op_c": "c",
                "triplet_stable": 0,
                "grad_vanishing": 1,
            },
        ],
    )
    blocklist = derive_triplet_blocklist(db)
    blocked = blocked_triplet_set(blocklist)
    assert ("a", "b", "c") in blocked
    assert ("a", "b", "z") not in blocked
