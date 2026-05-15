"""Smoke + behavior tests for component_fab.proposer.property_miner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from component_fab.proposer.property_miner import (
    DEFAULT_AXES,
    DEFAULT_META_DB,
    AxisLift,
    compute_axis_lifts,
    enumerate_candidates,
    extant_tuples,
    load_rows,
    run,
)


def _build_tiny_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE op_property_catalog (
                op_name TEXT PRIMARY KEY,
                eval_count INTEGER,
                s1_pass_count INTEGER,
                op_algebraic_space TEXT,
                op_dynamical_memory_length_class TEXT,
                op_dynamical_has_state INTEGER,
                op_activation_sparsity_pattern TEXT,
                op_geometric_receptive_field TEXT,
                op_spectral_preferred_basis TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO op_property_catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "alpha",
                    100,
                    40,
                    "euclidean",
                    "O(L)",
                    1,
                    "dense",
                    "global",
                    "content",
                ),
                (
                    "beta",
                    200,
                    90,
                    "euclidean",
                    "O(L^2)",
                    0,
                    "dense",
                    "global",
                    "content",
                ),
                (
                    "gamma",
                    50,
                    20,
                    "tropical",
                    "O(L)",
                    0,
                    "learned_structured",
                    "global",
                    "content",
                ),
                ("delta", 80, 30, "euclidean", "O(L)", 0, "top_k", "local", "identity"),
                (
                    "epsilon",
                    30,
                    5,
                    "clifford",
                    "O(L^2)",
                    0,
                    "dense",
                    "global",
                    "content",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def tiny_db(tmp_path: Path) -> Path:
    path = tmp_path / "meta.db"
    _build_tiny_db(path)
    return path


def test_load_rows_returns_dicts(tiny_db: Path) -> None:
    rows = load_rows(tiny_db)
    assert len(rows) == 5
    assert all("op_name" in r for r in rows)


def test_load_rows_missing_db_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_rows(tmp_path / "absent.db")


def test_compute_axis_lifts_aggregates_evals(tiny_db: Path) -> None:
    rows = load_rows(tiny_db)
    lifts = compute_axis_lifts(rows, ("op_algebraic_space",))
    by_space = lifts["op_algebraic_space"]
    eu = by_space["euclidean"]
    assert isinstance(eu, AxisLift)
    assert eu.n_ops == 3
    assert eu.total_evals == 100 + 200 + 80
    assert eu.total_s1_pass == 40 + 90 + 30
    assert eu.pass_rate == pytest.approx((40 + 90 + 30) / (100 + 200 + 80))


def test_extant_tuples_covers_each_row(tiny_db: Path) -> None:
    rows = load_rows(tiny_db)
    extant = extant_tuples(rows, ("op_algebraic_space", "op_dynamical_has_state"))
    assert ("euclidean", 1) in extant
    assert ("tropical", 0) in extant
    assert ("tropical", 1) not in extant


def test_enumerate_candidates_excludes_extant(tiny_db: Path) -> None:
    rows = load_rows(tiny_db)
    axes = ("op_algebraic_space", "op_dynamical_has_state")
    lifts = compute_axis_lifts(rows, axes)
    extant = extant_tuples(rows, axes)
    candidates = enumerate_candidates(
        lifts,
        extant,
        axes,
        min_axis_n_ops=1,
        min_axis_pass_rate=0.0,
        top_k_values_per_axis=4,
    )
    for c in candidates:
        tup = tuple(v for _, v in c.tuple_values)
        assert tup not in extant


def test_run_against_real_db_returns_well_formed_report() -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("research/meta_analysis.db not present in this environment")
    report = run(max_candidates=5)
    assert "candidates" in report
    assert report["n_rows"] > 0
    assert list(report["axes"]) == list(DEFAULT_AXES)
    for c in report["candidates"]:
        assert "tuple" in c
        assert "predicted_lift" in c
        assert len(c["tuple"]) == len(DEFAULT_AXES)
