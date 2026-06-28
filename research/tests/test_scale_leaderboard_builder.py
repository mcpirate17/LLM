"""Tests for scale_leaderboard_builder against a synthetic in-memory runs.db."""

from __future__ import annotations

import sqlite3

import pytest

from research.tools import scale_leaderboard_builder as slb


def _build_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE scale_run_evals (
            run_name TEXT, seed INTEGER, mixer TEXT, dim INTEGER, n_blocks INTEGER,
            n_params INTEGER, PRIMARY KEY (run_name, seed)
        );
        CREATE TABLE scale_run_probe_metrics (
            run_name TEXT, seed INTEGER, probe_family TEXT, metric_key TEXT,
            value_num REAL, PRIMARY KEY (run_name, seed, probe_family, metric_key)
        );
        CREATE TABLE scale_run_blimp (run_name TEXT PRIMARY KEY, blimp_overall REAL);
        CREATE TABLE scale_run_leaderboard (
            model TEXT PRIMARY KEY, active_m REAL, tokens_m REAL, seq TEXT
        );
        CREATE TABLE leaderboard (
            entry_id TEXT PRIMARY KEY, induction_screening_auc REAL,
            blimp_overall_accuracy REAL
        );
        """
    )
    # Two full runs (8 metrics) + one blimp-only partial run.
    fams = [
        (fam, key)
        for (fam, key) in slb.CAPABILITY_METRICS.values()
        if fam != "__blimp__"
    ]
    for run, base, params, active, tokens in [
        ("good_run", 0.9, 100_000_000, 40.0, 200.0),
        ("weak_run", 0.2, 30_000_000, 5.0, 100.0),
    ]:
        for seed in (0, 1):
            conn.execute(
                "INSERT INTO scale_run_evals VALUES (?,?,?,?,?,?)",
                (run, seed, f"{run}_mixer", 512, 8, params),
            )
            for fam, key in fams:
                conn.execute(
                    "INSERT INTO scale_run_probe_metrics VALUES (?,?,?,?,?)",
                    (run, seed, fam, key, base),
                )
        conn.execute("INSERT INTO scale_run_blimp VALUES (?,?)", (run, base))
        conn.execute(
            "INSERT INTO scale_run_leaderboard VALUES (?,?,?,?)",
            (run, active, tokens, "256"),
        )
    conn.execute("INSERT INTO scale_run_blimp VALUES (?,?)", ("partial_run", 0.7))
    # Nano leaderboard: perfectly monotone screen->outcome -> rho == 1.
    for i in range(30):
        conn.execute(
            "INSERT INTO leaderboard VALUES (?,?,?)",
            (f"e{i}", float(i), float(i) * 2.0),
        )
    conn.commit()
    return conn


def test_composite_ranking_and_split() -> None:
    conn = _build_db()
    runs = slb._load_runs(conn)
    scored = slb.score_runs(runs)
    by_run = {s.run.run: s for s in scored}
    # good_run beats weak_run on every metric -> higher composite.
    assert by_run["good_run"].composite > by_run["weak_run"].composite
    # partial_run has a single metric (blimp only).
    assert by_run["partial_run"].n_metrics == 1
    md = slb.render_markdown(scored, [], [], min_metrics=4)
    assert "Partial — fewer than 4 metrics" in md
    # partial_run appears in the Partial section, not the main ranking body.
    main, partial = md.split("Partial — fewer than 4 metrics")
    assert "partial_run" in partial and "partial_run" not in main
    assert "good_run" in main


def test_param_and_compute_efficiency() -> None:
    conn = _build_db()
    scored = slb.score_runs(slb._load_runs(conn))
    by_run = {s.run.run: s for s in scored}
    g = by_run["good_run"]
    # param_eff = composite / active_m; compute_eff uses 6*active*tokens.
    assert g.param_eff == pytest.approx(g.composite / 40.0)
    assert g.compute_eff is not None and g.compute_eff > 0
    # partial run has no active/tokens enrichment -> falls back / no compute_eff.
    assert by_run["partial_run"].compute_eff is None


def test_nano_predictivity_monotone() -> None:
    conn = _build_db()
    corrs = slb.nano_screen_predictivity(conn)
    hit = [
        c
        for c in corrs
        if c.feature == "induction_screening_auc"
        and c.target == "blimp_overall_accuracy"
    ]
    assert hit and hit[0].rho == pytest.approx(1.0)
    assert hit[0].n == 30


def test_writeback_table() -> None:
    conn = _build_db()
    scored = slb.score_runs(slb._load_runs(conn))
    slb.writeback_table(conn, scored)
    rows = conn.execute(
        "SELECT run, rank, composite FROM scale_run_leaderboard_auto ORDER BY rank"
    ).fetchall()
    assert rows[0]["run"] == "good_run"  # top composite ranks first
    # manual table untouched.
    assert conn.execute("SELECT COUNT(*) FROM scale_run_leaderboard").fetchone()[0] == 2
