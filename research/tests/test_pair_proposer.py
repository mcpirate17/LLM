"""Tests for the untapped pair proposer.

Verifies the read-only diff between healthy profiled pair compositions and
pair signatures observed in real programs — i.e., the "new math" surface
the grammar has not yet assembled.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from research.meta_analysis.pair_proposer import propose_untapped_pairs


def _build_meta_db(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE op_pair_profile_catalog (
            op_a TEXT, op_b TEXT, composition TEXT,
            shape_compatible INTEGER, algebraic_compatible INTEGER,
            output_mean REAL, output_std REAL, output_min REAL, output_max REAL,
            output_has_nan INTEGER, output_has_inf INTEGER, output_kurtosis REAL,
            grad_norm REAL, grad_max REAL, grad_min REAL,
            grad_has_nan INTEGER, grad_has_zero INTEGER,
            grad_vanishing INTEGER, grad_exploding INTEGER,
            jacobian_spectral_norm REAL, jacobian_condition_num REAL,
            lipschitz_estimate REAL,
            forward_time_us REAL, backward_time_us REAL,
            peak_memory_bytes INTEGER, flops_estimate INTEGER,
            stability_delta REAL, distribution_shift REAL, speed_overhead REAL,
            error TEXT, profiled_at REAL,
            profile_source_db_path TEXT, profile_source_mtime REAL,
            PRIMARY KEY (op_a, op_b, composition)
        )
        """
    )
    columns = (
        "op_a, op_b, composition, output_std, output_has_nan, "
        "grad_norm, grad_has_nan, grad_vanishing, grad_exploding, "
        "lipschitz_estimate, jacobian_spectral_norm, "
        "stability_delta, distribution_shift"
    )
    placeholders = ",".join("?" * 13)
    for r in rows:
        conn.execute(
            f"INSERT INTO op_pair_profile_catalog ({columns}) VALUES ({placeholders})",
            (
                r["op_a"],
                r["op_b"],
                r["composition"],
                r.get("output_std", 0.1),
                r.get("output_has_nan", 0),
                r.get("grad_norm", 1.0),
                r.get("grad_has_nan", 0),
                r.get("grad_vanishing", 0),
                r.get("grad_exploding", 0),
                r.get("lipschitz_estimate", 0.5),
                r.get("jacobian_spectral_norm"),
                r.get("stability_delta"),
                r.get("distribution_shift"),
            ),
        )
    conn.commit()
    conn.close()


def _build_runs_db(path: Path, observed_signatures: list[str]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE program_graph_pairs (
            result_id TEXT NOT NULL,
            graph_fingerprint TEXT,
            signature TEXT NOT NULL,
            PRIMARY KEY (result_id, signature)
        )
        """
    )
    conn.execute("INSERT INTO program_results (result_id) VALUES ('r1')")
    for sig in observed_signatures:
        conn.execute(
            "INSERT INTO program_graph_pairs (result_id, signature) VALUES (?, ?)",
            ("r1", sig),
        )
    conn.commit()
    conn.close()


def test_proposer_emits_only_unobserved_stable_pairs(tmp_path: Path):
    meta = tmp_path / "meta.db"
    runs = tmp_path / "runs.db"
    _build_meta_db(
        meta,
        [
            # Healthy + unobserved → should emit
            {"op_a": "linear", "op_b": "softmax_last", "composition": "sequential"},
            # Healthy but observed → should be filtered
            {"op_a": "rmsnorm", "op_b": "linear", "composition": "sequential"},
            # Unobserved but unhealthy (NaN) → filtered
            {
                "op_a": "linear",
                "op_b": "exp",
                "composition": "sequential",
                "output_has_nan": 1,
            },
            # Unobserved but vanishing → filtered
            {
                "op_a": "linear",
                "op_b": "tanh",
                "composition": "sequential",
                "grad_vanishing": 1,
            },
            # Unobserved but exploding → filtered
            {
                "op_a": "linear",
                "op_b": "abs",
                "composition": "sequential",
                "grad_exploding": 1,
            },
        ],
    )
    _build_runs_db(runs, observed_signatures=["rmsnorm->linear", "abs->abs"])

    candidates = propose_untapped_pairs(meta, runs)
    sigs = [c["signature"] for c in candidates]
    assert sigs == ["linear->softmax_last"]
    assert candidates[0]["novelty"] == "fully_untapped"
    assert candidates[0]["ar_binding_overlay"] == {
        "expected_ar_gain": None,
        "ar_gain_n": 0,
        "expected_binding_gain": None,
        "binding_gain_n": 0,
        "retention_risk": None,
        "collapse_risk": None,
        "holdout_required": True,
    }


def test_proposer_filters_near_zero_output_std(tmp_path: Path):
    meta = tmp_path / "meta.db"
    runs = tmp_path / "runs.db"
    _build_meta_db(
        meta,
        [
            {"op_a": "a", "op_b": "b", "composition": "sequential", "output_std": 1e-9},
            {"op_a": "c", "op_b": "d", "composition": "sequential", "output_std": 0.5},
        ],
    )
    _build_runs_db(runs, observed_signatures=[])
    candidates = propose_untapped_pairs(meta, runs, include_ar_binding_overlay=False)
    assert [c["signature"] for c in candidates] == ["c->d"]


def test_proposer_orders_by_stability_score(tmp_path: Path):
    meta = tmp_path / "meta.db"
    runs = tmp_path / "runs.db"
    _build_meta_db(
        meta,
        [
            # tight Lipschitz, balanced grad → low score (best)
            {
                "op_a": "good_a",
                "op_b": "good_b",
                "composition": "sequential",
                "grad_norm": 1.0,
                "lipschitz_estimate": 0.05,
            },
            # wild Lipschitz → high score (worst)
            {
                "op_a": "wild_a",
                "op_b": "wild_b",
                "composition": "sequential",
                "grad_norm": 5.0,
                "lipschitz_estimate": 10.0,
            },
        ],
    )
    _build_runs_db(runs, observed_signatures=[])
    candidates = propose_untapped_pairs(meta, runs, include_ar_binding_overlay=False)
    assert candidates[0]["signature"] == "good_a->good_b"
    assert candidates[1]["signature"] == "wild_a->wild_b"


def test_proposer_respects_limit(tmp_path: Path):
    meta = tmp_path / "meta.db"
    runs = tmp_path / "runs.db"
    _build_meta_db(
        meta,
        [
            {"op_a": f"a{i}", "op_b": "b", "composition": "sequential"}
            for i in range(20)
        ],
    )
    _build_runs_db(runs, observed_signatures=[])
    candidates = propose_untapped_pairs(
        meta, runs, limit=5, include_ar_binding_overlay=False
    )
    assert len(candidates) == 5
