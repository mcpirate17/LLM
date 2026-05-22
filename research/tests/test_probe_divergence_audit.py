"""Tests for the read-only probe divergence audit.

Uses an in-process sqlite fixture with the minimal schema the audit
reads. No real models trained, no real DB touched.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml

from research.tools import probe_divergence_audit as audit

pytestmark = pytest.mark.unit


def _make_fixture_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE leaderboard (
            result_id TEXT PRIMARY KEY,
            entry_id TEXT,
            tier TEXT,
            composite_score REAL,
            template_name TEXT,
            graph_fingerprint TEXT,
            is_reference INTEGER,
            reference_name TEXT
        );
        CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            wikitext_perplexity REAL,
            induction_screening_auc REAL,
            binding_screening_auc REAL,
            ar_gate_score REAL,
            hellaswag_acc REAL,
            blimp_overall_accuracy REAL
        );
        """
    )
    # 8 rows: induction & binding strongly correlated (both reflect attention
    # depth); AR independent (different mechanism). Two families with clear
    # between-family separation.
    rows = [
        # attention family — high on all three
        ("attn1", "screening", "softmax_attention_block", 100.0, 0.80, 0.78, 0.75),
        ("attn2", "screening", "softmax_attention_block", 110.0, 0.82, 0.80, 0.72),
        ("attn3", "screening", "softmax_attention_block", 105.0, 0.85, 0.83, 0.78),
        ("attn4", "screening", "softmax_attention_block", 95.0, 0.78, 0.76, 0.70),
        # ssm family — low induction/binding, mid AR
        ("ssm1", "screening", "mamba_block_v2", 220.0, 0.30, 0.28, 0.55),
        ("ssm2", "screening", "mamba_block_v2", 240.0, 0.32, 0.30, 0.52),
        ("ssm3", "screening", "mamba_block_v2", 235.0, 0.28, 0.26, 0.58),
        ("ssm4", "screening", "mamba_block_v2", 225.0, 0.34, 0.32, 0.50),
    ]
    for rid, tier, tmpl, ppl, ind, bnd, ar in rows:
        conn.execute(
            "INSERT INTO leaderboard VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, rid, tier, 250.0, tmpl, rid + "fp", 0, None),
        )
        conn.execute(
            "INSERT INTO program_results VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid, ppl, ind, bnd, ar, None, None),
        )
    conn.commit()
    conn.close()


def _make_scoring_yaml(path: Path) -> None:
    payload = {
        "base": {
            "w_cap_induction": 45.0,
            "w_cap_binding": 45.0,
            "w_cap_ar": 35.0,
            "cap_induction_anchor": 0.006,
            "cap_binding_anchor": 0.004,
            "cap_ar_anchor": 0.50,
        }
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_build_report_returns_matrix_and_findings(tmp_path: Path):
    db = tmp_path / "runs.db"
    yaml_path = tmp_path / "scoring_config.yaml"
    _make_fixture_db(db)
    _make_scoring_yaml(yaml_path)

    report = audit.build_report(db, scoring_yaml_path=yaml_path, bootstrap=50)

    assert report["coverage"]["rows"] == 8
    screening = report["tiers"]["screening"]
    assert screening["n_rows"] == 8

    # Induction ↔ binding should be near-perfect; AR should diverge.
    rho_map = screening["matrix"]["rho"]
    ib = rho_map.get("induction_screening_auc|binding_screening_auc") or rho_map.get(
        "binding_screening_auc|induction_screening_auc"
    )
    ia = rho_map.get("induction_screening_auc|ar_gate_score") or rho_map.get(
        "ar_gate_score|induction_screening_auc"
    )
    assert ib is not None and ib > 0.9
    assert ia is not None and ia < ib


def test_audit_finds_redundancy_cluster(tmp_path: Path):
    db = tmp_path / "runs.db"
    yaml_path = tmp_path / "scoring_config.yaml"
    _make_fixture_db(db)
    _make_scoring_yaml(yaml_path)
    report = audit.build_report(db, scoring_yaml_path=yaml_path, bootstrap=50)
    clusters = report["tiers"]["screening"]["redundancy_clusters"]
    flat = [sorted(g) for g in clusters]
    # induction + binding should land in the same cluster.
    assert any(
        {"induction_screening_auc", "binding_screening_auc"}.issubset(set(g))
        for g in flat
    )


def test_audit_high_signal_ratio_when_families_separate(tmp_path: Path):
    db = tmp_path / "runs.db"
    yaml_path = tmp_path / "scoring_config.yaml"
    _make_fixture_db(db)
    _make_scoring_yaml(yaml_path)
    report = audit.build_report(db, scoring_yaml_path=yaml_path, bootstrap=50)
    sr = report["tiers"]["screening"]["signal_ratios"]
    for col in (
        "induction_screening_auc",
        "binding_screening_auc",
        "ar_gate_score",
    ):
        assert sr[col]["signal_ratio"] > 0.8


def test_audit_writes_report_files(tmp_path: Path, monkeypatch):
    db = tmp_path / "runs.db"
    yaml_path = tmp_path / "scoring_config.yaml"
    out_dir = tmp_path / "reports"
    _make_fixture_db(db)
    _make_scoring_yaml(yaml_path)
    monkeypatch.setattr(
        audit.sys if hasattr(audit, "sys") else __import__("sys"),
        "argv",
        [
            "probe_divergence_audit",
            "--db",
            str(db),
            "--scoring-yaml",
            str(yaml_path),
            "--out",
            str(out_dir),
            "--bootstrap",
            "20",
        ],
        raising=False,
    )
    rc = audit.main()
    assert rc == 0
    files = list(out_dir.glob("*"))
    assert any(p.suffix == ".json" for p in files)
    assert any(p.suffix == ".md" for p in files)
    assert any(p.suffix == ".yaml" for p in files)
    # The YAML proposal must have a `base:` section even if all weights
    # collapsed to zero — schema is the deliverable, not the magnitude.
    yaml_proposal = next(p for p in files if p.name.startswith("weight_refit_proposal"))
    loaded = yaml.safe_load(yaml_proposal.read_text(encoding="utf-8"))
    assert "base" in loaded
    assert loaded["_meta"]["applied"] is False


def test_audit_does_not_mutate_db(tmp_path: Path):
    db = tmp_path / "runs.db"
    yaml_path = tmp_path / "scoring_config.yaml"
    _make_fixture_db(db)
    _make_scoring_yaml(yaml_path)
    mtime_before = db.stat().st_mtime
    audit.build_report(db, scoring_yaml_path=yaml_path, bootstrap=20)
    mtime_after = db.stat().st_mtime
    assert mtime_before == mtime_after


def test_audit_returns_findings_strings(tmp_path: Path):
    db = tmp_path / "runs.db"
    yaml_path = tmp_path / "scoring_config.yaml"
    _make_fixture_db(db)
    _make_scoring_yaml(yaml_path)
    report = audit.build_report(db, scoring_yaml_path=yaml_path, bootstrap=20)
    findings = report["findings"]
    assert isinstance(findings, list)
    # Expect at least the pairwise rho and per-probe signal_ratio bullets.
    joined = " | ".join(findings)
    assert "Spearman" in joined
    assert "signal_ratio" in joined
