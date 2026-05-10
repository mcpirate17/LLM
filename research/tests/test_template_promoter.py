"""Tests for the mined-chain promoter."""

from __future__ import annotations

import json
from pathlib import Path

from research.meta_analysis.template_promoter import (
    promote_mined_chains,
    write_promotion_registry,
)


def _build_report(
    path: Path, novel: list[dict], rare: list[dict] | None = None
) -> None:
    payload = {
        "metadata": {"top_k": 20},
        "novel_candidates": novel,
        "rare_candidates": rare or [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _record(
    *,
    name: str = "mined_block",
    chain: tuple[str, ...] = ("a", "b", "c"),
    n_total: int = 20,
    n_pass: int = 10,
    pass_rate: float = 0.5,
    lift: float = 1.5,
    anchor: str = "b",
    skeleton: str = "def f(): pass",
    covered_by: list[str] | None = None,
) -> dict:
    return {
        "proposed_template_name": name,
        "chain": list(chain),
        "length": len(chain),
        "n_total": n_total,
        "n_pass": n_pass,
        "pass_rate": pass_rate,
        "lift_vs_cohort": lift,
        "anchor_op": anchor,
        "covered_by_templates": covered_by or [],
        "code_skeleton": skeleton,
    }


def test_promoter_filters_by_support_lift_and_pass_rate(tmp_path: Path):
    report = tmp_path / "report.json"
    _build_report(
        report,
        novel=[
            _record(name="strong", n_total=20, lift=2.0, pass_rate=0.6),
            _record(name="thin_support", n_total=2, lift=2.0, pass_rate=0.6),
            _record(name="low_lift", n_total=20, lift=1.0, pass_rate=0.6),
            _record(name="low_pass", n_total=20, lift=2.0, pass_rate=0.1),
        ],
    )
    candidates = promote_mined_chains(report)
    names = [c["proposed_template_name"] for c in candidates]
    assert names == ["strong"]


def test_promoter_can_add_ar_binding_overlay(tmp_path: Path):
    report = tmp_path / "report.json"
    meta_db = tmp_path / "missing_meta.db"
    _build_report(report, novel=[_record(name="strong")])

    candidates = promote_mined_chains(
        report,
        include_ar_binding_overlay=True,
        meta_db_path=meta_db,
    )

    assert candidates[0]["ar_binding_overlay"] == {
        "expected_ar_gain": None,
        "ar_gain_n": 0,
        "expected_binding_gain": None,
        "binding_gain_n": 0,
        "retention_risk": None,
        "collapse_risk": None,
        "holdout_required": True,
    }


def test_promoter_dedupes_against_existing_templates(tmp_path: Path):
    report = tmp_path / "report.json"
    _build_report(
        report,
        novel=[
            _record(name="already_registered"),
            _record(name="genuinely_new"),
        ],
    )
    candidates = promote_mined_chains(
        report, existing_template_names=["already_registered", "transformer_block"]
    )
    assert [c["proposed_template_name"] for c in candidates] == ["genuinely_new"]


def test_promoter_orders_by_promotion_score(tmp_path: Path):
    report = tmp_path / "report.json"
    _build_report(
        report,
        novel=[
            _record(name="medium", n_total=10, lift=1.5, pass_rate=0.4),
            _record(name="strong", n_total=50, lift=2.5, pass_rate=0.6),
            _record(name="weak_but_passing", n_total=5, lift=1.3, pass_rate=0.35),
        ],
    )
    candidates = promote_mined_chains(report)
    names = [c["proposed_template_name"] for c in candidates]
    assert names[0] == "strong"
    # promotion_score is monotonically descending
    scores = [c["promotion_score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_promoter_skips_rare_by_default(tmp_path: Path):
    report = tmp_path / "report.json"
    _build_report(
        report,
        novel=[],
        rare=[
            _record(
                name="rare_candidate",
                covered_by=["transformer_block"],
                n_total=20,
                lift=1.5,
                pass_rate=0.5,
            ),
        ],
    )
    assert promote_mined_chains(report) == []
    promoted = promote_mined_chains(report, include_rare=True)
    assert [c["proposed_template_name"] for c in promoted] == ["rare_candidate"]


def test_promoter_handles_missing_report(tmp_path: Path):
    candidates = promote_mined_chains(tmp_path / "does_not_exist.json")
    assert candidates == []


def test_write_promotion_registry_round_trips(tmp_path: Path):
    candidates = [{"proposed_template_name": "x", "promotion_score": 1.0}]
    out = tmp_path / "out" / "registry.json"
    written = write_promotion_registry(candidates, out, metadata={"source": "test"})
    assert written == out
    decoded = json.loads(out.read_text())
    assert decoded["count"] == 1
    assert decoded["candidates"] == candidates
    assert decoded["metadata"] == {"source": "test"}
