"""Tests for WS-7: fidelity ladder, transplant gate, ARIA registration."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.state.ledger import Ledger
from component_fab.state.aria_registration import (
    aria_registration_row,
    read_handoff,
    register_promotion,
)
from component_fab.state.fidelity import (
    RungScore,
    append_rung_scores,
    compute_fidelity_report,
    demoted_metrics,
    read_rung_scores,
    write_fidelity_report,
)
from component_fab.validator.transplant import (
    TRANSPLANT_HOSTS,
    TransplantScorecard,
    transplant_metadata_for_spec,
    transplant_portability,
)
from component_fab.tools.run_fidelity import _candidate_specs, _existing_rungs
from component_fab.tests.conftest import make_spec


def _spec(axes: dict, name: str = "cand") -> ProposalSpec:
    return make_spec(axes, name, anchor_witness_op="x", rationale="t")


# --------------------------------------------------------------------------- #
# Fidelity ladder
# --------------------------------------------------------------------------- #
def test_rung_score_store_roundtrip(tmp_path: Path):
    store = tmp_path / "scores.jsonl"
    append_rung_scores([RungScore("a", "R0", {"binding": 0.5})], store)
    append_rung_scores([RungScore("a", "R1", {"binding": 0.6})], store)
    rows = read_rung_scores(store)
    assert len(rows) == 2
    assert {r.rung for r in rows} == {"R0", "R1"}


def test_fidelity_flags_weak_metric():
    # 'binding' tracks perfectly R0->R1 (strong); 'learning' is rank-inverted (weak).
    records: list[RungScore] = []
    for i in range(10):
        records.append(
            RungScore(f"p{i}", "R0", {"binding": float(i), "learning": float(i)})
        )
        records.append(
            RungScore(f"p{i}", "R1", {"binding": float(i), "learning": float(10 - i)})
        )
    report = compute_fidelity_report(records, min_pairs=8)
    by = {m.metric: m for m in report.metrics}
    assert by["binding"].spearman == 1.0 and not by["binding"].weak
    assert by["learning"].spearman is not None and by["learning"].weak
    assert "learning" in demoted_metrics(report)
    assert "binding" not in demoted_metrics(report)


def test_fidelity_insufficient_pairs():
    records = [
        RungScore("a", "R0", {"binding": 0.1}),
        RungScore("a", "R1", {"binding": 0.2}),
    ]
    report = compute_fidelity_report(records, min_pairs=8)
    assert report.metrics[0].spearman is None
    assert any("Not enough" in f for f in report.findings)


def test_fidelity_report_writes(tmp_path: Path):
    records = [
        RungScore(f"p{i}", r, {"binding": float(i)})
        for i in range(10)
        for r in ("R0", "R1")
    ]
    report = compute_fidelity_report(records, min_pairs=8)
    out = write_fidelity_report(report, tmp_path / "fid.json")
    assert out.exists()


def test_fidelity_candidate_selection_skips_fully_scored_pairs(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for idx, score in enumerate((0.9, 0.8, 0.7)):
        ledger.record_grade(
            proposal_id=f"p{idx}",
            name=f"p{idx}",
            category="lane",
            synthesis_kind="novel_hybrid",
            cycle=1,
            composite_score=score,
            smoke_pass=True,
            learned_signal=True,
            metadata={
                "math_axes": {"op_invention_mechanism": "causal_fast_weight_memory"}
            },
        )

    existing = _existing_rungs(
        [
            RungScore("p0", "R0", {}),
            RungScore("p0", "R1", {}),
            RungScore("p1", "R0", {}),
        ]
    )

    specs = _candidate_specs(ledger, 2, existing_rungs=existing)
    assert [spec.proposal_id for spec in specs] == ["p1", "p2"]


# --------------------------------------------------------------------------- #
# Transplant gate
# --------------------------------------------------------------------------- #
def test_all_hosts_build_and_preserve_shape():
    dim = 16
    x = torch.randn(2, 12, dim)
    for name, build in TRANSPLANT_HOSTS.items():
        host = build(lambda d: nn.Linear(d, d), lambda d: nn.Linear(d, d), dim)
        out = host(x)
        assert out.shape == x.shape, f"host {name} changed shape"


def test_transplant_portability_structure():
    card = transplant_portability(
        lambda d: nn.Linear(d, d), seeds=(0, 1), dim=16, seq_len=16, n_steps=5
    )
    assert isinstance(card, TransplantScorecard)
    assert card.n_hosts == len(TRANSPLANT_HOSTS)
    assert 0.0 <= card.portability <= 1.0
    md = card.to_metadata()
    assert "transplant_portability" in md
    assert "transplant_per_host_delta" in md


def test_transplant_metadata_unbuildable_spec_skips():
    md = transplant_metadata_for_spec(
        _spec({"op_algebraic_space": "euclidean"}), seeds=(0, 1), dim=16, n_steps=4
    )
    assert "transplant_skipped_reason" in md
    assert "mechanism_unbuildable" in md["transplant_skipped_reason"]


def test_transplant_metadata_buildable_spec_scores():
    md = transplant_metadata_for_spec(
        _spec({"op_invention_mechanism": "causal_fast_weight_memory"}),
        seeds=(0, 1),
        dim=16,
        seq_len=16,
        n_steps=5,
    )
    assert "transplant_portability" in md
    assert md["transplant_n_hosts"] == len(TRANSPLANT_HOSTS)


# --------------------------------------------------------------------------- #
# ARIA registration handoff
# --------------------------------------------------------------------------- #
def test_registration_row_carries_recipe_and_axes():
    spec = _spec(
        {"op_algebraic_space": "tropical", "op_dynamical_has_state": 1}, name="trop"
    )
    row = aria_registration_row(spec, evidence={"composite": 0.7})
    assert row["op_name"] == "trop"
    assert row["math_axes"]["op_algebraic_space"] == "tropical"
    assert row["declared_axes"]["op_algebraic_space"] == "tropical"
    assert row["evidence"]["composite"] == 0.7
    assert row["source"] == "component_fab"


def test_register_promotion_roundtrips(tmp_path: Path):
    handoff = tmp_path / "aria_handoff.jsonl"
    spec = _spec({"op_invention_mechanism": "causal_fast_weight_memory"}, name="fw")
    register_promotion(
        spec, evidence={"transplant_portability": 0.75}, handoff_path=handoff
    )
    rows = read_handoff(handoff)
    assert len(rows) == 1
    assert rows[0]["op_name"] == "fw"
    assert rows[0]["evidence"]["transplant_portability"] == 0.75
    # latest-per-id dedupe
    register_promotion(
        spec, evidence={"transplant_portability": 0.9}, handoff_path=handoff
    )
    rows = read_handoff(handoff)
    assert len(rows) == 1 and rows[0]["evidence"]["transplant_portability"] == 0.9
