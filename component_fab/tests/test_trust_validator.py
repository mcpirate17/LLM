"""Tests for evidence-tier fab trust decisions."""

from __future__ import annotations

import json
from pathlib import Path

from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.state.ledger import PROMOTION_PROMOTED, PROMOTION_REJECTED, Ledger
from component_fab.validator.trust import (
    TRUST_PROMISING,
    TRUST_REJECTED,
    TRUST_SCREENED,
    TRUST_TRUSTED,
    TrustThresholds,
    build_trust_report,
    decide_trust,
)
from component_fab.tests.conftest import make_spec


def _spec(pid: str = "candidate_abc") -> ProposalSpec:
    return make_spec(
        {
            "op_search_track": "invention",
            "op_invention_mechanism": "novel_memory",
            "op_algebraic_space": "novel_memory",
        },
        pid,
        name="candidate",
        category="lane",
    )


def _promoted_ledger(tmp_path: Path, *, status: str = PROMOTION_PROMOTED) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        proposal_id="candidate_abc",
        name="candidate",
        category="lane",
        synthesis_kind="novel_hybrid",
        cycle=1,
        composite_score=0.9,
        smoke_pass=True,
        learned_signal=True,
        metadata={"lm_binding_mean_margin": -0.2},
    )
    ledger.record_promotion("candidate_abc", status)
    return ledger


def _tier2(seed_count: int = 2) -> dict:
    tasks = {
        "long_gap_recall": {"delta": 0.1, "beats": True},
        "compositional_binding": {"delta": 0.2, "beats": True},
        "multi_query_kv_recall": {"delta": 0.05, "beats": True},
    }
    return {
        "seed_count": seed_count,
        "results": {
            "candidate_abc": {
                "status": "ok",
                "per_task": tasks,
                "pass_count": 3,
                "n_tasks": 3,
                "tier2_passed": True,
                "tier2_passed_niche": True,
            }
        },
    }


def _blimp(seed_count: int = 2, delta: float = 0.01) -> dict:
    return {
        "seed_count": seed_count,
        "softmax_baseline_blimp": 0.55,
        "baselines": {
            "softmax_attention": {
                "wikitext": {
                    "post_train_ppl": 30.0,
                }
            }
        },
        "results": {
            "candidate_abc": {
                "status": "ok",
                "blimp_overall_accuracy": 0.56,
                "delta_vs_softmax_blimp": delta,
                "wikitext_post_ppl": 31.0,
            }
        },
    }


def test_internal_promotion_without_downstream_evidence_is_screened(
    tmp_path: Path,
) -> None:
    ledger = _promoted_ledger(tmp_path)
    decision = decide_trust(
        "candidate_abc",
        entry=ledger.entries["candidate_abc"],
        spec=_spec(),
    )
    assert decision.trust_tier == TRUST_SCREENED
    assert any("negative LM-binding" in reason for reason in decision.reasons)


def test_two_seed_tier2_and_blimp_evidence_certifies_trusted(tmp_path: Path) -> None:
    ledger = _promoted_ledger(tmp_path)
    decision = decide_trust(
        "candidate_abc",
        entry=ledger.entries["candidate_abc"],
        spec=_spec(),
        tier2_summary=_tier2(seed_count=2),
        blimp_summary=_blimp(seed_count=2),
        thresholds=TrustThresholds(min_seed_count=2),
    )
    assert decision.trust_tier == TRUST_TRUSTED
    assert decision.evidence_status == "sufficient_downstream_evidence"


def test_single_seed_positive_evidence_is_promising_not_trusted(tmp_path: Path) -> None:
    ledger = _promoted_ledger(tmp_path)
    decision = decide_trust(
        "candidate_abc",
        entry=ledger.entries["candidate_abc"],
        spec=_spec(),
        tier2_summary=_tier2(seed_count=1),
        blimp_summary=_blimp(seed_count=1),
        thresholds=TrustThresholds(min_seed_count=2),
    )
    assert decision.trust_tier == TRUST_PROMISING
    assert decision.evidence_status == "partial_downstream_evidence"
    assert decision.reasons == (
        "downstream evidence is positive but seed count is insufficient",
    )


def test_missing_blimp_evidence_reports_complementary_tier_gap(tmp_path: Path) -> None:
    ledger = _promoted_ledger(tmp_path)
    decision = decide_trust(
        "candidate_abc",
        entry=ledger.entries["candidate_abc"],
        spec=_spec(),
        tier2_summary=_tier2(seed_count=2),
        thresholds=TrustThresholds(min_seed_count=2),
    )
    assert decision.trust_tier == TRUST_PROMISING
    assert decision.reasons == (
        "positive downstream evidence is missing the complementary tier",
    )


def test_rejected_ledger_status_dominates_downstream_evidence(tmp_path: Path) -> None:
    ledger = _promoted_ledger(tmp_path, status=PROMOTION_REJECTED)
    decision = decide_trust(
        "candidate_abc",
        entry=ledger.entries["candidate_abc"],
        spec=_spec(),
        tier2_summary=_tier2(seed_count=2),
        blimp_summary=_blimp(seed_count=2),
    )
    assert decision.trust_tier == TRUST_REJECTED


def test_build_trust_report_counts_decisions(tmp_path: Path) -> None:
    ledger = _promoted_ledger(tmp_path)
    report = build_trust_report(
        ["candidate_abc"],
        ledger=ledger,
        proposals_by_id={"candidate_abc": _spec()},
        tier2_summary=_tier2(),
        blimp_summary=_blimp(),
    )
    assert report["counts"] == {TRUST_TRUSTED: 1}
    assert report["decisions"][0]["novelty"]["status"] == "mechanism_invention"


def test_run_trust_audit_dry_run_writes_no_artifact(tmp_path: Path, capsys) -> None:
    from component_fab.tools.run_trust_audit import main

    ledger = _promoted_ledger(tmp_path)
    tier2_path = tmp_path / "tier2.json"
    blimp_path = tmp_path / "blimp.json"
    winners_path = tmp_path / "saved_winners.json"
    tier2_path.write_text(json.dumps(_tier2()), encoding="utf-8")
    blimp_path.write_text(json.dumps(_blimp()), encoding="utf-8")
    winners_path.write_text(json.dumps({"winners": []}), encoding="utf-8")

    out_path = tmp_path / "audit.json"
    exit_code = main(
        [
            "--proposal-id",
            "candidate_abc",
            "--ledger",
            str(ledger.path),
            "--tier2",
            str(tier2_path),
            "--blimp",
            str(blimp_path),
            "--saved-winners",
            str(winners_path),
            "--output",
            str(out_path),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not out_path.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"][TRUST_TRUSTED] == 1
