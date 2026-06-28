from __future__ import annotations

import json
from pathlib import Path

from component_fab.tests.conftest import base_dynamic_axes
from component_fab.improver.ranking import rank_proposals
from component_fab.proposer.dynamic import enumerate_dynamic_proposals
from component_fab.proposer.tier2_feedback import (
    WEAK_FAIL_BROAD_KV,
    WEAK_FAIL_COMPOSITIONAL,
    WEAK_FAIL_LONG_GAP,
    WEAK_NARROW_DISTRACTOR_ONLY,
    load_tier2_feedback,
    tier2_score_multiplier,
)
from component_fab.state.ledger import Ledger


_PID = "dynamic_candidate_abc123"


def _tier2_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "tier2.json"
    path.write_text(
        json.dumps(
            {
                "results": {
                    _PID: {
                        "status": "ok",
                        "name": "dynamic_candidate",
                        "pass_count": 1,
                        "n_tasks": 6,
                        "tier2_passed": False,
                        "tier2_passed_niche": False,
                        "per_task": {
                            "distractor_kv_recall": {
                                "candidate_eval_acc": 0.2,
                                "baseline_max": 0.1,
                                "delta": 0.1,
                                "beats": True,
                            },
                            "long_gap_recall": {
                                "candidate_eval_acc": 0.0,
                                "baseline_max": 0.1,
                                "delta": -0.1,
                                "beats": False,
                            },
                            "compositional_binding": {
                                "candidate_eval_acc": 0.1,
                                "baseline_max": 0.2,
                                "delta": -0.1,
                                "beats": False,
                            },
                            "multi_query_kv_recall": {
                                "candidate_eval_acc": 0.1,
                                "baseline_max": 0.3,
                                "delta": -0.2,
                                "beats": False,
                            },
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _seed_ledger(ledger: Ledger) -> Ledger:
    ledger.record_grade(
        proposal_id=_PID,
        name="dynamic_candidate",
        category="lane",
        synthesis_kind="novel_hybrid",
        cycle=1,
        composite_score=0.8,
        smoke_pass=True,
        learned_signal=True,
        metadata={
            "math_axes": base_dynamic_axes(),
            "can_bind": True,
            "erf_density": 0.08,
            "nb_max_accuracy": 0.7,
        },
    )
    return ledger


def test_tier2_feedback_classifies_narrow_distractor_failures(tmp_path: Path) -> None:
    feedback = load_tier2_feedback([_tier2_artifact(tmp_path)])[_PID]

    assert WEAK_NARROW_DISTRACTOR_ONLY in feedback.signatures
    assert WEAK_FAIL_LONG_GAP in feedback.signatures
    assert WEAK_FAIL_COMPOSITIONAL in feedback.signatures
    assert WEAK_FAIL_BROAD_KV in feedback.signatures
    assert tier2_score_multiplier(feedback) == 0.55


def test_dynamic_proposer_uses_tier2_task_specific_repairs(
    tmp_path: Path, tmp_ledger: Ledger
) -> None:
    feedback_by_id = load_tier2_feedback([_tier2_artifact(tmp_path)])
    specs = enumerate_dynamic_proposals(
        [],
        _seed_ledger(tmp_ledger),
        max_specs=16,
        include_anchor_fallback=False,
        tier2_feedback_by_id=feedback_by_id,
    )

    names = {spec.name for spec in specs}
    assert any("repair_long_gap_memory" in name for name in names)
    assert any("repair_compositional_tensor" in name for name in names)
    assert any("repair_broad_kv_lookup" in name for name in names)
    assert any("escape_distractor_only" in name for name in names)


def test_rank_proposals_downgrades_narrow_tier2_candidate(tmp_path: Path) -> None:
    feedback_by_id = load_tier2_feedback([_tier2_artifact(tmp_path)])
    solo = {
        "proposal_id": _PID,
        "name": "dynamic_candidate",
        "category": "lane",
        "synthesis_kind": "novel_hybrid",
        "smoke": {
            "forward_passed": True,
            "backward_passed": True,
            "output_finite": True,
            "param_grad_finite": True,
        },
        "property_cross_check": {},
        "promoted": False,
    }

    ranked = rank_proposals([solo], tier2_feedback_by_id=feedback_by_id)
    assert ranked[0].components["tier2_multiplier"] == 0.55
    assert ranked[0].composite_score < 0.6
