from __future__ import annotations

from component_fab.improver.ranking import rank_proposals
from component_fab.proposer.nas_screen import (
    NasScreenResult,
    nas_score_multiplier,
)


def _solo() -> dict:
    return {
        "proposal_id": "candidate_abc",
        "name": "candidate",
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


def test_nas_score_multiplier_respects_gate_and_rank_score() -> None:
    rejected = NasScreenResult(
        proposal_id="candidate_abc",
        available=True,
        gate_pass=False,
        downstream_gate_pass=True,
        rank_score=2.0,
        source="test",
    )
    boosted = NasScreenResult(
        proposal_id="candidate_abc",
        available=True,
        gate_pass=True,
        downstream_gate_pass=True,
        rank_score=1.3,
        source="test",
    )

    assert nas_score_multiplier(rejected) == 0.55
    assert nas_score_multiplier(boosted) == 1.08


def test_rank_proposals_applies_nas_multiplier() -> None:
    result = NasScreenResult(
        proposal_id="candidate_abc",
        available=True,
        gate_pass=False,
        downstream_gate_pass=True,
        rank_score=2.0,
        source="test",
    )

    ranked = rank_proposals([_solo()], nas_screen_by_id={"candidate_abc": result})
    assert ranked[0].components["nas_multiplier"] == 0.55
    assert ranked[0].composite_score < 0.6
