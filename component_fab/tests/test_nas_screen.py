from __future__ import annotations

import json
from pathlib import Path

from component_fab.improver.ranking import rank_proposals
from component_fab.proposer.nas_screen import (
    NasScreenResult,
    nas_calibration_context,
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


def test_nas_calibration_context_reports_ppv_npv_roc(
    tmp_path: Path, monkeypatch
) -> None:
    from component_fab.proposer import nas_screen

    oracle_meta = tmp_path / "oracle_meta.json"
    predictor = tmp_path / "predictor.json"
    oracle_meta.write_text(
        json.dumps(
            {
                "thresholds": {"ar_gate": 0.9},
                "selected_per_axis": {
                    "ar_gate": {
                        "kind": "gbm",
                        "leave_family_out_roc": {"gbm": 0.89},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    predictor.write_text(
        json.dumps(
            {
                "graph_predictor": {
                    "val_metrics_selected_threshold": {
                        "roc_auc": 0.91,
                        "precision_ppv": 0.72,
                        "npv": 0.84,
                        "threshold": 0.5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(nas_screen, "_ORACLE_META", oracle_meta)
    monkeypatch.setattr(nas_screen, "_PREDICTOR_REPORT", predictor)

    context = nas_calibration_context()
    assert context["oracle_axes"]["ar_gate"]["leave_family_out_roc"]["gbm"] == 0.89
    assert context["graph_predictor_selected"]["precision_ppv"] == 0.72
    assert context["graph_predictor_selected"]["npv"] == 0.84
    assert context["graph_predictor_selected"]["roc_auc"] == 0.91
