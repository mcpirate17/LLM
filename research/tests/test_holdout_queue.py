"""Tests for AR/binding holdout queue construction."""

from __future__ import annotations

import json
from pathlib import Path

from research.meta_analysis.holdout_queue import (
    build_holdout_queue,
    write_holdout_queue,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _overlay(*, holdout_required: bool, retention_risk: float = 0.0) -> dict:
    return {
        "expected_ar_gain": 0.1,
        "ar_gain_n": 30,
        "expected_binding_gain": 0.2,
        "binding_gain_n": 30,
        "retention_risk": retention_risk,
        "collapse_risk": 0.0,
        "holdout_required": holdout_required,
    }


def test_holdout_queue_blocks_structurally_invalid_templates(tmp_path: Path):
    validated = tmp_path / "validated.json"
    pairs = tmp_path / "pairs.json"
    _write_json(
        validated,
        {
            "candidates": [
                {
                    "proposed_template_name": "bad_tpl",
                    "chain": ["a", "add"],
                    "promotion_score": 10.0,
                    "ar_binding_overlay": _overlay(holdout_required=True),
                    "validation": {
                        "validate_passed": False,
                        "compile_passed": False,
                        "failure_mode": "build",
                    },
                }
            ]
        },
    )
    _write_json(pairs, {"candidates": []})

    payload = build_holdout_queue(
        promoted_templates_path=tmp_path / "missing_promoted.json",
        validated_templates_path=validated,
        pair_proposals_path=pairs,
        created_at=1.0,
    )

    item = payload["items"][0]
    assert item["status"] == "blocked_structural"
    assert "compile_validation_failed" in item["blockers"]
    assert "overlay_holdout_required" in item["blockers"]
    assert payload["metadata"]["status_counts"] == {"blocked_structural": 1}


def test_holdout_queue_marks_compile_passing_template_ready_for_holdout(
    tmp_path: Path,
):
    validated = tmp_path / "validated.json"
    pairs = tmp_path / "pairs.json"
    _write_json(
        validated,
        {
            "candidates": [
                {
                    "proposed_template_name": "ready_tpl",
                    "chain": ["linear_proj", "rmsnorm"],
                    "promotion_score": 3.0,
                    "ar_binding_overlay": _overlay(holdout_required=True),
                    "validation": {
                        "validate_passed": True,
                        "compile_passed": True,
                        "backward_passed": True,
                    },
                }
            ]
        },
    )
    _write_json(pairs, {"candidates": []})

    payload = build_holdout_queue(
        validated_templates_path=validated,
        pair_proposals_path=pairs,
        created_at=1.0,
    )

    assert payload["items"][0]["status"] == "ready_for_holdout"
    assert payload["items"][0]["next_step"] == "run_small_training_holdout"


def test_holdout_queue_adds_pair_schema_work_items(tmp_path: Path):
    promoted = tmp_path / "promoted.json"
    pairs = tmp_path / "pairs.json"
    _write_json(promoted, {"candidates": []})
    _write_json(
        pairs,
        {
            "candidates": [
                {
                    "signature": "a->b",
                    "op_a": "a",
                    "op_b": "b",
                    "composition": "sequential",
                    "stability_score": 0.1,
                    "ar_binding_overlay": _overlay(holdout_required=False),
                }
            ]
        },
    )

    payload = build_holdout_queue(
        promoted_templates_path=promoted,
        validated_templates_path=tmp_path / "missing_validated.json",
        pair_proposals_path=pairs,
        created_at=1.0,
    )

    item = payload["items"][0]
    assert item["candidate_type"] == "pair"
    assert item["status"] == "needs_motif_schema"
    assert "motif_schema_required" in item["blockers"]


def test_write_holdout_queue_round_trips(tmp_path: Path):
    payload = {"metadata": {"status_counts": {}}, "items": []}
    out = write_holdout_queue(payload, tmp_path / "queue.json")
    assert json.loads(out.read_text()) == payload
