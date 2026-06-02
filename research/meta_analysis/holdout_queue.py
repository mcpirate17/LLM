"""Build holdout/triage queues from AR/binding proposal registries.

The queue is the hand-off between advisory mining artifacts and live grammar
changes. It keeps risky-but-interesting candidates out of production template
registration until they pass structural validation and small holdout runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


DEFAULT_PROMOTED_TEMPLATES = Path(
    "research/data/synthesis_candidates/promoted_template_candidates.json"
)
DEFAULT_VALIDATED_TEMPLATES = Path(
    "research/data/synthesis_candidates/validated_template_candidates.json"
)
DEFAULT_PAIR_PROPOSALS = Path(
    "research/data/synthesis_candidates/untapped_pair_proposals.json"
)
DEFAULT_OUTPUT = Path(
    "research/data/synthesis_candidates/ar_binding_holdout_queue.json"
)


def build_holdout_queue(
    *,
    promoted_templates_path: str | Path = DEFAULT_PROMOTED_TEMPLATES,
    validated_templates_path: str | Path = DEFAULT_VALIDATED_TEMPLATES,
    pair_proposals_path: str | Path = DEFAULT_PAIR_PROPOSALS,
    max_pairs: int = 50,
    created_at: float | None = None,
) -> dict[str, Any]:
    """Return a structured queue for template and pair follow-up work."""
    validated_templates = _load_validated_template_candidates(
        promoted_templates_path=Path(promoted_templates_path),
        validated_templates_path=Path(validated_templates_path),
    )
    pair_candidates = _load_pair_candidates(
        Path(pair_proposals_path), max_pairs=max_pairs
    )

    template_items = [_template_queue_item(c) for c in validated_templates]
    pair_items = [_pair_queue_item(c) for c in pair_candidates]
    items = sorted(
        template_items + pair_items,
        key=lambda item: (
            _status_rank(str(item["status"])),
            -float(item["priority_score"]),
            str(item["candidate_id"]),
        ),
    )
    return {
        "metadata": {
            "created_at": time.time() if created_at is None else float(created_at),
            "promoted_templates_source": str(promoted_templates_path),
            "validated_templates_source": str(validated_templates_path),
            "pair_proposals_source": str(pair_proposals_path),
            "max_pairs": int(max_pairs),
            "n_templates": len(template_items),
            "n_pairs": len(pair_items),
            "status_counts": _status_counts(items),
        },
        "items": items,
    }


def write_holdout_queue(payload: dict[str, Any], output_path: str | Path) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _load_validated_template_candidates(
    *,
    promoted_templates_path: Path,
    validated_templates_path: Path,
) -> list[dict[str, Any]]:
    if validated_templates_path.exists():
        payload = json.loads(validated_templates_path.read_text(encoding="utf-8"))
        candidates = payload.get("candidates") or []
        return [c for c in candidates if isinstance(c, dict)]
    if not promoted_templates_path.exists():
        return []
    payload = json.loads(promoted_templates_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates") or []
    return [c for c in candidates if isinstance(c, dict)]


def _load_pair_candidates(path: Path, *, max_pairs: int) -> list[dict[str, Any]]:
    if not path.exists() or max_pairs <= 0:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates") or []
    return [c for c in candidates if isinstance(c, dict)][:max_pairs]


def _template_queue_item(candidate: dict[str, Any]) -> dict[str, Any]:
    overlay = dict(candidate.get("ar_binding_overlay") or {})
    validation = dict(candidate.get("validation") or {})
    validate_passed = bool(validation.get("validate_passed"))
    compile_passed = bool(validation.get("compile_passed"))
    backward_passed = bool(validation.get("backward_passed"))
    holdout_required = bool(overlay.get("holdout_required", True))

    blockers: list[str] = []
    if not validate_passed:
        blockers.append("structural_validation_failed")
    if not compile_passed:
        blockers.append("compile_validation_failed")
    if compile_passed and not backward_passed:
        blockers.append("runtime_smoke_failed")
    if holdout_required:
        blockers.append("overlay_holdout_required")

    if compile_passed and backward_passed and holdout_required:
        status = "ready_for_holdout"
        next_step = "run_small_training_holdout"
    elif compile_passed and backward_passed:
        status = "registration_candidate"
        next_step = "human_review_then_feature_flag_registration"
    else:
        status = "blocked_structural"
        next_step = "adapt_code_skeleton_for_multi_input_or_shape_requirements"

    return {
        "candidate_type": "template",
        "candidate_id": str(candidate.get("proposed_template_name") or ""),
        "status": status,
        "priority_score": _priority_score(candidate, overlay),
        "chain": list(candidate.get("chain") or []),
        "overlay": overlay,
        "validation": validation,
        "blockers": blockers,
        "next_step": next_step,
    }


def _pair_queue_item(candidate: dict[str, Any]) -> dict[str, Any]:
    overlay = dict(candidate.get("ar_binding_overlay") or {})
    blockers = ["motif_schema_required", "holdout_generation_required"]
    if overlay.get("holdout_required", True):
        blockers.append("overlay_holdout_required")
    return {
        "candidate_type": "pair",
        "candidate_id": str(candidate.get("signature") or ""),
        "status": "needs_motif_schema",
        "priority_score": _priority_score(candidate, overlay),
        "composition": candidate.get("composition"),
        "op_a": candidate.get("op_a"),
        "op_b": candidate.get("op_b"),
        "overlay": overlay,
        "stability_score": candidate.get("stability_score"),
        "blockers": blockers,
        "next_step": "define_motif_class_and_activation_rules_then_holdout",
    }


def _priority_score(candidate: dict[str, Any], overlay: dict[str, Any]) -> float:
    promotion = _finite(candidate.get("promotion_score"))
    stability = _finite(candidate.get("stability_score"))
    ar_gain = _finite(overlay.get("expected_ar_gain")) or 0.0
    binding_gain = _finite(overlay.get("expected_binding_gain")) or 0.0
    retention_risk = _finite(overlay.get("retention_risk")) or 0.0
    collapse_risk = _finite(overlay.get("collapse_risk")) or 0.0
    base = (
        promotion if promotion is not None else 1.0 / (1.0 + max(stability or 0.0, 0.0))
    )
    return round(
        max(0.0, base + ar_gain + binding_gain - retention_risk - collapse_risk), 6
    )


def _status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _status_rank(status: str) -> int:
    order = {
        "registration_candidate": 0,
        "ready_for_holdout": 1,
        "needs_motif_schema": 2,
        "blocked_structural": 3,
    }
    return order.get(status, 9)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None
