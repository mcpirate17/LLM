"""Hypothesis preregistration models and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


class PreregistrationError(ValueError):
    """Raised when preregistration payload is incomplete."""


@dataclass
class HypothesisPreregistration:
    hypothesis: Dict[str, Any]
    analysis_plan: Dict[str, Any]
    falsification_conditions: List[str] = field(default_factory=list)
    confounders_checklist: List[Dict[str, Any]] = field(default_factory=list)
    exploratory: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "analysis_plan": self.analysis_plan,
            "falsification_conditions": self.falsification_conditions,
            "confounders_checklist": self.confounders_checklist,
            "exploratory": self.exploratory,
        }


def validate_preregistration(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise PreregistrationError("Preregistration must be a dict.")

    required_top = [
        "hypothesis",
        "analysis_plan",
        "falsification_conditions",
        "confounders_checklist",
    ]
    for key in required_top:
        if key not in payload:
            raise PreregistrationError(f"Missing preregistration field: {key}")

    hypothesis = payload.get("hypothesis") or {}
    for key in ("statement", "variables", "expected_direction", "success_criteria"):
        if key not in hypothesis:
            raise PreregistrationError(f"Missing hypothesis field: {key}")

    analysis = payload.get("analysis_plan") or {}
    for key in (
        "primary_metrics",
        "secondary_metrics",
        "thresholds",
        "baseline_comparison",
    ):
        if key not in analysis:
            raise PreregistrationError(f"Missing analysis_plan field: {key}")

    if (
        not isinstance(payload.get("falsification_conditions"), list)
        or not payload["falsification_conditions"]
    ):
        raise PreregistrationError("falsification_conditions must be a non-empty list.")
    if (
        not isinstance(payload.get("confounders_checklist"), list)
        or not payload["confounders_checklist"]
    ):
        raise PreregistrationError("confounders_checklist must be a non-empty list.")
