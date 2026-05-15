"""Composite scoring + leaderboard for proposed components.

A scorecard from ``solo.validate_solo`` (and optionally
``in_context.validate_in_context``) is reduced to a single composite
score so promoted candidates can be ranked.

The composite weights:
- 30% smoke (all checks pass = 1.0, else 0.0)
- 30% cross-check pass ratio (fraction of declared properties matched)
- 40% learning signal (log10 of probe loss-ratio, clamped to [0, 1])

These weights are intentional starting points; tune as the catalog grows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

_SMOKE_KEYS_REQUIRED = (
    "forward_passed",
    "backward_passed",
    "output_finite",
    "param_grad_finite",
)


@dataclass(frozen=True, slots=True)
class RankedEntry:
    proposal_id: str
    name: str
    category: str
    synthesis_kind: str
    composite_score: float
    components: dict[str, float]
    promoted: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def smoke_subscore(smoke: dict[str, Any]) -> float:
    return 1.0 if all(smoke.get(key) for key in _SMOKE_KEYS_REQUIRED) else 0.0


def cross_check_subscore(cross: dict[str, Any]) -> float:
    consistent_keys = [k for k in cross if k.endswith("_consistent")]
    if not consistent_keys:
        return 1.0
    passed = sum(1 for k in consistent_keys if cross.get(k) is True)
    return passed / len(consistent_keys)


def learning_subscore(probe_scorecard: dict[str, Any] | None) -> float:
    """Read aggregate_loss_ratio off the multi-task in-context scorecard."""
    if not probe_scorecard:
        return 0.0
    ratio = float(probe_scorecard.get("aggregate_loss_ratio") or 0.0)
    if ratio <= 1.0:
        return 0.0
    return min(1.0, math.log10(ratio) / 2.0)


def composite_score(
    solo_scorecard: dict[str, Any],
    probe_scorecard: dict[str, Any] | None = None,
    *,
    smoke_weight: float = 0.3,
    cross_weight: float = 0.3,
    learn_weight: float = 0.4,
) -> tuple[float, dict[str, float]]:
    smoke = smoke_subscore(solo_scorecard.get("smoke", {}))
    cross = cross_check_subscore(solo_scorecard.get("property_cross_check", {}))
    learn = learning_subscore(probe_scorecard)
    components = {
        "smoke": smoke,
        "cross_check": cross,
        "learning": learn,
    }
    score = smoke_weight * smoke + cross_weight * cross + learn_weight * learn
    return score, components


def rank_proposals(
    solo_scorecards: Sequence[dict[str, Any]],
    probe_scorecards_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[RankedEntry]:
    probe_map = probe_scorecards_by_id or {}
    out: list[RankedEntry] = []
    for solo in solo_scorecards:
        proposal_id = str(solo.get("proposal_id") or "")
        probe = probe_map.get(proposal_id)
        score, components = composite_score(solo, probe)
        out.append(
            RankedEntry(
                proposal_id=proposal_id,
                name=str(solo.get("name") or ""),
                category=str(solo.get("category") or ""),
                synthesis_kind=str(solo.get("synthesis_kind") or ""),
                composite_score=score,
                components=components,
                promoted=bool(solo.get("promoted")),
                notes=tuple(solo.get("notes") or ()),
            )
        )
    out.sort(key=lambda e: e.composite_score, reverse=True)
    return out


def leaderboard_to_json(ranked: Iterable[RankedEntry]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index,
            "proposal_id": entry.proposal_id,
            "name": entry.name,
            "category": entry.category,
            "synthesis_kind": entry.synthesis_kind,
            "composite_score": round(entry.composite_score, 4),
            "components": {k: round(v, 4) for k, v in entry.components.items()},
            "promoted": entry.promoted,
            "notes": list(entry.notes),
        }
        for index, entry in enumerate(ranked, start=1)
    ]
