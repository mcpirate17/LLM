"""Composite scoring + leaderboard for proposed components.

A scorecard from ``solo.validate_solo`` (and optionally
``in_context.validate_in_context`` + ``capability.validate_capabilities``)
is reduced to a single composite score so promoted candidates can be
ranked.

The composite weights (post-2026-05-15 rebalance toward binding +
induction, the dominant signals for downstream BLiMP wins; see
``_BINDING_WEIGHTS`` — the single source of truth):
- 30% smoke (all checks pass = 1.0, else 0.0)
- 30% cross-check pass ratio (fraction of declared properties matched)
- 10% learning signal (log10 of probe loss-ratio, clamped to [0, 1])
- 30% binding signal (capability AR-probe binds + relative_recall mean)

The binding subscore replaces the ad-hoc ``+0.2 if can_bind`` bonus that
``_grade_active_specs`` applied on top of the old 3-component composite.
Passing ``capability_scorecard=None`` reproduces the legacy 0/0.3/0.3/0.4
weighting so older callers still work.

Two additional stability-gated bonuses reward the user's "maximum mixing" and
"minimum steps" objectives: ``_MIXING_BONUS`` (intrinsic reach+breadth) and
``_LEARN_SPEED_BONUS`` (fewer steps to cross the AR recall bar). Both are zero
unless the candidate clears ``STABILITY_FLOOR`` on some capability axis, so a
broad mixer that cannot bind/induce never wins on mixing alone. They are also
first-class Pareto objectives (``mixing``, ``learning_speed`` in
``OBJECTIVE_KEYS``).
"""

from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from component_fab.proposer.tier2_feedback import (
    Tier2Feedback,
    tier2_score_multiplier,
)
from component_fab.proposer.nas_screen import NasScreenResult, nas_score_multiplier
from component_fab.harness.state_tracking_suite import AXES as _STATE_TRACK_AXES

# SSM-favoured probe tasks (state-tracking + copy/compression) — the axis the
# binding composite is blind to. The 2026-06-07 SSM-fair cohort showed non-QKV
# mechanisms (routing / compression / selective-scan) Pareto-beat attention here
# while attention keeps recall/induction. These task names index the per-task
# loss ratios already in the in_context probe scorecard, so the subscore is free.
_SSM_FAVOURED_TASKS: tuple[str, ...] = (
    _STATE_TRACK_AXES["state_tracking"] + _STATE_TRACK_AXES["copy_compression"]
)
# Additive composite bonus for SSM-favoured state-tracking (modest: the cohort
# separation is real but ~9% at the axis top — nudge promotion, don't dominate).
_STATE_TRACK_BONUS = 0.15
# "Maximum mixing" + "minimum steps" bonuses (2026-07-01). Stability-gated like
# the orthogonality lift: a broad/fast lane that binds nothing earns no credit.
_MIXING_BONUS = 0.12
_LEARN_SPEED_BONUS = 0.10
# Reference budget for the speed map: the largest DEFAULT AR probe trains for 60
# steps, so reaching the recall bar at step s scores 1 - s/60.
_LEARN_SPEED_REF_STEPS = 60.0

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


def state_tracking_subscore(probe_scorecard: dict[str, Any] | None) -> float:
    """Mean SSM-favoured loss-reduction off the in_context probe ``per_task``.

    Reads the per-task loss ratios already computed by ``validate_in_context``,
    keeps only the SSM-favoured tasks (state-tracking + copy/compression — the
    axis the binding composite ignores), and maps the mean ratio through the
    same ``log10(ratio)/2`` clamp as ``learning_subscore``. Returns 0.0 when the
    scorecard, ``per_task``, or trained tasks are missing. Free: no new compute.
    """
    if not probe_scorecard:
        return 0.0
    per_task = probe_scorecard.get("per_task") or {}
    ratios = [
        float(per_task[name].get("loss_ratio_initial_over_final") or 0.0)
        for name in _SSM_FAVOURED_TASKS
        if name in per_task and per_task[name].get("trained_successfully")
    ]
    if not ratios:
        return 0.0
    mean_ratio = sum(ratios) / len(ratios)
    if mean_ratio <= 1.0:
        return 0.0
    return min(1.0, math.log10(mean_ratio) / 2.0)


def binding_subscore(capability_scorecard: dict[str, Any] | None) -> float:
    """Average of AR-probe binds + mean relative_recall, clipped to [0, 1].

    Skips degenerate cards (eliminated by an earlier gate, or no probes).
    Each AR probe contributes a 1.0 if it bound (``binds_per_probe[name]``
    is True) and its ``relative_recall_per_probe[name]`` clipped to [0, 1].
    Final score is the mean over (2 × n_probes) entries — gives equal
    weight to a clean bind and to the recall margin.
    """
    if not capability_scorecard:
        return 0.0
    binds = capability_scorecard.get("binds_per_probe") or {}
    recalls = capability_scorecard.get("relative_recall_per_probe") or {}
    if not binds and not recalls:
        return 0.0
    bind_mean = sum(1.0 if v else 0.0 for v in binds.values()) / max(1, len(binds))
    recall_mean = sum(max(0.0, min(1.0, float(v))) for v in recalls.values()) / max(
        1, len(recalls)
    )
    return 0.5 * bind_mean + 0.5 * recall_mean


def mixing_subscore(capability_scorecard: dict[str, Any] | None) -> float:
    """Intrinsic reach+breadth mixing signal off the capability scorecard.

    Already measured at init-time (no training) by ``validate_capabilities``
    via :func:`measure_mixing_quality`. Returns 0.0 for eliminated/missing
    cards. A pure-mixer earns this only behind ``STABILITY_FLOOR`` (composite)
    or the stability gate (objective vector).
    """
    if not capability_scorecard:
        return 0.0
    return max(0.0, min(1.0, float(capability_scorecard.get("mixing_subscore") or 0.0)))


def learning_speed_subscore(capability_scorecard: dict[str, Any] | None) -> float:
    """How few steps a lane needs to cross the AR recall bar (higher = faster).

    Reads ``mean_steps_to_threshold`` (mean first-to-threshold checkpoint step
    across AR probes; ``None`` when no probe reaches the bar) and maps it through
    ``1 - mean/REF`` so a lane that binds in a handful of steps scores near 1 and
    one that needs the full budget scores near 0. Scores the user's "minimum
    steps" objective directly. 0.0 for eliminated/missing cards or non-learners.
    """
    if not capability_scorecard:
        return 0.0
    mean_steps = capability_scorecard.get("mean_steps_to_threshold")
    if mean_steps is None:
        return 0.0
    return max(0.0, 1.0 - float(mean_steps) / _LEARN_SPEED_REF_STEPS)


def orthogonality_subscore(solo_scorecard: dict[str, Any]) -> float:
    """Orthogonality signal (distance from state degeneracy with clones + baselines).

    The radius (min z-scored Euclidean distance to the ledger catalog spectra) is
    computed and attached by ``annotate_niche_metadata`` in
    ``runner/niche.py``. ``composite_score`` only adds the lift when
    the candidate clears ``meets_stability_floor`` — so non-functional states
    never score for distinctness alone.
    """
    radius = float(
        solo_scorecard.get("metadata", {}).get("orthogonality_radius") or 0.0
    )
    return min(1.0, radius / 4.0)  # radius [0, 4+] -> [0, 1]


_LEGACY_WEIGHTS = (0.3, 0.3, 0.4, 0.0)
# Day-6 reweight (2026-05-15): align weights with sequence-mixing capability.
_BINDING_WEIGHTS = (0.3, 0.3, 0.1, 0.3)
# Orthogonality lift: reward mechanistic distinctness, but ONLY for functional models
# (gated by the shared stability floor below).
_ORTHOGONALITY_LIFT = 0.20

# Single source of truth for the orthogonality stability floor, shared by the scalar
# composite bonus and the Pareto orthogonality objective. A candidate must clear it on
# *some* capability axis before its distinctness earns any credit.
STABILITY_FLOOR = 0.05


def meets_stability_floor(
    *, binding: float, induction: float, learning: float, state_tracking: float
) -> bool:
    """True if any capability axis clears ``STABILITY_FLOOR``."""
    return max(binding, induction, learning, state_tracking) >= STABILITY_FLOOR


def composite_score(
    solo_scorecard: dict[str, Any],
    probe_scorecard: dict[str, Any] | None = None,
    capability_scorecard: dict[str, Any] | None = None,
    *,
    weights: tuple[float, float, float, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Weighted sum over (smoke, cross_check, learning, binding) subscores.

    Defaults to ``_BINDING_WEIGHTS`` (0.3/0.3/0.1/0.3) when a capability
    scorecard is supplied.
    """
    if weights is None:
        weights = _BINDING_WEIGHTS if capability_scorecard else _LEGACY_WEIGHTS
    smoke_w, cross_w, learn_w, bind_w = weights
    smoke = smoke_subscore(solo_scorecard.get("smoke", {}))
    cross = cross_check_subscore(solo_scorecard.get("property_cross_check", {}))
    learn = learning_subscore(probe_scorecard)
    bind = binding_subscore(capability_scorecard)
    state_track = state_tracking_subscore(probe_scorecard)
    orthogonality = orthogonality_subscore(solo_scorecard)
    mixing = mixing_subscore(capability_scorecard)
    learn_speed = learning_speed_subscore(capability_scorecard)

    components = {
        "smoke": smoke,
        "cross_check": cross,
        "learning": learn,
        "binding": bind,
        "state_tracking": state_track,
        "orthogonality": orthogonality,
        "mixing": mixing,
        "learning_speed": learn_speed,
    }
    score = smoke_w * smoke + cross_w * cross + learn_w * learn + bind_w * bind

    # Additive bonuses
    score += _STATE_TRACK_BONUS * state_track

    # Degeneracy penalty: reward mechanistic distinctness (orthogonality), plus
    # the mixing/learning-speed bonuses — all gated by the stability floor so a
    # broad or fast lane that binds/induces nothing earns no credit.
    induction = float((capability_scorecard or {}).get("ind_max_accuracy") or 0.0)
    if meets_stability_floor(
        binding=bind, induction=induction, learning=learn, state_tracking=state_track
    ):
        score += _ORTHOGONALITY_LIFT * orthogonality
        score += _MIXING_BONUS * mixing
        score += _LEARN_SPEED_BONUS * learn_speed

    return score, components


def rank_proposals(
    solo_scorecards: Sequence[dict[str, Any]],
    probe_scorecards_by_id: dict[str, dict[str, Any]] | None = None,
    capability_scorecards_by_id: dict[str, dict[str, Any]] | None = None,
    tier2_feedback_by_id: dict[str, Tier2Feedback] | None = None,
    nas_screen_by_id: dict[str, NasScreenResult] | None = None,
) -> list[RankedEntry]:
    probe_map = probe_scorecards_by_id or {}
    cap_map = capability_scorecards_by_id or {}
    tier2_map = tier2_feedback_by_id or {}
    nas_map = nas_screen_by_id or {}
    out: list[RankedEntry] = []
    for solo in solo_scorecards:
        proposal_id = str(solo.get("proposal_id") or "")
        probe = probe_map.get(proposal_id)
        capability = cap_map.get(proposal_id)
        score, components = composite_score(solo, probe, capability)
        feedback = tier2_map.get(proposal_id)
        multiplier = tier2_score_multiplier(feedback)
        if multiplier != 1.0:
            score *= multiplier
            components = dict(components)
            components["tier2_multiplier"] = multiplier
        nas_multiplier = nas_score_multiplier(nas_map.get(proposal_id))
        if nas_multiplier != 1.0:
            score *= nas_multiplier
            components = dict(components)
            components["nas_multiplier"] = nas_multiplier
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


# --------------------------------------------------------------------------- #
# WS-4: Pareto / niche objectives (multi-objective, no scalar collapse)
# --------------------------------------------------------------------------- #
# Objective vector — ALL dimensions "higher is better".
OBJECTIVE_KEYS: tuple[str, ...] = (
    "binding",
    "induction",
    "learning",
    "state_tracking",
    "orthogonality",
    "mixing",
    "learning_speed",
    "efficiency",
)


def objective_vector(
    probe_scorecard: dict[str, Any] | None = None,
    capability_scorecard: dict[str, Any] | None = None,
    *,
    param_count: int = 0,
    orthogonality: float = 0.0,
) -> dict[str, float]:
    """Build the promotion-time objective vector for Pareto/niche ranking.

    Every returned dimension is maximized. ``efficiency`` is stored as
    ``-param_count``.

    ``orthogonality`` is gated by ``STABILITY_FLOOR``: a candidate with no
    measurable capability gets zero credit, so it cannot ride the orthogonality
    axis onto the Pareto front.
    """
    cap = capability_scorecard or {}
    binding = binding_subscore(capability_scorecard)
    induction = float(cap.get("ind_max_accuracy") or 0.0)
    learning = learning_subscore(probe_scorecard)
    state_tracking = state_tracking_subscore(probe_scorecard)
    mixing = mixing_subscore(capability_scorecard)
    learn_speed = learning_speed_subscore(capability_scorecard)
    stable = meets_stability_floor(
        binding=binding,
        induction=induction,
        learning=learning,
        state_tracking=state_tracking,
    )
    return {
        "binding": binding,
        "induction": induction,
        "learning": learning,
        "state_tracking": state_tracking,
        "orthogonality": float(orthogonality) if stable else 0.0,
        "mixing": mixing if stable else 0.0,
        "learning_speed": learn_speed if stable else 0.0,
        "efficiency": -float(param_count),
    }


def _as_objective_list(vec: dict[str, float]) -> tuple[float, ...]:
    return tuple(float(vec.get(k, 0.0)) for k in OBJECTIVE_KEYS)


def non_dominated_sort(vectors: Sequence[dict[str, float]]) -> list[int]:
    """Assign each vector a Pareto front index, where ``0`` is non-dominated."""
    if not vectors:
        return []
    arrs = np.asarray([_as_objective_list(v) for v in vectors], dtype=float)
    fronts = [-1] * len(arrs)
    remaining = np.arange(len(arrs))
    front = 0
    while remaining.size:
        sub = arrs[remaining]
        # dominates[j, i]: j >= i on all objectives and > on at least one
        dominates = (sub[:, None, :] >= sub[None, :, :]).all(-1) & (
            sub[:, None, :] > sub[None, :, :]
        ).any(-1)
        dominated = dominates.any(axis=0)
        if dominated.all():  # numerical safety — should be unreachable
            dominated[:] = False
        for i in remaining[~dominated]:
            fronts[int(i)] = front
        remaining = remaining[dominated]
        front += 1
    return fronts


def pareto_front_indices(vectors: Sequence[dict[str, float]]) -> list[int]:
    """Indices of the non-dominated (front-0) objective vectors."""
    fronts = non_dominated_sort(vectors)
    return [i for i, f in enumerate(fronts) if f == 0]


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
