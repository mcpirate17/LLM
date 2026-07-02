"""Math-knob catalog and composition enumerator.

This layer is where component_fab stops treating math choices as one
exclusive label. A knob is a concrete mechanism that can be stacked onto
an anchor lane: calculus features, low-rank linear algebra adapters, sparse
matrix preconditioning, and future families.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from ..proposer.spec_generator import ProposalSpec, build_spec_from_axes
from ..state.axis_lift import AxisLiftReport, load_axis_lift
from ..state.ledger import (
    Ledger,
    LedgerEntry,
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
)
from .axis_variants import DEFAULT_META_DB, AnchorAxes, anchor_axes_for_op
from .math_knobs import AUTO_DEEPENING_MATH_KNOBS, DEFAULT_MATH_KNOBS, MathKnob

__all__ = [
    "DEFAULT_MATH_KNOBS",
    "KnobStackScore",
    "MathKnob",
    "auto_deepen_math_knobs",
    "enumerate_adaptive_math_knob_compositions",
    "enumerate_math_knob_compositions",
    "score_knob_stack",
]


@dataclass(frozen=True, slots=True)
class KnobStackScore:
    knob_ids: tuple[str, ...]
    score: float
    attempts: int
    rejected: bool
    reason: str


_DEFAULT_KNOB_IDS = tuple(knob.knob_id for knob in DEFAULT_MATH_KNOBS)


def auto_deepen_math_knobs(
    knobs: Sequence[MathKnob],
    *,
    siblings: Sequence[MathKnob] = AUTO_DEEPENING_MATH_KNOBS,
    max_new: int = 12,
) -> tuple[MathKnob, ...]:
    """Add natural sibling knobs for families represented by exactly one knob.

    NM-12 is a proposer meta-rule: when a cycle would otherwise see a family as
    a one-operator alias, emit bounded sibling axis values for that family. The
    generator still owns buildability; these siblings are aliases over the
    existing family lane classes, but they give the search distinct cells to try
    and measure instead of relying on hand-curated variants.
    """
    if max_new <= 0:
        return tuple(knobs)
    by_family: dict[str, list[MathKnob]] = {}
    seen_ids = {knob.knob_id for knob in knobs}
    for knob in knobs:
        by_family.setdefault(knob.family, []).append(knob)

    out = list(knobs)
    added = 0
    for sibling in siblings:
        if sibling.knob_id in seen_ids:
            continue
        family_members = by_family.get(sibling.family, [])
        if len(family_members) != 1:
            continue
        source = sibling.axes.get("op_math_deepening_source")
        if source and source != family_members[0].knob_id:
            continue
        out.append(sibling)
        seen_ids.add(sibling.knob_id)
        added += 1
        if added >= max_new:
            break
    return tuple(out)


def _knobs_from_name(
    name: str, knob_ids: Sequence[str] = _DEFAULT_KNOB_IDS
) -> tuple[str, ...]:
    found = [knob_id for knob_id in knob_ids if knob_id in name]
    return tuple(sorted(found))


def _knobs_from_metadata(metadata: dict[str, Any]) -> tuple[str, ...]:
    # NOTE: two knob vocabularies meet here. ``math_knobs`` may carry
    # DEFAULT_MATH_KNOBS ids (calculus_finite_difference, ...) or the
    # axis_variants/dynamic literals (info_geom_fisher, spectral_chebyshev,
    # tensor_tucker) that code_generator dispatches on. Unification is
    # deferred — code_generator owns those literals.
    raw = metadata.get("math_knobs")
    if isinstance(raw, str):
        return tuple(sorted(part for part in raw.split("+") if part))
    if isinstance(raw, (list, tuple)):
        return tuple(sorted(str(part) for part in raw if str(part)))
    return ()


def _metadata_score(metadata: dict[str, Any]) -> float:
    eliminated_by = metadata.get("eliminated_by")
    if eliminated_by:
        if eliminated_by == "s05_causality_stability":
            return -0.6
        if eliminated_by == "erf_density":
            return -0.35
        if eliminated_by == "nano_bind":
            return -0.2
        return -0.1
    # WS-1 gate calibration (2026-06-10, catalog/gate_calibration.json):
    # among survivors erf_density is anti-predictive (residual AUC 0.420)
    # and can_bind is noise (0.481); only nb_max_accuracy weakly tracks the
    # learned label. Positive evidence therefore reads nb_max_accuracy
    # alone, capped so score_knob_stack magnitudes stay in the historical
    # ballpark.
    nb_max_accuracy = float(metadata.get("nb_max_accuracy") or 0.0)
    return 0.08 + min(0.34, max(0.0, nb_max_accuracy) * 0.4)


def _entry_capability_score(entry: LedgerEntry) -> float:
    if not entry.metadata_history:
        return 0.0
    return max(_metadata_score(metadata) for metadata in entry.metadata_history)


def _stack_stats(
    ledger: Ledger | None, knob_ids: Sequence[str] = _DEFAULT_KNOB_IDS
) -> dict[tuple[str, ...], list[LedgerEntry]]:
    if ledger is None:
        return {}
    out: dict[tuple[str, ...], list[LedgerEntry]] = {}
    for entry in ledger.all_entries():
        stack = ()
        for metadata in reversed(entry.metadata_history):
            stack = _knobs_from_metadata(metadata)
            if stack:
                break
        if not stack:
            stack = _knobs_from_name(entry.name, knob_ids=knob_ids)
        if not stack:
            continue
        out.setdefault(stack, []).append(entry)
    return out


def _axis_lift_bonus(
    knob_ids: tuple[str, ...], axis_lift: AxisLiftReport | None
) -> float:
    """Mean per-knob lift over the global pass rate, mapped to a small bonus.

    Returns 0 when ``axis_lift`` is missing or none of the knobs have a row.
    Otherwise: mean_lift - 1.0, capped to [-0.5, 1.0]. A multiscale_wavelet-
    style 8x lift contributes the cap; a 0.4x knob contributes -0.5.
    """
    if axis_lift is None or not knob_ids:
        return 0.0
    by_value = {r.value: r.lift for r in axis_lift.by_axis.get("math_knob", [])}
    lifts = [by_value[k] for k in knob_ids if k in by_value]
    if not lifts:
        return 0.0
    mean_lift = sum(lifts) / len(lifts)
    delta = mean_lift - 1.0
    return max(-0.5, min(1.0, delta))


def score_knob_stack(
    knob_ids: tuple[str, ...],
    ledger: Ledger | None,
    *,
    reject_below: float = 0.35,
    axis_lift: AxisLiftReport | None = None,
    stats: dict[tuple[str, ...], list[LedgerEntry]] | None = None,
) -> KnobStackScore:
    """Score a knob stack from exact and subset ledger history.

    The score is intentionally simple and auditable:
    - exact promoted stacks get a strong boost
    - exact rejected stacks below ``reject_below`` are skipped
    - otherwise, exact history dominates, with subset history as exploration prior
    - when an ``axis_lift`` report is provided, per-knob lift mean folds
      into the score so knobs with empirical lift over the global pass rate
      get sampled more often (and underperformers get sampled less).

    ``stats`` lets batch callers precompute ``_stack_stats(ledger)`` once;
    it is a full ledger scan, so recomputing it per candidate stack is
    O(specs x ledger).
    """
    if stats is None:
        stats = _stack_stats(ledger)
    exact = stats.get(tuple(sorted(knob_ids)), [])
    exact_attempts = len(exact)
    exact_best = max((entry.best_composite() for entry in exact), default=0.0)
    exact_capability = max(
        (_entry_capability_score(entry) for entry in exact), default=0.0
    )
    exact_promoted = any(
        entry.promotion_status == PROMOTION_PROMOTED for entry in exact
    )
    exact_rejected = bool(exact) and all(
        entry.promotion_status == PROMOTION_REJECTED for entry in exact
    )
    if exact_rejected and exact_best <= reject_below:
        return KnobStackScore(
            knob_ids=knob_ids,
            score=-1.0,
            attempts=exact_attempts,
            rejected=True,
            reason=f"exact stack rejected below {reject_below}",
        )

    subset_scores: list[float] = []
    for size in range(1, len(knob_ids)):
        for subset in combinations(tuple(sorted(knob_ids)), size):
            entries = stats.get(subset, [])
            if entries:
                subset_scores.append(max(entry.best_composite() for entry in entries))
    subset_prior = sum(subset_scores) / len(subset_scores) if subset_scores else 0.0
    novelty_bonus = 0.05 if not exact else 0.0
    depth_penalty = 0.02 * max(0, len(knob_ids) - 1)
    promotion_bonus = 0.25 if exact_promoted else 0.0
    lift_bonus = _axis_lift_bonus(knob_ids, axis_lift)
    score = (
        max(exact_best, subset_prior * 0.75)
        + exact_capability
        + novelty_bonus
        + promotion_bonus
        + lift_bonus
    )
    score -= depth_penalty
    reason = "exact history" if exact else "subset prior" if subset_scores else "novel"
    if axis_lift is not None and lift_bonus != 0.0:
        reason = f"{reason} + axis_lift({lift_bonus:+.2f})"
    return KnobStackScore(
        knob_ids=knob_ids,
        score=score,
        attempts=exact_attempts,
        rejected=False,
        reason=reason,
    )


def _spec_for_knobs(anchor: AnchorAxes, knobs: tuple[MathKnob, ...]) -> ProposalSpec:
    knob_ids = tuple(knob.knob_id for knob in knobs)
    axes = dict(anchor.axes)
    axes["op_math_knobs"] = "+".join(knob_ids)
    axes["op_math_family"] = "composite" if len(knobs) > 1 else knobs[0].family
    for knob in knobs:
        axes.update(knob.axes)
    name = f"compose_{anchor.op_name}_{'__'.join(knob_ids)}"
    notes = tuple(
        [f"anchor={anchor.op_name}", f"math_knobs={'+'.join(knob_ids)}"]
        + [knob.rationale for knob in knobs]
    )
    # This path historically emits math_axes WITHOUT the mirrored
    # synthesis_kind and fingerprints the pre-dispatch axes — preserved
    # for proposal_id stability.
    return build_spec_from_axes(
        name,
        axes,
        witness_ops=(anchor.op_name,),
        anchor_axes=anchor.axes,
        notes=notes,
        dispatched_math_axes=False,
    )


def enumerate_math_knob_compositions(
    anchor_op_names: Sequence[str],
    *,
    knobs: Sequence[MathKnob] = DEFAULT_MATH_KNOBS,
    min_depth: int = 1,
    max_depth: int = 3,
    db_path: Path | str = DEFAULT_META_DB,
) -> list[ProposalSpec]:
    """Generate specs for every compatible knob stack on each anchor."""
    if min_depth <= 0:
        raise ValueError("min_depth must be positive")
    if max_depth < min_depth:
        raise ValueError("max_depth must be >= min_depth")
    max_depth = min(max_depth, len(knobs))

    anchors: list[AnchorAxes] = []
    for name in anchor_op_names:
        anchor = anchor_axes_for_op(name, db_path=db_path)
        if anchor is not None:
            anchors.append(anchor)

    specs: list[ProposalSpec] = []
    for anchor in anchors:
        for depth in range(min_depth, max_depth + 1):
            for combo in combinations(tuple(knobs), depth):
                specs.append(_spec_for_knobs(anchor, combo))
    return specs


def enumerate_adaptive_math_knob_compositions(
    anchor_op_names: Sequence[str],
    ledger: Ledger,
    *,
    knobs: Sequence[MathKnob] = DEFAULT_MATH_KNOBS,
    min_depth: int = 1,
    max_depth: int = 3,
    max_specs: int = 48,
    db_path: Path | str = DEFAULT_META_DB,
    axis_lift: AxisLiftReport | None = None,
    include_auto_deepening: bool = True,
    max_auto_deepened_knobs: int = 13,
) -> list[ProposalSpec]:
    """Generate knob specs with ledger-guided pruning and ranking.

    When ``axis_lift`` is omitted the loader auto-discovers
    ``component_fab/catalog/axis_lift.json`` (refreshed by
    ``research/tools/fab_daily_loop.py`` via ``state.axis_lift``). Pass an
    empty report explicitly if disabling is needed.
    """
    if axis_lift is None:
        axis_lift = load_axis_lift()
    if include_auto_deepening:
        knobs = auto_deepen_math_knobs(knobs, max_new=max_auto_deepened_knobs)
    specs = enumerate_math_knob_compositions(
        anchor_op_names,
        knobs=knobs,
        min_depth=min_depth,
        max_depth=max_depth,
        db_path=db_path,
    )
    if max_specs <= 0:
        return []
    ranked: list[tuple[KnobStackScore, ProposalSpec]] = []
    stats = _stack_stats(ledger)
    for spec in specs:
        raw = str(spec.math_axes.get("op_math_knobs") or "")
        knob_ids = tuple(part for part in raw.split("+") if part)
        score = score_knob_stack(knob_ids, ledger, axis_lift=axis_lift, stats=stats)
        if score.rejected:
            continue
        ranked.append((score, spec))
    ranked.sort(
        key=lambda pair: (
            pair[0].score,
            -len(str(pair[1].math_axes.get("op_math_knobs") or "").split("+")),
            pair[1].name,
        ),
        reverse=True,
    )
    return [spec for _, spec in ranked[:max_specs]]
