"""Single source of truth for the breakthrough-tier gates.

Used by:
- ``runner/_eval_registry.apply_breakthrough_logic`` — initial promotion gate
  on the validation pipeline output.
- ``runner/_helpers_benchmark.handle_breakthrough`` — trajectory-composite
  fallback promotion.
- ``leaderboard_rescore.rescore_entry`` — post-rescore demotion check.
- ``notebook/leaderboard_maintenance._effective_fingerprint_tier`` — sync-time
  demotion check after a rescreen writes new program_results.

Background: the prior ``trajectory_composite > 300.0`` hardcode in
``_helpers_benchmark.py`` and the ``val_baseline_ratio < 1.0`` lone-axis check
in ``_eval_registry.py`` together promoted candidates with no real capability
signal (the 2026-05-03 d904 ``Gated-MLP`` incident: composite=499 with
hellaswag=0.22, induction_auc=0.016, binding_composite=0.008). The capability
floor below blocks that family of false positives.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

# Composite floor for the trajectory-aware promotion fallback. 450 sits above
# all current non-real-architecture composites (top false positive: d904 at
# 499 originally, 335 after rescreen) and below the real reference architecture
# breakthrough (b0c38826 at 515).
BREAKTHROUGH_COMPOSITE_FLOOR = 450.0

# Minimum capability signal across all probes. The real breakthrough has every
# signal >> 0.10 (induction_auc=0.966, binding_v2=0.921, induction_v2=0.994).
# d904's strongest capability metric is binding_v2=0.091, so 0.10 is the
# narrowest gap that separates them.
BREAKTHROUGH_CAPABILITY_FLOOR = 0.10

# Names of capability metrics tracked. Helpers below accept any subset; missing
# values are treated as 0.0 so all-None rows fail the floor.
CAPABILITY_METRIC_NAMES: Tuple[str, ...] = (
    "induction_auc",
    "binding_composite",
    "induction_v2_investigation_auc",
    "binding_v2_investigation_auc",
)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _max_capability_signal(values: Iterable[Any]) -> float:
    """Return the max non-None float in ``values``, or 0.0 if none populated."""
    floats = [v for v in (_coerce_float(x) for x in values) if v is not None]
    return max(floats) if floats else 0.0


def passes_capability_floor(
    *,
    induction_auc: Any = None,
    binding_composite: Any = None,
    induction_v2_investigation_auc: Any = None,
    binding_v2_investigation_auc: Any = None,
    floor: float = BREAKTHROUGH_CAPABILITY_FLOOR,
) -> bool:
    """True iff at least one capability metric meets the floor."""
    signal = _max_capability_signal(
        (
            induction_auc,
            binding_composite,
            induction_v2_investigation_auc,
            binding_v2_investigation_auc,
        )
    )
    return signal >= float(floor)


def passes_breakthrough_gates(
    *,
    composite_score: Any = None,
    val_baseline_ratio: Any = None,
    induction_auc: Any = None,
    binding_composite: Any = None,
    induction_v2_investigation_auc: Any = None,
    binding_v2_investigation_auc: Any = None,
    composite_floor: float = BREAKTHROUGH_COMPOSITE_FLOOR,
    capability_floor: float = BREAKTHROUGH_CAPABILITY_FLOOR,
) -> Tuple[bool, Optional[str]]:
    """Return ``(passes, reason_failed)`` for the breakthrough-tier criteria.

    On pass returns ``(True, None)``. On fail returns ``(False, reason)`` where
    ``reason`` is one of ``composite_below_floor``, ``no_baseline_improvement``,
    ``capability_signal_below_floor``.
    """
    composite = _coerce_float(composite_score)
    if composite is None or composite < float(composite_floor):
        return False, "composite_below_floor"

    baseline = _coerce_float(val_baseline_ratio)
    if baseline is not None and baseline >= 1.0:
        return False, "no_baseline_improvement"

    if not passes_capability_floor(
        induction_auc=induction_auc,
        binding_composite=binding_composite,
        induction_v2_investigation_auc=induction_v2_investigation_auc,
        binding_v2_investigation_auc=binding_v2_investigation_auc,
        floor=capability_floor,
    ):
        return False, "capability_signal_below_floor"

    return True, None


def passes_breakthrough_from_row(
    row: dict,
    *,
    composite_score: Any = None,
) -> Tuple[bool, Optional[str]]:
    """Convenience: extract gate inputs from a leaderboard row dict.

    ``composite_score`` override is for callers that have just recomputed the
    score and want to gate against the new value rather than the stale row.
    """
    return passes_breakthrough_gates(
        composite_score=composite_score
        if composite_score is not None
        else row.get("composite_score"),
        val_baseline_ratio=row.get("validation_baseline_ratio"),
        induction_auc=row.get("induction_auc"),
        binding_composite=row.get("binding_composite"),
        induction_v2_investigation_auc=row.get("induction_v2_investigation_auc"),
        binding_v2_investigation_auc=row.get("binding_v2_investigation_auc"),
    )
