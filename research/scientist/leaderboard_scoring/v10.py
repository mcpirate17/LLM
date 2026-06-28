"""v10 — three equal buckets (capability / loss / understanding), all S-curves.

Three structural changes from v9:
  1. Binding/induction/AR are unrolled from one 85pt S-curve into three
     independent 25pt S-curves, sitting alongside four trajectory metrics
     (erf_density, id_collapse_rate, erf_decay_slope, logit_margin_velocity)
     at equal 25pt weight — all seven form a 175pt capability tier.
  2. Loss tier (perf_short/medium/long + learn_eff + early_conv) and
     understanding tier (blimp + tinystories + cross_task + diagnostic +
     hellaswag + hierarchy) are sized to ~175pts each so capability,
     loss, and understanding contribute roughly equally.
  3. binding_all_below_penalty (×0.50) and binding_screening_composite_boost (×1.15)
     are retired. With understanding properly weighted at 175pts a
     ppl-only graph already loses on the additive scoreboard; the
     multiplicative gate was redundant and penalized SSMs (Mamba-class)
     that bind weakly by architecture but are otherwise capable.

Anchors come from the 2026-04-25 distribution analysis on the 16k+ rows
that have at least one Gemini metric populated. Lower-is-better metrics
(id_collapse_rate, erf_decay_slope, icld_velocity) are negated before
S-curve normalization.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from ._config import _LOSS_TIER_BD_KEYS, _UND_TIER_BD_KEYS, _V10_CONFIG
from ._utils import (
    _cv_penalty_multiplier,
    _effective_probe_pair,
    _scurve_higher_better,
    _scurve_lower_better,
)
from .ar_validation import _ar_cascade_fraction
from .generic import _compute_composite_generic


def _score_capability_tier_v10(
    cfg: Dict[str, float],
    *,
    inv_failed: bool,
    effective_ar_legacy_auc: Optional[float],
    effective_induction_screening_auc: Optional[float],
    effective_binding_screening_auc: Optional[float],
    erf_density: Optional[float],
    id_collapse_rate: Optional[float],
    erf_decay_slope: Optional[float],
    logit_margin_velocity: Optional[float],
    ar_gate_score: Optional[float] = None,
    ar_validation_rank_score: Optional[float] = None,
    ar_validation_held_pair: Optional[float] = None,
    ar_validation_held_class: Optional[float] = None,
) -> tuple[float, Dict[str, float]]:
    """v10 capability tier — trajectory, binding, AR gate, and full-AR signals.

    ``cap_ar`` stays tied to ``ar_gate_score`` for screening/investigation
    continuity. ``cap_legacy_ar`` gives the full AR probe a separate champion
    path when longer training makes that harder test meaningful.
    """
    bd: Dict[str, float] = {}
    total = 0.0

    if inv_failed:
        for k in (
            "cap_ar",
            "cap_legacy_ar",
            "cap_induction",
            "cap_binding",
            "cap_erf_density",
            "cap_id_collapse",
            "cap_erf_decay",
            "cap_logit_margin",
        ):
            bd[k] = 0.0
        return 0.0, bd

    ar_fraction, nano_gate_fraction, ar_rank_signal = _ar_cascade_fraction(
        cfg=cfg,
        ar_gate_score=ar_gate_score,
        ar_validation_rank_score=ar_validation_rank_score,
        ar_validation_held_pair=ar_validation_held_pair,
        ar_validation_held_class=ar_validation_held_class,
    )
    pairs = (
        ("cap_ar", ar_fraction),
        (
            "cap_legacy_ar",
            _scurve_higher_better(
                effective_ar_legacy_auc,
                cfg.get("legacy_ar_anchor", 0.15),
            ),
        ),
        (
            "cap_induction",
            _scurve_higher_better(
                effective_induction_screening_auc, cfg["cap_induction_anchor"]
            ),
        ),
        (
            "cap_binding",
            _scurve_higher_better(
                effective_binding_screening_auc, cfg["cap_binding_anchor"]
            ),
        ),
        (
            "cap_erf_density",
            _scurve_higher_better(erf_density, cfg["cap_erf_density_anchor"]),
        ),
        (
            "cap_id_collapse",
            _scurve_lower_better(id_collapse_rate, cfg["cap_id_collapse_anchor"]),
        ),
        (
            "cap_erf_decay",
            _scurve_lower_better(erf_decay_slope, cfg["cap_erf_decay_anchor"]),
        ),
        (
            "cap_logit_margin",
            _scurve_higher_better(
                logit_margin_velocity, cfg["cap_logit_margin_anchor"]
            ),
        ),
    )
    for key, frac in pairs:
        weight = cfg.get(f"w_{key}", 0.0)
        if key == "cap_legacy_ar":
            weight = cfg.get("w_cap_legacy_ar", cfg.get("w_legacy_ar", weight))
        pts = weight * frac
        bd[key] = pts
        total += pts
    bd["cap_ar_nano_gate_fraction"] = nano_gate_fraction
    bd["cap_ar_rank_signal"] = ar_rank_signal
    bd["cap_ar_signal_fraction"] = ar_fraction
    return total, bd


def _score_trajectory_aux_v10(
    cfg: Dict[str, float],
    *,
    inv_failed: bool,
    is_investigated: bool,
    is_validation: bool,
    erf_variance: Optional[float],
    icld_velocity: Optional[float],
) -> tuple[float, Dict[str, float]]:
    """v10 aux trajectory — erf_variance always, icld only post-screening."""
    bd: Dict[str, float] = {}
    total = 0.0
    if inv_failed:
        bd["aux_erf_variance"] = 0.0
        bd["aux_icld"] = 0.0
        return 0.0, bd

    var_pts = cfg["w_aux_erf_variance"] * _scurve_higher_better(
        erf_variance, cfg["aux_erf_variance_anchor"]
    )
    bd["aux_erf_variance"] = var_pts
    total += var_pts

    icld_pts = 0.0
    if is_investigated or is_validation:
        icld_pts = cfg["w_aux_icld"] * _scurve_lower_better(
            icld_velocity, cfg["aux_icld_anchor"]
        )
    bd["aux_icld"] = icld_pts
    total += icld_pts
    return total, bd


def compute_composite_v10(
    *,
    decompose: bool = False,
    **kw: Any,
) -> Union[float, Dict[str, Any]]:
    """Composite v10 — three equal tiers, all S-curves, no binding gates.

    Pulls trajectory metrics from kw (fp_jacobian_erf_density,
    fp_id_collapse_rate, fp_jacobian_erf_decay_slope,
    fp_logit_margin_velocity, fp_jacobian_erf_variance, fp_icld_velocity)
    and routes them through the new capability + aux-trajectory scorers.
    Everything else (loss curves, understanding, efficiency, novelty,
    robustness, long-context) flows through the existing v8 generic
    scorer with the v10 weights.
    """
    cfg = _V10_CONFIG
    base = _compute_composite_generic(cfg, decompose=True, **kw)
    base_score = float(base["composite_score"])
    base_bd = base.get("breakdown") or {}

    # Determine tier flags consistent with the generic scorer.
    tier = kw.get("tier")
    inv_failed = tier in ("investigation_failed", "screened_out")
    is_investigated = (
        tier in ("investigation", "validation", "breakthrough")
        if tier
        else (kw.get("ppl_investigation") is not None)
    )
    is_validation = (
        tier in ("validation", "breakthrough")
        if tier
        else (kw.get("ppl_validation") is not None)
    )

    ar_legacy_timed_out = kw.get("ar_legacy_timed_out")
    effective_ar = None if ar_legacy_timed_out else kw.get("ar_legacy_auc")
    eff_ind, eff_bind = _effective_probe_pair(kw)

    cap_pts, cap_bd = _score_capability_tier_v10(
        cfg,
        inv_failed=inv_failed,
        effective_ar_legacy_auc=effective_ar,
        effective_induction_screening_auc=eff_ind,
        effective_binding_screening_auc=eff_bind,
        erf_density=kw.get("fp_jacobian_erf_density"),
        id_collapse_rate=kw.get("fp_id_collapse_rate"),
        erf_decay_slope=kw.get("fp_jacobian_erf_decay_slope"),
        logit_margin_velocity=kw.get("fp_logit_margin_velocity"),
        ar_gate_score=kw.get("ar_gate_score"),
        ar_validation_rank_score=kw.get("ar_validation_rank_score"),
        ar_validation_held_pair=kw.get("ar_validation_held_pair_acc"),
        ar_validation_held_class=kw.get("ar_validation_held_class_acc"),
    )
    aux_pts, aux_bd = _score_trajectory_aux_v10(
        cfg,
        inv_failed=inv_failed,
        is_investigated=is_investigated,
        is_validation=is_validation,
        erf_variance=kw.get("fp_jacobian_erf_variance"),
        icld_velocity=kw.get("fp_icld_velocity"),
    )

    # Score-stability (CV) penalty — only fires at validation/breakthrough,
    # only when the per-tier CV is populated (n>=2 runs for that tier).
    apply_cv_penalty = is_validation and not inv_failed
    loss_pen = und_pen = cap_pen = 1.0
    if apply_cv_penalty:
        loss_pen = _cv_penalty_multiplier(
            kw.get("cv_loss"), cfg["cv_lambda_loss"], cfg["cv_penalty_floor"]
        )
        und_pen = _cv_penalty_multiplier(
            kw.get("cv_understanding"), cfg["cv_lambda_und"], cfg["cv_penalty_floor"]
        )
        cap_pen = _cv_penalty_multiplier(
            kw.get("cv_capability"), cfg["cv_lambda_cap"], cfg["cv_penalty_floor"]
        )

    # Decompose base_score into loss-tier, understanding-tier, and the
    # rest (legacy: routing/compression/sparsity/adaptive/novelty/ncd/
    # robustness/long_context/binding-rollup).  Apply per-tier CV
    # penalty, then recompose.
    loss_sum = sum(float(base_bd.get(k, 0.0) or 0.0) for k in _LOSS_TIER_BD_KEYS)
    und_sum = sum(float(base_bd.get(k, 0.0) or 0.0) for k in _UND_TIER_BD_KEYS)
    legacy_sum = base_score - loss_sum - und_sum

    base_score_penalized = loss_sum * loss_pen + und_sum * und_pen + legacy_sum
    cap_pts_penalized = cap_pts * cap_pen
    composite = base_score_penalized + cap_pts_penalized + aux_pts

    if decompose:
        bd: Dict[str, Any] = dict(base_bd)
        bd.update(cap_bd)
        bd.update(aux_bd)
        # Apply penalty in-place so callers reading individual tier
        # points see the penalized values.
        if apply_cv_penalty:
            for k in _LOSS_TIER_BD_KEYS:
                if k in bd and bd[k]:
                    bd[k] = float(bd[k]) * loss_pen
            for k in _UND_TIER_BD_KEYS:
                if k in bd and bd[k]:
                    bd[k] = float(bd[k]) * und_pen
            for k in cap_bd:
                if bd.get(k):
                    bd[k] = float(bd[k]) * cap_pen
        bd["_v10_capability_total"] = cap_pts_penalized
        bd["_v10_aux_trajectory_total"] = aux_pts
        bd["_v10_base_v8style_total"] = base_score_penalized
        bd["_cv_penalty_loss"] = loss_pen
        bd["_cv_penalty_und"] = und_pen
        bd["_cv_penalty_cap"] = cap_pen
        bd["_cv_penalty_applied"] = bool(apply_cv_penalty)
        return {"composite_score": composite, "breakdown": bd}
    return composite
