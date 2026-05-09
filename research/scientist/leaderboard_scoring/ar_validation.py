"""AR Validation / AR Gate cascade scoring.

Splits the AR signal into a screening gate (early-stage discriminator) and a
rank-order tier (post-validation ordering). The cascade returns the higher of
the two so models with stronger validation signal aren't capped by the gate.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from ._utils import _clamp01, _safe_float01, _scurve_higher_better


def _ar_validation_validation_signal(kw: Dict[str, Any]) -> Optional[float]:
    """Return normalized AR Validation/easy25 signal in [0, 1], if present."""
    value = kw.get("ar_validation_rank_score")
    if value is None:
        value = kw.get("champion_ar_validation_score")
    if value is None:
        pair = kw.get("ar_validation_held_pair_acc")
        held_class = kw.get("ar_validation_held_class_acc")
        speed = kw.get("ar_validation_learning_speed_score")
        if pair is None and held_class is None and speed is None:
            return None
        try:
            raw = (
                6.0 * _clamp01(float(pair or 0.0))
                + 2.0 * _clamp01(float(held_class or 0.0))
                + 2.0 * _clamp01(float(speed or 0.0))
            )
        except (TypeError, ValueError):
            return None
    else:
        try:
            raw = float(value)
        except (TypeError, ValueError):
            return None
    if not math.isfinite(raw):
        return None
    return _clamp01(raw if raw <= 1.0 else raw / 10.0)


def _score_ar_validation_validation_tier(
    cfg: Dict[str, float],
    *,
    inv_failed: bool,
    is_validation: bool,
    kw: Dict[str, Any],
) -> tuple[float, Dict[str, float]]:
    """Validation-tier AR Validation/easy25 rank-order credit.

    Missing AR Validation contributes zero for compatibility with pre-backfill rows.
    The score is intentionally validation-only: AR Gate remains the early
    screening gate, while AR Validation orders candidates after they reach validation.
    """
    bd = {
        "cap_ar_validation_validation": 0.0,
        "_ar_validation_validation_signal": 0.0,
    }
    signal = _ar_validation_validation_signal(kw)
    if inv_failed or not is_validation or signal is None:
        return 0.0, bd
    bd["_ar_validation_validation_signal"] = signal
    weight = float(cfg.get("w_cap_ar_validation_validation", 0.0))
    anchor = float(cfg.get("cap_ar_validation_validation_anchor", 0.55))
    pts = weight * _scurve_higher_better(signal, anchor)
    bd["cap_ar_validation_validation"] = pts
    return pts, bd


def _ar_validation_rank_signal(
    *,
    ar_validation_rank_score: Optional[float],
    ar_validation_held_pair: Optional[float],
    ar_validation_held_class: Optional[float],
) -> Optional[float]:
    if ar_validation_rank_score is not None:
        try:
            score = float(ar_validation_rank_score)
        except (TypeError, ValueError):
            score = math.nan
        if math.isfinite(score):
            return _clamp01(score if score <= 1.0 else score / 10.0)
    pair = _safe_float01(ar_validation_held_pair)
    held_class = _safe_float01(ar_validation_held_class)
    if pair is None and held_class is None:
        return None
    return 0.7 * (pair or 0.0) + 0.3 * (held_class or 0.0)


def _ar_gate_gate_fraction(value: Optional[float], cfg: Dict[str, float]) -> float:
    """Saturation-aware AR Gate credit.

    AR Gate remains a real gate below the learner threshold, but it saturates
    too early to rank mature learners. Above the gate, credit intentionally
    plateaus unless a harder AR probe is present.
    """
    v = _safe_float01(value)
    if v is None:
        return 0.0
    gate = float(cfg.get("ar_gate_gate_threshold", 0.30))
    plateau_start = float(cfg.get("ar_gate_gate_plateau_start", 0.90))
    below_max = float(cfg.get("ar_gate_gate_below_max", 0.20))
    floor = float(cfg.get("ar_gate_gate_floor", 0.35))
    plateau = float(cfg.get("ar_gate_gate_plateau", 0.45))
    if v < gate:
        return below_max * (v / max(gate, 1e-9))
    if v < plateau_start:
        return floor + (plateau - floor) * (
            (v - gate) / max(plateau_start - gate, 1e-9)
        )
    return plateau


def _ar_cascade_fraction(
    *,
    cfg: Dict[str, float],
    ar_gate_score: Optional[float],
    ar_validation_rank_score: Optional[float],
    ar_validation_held_pair: Optional[float],
    ar_validation_held_class: Optional[float],
) -> tuple[float, float, float]:
    """Return (fraction, nano_gate_fraction, rank_signal).

    AR Validation is the rank-order signal when available. AR Gate remains the
    early screening gate and fallback for rows not yet run through AR Validation.
    """
    nano_gate = _ar_gate_gate_fraction(ar_gate_score, cfg)
    rank_signal = _ar_validation_rank_signal(
        ar_validation_rank_score=ar_validation_rank_score,
        ar_validation_held_pair=ar_validation_held_pair,
        ar_validation_held_class=ar_validation_held_class,
    )
    if rank_signal is None:
        return nano_gate, nano_gate, 0.0
    rank_floor = float(cfg.get("ar_validation_rank_floor", 0.35))
    rank_span = float(cfg.get("ar_validation_rank_span", 0.65))
    return max(nano_gate, rank_floor + rank_span * rank_signal), nano_gate, rank_signal
