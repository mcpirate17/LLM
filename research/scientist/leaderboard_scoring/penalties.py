"""Multiplicative penalty/boost stage applied after additive component scoring."""

from __future__ import annotations

from typing import Dict, Optional

from ..thresholds import (
    BINDING_BINDING_AUC_SOFT_GATE,
    BINDING_INDUCTION_SOFT_GATE,
    BINDING_LOCAL_ONLY_PENALTY,
)


def _apply_scoring_penalties(
    score: float,
    *,
    inv_failed: bool,
    param_count: Optional[float],
    induction_screening_auc: Optional[float],
    binding_screening_auc_val: Optional[float],
    effective_ar_legacy_auc: Optional[float],  # legacy passthrough; soft-gate ignores
    ar_legacy_above_chance: Optional[bool],
    cfg: Optional[Dict[str, float]] = None,
    ar_gate_score: Optional[float] = None,
) -> tuple[float, float, float]:
    """Apply binding soft gate and param-size penalties.

    Returns (score, binding_pen, param_pen). The soft gate uses ``ar_gate_score``
    (threshold 0.30); ``effective_ar_legacy_auc`` retains the parameter slot for
    back-compat but no longer participates in the gate or composite boost.
    """
    _ = effective_ar_legacy_auc
    cfg = cfg or {}
    _induction_below = (
        induction_screening_auc is not None
        and induction_screening_auc < BINDING_INDUCTION_SOFT_GATE
    )
    _binding_below = (
        binding_screening_auc_val is not None
        and binding_screening_auc_val < BINDING_BINDING_AUC_SOFT_GATE
    )
    # ar_gate soft gate: pass at >=0.30 (V4 calibration). Legacy ar_legacy_above_chance
    # still respected as a "model showed retrieval signal" override.
    _AR_GATE_SOFT_GATE = 0.30
    _ar_below = (
        ar_gate_score is not None
        and ar_gate_score < _AR_GATE_SOFT_GATE
        and not ar_legacy_above_chance
    )
    _signals = sum(
        [
            induction_screening_auc is not None,
            binding_screening_auc_val is not None,
            ar_gate_score is not None,
        ]
    )
    _all_below = _signals >= 2 and all(
        [
            _induction_below or induction_screening_auc is None,
            _binding_below or binding_screening_auc_val is None,
            _ar_below or ar_gate_score is None,
        ]
    )
    binding_penalty = 1.0
    if not inv_failed and _signals >= 2 and _all_below:
        binding_penalty = float(
            cfg.get("binding_all_below_penalty", BINDING_LOCAL_ONLY_PENALTY)
        )
        score *= binding_penalty

    # v8.1 optional boost: reward graphs that clear a binding_screening_composite floor.
    # binding_screening_composite matches the understanding gate convention:
    #   0.4 * ar + 0.3 * induction + 0.3 * binding
    boost_mult = float(cfg.get("binding_screening_composite_boost", 1.0))
    boost_floor = float(cfg.get("binding_screening_composite_boost_floor", 0.0))
    if not inv_failed and boost_mult > 1.0 and boost_floor > 0.0:
        _ind = (
            float(induction_screening_auc)
            if induction_screening_auc is not None
            else 0.0
        )
        _bind = (
            float(binding_screening_auc_val)
            if binding_screening_auc_val is not None
            else 0.0
        )
        _nai = float(ar_gate_score) if ar_gate_score is not None else 0.0
        _composite = 0.4 * _nai + 0.3 * _ind + 0.3 * _bind
        if _composite >= boost_floor:
            score *= boost_mult
            # Fold the boost into binding_penalty so the breakdown reflects
            # it as part of the binding-path multiplier rather than a
            # separate surprise.
            binding_penalty *= boost_mult

    _TARGET_PARAMS = 5_000_000
    param_penalty = 1.0
    if param_count is not None and param_count > _TARGET_PARAMS:
        param_penalty = 1.0 / ((param_count / _TARGET_PARAMS) ** 0.13)
        score *= param_penalty

    return max(0.0, score), binding_penalty, param_penalty
