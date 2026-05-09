"""Generic composite scorer parameterized by config dict.

Both v7 and v8 (and through v10's reuse, all later versions) delegate to
``_compute_composite_generic``. The config dict controls component weights
and frontier reference values.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from ..thresholds import INSUFFICIENT_LEARNING_LR
from .components import (
    _score_efficiency,
    _score_novelty_ncd,
    _score_performance_curves,
    _score_robustness_linguistics,
    _score_speed_convergence,
    _score_understanding_v8,
)
from .penalties import _apply_scoring_penalties


def _compute_composite_generic(
    cfg: Dict[str, float],
    *,
    decompose: bool = False,
    # S-curved inputs (perplexity, efficiency)
    ppl_screening: Optional[float] = None,
    ppl_investigation: Optional[float] = None,
    ppl_validation: Optional[float] = None,
    param_count: Optional[float] = None,
    ppl_at_100: Optional[float] = None,
    ppl_at_500: Optional[float] = None,
    ppl_at_1000: Optional[float] = None,
    long_ctx_score: Optional[float] = None,
    # Additive inputs (routing/compression/sparsity)
    routing_savings: Optional[float] = None,
    routing_fast_fraction: Optional[float] = None,
    routing_balance_score: Optional[float] = None,
    compression_ratio: Optional[float] = None,
    quant_quality_per_byte: Optional[float] = None,
    n_sparse_ops: Optional[int] = None,
    activation_sparsity: Optional[float] = None,
    recursion_savings: Optional[float] = None,
    depth_savings: Optional[float] = None,
    # Novelty
    screening_nov: Optional[float] = None,
    novelty_confidence: Optional[float] = None,
    is_reference: bool = False,
    # NCD
    ncd_score: Optional[float] = None,
    # Robustness
    spectral_norm: Optional[float] = None,
    robustness_noise: Optional[float] = None,
    robustness_score: Optional[float] = None,
    quant_retention: Optional[float] = None,
    # Loss ratio fallback
    screening_lr: Optional[float] = None,
    # Speed/throughput
    throughput_tok_s: Optional[float] = None,
    forward_time_ms: Optional[float] = None,
    # HellaSwag
    hellaswag_acc_screening: Optional[float] = None,
    hellaswag_acc_investigation: Optional[float] = None,
    hellaswag_acc_validation: Optional[float] = None,
    # Binding probes
    ar_legacy_auc: Optional[float] = None,
    ar_legacy_timed_out: Optional[bool] = None,
    ar_legacy_above_chance: Optional[bool] = None,
    ar_gate_score: Optional[
        float
    ] = None,  # post-V4 replacement for ar_legacy_auc weight
    induction_screening_auc: Optional[float] = None,
    binding_screening_auc: Optional[float] = None,
    # v2 investigation-tier probes (override induction_screening_auc/binding_screening_auc when
    # present; pre-backfill rows leave these None and fall back to v1)
    induction_intermediate_inv_auc: Optional[float] = None,
    binding_intermediate_inv_auc: Optional[float] = None,
    # BLiMP
    blimp_accuracy: Optional[float] = None,
    # v8 understanding metrics (ignored when weights are 0)
    tinystories_score: Optional[float] = None,
    cross_task_score: Optional[float] = None,
    diagnostic_score: Optional[float] = None,
    hierarchy_fitness: Optional[float] = None,
    # Tier context
    tier: Optional[str] = None,
    **kwargs: Any,
) -> Union[float, Dict[str, Any]]:
    """Generic composite scoring -- parameterized by config dict.

    Both v7 and v8 delegate to this. The config dict controls component
    weights and frontier reference values.
    """
    # Hard gate: model that didn't learn at screening
    if screening_lr is not None and screening_lr > INSUFFICIENT_LEARNING_LR:
        result = max(0.0, 10.0)
        if decompose:
            return {
                "composite_score": result,
                "breakdown": {"insufficient_learning_cap": 10.0},
            }
        return result

    _inv_failed = tier in ("investigation_failed", "screened_out")
    _is_investigated = (
        tier in ("investigation", "validation", "breakthrough")
        if tier
        else (ppl_investigation is not None)
    )
    _is_validation = (
        tier in ("validation", "breakthrough") if tier else (ppl_validation is not None)
    )
    _effective_ar_legacy_auc = None if ar_legacy_timed_out else ar_legacy_auc
    # v2 investigation probes override v1 when present. Rows pre-backfill
    # have None v2 values and keep scoring against v1 for continuity.
    _effective_induction_screening_auc = (
        induction_intermediate_inv_auc
        if induction_intermediate_inv_auc is not None
        else induction_screening_auc
    )
    _effective_binding_screening_auc = (
        binding_intermediate_inv_auc
        if binding_intermediate_inv_auc is not None
        else binding_screening_auc
    )

    # Score each component family
    perf_pts, perf_bd = _score_performance_curves(
        cfg,
        inv_failed=_inv_failed,
        is_investigated=_is_investigated,
        is_validation=_is_validation,
        ppl_screening=ppl_screening,
        ppl_investigation=ppl_investigation,
        ppl_validation=ppl_validation,
        param_count=param_count,
        ppl_at_500=ppl_at_500,
        ppl_at_1000=ppl_at_1000,
        screening_lr=screening_lr,
    )
    eff_pts, eff_bd = _score_efficiency(
        routing_savings=routing_savings,
        compression_ratio=compression_ratio,
        quant_quality_per_byte=quant_quality_per_byte,
        n_sparse_ops=n_sparse_ops,
        activation_sparsity=activation_sparsity,
        recursion_savings=recursion_savings,
        depth_savings=depth_savings,
    )
    nov_pts, nov_bd = _score_novelty_ncd(
        screening_nov=screening_nov,
        novelty_confidence=novelty_confidence,
        is_reference=is_reference,
        ncd_score=ncd_score,
        novelty_valid=bool(kwargs.get("novelty_valid_for_promotion")),
        analyses_succeeded=int(kwargs.get("analyses_succeeded", 0)),
    )
    robust_pts, robust_bd = _score_robustness_linguistics(
        cfg,
        inv_failed=_inv_failed,
        is_investigated=_is_investigated,
        is_validation=_is_validation,
        spectral_norm=spectral_norm,
        robustness_noise=robustness_noise,
        robustness_score_val=robustness_score,
        quant_retention=quant_retention,
        long_ctx_score=long_ctx_score,
        blimp_accuracy=blimp_accuracy,
        effective_ar_legacy_auc=_effective_ar_legacy_auc,
        induction_screening_auc=_effective_induction_screening_auc,
        binding_screening_auc_val=_effective_binding_screening_auc,
        ar_gate_score=ar_gate_score,
    )
    understand_pts, understand_bd = _score_understanding_v8(
        cfg,
        is_investigated=_is_investigated,
        is_validation=_is_validation,
        inv_failed=_inv_failed,
        tinystories_score=tinystories_score,
        cross_task_score=cross_task_score,
        diagnostic_score=diagnostic_score,
        hellaswag_acc_investigation=hellaswag_acc_investigation,
        hellaswag_acc_validation=hellaswag_acc_validation,
        hierarchy_fitness=hierarchy_fitness,
        ppl_screening=ppl_screening,
        ppl_investigation=ppl_investigation,
        ppl_validation=ppl_validation,
    )
    speed_pts, speed_bd = _score_speed_convergence(
        cfg,
        inv_failed=_inv_failed,
        throughput_tok_s=throughput_tok_s,
        forward_time_ms=forward_time_ms,
        ppl_at_100=ppl_at_100,
        ppl_at_500=ppl_at_500,
    )

    score = perf_pts + eff_pts + nov_pts + robust_pts + understand_pts + speed_pts

    final, binding_pen, param_pen = _apply_scoring_penalties(
        score,
        inv_failed=_inv_failed,
        param_count=param_count,
        induction_screening_auc=_effective_induction_screening_auc,
        binding_screening_auc_val=_effective_binding_screening_auc,
        effective_ar_legacy_auc=_effective_ar_legacy_auc,
        ar_legacy_above_chance=ar_legacy_above_chance,
        cfg=cfg,
        ar_gate_score=ar_gate_score,
    )

    if decompose:
        _bd: Dict[str, Any] = {}
        for d in (perf_bd, eff_bd, nov_bd, robust_bd, understand_bd, speed_bd):
            _bd.update(d)
        _bd["binding_local_only_penalty"] = binding_pen if binding_pen < 1.0 else 0.0
        _bd["param_size_penalty"] = param_pen if param_pen < 1.0 else 0.0
        if _inv_failed:
            _bd["_investigation_failed_penalty"] = True
        if param_pen < 1.0:
            _bd["param_size_penalty_multiplier"] = param_pen
        return {"composite_score": final, "breakdown": _bd}
    return final
