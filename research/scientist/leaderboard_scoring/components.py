"""Per-component scorers consumed by the generic composite.

Each ``_score_*`` function returns ``(total, breakdown_dict)`` so the generic
scorer can sum tier totals while preserving per-key contributions for the
``decompose=True`` path.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..thresholds import GPT2_REF
from ._utils import _scurve


def _score_performance_curves(
    cfg: Dict[str, float],
    *,
    inv_failed: bool,
    is_investigated: bool,
    is_validation: bool,
    ppl_screening: Optional[float],
    ppl_investigation: Optional[float],
    ppl_validation: Optional[float],
    param_count: Optional[float],
    ppl_at_500: Optional[float],
    ppl_at_1000: Optional[float],
    screening_lr: Optional[float],
) -> tuple[float, Dict[str, float]]:
    """S-curved performance components: perf short/med/long, param_eff, learn_eff."""
    bd: Dict[str, float] = {}
    total = 0.0

    perf_short = 0.0
    if not inv_failed:
        if ppl_screening is not None and ppl_screening > 0:
            perf_short = cfg["w_perf_short"] * _scurve(cfg["ppl_1000"] / ppl_screening)
        elif screening_lr is not None:
            perf_short = cfg["w_perf_short"] * _scurve(max(0.01, 1.0 - screening_lr))
    total += perf_short
    bd["perf_short"] = perf_short

    perf_med = 0.0
    if (
        not inv_failed
        and is_investigated
        and ppl_investigation is not None
        and ppl_investigation > 0
    ):
        perf_med = cfg["w_perf_medium"] * _scurve(cfg["ppl_2500"] / ppl_investigation)
    total += perf_med
    bd["perf_medium"] = perf_med

    perf_long = 0.0
    if (
        not inv_failed
        and is_validation
        and ppl_validation is not None
        and ppl_validation > 0
    ):
        perf_long = cfg["w_perf_long"] * _scurve(cfg["ppl_10000"] / ppl_validation)
    total += perf_long
    bd["perf_long"] = perf_long

    param_eff_pts = 0.0
    if (
        not inv_failed
        and param_count is not None
        and param_count > 0
        and ppl_screening is not None
        and ppl_screening > 0
    ):
        model_eff = (cfg["ppl_1000"] / ppl_screening) * (
            cfg["avg_params"] / param_count
        )
        param_eff_pts = cfg["w_param_eff"] * _scurve(model_eff / cfg["param_eff"])
    total += param_eff_pts
    bd["param_efficiency"] = param_eff_pts

    learn_eff_pts = 0.0
    if (
        not inv_failed
        and ppl_at_500 is not None
        and ppl_at_1000 is not None
        and ppl_at_1000 > 0
    ):
        learn_eff_pts = cfg["w_learn_eff"] * _scurve(
            (ppl_at_500 / ppl_at_1000) / cfg["learn_eff"]
        )
    total += learn_eff_pts
    bd["learning_efficiency"] = learn_eff_pts

    return total, bd


def _score_efficiency(
    *,
    routing_savings: Optional[float],
    compression_ratio: Optional[float],
    quant_quality_per_byte: Optional[float],
    n_sparse_ops: Optional[int],
    activation_sparsity: Optional[float],
    recursion_savings: Optional[float],
    depth_savings: Optional[float],
) -> tuple[float, Dict[str, float]]:
    """Additive efficiency components: routing, compression, sparsity, adaptive."""
    bd: Dict[str, float] = {}
    total = 0.0

    routing_pts = 50.0 * routing_savings if routing_savings is not None else 0.0
    bd["routing_savings"] = routing_pts
    total += routing_pts

    comp_pts = 0.0
    if compression_ratio is not None:
        comp_pts = 20.0 * max(0, 1.0 - compression_ratio)
        if quant_quality_per_byte is not None:
            comp_pts += 10.0 * max(0, quant_quality_per_byte)
    comp_pts = min(30.0, comp_pts)
    bd["compression"] = comp_pts
    total += comp_pts

    sparsity_pts = 0.0
    if n_sparse_ops is not None and n_sparse_ops > 0:
        sparsity_pts += min(20.0, n_sparse_ops * 6.0)
    if activation_sparsity is not None and activation_sparsity > 0.3:
        sparsity_pts += 10.0 * min(1.0, (activation_sparsity - 0.3) / 0.5)
    bd["sparsity"] = sparsity_pts
    total += sparsity_pts

    adaptive_pts = 0.0
    if recursion_savings is not None and recursion_savings > 0:
        adaptive_pts += 15.0 * min(1.0, recursion_savings / 0.5)
    if depth_savings is not None and depth_savings > 0:
        adaptive_pts += 10.0 * min(1.0, depth_savings / 0.5)
    bd["adaptive_computation"] = adaptive_pts
    total += adaptive_pts

    return total, bd


def _score_novelty_ncd(
    *,
    screening_nov: Optional[float],
    novelty_confidence: Optional[float],
    is_reference: bool,
    ncd_score: Optional[float],
    novelty_valid: bool,
    analyses_succeeded: int,
) -> tuple[float, Dict[str, float]]:
    """Novelty (max 40pts) and NCD (max 15pts)."""
    bd: Dict[str, float] = {}
    total = 0.0

    eff_nov = (
        0.0 if is_reference else (screening_nov if screening_nov is not None else 0.0)
    )
    conf = (
        0.0
        if is_reference
        else (novelty_confidence if novelty_confidence is not None else 1.0)
    )
    raw = 40.0 * eff_nov * conf
    if not is_reference and not novelty_valid:
        novelty_pts = min(40.0, raw * (0.4 + 0.6 * (analyses_succeeded / 4)))
    else:
        novelty_pts = raw
    bd["novelty"] = novelty_pts
    total += novelty_pts

    ncd_pts = (
        15.0 * max(0, 1.0 - ncd_score)
        if ncd_score is not None and ncd_score > 0
        else 0.0
    )
    bd["ncd"] = ncd_pts
    total += ncd_pts

    return total, bd


def _score_robustness_linguistics(
    cfg: Dict[str, float],
    *,
    inv_failed: bool,
    is_investigated: bool,
    is_validation: bool,
    spectral_norm: Optional[float],
    robustness_noise: Optional[float],
    robustness_score_val: Optional[float],
    quant_retention: Optional[float],
    long_ctx_score: Optional[float],
    blimp_accuracy: Optional[float],
    effective_ar_legacy_auc: Optional[float],
    induction_screening_auc: Optional[float],
    binding_screening_auc_val: Optional[float],
    ar_gate_score: Optional[float] = None,
) -> tuple[float, Dict[str, float]]:
    """Robustness (40pts), long context (25pts), binding probes, BLiMP (40pts).

    Binding composite: ``0.4 * ar_gate_score + 0.3 * induction + 0.3 * binding``.
    ``effective_ar_legacy_auc`` is retained as a parameter slot but contributes zero.
    """
    _ = effective_ar_legacy_auc
    bd: Dict[str, float] = {}
    total = 0.0

    robust_pts = 0.0
    if is_investigated:
        if spectral_norm is not None:
            robust_pts += 10.0 * max(0, 1.0 - (spectral_norm / 20.0))
        if robustness_noise is not None:
            robust_pts += 15.0 * max(0, 1.0 - robustness_noise)
        if robustness_score_val is not None:
            robust_pts += 15.0 * robustness_score_val
        if quant_retention is not None:
            robust_pts += 15.0 * max(0, quant_retention - 0.5) / 0.5
    robust_pts = min(40.0, robust_pts)
    bd["robustness"] = robust_pts
    total += robust_pts

    long_ctx_pts = 0.0
    if (
        (is_investigated or is_validation)
        and long_ctx_score is not None
        and long_ctx_score > 0
        and cfg["long_ctx"] > 0
    ):
        long_ctx_pts = 25.0 * _scurve(long_ctx_score / cfg["long_ctx"])
    bd["long_context"] = long_ctx_pts
    total += long_ctx_pts

    binding_pts = 0.0
    if not inv_failed:
        _bc, _bc_n = 0.0, 0
        if ar_gate_score is not None:
            _bc += 0.4 * ar_gate_score
            _bc_n += 1
        if induction_screening_auc is not None:
            _bc += 0.3 * induction_screening_auc
            _bc_n += 1
        if binding_screening_auc_val is not None:
            _bc += 0.3 * binding_screening_auc_val
            _bc_n += 1
        if _bc > 0 and _bc_n > 0:
            binding_pts = cfg["w_binding"] * _scurve(_bc / cfg["binding"], k=6)
    binding_pts = min(cfg["w_binding"], binding_pts)
    bd["binding"] = binding_pts
    total += binding_pts

    w_blimp = cfg.get("w_blimp", 40.0)
    blimp_pts = 0.0
    if not inv_failed and blimp_accuracy is not None and blimp_accuracy > 0.50:
        blimp_pts = min(w_blimp, w_blimp * _scurve(blimp_accuracy / cfg["blimp"], k=6))
    bd["blimp"] = blimp_pts
    total += blimp_pts

    return total, bd


_CROSS_TASK_COMPETENCE_PPL_THRESHOLD = 200.0


def _cross_task_has_competence(
    *,
    ppl_screening: Optional[float],
    ppl_investigation: Optional[float],
    ppl_validation: Optional[float],
    threshold: float = _CROSS_TASK_COMPETENCE_PPL_THRESHOLD,
) -> bool:
    """Return true when reproduced PPL shows enough learning to credit balance."""
    ppls = []
    for value in (ppl_validation, ppl_investigation, ppl_screening):
        if value is None:
            continue
        try:
            ppl = float(value)
        except (TypeError, ValueError):
            continue
        if ppl > 0:
            ppls.append(ppl)

    # Some direct/unit callers only pass cross_task_score. Preserve that API
    # surface; production score kwargs include at least one PPL when available.
    if not ppls:
        return True
    return min(ppls) <= threshold


def _score_understanding_v8(
    cfg: Dict[str, float],
    *,
    is_investigated: bool,
    is_validation: bool,
    inv_failed: bool,
    tinystories_score: Optional[float],
    cross_task_score: Optional[float],
    diagnostic_score: Optional[float],
    hellaswag_acc_investigation: Optional[float],
    hellaswag_acc_validation: Optional[float],
    hierarchy_fitness: Optional[float],
    ppl_screening: Optional[float] = None,
    ppl_investigation: Optional[float] = None,
    ppl_validation: Optional[float] = None,
) -> tuple[float, Dict[str, float]]:
    """v8 understanding components (weights are 0 in v7 config)."""
    bd: Dict[str, float] = {}
    total = 0.0

    w_ts = cfg["w_tinystories"]
    if w_ts > 0:
        pts = 0.0
        if (
            (is_investigated or is_validation)
            and tinystories_score is not None
            and tinystories_score > 0
        ):
            pts = min(w_ts, w_ts * _scurve(tinystories_score / cfg["tinystories"]))
        bd["tinystories"] = pts
        total += pts

    w_ct = cfg["w_cross_task"]
    if w_ct > 0:
        pts = 0.0
        if (
            is_investigated
            and cross_task_score is not None
            and cross_task_score > 0
            and _cross_task_has_competence(
                ppl_screening=ppl_screening,
                ppl_investigation=ppl_investigation,
                ppl_validation=ppl_validation,
            )
        ):
            pts = min(w_ct, w_ct * _scurve(cross_task_score / cfg["cross_task"]))
        bd["cross_task"] = pts
        total += pts

    w_diag = cfg["w_diagnostic"]
    if w_diag > 0:
        pts = 0.0
        if (
            (is_investigated or is_validation)
            and diagnostic_score is not None
            and diagnostic_score > 0
        ):
            pts = min(w_diag, w_diag * _scurve(diagnostic_score / cfg["diagnostic"]))
        bd["diagnostic"] = pts
        total += pts

    w_hs = cfg["w_hellaswag"]
    if w_hs > 0:
        pts = 0.0
        _acc = hellaswag_acc_investigation or hellaswag_acc_validation
        # Gate was 0.26 (above random chance for 4-way); cohort p99 sits at
        # 0.265 so 99% of rows scored 0. Lowered to strictly above random
        # (0.25) so the metric actually contributes signal. The anchor (0.30)
        # and steep k=6 already make sub-random scores irrelevant.
        if is_investigated and _acc is not None and _acc > 0.25:
            pts = min(w_hs, w_hs * _scurve(_acc / cfg["hellaswag"], k=6))
        bd["hellaswag"] = pts
        total += pts

    w_hier = cfg["w_hierarchy"]
    if w_hier > 0:
        pts = 0.0
        if is_investigated and hierarchy_fitness is not None and hierarchy_fitness > 0:
            pts = min(w_hier, w_hier * _scurve(hierarchy_fitness / cfg["hierarchy"]))
        bd["hierarchy"] = pts
        total += pts

    return total, bd


def _score_speed_convergence(
    cfg: Dict[str, float],
    *,
    inv_failed: bool,
    throughput_tok_s: Optional[float],
    forward_time_ms: Optional[float],
    ppl_at_100: Optional[float],
    ppl_at_500: Optional[float],
) -> tuple[float, Dict[str, float]]:
    """Speed/latency (max 25pts) and early convergence (max 10pts)."""
    bd: Dict[str, float] = {}
    total = 0.0

    speed_pts = 0.0
    if not inv_failed:
        if throughput_tok_s is not None and throughput_tok_s > 0:
            speed_pts = 25.0 * _scurve(throughput_tok_s / GPT2_REF["throughput_tok_s"])
        elif forward_time_ms is not None and forward_time_ms > 0:
            speed_pts = 25.0 * _scurve(GPT2_REF["forward_time_ms"] / forward_time_ms)
    bd["speed"] = speed_pts
    total += speed_pts

    convergence_pts = 0.0
    if (
        not inv_failed
        and ppl_at_100 is not None
        and ppl_at_500 is not None
        and ppl_at_500 > 0
    ):
        convergence_pts = 10.0 * _scurve((ppl_at_100 / ppl_at_500) / cfg["convergence"])
    bd["early_convergence"] = convergence_pts
    total += convergence_pts

    return total, bd
