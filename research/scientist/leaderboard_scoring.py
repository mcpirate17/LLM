"""Leaderboard scoring functions — pure arithmetic, no I/O.

All functions are static / module-level. Candidates for future Cython port.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Sequence, Union

from .thresholds import (
    BINDING_AR_SOFT_GATE,
    BINDING_BINDING_AUC_SOFT_GATE,
    BINDING_INDUCTION_SOFT_GATE,
    BINDING_LOCAL_ONLY_PENALTY,
    GPT2_REF,
    INSUFFICIENT_LEARNING_LR,
)


def compute_efficiency_multiple(
    loss_ratio: Optional[float] = None,
    param_count: Optional[float] = None,
    flops_forward: Optional[float] = None,
    throughput_tok_s: Optional[float] = None,
    peak_memory_mb: Optional[float] = None,
    forward_time_ms: Optional[float] = None,
    is_moe: bool = False,
) -> Optional[Dict[str, float]]:
    """Geometric mean of per-dimension ratios vs GPT-2.

    All ratios >1.0 = better than GPT-2. Requires at least 3 of 6
    dimensions to return a result (graceful with missing data).

    For MoE models (is_moe=True), total param count is excluded from
    the geomean since MoE activates only a fraction of params per token.
    Returns dict with per-dimension ratios and ``geomean``, or None.
    """
    ref = GPT2_REF
    ratios: Dict[str, float] = {}

    if loss_ratio is not None and loss_ratio > 0:
        ratios["x_quality"] = ref["loss_ratio"] / loss_ratio
    # MoE: skip param count penalty — total params != active params
    if param_count is not None and param_count > 0 and not is_moe:
        ratios["x_params"] = ref["param_count"] / param_count
    if flops_forward is not None and flops_forward > 0:
        ratios["x_flops"] = ref["flops_forward"] / flops_forward
    if throughput_tok_s is not None and throughput_tok_s > 0:
        ratios["x_throughput"] = throughput_tok_s / ref["throughput_tok_s"]
    if peak_memory_mb is not None and peak_memory_mb > 0:
        ratios["x_memory"] = ref["peak_memory_mb"] / peak_memory_mb
    if forward_time_ms is not None and forward_time_ms > 0:
        ratios["x_latency"] = ref["forward_time_ms"] / forward_time_ms

    if len(ratios) < 3:
        return None

    geomean = 1.0
    for v in ratios.values():
        geomean *= v
    geomean = geomean ** (1.0 / len(ratios))
    ratios["geomean"] = geomean
    ratios["n_dimensions"] = float(len(ratios) - 1)
    return ratios


_PR_SELECT_COLS = (
    "result_id, novelty_confidence, loss_improvement_rate, final_loss, "
    "param_count, n_train_steps, behavioral_novelty, structural_novelty, "
    "fp_cka_vs_transformer, wikitext_perplexity, wikitext_score, "
    "wikitext_ppl_200, wikitext_ppl_500, wikitext_eval_steps, "
    "routing_savings_ratio, compression_ratio, activation_sparsity_score, "
    "depth_savings_ratio, recursion_depth_ratio, "
    "fp_jacobian_spectral_norm, validation_robustness_score, "
    "ncd_description_length_per_param, novelty_valid_for_promotion, "
    "fingerprint_json, hellaswag_acc, ar_auc, ar_final_acc, ar_timed_out, "
    "ar_above_chance, induction_auc, binding_auc, blimp_overall_accuracy, "
    "tinystories_score, cross_task_score, diagnostic_score, "
    "fp_gromov_delta, fp_hierarchy_fitness"
)


def _pr_dict_to_score_kwargs(
    pr_dict: Dict[str, Any],
    d: Dict[str, Any],
    is_reference: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build kwargs for ``compute_composite_v7`` from a pr_dict + leaderboard row.

    Pure logic — no I/O. Shared by both single and batch paths.
    """
    # Extract analyses_succeeded from fingerprint JSON.
    _analyses_succeeded = 0
    _fp_json_str = pr_dict.pop("fingerprint_json", None)
    if _fp_json_str:
        try:
            _fp_data = json.loads(_fp_json_str)
            _analyses_succeeded = int(_fp_data.get("analyses_succeeded", 0))
        except (ValueError, TypeError):
            pass

    ppl_final = pr_dict.get("wikitext_perplexity") or d.get("wikitext_perplexity")
    ppl_200 = pr_dict.get("wikitext_ppl_200")
    ppl_500 = pr_dict.get("wikitext_ppl_500") or d.get("ppl_500")
    eval_steps = pr_dict.get("wikitext_eval_steps")

    ppl_screening = ppl_final
    ppl_at_1000 = ppl_final if (eval_steps is not None and eval_steps >= 1000) else None

    tier = d.get("tier") or "screening"

    _inv_tiers = ("investigation", "investigation_failed", "validation", "breakthrough")
    _val_tiers = ("validation", "breakthrough")
    ppl_investigation = d.get("wikitext_perplexity") if tier in _inv_tiers else None
    ppl_validation = d.get("wikitext_perplexity") if tier in _val_tiers else None

    kw: Dict[str, Any] = {
        "ppl_screening": ppl_screening,
        "ppl_investigation": ppl_investigation,
        "ppl_validation": ppl_validation,
        "param_count": pr_dict.get("param_count") or d.get("param_count"),
        "ppl_at_100": ppl_200,
        "ppl_at_500": ppl_500,
        "ppl_at_1000": ppl_at_1000,
        "screening_lr": d.get("screening_loss_ratio"),
        "tier": tier,
        "routing_savings": d.get("routing_savings_ratio")
        or pr_dict.get("routing_savings_ratio"),
        "compression_ratio": d.get("compression_ratio")
        or pr_dict.get("compression_ratio"),
        "quant_quality_per_byte": d.get("quant_quality_per_byte"),
        "n_sparse_ops": d.get("n_sparse_ops"),
        "activation_sparsity": d.get("activation_sparsity_score")
        or pr_dict.get("activation_sparsity_score"),
        "recursion_savings": pr_dict.get("recursion_depth_ratio"),
        "depth_savings": pr_dict.get("depth_savings_ratio"),
        "screening_nov": d.get("screening_novelty"),
        "novelty_confidence": d.get("novelty_confidence")
        or pr_dict.get("novelty_confidence"),
        "novelty_valid_for_promotion": bool(pr_dict.get("novelty_valid_for_promotion")),
        "analyses_succeeded": _analyses_succeeded,
        "ncd_score": pr_dict.get("ncd_description_length_per_param"),
        "spectral_norm": d.get("fp_jacobian_spectral_norm")
        or pr_dict.get("fp_jacobian_spectral_norm"),
        "robustness_noise": d.get("robustness_noise_score"),
        "robustness_score": pr_dict.get("validation_robustness_score"),
        "quant_retention": d.get("quant_int8_retention"),
        "long_ctx_score": d.get("robustness_long_ctx_combined_score"),
        "is_reference": is_reference,
        # Binding probes
        "ar_auc": pr_dict.get("ar_auc") or d.get("ar_auc"),
        "ar_timed_out": bool(pr_dict.get("ar_timed_out"))
        if pr_dict.get("ar_timed_out") is not None
        else None,
        "ar_above_chance": bool(pr_dict.get("ar_above_chance"))
        if pr_dict.get("ar_above_chance") is not None
        else None,
        "induction_auc": pr_dict.get("induction_auc") or d.get("induction_auc"),
        "binding_auc": pr_dict.get("binding_auc") or d.get("binding_auc"),
        # BLiMP
        "blimp_accuracy": pr_dict.get("blimp_overall_accuracy")
        or d.get("blimp_overall_accuracy"),
        # HellaSwag commonsense reasoning
        "hellaswag_acc_screening": pr_dict.get("hellaswag_acc")
        or d.get("hellaswag_acc"),
        "hellaswag_acc_investigation": d.get("hellaswag_acc")
        if tier in _inv_tiers
        else None,
        "hellaswag_acc_validation": d.get("hellaswag_acc")
        if tier in _val_tiers
        else None,
        # v8 understanding metrics
        "tinystories_score": pr_dict.get("tinystories_score"),
        "cross_task_score": pr_dict.get("cross_task_score"),
        "diagnostic_score": pr_dict.get("diagnostic_score"),
        "hierarchy_fitness": pr_dict.get("fp_hierarchy_fitness"),
    }
    if extra:
        kw.update(extra)
    return kw


def build_score_kwargs(
    conn: Any,
    notebook: Any,
    result_id: str,
    d: Dict[str, Any],
    is_reference: bool,
    **extra: Any,
) -> Dict[str, Any]:
    """Build kwargs for ``compute_composite_v7`` from a leaderboard row.

    Queries program_results for fields not on the leaderboard row.
    ``conn`` is a sqlite3 connection, ``notebook`` provides helper methods.
    """
    pr = conn.execute(
        f"SELECT {_PR_SELECT_COLS} FROM program_results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    pr_dict = dict(pr) if pr else {}
    return _pr_dict_to_score_kwargs(pr_dict, d, is_reference, extra or None)


def prefetch_program_results(
    conn: Any,
    result_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Batch-fetch program_results rows for multiple result_ids in one query.

    Returns ``{result_id: row_dict}``. Missing IDs are absent from the dict.
    Use with ``build_score_kwargs_from_prefetch`` to avoid N+1 queries.
    """
    if not result_ids:
        return {}
    # SQLite supports up to 999 variables; chunk if needed.
    out: Dict[str, Dict[str, Any]] = {}
    ids = list(result_ids)
    chunk_size = 900
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT {_PR_SELECT_COLS} FROM program_results "
            f"WHERE result_id IN ({placeholders})",
            chunk,
        ).fetchall()
        for row in rows:
            out[row["result_id"]] = dict(row)
    return out


def build_score_kwargs_from_prefetch(
    pr_dict: Dict[str, Any],
    d: Dict[str, Any],
    is_reference: bool,
    **extra: Any,
) -> Dict[str, Any]:
    """Like ``build_score_kwargs`` but uses a pre-fetched pr_dict (no SQL).

    Caller is responsible for fetching via ``prefetch_program_results``.
    Pass a copy if you need to reuse the pr_dict (fingerprint_json is popped).
    """
    return _pr_dict_to_score_kwargs(dict(pr_dict), d, is_reference, extra or None)


def reference_novelty_for_display(novelty: Optional[float]) -> Optional[float]:
    """Compress reference novelty values for dashboard display."""
    if novelty is None:
        return None
    try:
        value = float(novelty)
    except (TypeError, ValueError):
        return None
    value = max(0.0, min(1.0, value))
    return min(0.35, value * 0.4)


def compute_pre_investigation_score(
    row: Dict[str, Any],
    best_ref_lr: Optional[float] = None,
) -> float:
    """Stage B composite readiness score (0-100 scale).

    Components:
    - Performance (40pts): loss_ratio, discovery_loss_ratio, loss_improvement_rate
    - Stability (20pts): stability_score, spectral_norm (Gaussian around 1.0), grad_norm_std
    - Novelty (20pts): novelty_score * confidence, structural_novelty, behavioral_novelty
    - Fingerprint quality (10pts): fp_intrinsic_dim, fp_isotropy, fp_rank_ratio
    - Efficiency (10pts): throughput_tok_s, peak_memory_mb
    - Reference penalty (-20pts): if loss_ratio > 1.5 * best_reference_lr
    """
    score = 0.0

    # -- Performance (40 pts) --
    lr = row.get("loss_ratio")
    if lr is not None and lr > 0:
        score += max(0, min(40, 40 * (1.0 - float(lr))))

    dlr = row.get("discovery_loss_ratio")
    if dlr is not None and dlr > 0:
        score += max(0, min(5, 5 * (1.0 - float(dlr))))

    lir = row.get("loss_improvement_rate")
    if lir is not None and float(lir) > 0:
        score += min(5, float(lir) * 10)

    score = min(40, score)

    # -- Stability (20 pts) --
    stab = row.get("stability_score")
    if stab is not None:
        score += min(10, float(stab) * 10)

    sn = row.get("fp_jacobian_spectral_norm")
    if sn is not None and float(sn) > 0:
        log_sn = math.log(float(sn))
        score += max(0, min(6, 6 * math.exp(-log_sn * log_sn / 2.0)))

    gns = row.get("grad_norm_std")
    if gns is not None:
        score += max(0, min(4, 4 * max(0, 1.0 - float(gns))))

    # -- Novelty (20 pts) --
    ns = row.get("novelty_score")
    nc = row.get("novelty_confidence")
    if ns is not None:
        conf = float(nc) if nc is not None else 0.5
        score += min(10, float(ns) * conf * 10)

    sn_nov = row.get("structural_novelty")
    if sn_nov is not None:
        score += min(5, float(sn_nov) * 5)

    bn = row.get("behavioral_novelty")
    if bn is not None:
        score += min(5, float(bn) * 5)

    # -- Fingerprint quality (10 pts) --
    fid = row.get("fp_intrinsic_dim")
    if fid is not None and float(fid) > 0:
        score += min(4, float(fid) / 5.0)

    fiso = row.get("fp_isotropy")
    if fiso is not None:
        score += min(3, float(fiso) * 3)

    frr = row.get("fp_rank_ratio")
    if frr is not None:
        score += min(3, float(frr) * 3)

    # -- Efficiency (10 pts) --
    tp = row.get("throughput_tok_s")
    if tp is not None and float(tp) > 0:
        score += min(5, float(tp) / 2000.0)

    mem = row.get("peak_memory_mb")
    if mem is not None and float(mem) > 0:
        score += max(0, min(5, 5 * (1.0 - float(mem) / 600.0)))

    # -- Reference penalty (-20 pts) --
    if best_ref_lr is not None and lr is not None:
        if float(lr) > 1.5 * float(best_ref_lr):
            score -= 20

    return max(0, min(100, round(score, 2)))


# ---------------------------------------------------------------------------
# v7/v8 scoring: unified composite via _compute_composite_generic
# See BASELINE_SCORING_DO_NOT_DELETE.md for full design rationale.
# ---------------------------------------------------------------------------

# Frontier reference values shared by v7 and v8 (measured 2026-03-23 on wiki103)
_FRONTIER_COMMON: Dict[str, float] = {
    "ppl_1000": 10.0,  # screening anchor
    "ppl_2500": 8.6,  # investigation anchor
    "ppl_10000": 5.6,  # validation anchor
    "param_eff": 1.09,  # (frontier_ppl/model_ppl) * (frontier_params/model_params)
    "learn_eff": 1.13,  # ppl@500 / ppl@1000
    "long_ctx": 0.375,  # average long context score
    "avg_params": 28_561_920,  # average params across 4 refs
    "convergence": 1.30,  # avg ppl@100 / ppl@500 across 4 refs
    "binding": 0.15,  # binding composite for refs at nano scale
    "blimp": 0.60,  # BLiMP frontier at GPT-2 nano scale
}

_V7_CONFIG: Dict[str, float] = {
    **_FRONTIER_COMMON,
    # Component max points
    "w_perf_short": 50.0,
    "w_perf_medium": 75.0,
    "w_perf_long": 100.0,
    "w_param_eff": 50.0,
    "w_learn_eff": 25.0,
    "w_binding": 120.0,
    # v8-only understanding components (0 = disabled in v7)
    "w_tinystories": 0.0,
    "w_cross_task": 0.0,
    "w_diagnostic": 0.0,
    "w_hellaswag": 0.0,
    "w_hierarchy": 0.0,
    # v8-only frontier values (unused when weight=0, present for type safety)
    "tinystories": 0.45,
    "cross_task": 0.60,
    "diagnostic": 0.35,
    "hellaswag": 0.30,
    "hierarchy": 0.50,
}

_V8_CONFIG: Dict[str, float] = {
    **_FRONTIER_COMMON,
    # Component max points (reduced perplexity weights)
    "w_perf_short": 35.0,
    "w_perf_medium": 50.0,
    "w_perf_long": 65.0,
    "w_param_eff": 30.0,
    "w_learn_eff": 20.0,
    "w_binding": 85.0,
    # v8-only understanding components
    "w_tinystories": 30.0,
    "w_cross_task": 30.0,
    "w_diagnostic": 45.0,
    "w_hellaswag": 30.0,
    "w_hierarchy": 15.0,
    # v8-only frontier values
    "tinystories": 0.45,
    "cross_task": 0.60,
    "diagnostic": 0.35,
    "hellaswag": 0.30,
    "hierarchy": 0.50,
}


def _scurve(ratio: float, k: float = 4.0) -> float:
    """Sigmoid S-curve centered at ratio=1.0.

    Returns 0-1:
      ratio=1.0 → 0.5 (frontier parity)
      ratio>1.0 → approaches 1.0 (better than frontier)
      ratio<1.0 → approaches 0.0 (worse than frontier)

    k controls steepness. k=4 gives:
      ratio=0.5 → 0.12,  ratio=1.5 → 0.88,  ratio=2.0 → 0.98
    """
    return 1.0 / (1.0 + math.exp(-k * (ratio - 1.0)))


# V7 frontier anchors — empirical reference metrics from 4 baseline architectures.
_V7_FRONTIER_PPL_1000 = 10.0  # screening anchor
_V7_FRONTIER_PPL_2500 = 8.6  # investigation anchor
_V7_FRONTIER_PPL_10000 = 5.6  # validation anchor
_V7_FRONTIER_PARAM_EFF = (
    _V7_FRONTIER_PPL_1000 / 28_561_920
)  # ppl/param ratio at screening
_V7_FRONTIER_LEARN_EFF = 1.13  # ppl@500 / ppl@1000
_V7_FRONTIER_LONG_CTX = 0.375  # average long context score
_V7_FRONTIER_AVG_PARAMS = 28_561_920  # average params across 4 refs
_V7_FRONTIER_CONVERGENCE = 1.30  # avg ppl@100 / ppl@500 across 4 refs
_V7_FRONTIER_BINDING = 0.35  # composite binding score across 4 refs


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
    ar_auc: Optional[float] = None,
    ar_timed_out: Optional[bool] = None,
    ar_above_chance: Optional[bool] = None,
    induction_auc: Optional[float] = None,
    binding_auc: Optional[float] = None,
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
    score = 0.0
    _bd: Optional[Dict[str, Any]] = {} if decompose else None

    def _track(key: str, pts: float) -> None:
        if _bd is not None:
            _bd[key] = pts

    # Hard gate: model that didn't learn at screening
    _best_lr = screening_lr
    if _best_lr is not None and _best_lr > INSUFFICIENT_LEARNING_LR:
        result = max(0.0, 10.0)
        if decompose:
            return {
                "composite_score": result,
                "breakdown": {"insufficient_learning_cap": 10.0},
            }
        return result

    # Hard gate: screened_out or investigation_failed
    _inv_failed = tier in ("investigation_failed", "screened_out")

    # Tier helpers
    _is_investigated = (
        tier in ("investigation", "validation", "breakthrough")
        if tier
        else (ppl_investigation is not None)
    )
    _is_validation = (
        tier in ("validation", "breakthrough") if tier else (ppl_validation is not None)
    )

    # -- S-CURVED COMPONENTS --

    # 1. Performance (short)
    perf_short = 0.0
    if not _inv_failed:
        if ppl_screening is not None and ppl_screening > 0:
            ratio = cfg["ppl_1000"] / ppl_screening
            perf_short = cfg["w_perf_short"] * _scurve(ratio)
        elif screening_lr is not None:
            perf_short = cfg["w_perf_short"] * _scurve(max(0.01, 1.0 - screening_lr))
    score += perf_short
    _track("perf_short", perf_short)

    # 2. Performance (medium)
    perf_med = 0.0
    if not _inv_failed and ppl_investigation is not None and ppl_investigation > 0:
        ratio = cfg["ppl_2500"] / ppl_investigation
        perf_med = cfg["w_perf_medium"] * _scurve(ratio)
    score += perf_med
    _track("perf_medium", perf_med)

    # 3. Performance (long)
    perf_long = 0.0
    if not _inv_failed and ppl_validation is not None and ppl_validation > 0:
        ratio = cfg["ppl_10000"] / ppl_validation
        perf_long = cfg["w_perf_long"] * _scurve(ratio)
    score += perf_long
    _track("perf_long", perf_long)

    # 4. Parameter efficiency
    param_eff_pts = 0.0
    if (
        not _inv_failed
        and param_count is not None
        and param_count > 0
        and ppl_screening is not None
        and ppl_screening > 0
    ):
        model_eff = (cfg["ppl_1000"] / ppl_screening) * (
            cfg["avg_params"] / param_count
        )
        ratio = model_eff / cfg["param_eff"]
        param_eff_pts = cfg["w_param_eff"] * _scurve(ratio)
    score += param_eff_pts
    _track("param_efficiency", param_eff_pts)

    # 5. Learning efficiency
    learn_eff_pts = 0.0
    if (
        not _inv_failed
        and ppl_at_500 is not None
        and ppl_at_1000 is not None
        and ppl_at_1000 > 0
    ):
        model_conv = ppl_at_500 / ppl_at_1000
        ratio = model_conv / cfg["learn_eff"]
        learn_eff_pts = cfg["w_learn_eff"] * _scurve(ratio)
    score += learn_eff_pts
    _track("learning_efficiency", learn_eff_pts)

    # -- ADDITIVE COMPONENTS --

    # 6. Routing savings -- max 50pts
    routing_pts = 0.0
    if routing_savings is not None:
        routing_pts = 50.0 * routing_savings
    _track("routing_savings", routing_pts)
    score += routing_pts

    # 7. Compression -- max 30pts
    comp_pts = 0.0
    if compression_ratio is not None:
        comp_pts = 20.0 * max(0, 1.0 - compression_ratio)
        if quant_quality_per_byte is not None:
            comp_pts += 10.0 * max(0, quant_quality_per_byte)
    comp_pts = min(30.0, comp_pts)
    _track("compression", comp_pts)
    score += comp_pts

    # 8. Activation sparsity -- max 30pts (20 structural + 10 activation)
    sparsity_pts = 0.0
    if n_sparse_ops is not None and n_sparse_ops > 0:
        sparsity_pts += min(20.0, n_sparse_ops * 6.0)
    if activation_sparsity is not None and activation_sparsity > 0.3:
        sparsity_pts += 10.0 * min(1.0, (activation_sparsity - 0.3) / 0.5)
    _track("sparsity", sparsity_pts)
    score += sparsity_pts

    # 9. Adaptive computation -- max 25pts
    adaptive_pts = 0.0
    if recursion_savings is not None and recursion_savings > 0:
        adaptive_pts += 15.0 * min(1.0, recursion_savings / 0.5)
    if depth_savings is not None and depth_savings > 0:
        adaptive_pts += 10.0 * min(1.0, depth_savings / 0.5)
    _track("adaptive_computation", adaptive_pts)
    score += adaptive_pts

    # 10. Novelty -- max 40pts
    novelty_pts = 0.0
    eff_nov = (
        0.0 if is_reference else (screening_nov if screening_nov is not None else 0.0)
    )
    conf = (
        0.0
        if is_reference
        else (novelty_confidence if novelty_confidence is not None else 1.0)
    )
    raw_novelty_pts = 40.0 * eff_nov * conf
    _novelty_valid = bool(kwargs.get("novelty_valid_for_promotion"))
    if not is_reference and not _novelty_valid:
        _analyses_succeeded = int(kwargs.get("analyses_succeeded", 0))
        novelty_pts = raw_novelty_pts * (0.4 + 0.6 * (_analyses_succeeded / 4))
        novelty_pts = min(novelty_pts, 40.0)
    else:
        novelty_pts = raw_novelty_pts
    _track("novelty", novelty_pts)
    score += novelty_pts

    # 11. NCD -- max 15pts
    ncd_pts = 0.0
    if ncd_score is not None and ncd_score > 0:
        ncd_pts = 15.0 * max(0, 1.0 - ncd_score)
    _track("ncd", ncd_pts)
    score += ncd_pts

    # 12. Robustness -- max 40pts, investigation+ only
    robust_pts = 0.0
    if _is_investigated:
        if spectral_norm is not None:
            robust_pts += 10.0 * max(0, 1.0 - (spectral_norm / 20.0))
        if robustness_noise is not None:
            robust_pts += 15.0 * max(0, 1.0 - robustness_noise)
        if robustness_score is not None:
            robust_pts += 15.0 * robustness_score
        if quant_retention is not None:
            robust_pts += 15.0 * max(0, quant_retention - 0.5) / 0.5
    robust_pts = min(40.0, robust_pts)
    _track("robustness", robust_pts)
    score += robust_pts

    # 13. Long context -- S-curved, max 25pts, validation tier only
    long_ctx_pts = 0.0
    if (
        _is_validation
        and long_ctx_score is not None
        and long_ctx_score > 0
        and cfg["long_ctx"] > 0
    ):
        ratio = long_ctx_score / cfg["long_ctx"]
        long_ctx_pts = 25.0 * _scurve(ratio)
    _track("long_context", long_ctx_pts)
    score += long_ctx_pts

    # 14. Early convergence -- max 10pts
    convergence_pts = 0.0
    if (
        not _inv_failed
        and ppl_at_100 is not None
        and ppl_at_500 is not None
        and ppl_at_500 > 0
    ):
        conv_ratio = ppl_at_100 / ppl_at_500
        convergence_pts = 10.0 * _scurve(conv_ratio / cfg["convergence"])
    _track("early_convergence", convergence_pts)
    score += convergence_pts

    # 15. Speed/Latency -- max 25pts
    speed_pts = 0.0
    if not _inv_failed:
        if throughput_tok_s is not None and throughput_tok_s > 0:
            ratio = throughput_tok_s / GPT2_REF["throughput_tok_s"]
            speed_pts = 25.0 * _scurve(ratio)
        elif forward_time_ms is not None and forward_time_ms > 0:
            ratio = GPT2_REF["forward_time_ms"] / forward_time_ms
            speed_pts = 25.0 * _scurve(ratio)
    _track("speed", speed_pts)
    score += speed_pts

    # 16. Binding probes
    _effective_ar_auc = None if ar_timed_out else ar_auc
    binding_pts = 0.0
    if not _inv_failed:
        _bc = 0.0
        _bc_n = 0
        if _effective_ar_auc is not None:
            _bc += 0.4 * _effective_ar_auc
            _bc_n += 1
        if induction_auc is not None:
            _bc += 0.3 * induction_auc
            _bc_n += 1
        if binding_auc is not None:
            _bc += 0.3 * binding_auc
            _bc_n += 1
        if _bc > 0 and _bc_n > 0:
            ratio = _bc / cfg["binding"]
            binding_pts = cfg["w_binding"] * _scurve(ratio, k=6)
    binding_pts = min(cfg["w_binding"], binding_pts)
    _track("binding", binding_pts)
    score += binding_pts

    # 17. BLiMP linguistic competence -- max 40pts
    blimp_pts = 0.0
    if not _inv_failed and blimp_accuracy is not None and blimp_accuracy > 0.50:
        ratio = blimp_accuracy / cfg["blimp"]
        blimp_pts = 40.0 * _scurve(ratio, k=6)
    blimp_pts = min(40.0, blimp_pts)
    _track("blimp", blimp_pts)
    score += blimp_pts

    # -- v8 UNDERSTANDING COMPONENTS (weights are 0 in v7 config) --

    # 18. TinyStories -- validation+ only
    w_ts = cfg["w_tinystories"]
    if w_ts > 0:
        tinystories_pts = 0.0
        if _is_validation and tinystories_score is not None and tinystories_score > 0:
            ratio = tinystories_score / cfg["tinystories"]
            tinystories_pts = w_ts * _scurve(ratio)
        tinystories_pts = min(w_ts, tinystories_pts)
        _track("tinystories", tinystories_pts)
        score += tinystories_pts

    # 19. Cross-task robustness -- investigation+ only
    w_ct = cfg["w_cross_task"]
    if w_ct > 0:
        cross_task_pts = 0.0
        if _is_investigated and cross_task_score is not None and cross_task_score > 0:
            ratio = cross_task_score / cfg["cross_task"]
            cross_task_pts = w_ct * _scurve(ratio)
        cross_task_pts = min(w_ct, cross_task_pts)
        _track("cross_task", cross_task_pts)
        score += cross_task_pts

    # 20. Diagnostic tasks -- validation+ only
    w_diag = cfg["w_diagnostic"]
    if w_diag > 0:
        diag_pts = 0.0
        if _is_validation and diagnostic_score is not None and diagnostic_score > 0:
            ratio = diagnostic_score / cfg["diagnostic"]
            diag_pts = w_diag * _scurve(ratio)
        diag_pts = min(w_diag, diag_pts)
        _track("diagnostic", diag_pts)
        score += diag_pts

    # 21. HellaSwag reasoning -- investigation+ only
    w_hs = cfg["w_hellaswag"]
    if w_hs > 0:
        hellaswag_pts = 0.0
        _hellaswag_acc = hellaswag_acc_investigation or hellaswag_acc_validation
        if _is_investigated and _hellaswag_acc is not None and _hellaswag_acc > 0.26:
            ratio = _hellaswag_acc / cfg["hellaswag"]
            hellaswag_pts = w_hs * _scurve(ratio, k=6)
        hellaswag_pts = min(w_hs, hellaswag_pts)
        _track("hellaswag", hellaswag_pts)
        score += hellaswag_pts

    # 22. Hierarchy probe -- investigation+ only
    w_hier = cfg["w_hierarchy"]
    if w_hier > 0:
        hierarchy_pts = 0.0
        if _is_investigated and hierarchy_fitness is not None and hierarchy_fitness > 0:
            ratio = hierarchy_fitness / cfg["hierarchy"]
            hierarchy_pts = w_hier * _scurve(ratio)
        hierarchy_pts = min(w_hier, hierarchy_pts)
        _track("hierarchy", hierarchy_pts)
        score += hierarchy_pts

    # -- PENALTIES --

    # Binding soft gate -- 3-signal AND
    _binding_penalty = 1.0
    _induction_below = (
        induction_auc is not None and induction_auc < BINDING_INDUCTION_SOFT_GATE
    )
    _binding_below = (
        binding_auc is not None and binding_auc < BINDING_BINDING_AUC_SOFT_GATE
    )
    _ar_below = (
        _effective_ar_auc is not None
        and _effective_ar_auc < BINDING_AR_SOFT_GATE
        and not ar_above_chance
    )
    _binding_signals_measured = sum(
        [
            induction_auc is not None,
            binding_auc is not None,
            _effective_ar_auc is not None,
        ]
    )
    _all_below = _binding_signals_measured >= 2 and all(
        [
            _induction_below or induction_auc is None,
            _binding_below or binding_auc is None,
            _ar_below or _effective_ar_auc is None,
        ]
    )
    if not _inv_failed and _binding_signals_measured >= 2 and _all_below:
        _binding_penalty = BINDING_LOCAL_ONLY_PENALTY
        score *= _binding_penalty
    _track(
        "binding_local_only_penalty",
        _binding_penalty if _binding_penalty < 1.0 else 0.0,
    )

    # Param-size penalty
    _TARGET_PARAMS = 5_000_000  # ~5M = GPT-2 small block budget
    param_penalty = 1.0
    if param_count is not None and param_count > _TARGET_PARAMS:
        param_ratio = param_count / _TARGET_PARAMS
        param_penalty = 1.0 / (param_ratio**0.13)
        score *= param_penalty
    _track("param_size_penalty", param_penalty if param_penalty < 1.0 else 0.0)

    final = max(0.0, score)

    if decompose:
        if _inv_failed:
            _bd["_investigation_failed_penalty"] = True  # type: ignore[assignment]
        if param_penalty < 1.0:
            _bd["param_size_penalty_multiplier"] = param_penalty  # type: ignore[assignment]
        return {"composite_score": final, "breakdown": _bd}
    return final


def compute_composite_v7(
    *,
    decompose: bool = False,
    **kw: Any,
) -> Union[float, Dict[str, Any]]:
    """Composite score v7 -- 18 components, 710pt max (before penalties).

    8 S-curved components use frontier reference averages as anchor.
    7 additive components use old v5 formulas (frontier refs score 0).
    1 binding range component (120pt, S-curved).
    1 binding soft penalty (0.80x, 3-signal AND).
    1 multiplicative param-size penalty for oversized models.

    See BASELINE_SCORING_DO_NOT_DELETE.md for the full spec.
    """
    return _compute_composite_generic(_V7_CONFIG, decompose=decompose, **kw)


def compute_composite_v8(
    *,
    decompose: bool = False,
    **kw: Any,
) -> Union[float, Dict[str, Any]]:
    """Composite score v8 -- rebalanced for understanding capability.

    Key changes from v7:
    - Perplexity-derived: 310pts -> 210pts (46% -> 26%)
    - Binding probes: 120pts -> 85pts
    - NEW: TinyStories (30pts), cross-task (30pts), diagnostic (45pts),
      HellaSwag (30pts), hierarchy (15pts) = 150pts of new understanding
    - Understanding hard gate added at promotion (separate from scoring)
    """
    return _compute_composite_generic(_V8_CONFIG, decompose=decompose, **kw)


# ---------------------------------------------------------------------------
# Version dispatcher
# ---------------------------------------------------------------------------
SCORING_VERSION = "v8"


def compute_composite(
    *, decompose: bool = False, **kw: Any
) -> Union[float, Dict[str, Any]]:
    """Dispatch to the active scoring version."""
    if SCORING_VERSION == "v8":
        return compute_composite_v8(decompose=decompose, **kw)
    return compute_composite_v7(decompose=decompose, **kw)
