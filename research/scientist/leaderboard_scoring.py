"""Leaderboard scoring functions — pure arithmetic, no I/O.

All functions are static / module-level. Candidates for future Cython port.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Sequence, Union

from .thresholds import (
    INSUFFICIENT_LEARNING_LR,
    SPECTRAL_NORM_FLOOR,
    STRUCTURAL_ONLY_NOVELTY_CAP,
)


# GPT-2 reference metrics (measured on d_model=256, 6-layer config)
_GPT2_REF = {
    "loss_ratio": 0.2646,
    "param_count": 9_767_424,
    "flops_forward": 19_534_848,
    "throughput_tok_s": 1_200_845,
    "peak_memory_mb": 115.0,
    "forward_time_ms": 0.43,
}

_WIKITEXT_REF_SCORE_FLOOR = 0.5868
_WIKITEXT_REF_PPL_CEILING = 72.68


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
    ref = _GPT2_REF
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
    "fingerprint_json"
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


def compute_composite_score(
    screening_lr: Optional[float] = None,
    screening_nov: Optional[float] = None,
    inv_lr: Optional[float] = None,
    inv_robust: Optional[float] = None,
    val_lr: Optional[float] = None,
    val_baseline: Optional[float] = None,
    val_std: Optional[float] = None,
    robustness_score: Optional[float] = None,
    is_unstable: Optional[bool] = None,
    novelty_confidence: Optional[float] = None,
    scaling_param_efficiency: Optional[float] = None,
    is_reference: bool = False,
    routing_savings: Optional[float] = None,
    compression_ratio: Optional[float] = None,
    entropy: Optional[float] = None,
    discovery_lr: Optional[float] = None,
    spectral_norm: Optional[float] = None,
    robustness_noise: Optional[float] = None,
    quant_retention: Optional[float] = None,
    long_ctx_score: Optional[float] = None,
    init_std: Optional[float] = None,
    loss_improvement_rate: Optional[float] = None,
    quant_quality_per_byte: Optional[float] = None,
    ncd_score: Optional[float] = None,
    n_routing_ops: Optional[int] = None,
    n_sparse_ops: Optional[int] = None,
    n_moe_ops: Optional[int] = None,
    recursion_savings: Optional[float] = None,
    depth_savings: Optional[float] = None,
    activation_sparsity: Optional[float] = None,
    max_viable_seq_len: Optional[int] = None,
    long_ctx_scaling: Optional[float] = None,
    long_ctx_passkey: Optional[float] = None,
    long_ctx_multi_hop: Optional[float] = None,
    long_ctx_assoc: Optional[float] = None,
    routing_expert_count: Optional[int] = None,
    routing_confidence_mean: Optional[float] = None,
    routing_drop_rate: Optional[float] = None,
    routing_fast_fraction: Optional[float] = None,
    routing_balance_score: Optional[float] = None,
    replication_n: Optional[int] = None,
    replication_loss_mean: Optional[float] = None,
    replication_loss_std: Optional[float] = None,
    replication_best_vs_mean_gap: Optional[float] = None,
    wikitext_score: Optional[float] = None,
    peak_ppl: Optional[float] = None,
    ppl_500: Optional[float] = None,
    steps_to_divergence: Optional[int] = None,
    investigation_passed: Optional[bool] = None,
    validation_passed: Optional[bool] = None,
    decompose: bool = False,
    **kwargs: Any,
) -> Union[float, Dict[str, Any]]:
    """Compute "Total Scientific Utility" -- an open-ended additive score.

    Scoring Contract
    ================
    - The composite score is a **ranking signal**, not a scientific claim.
    - Event occurrence (investigated, validated) does **not** affect score.
      Only measurable outputs from evaluation stages affect score.
    - **First-class contributors**: loss performance, novelty, routing
      savings, sparsity, compression, learning efficiency, scaling.
    - Loss is important but **not dominant** -- capped at 100pts in a
      system where exotic architectures can reach 300+.
    - **Novelty confidence** (``novelty_confidence``) scales the novelty
      contribution -- structural-only novelty (conf=0.2) contributes at
      most 8 of the 40 available novelty points.
    - **Promotion eligibility** is separate from score. A high-scoring
      entry can still be ``novelty_valid_for_promotion=False``.
    - ``decompose=True`` returns a dict with per-component breakdown
      for full auditability.

    Score Budget (approximate maximums)
    ------------------------------------
    Performance:       100  |  Novelty:            40
    Discovery:          20  |  Learning efficiency: 20
    Scaling efficiency: ~90 |  Routing savings:     50
    Compression:        30  |  NCD:                 15
    Routing ops:        15  |  Sparsity:            30
    MoE quality:       ~25  |  Adaptive compute:    25
    Robustness:         40  |  Long context:        70
    Seq-len bonus:      20  |
    """
    score = 0.0
    _bd: Dict[str, float] = {} if decompose else None  # type: ignore[assignment]

    def _track(key: str, before: float) -> None:
        if _bd is not None:
            _bd[key] = score - before

    # 1. Performance Utility (Primary)
    # When replicated evidence exists (n >= 3), replace the single-run
    # screening_lr with loss_mean so ALL downstream references use the
    # aggregate.  Higher tiers already have multi-run evidence.
    _repl_n = int(replication_n or 0) if replication_n is not None else 0
    if _repl_n >= 3 and replication_loss_mean is not None and not is_reference:
        screening_lr = replication_loss_mean
        if _bd is not None:
            _bd["screening_lr_replaced_by_mean"] = True
            _bd["original_screening_lr"] = kwargs.get(
                "_original_screening_lr", screening_lr
            )

    _s0 = score
    if val_baseline is not None:
        perf_lr = val_baseline
        perf_confidence = 1.0
    elif val_lr is not None:
        perf_lr = val_lr
        perf_confidence = 1.0
    elif inv_lr is not None:
        perf_lr = inv_lr
        perf_confidence = 0.85
    elif screening_lr is not None:
        perf_lr = screening_lr
        perf_confidence = 0.65
    else:
        perf_lr = None
        perf_confidence = 0.0

    # HARD GATE: model that didn't learn cannot score high.
    # loss_ratio is normalized (final_loss / ln(vocab_size)), so:
    #   0.9 = model barely improved from random (10% reduction)
    #   0.95 = essentially no learning
    # Prevents inflated scores from novelty/routing when model is broken.
    _insufficient_learning_cap: Optional[float] = None
    if perf_lr is not None and perf_lr > INSUFFICIENT_LEARNING_LR:
        _insufficient_learning_cap = 10.0
    elif perf_lr is not None and perf_lr > 0.9:
        _insufficient_learning_cap = 20.0

    if perf_lr is not None:
        perf_norm = max(0.0, min(1.0, 1.0 - perf_lr))
        score += 100.0 * (perf_norm**1.6) * perf_confidence
    _track("performance", _s0)

    # Discovery channel (random tokens)
    _s0 = score
    if discovery_lr is not None:
        score += 20.0 * max(0, 1.0 - discovery_lr)
    _track("discovery", _s0)

    # Learning Efficiency
    _s0 = score
    if loss_improvement_rate is not None:
        score += 20.0 * max(0, min(1.0, loss_improvement_rate))
    _track("learning_efficiency", _s0)

    # 2. Novelty Utility — source-aware scoring
    # Full CKA + behavioral blend: up to 40 points
    # Structural-only (pre-completion or CKA degenerate): capped at 15 points
    _s0 = score
    eff_nov = (
        1.0 if is_reference else (screening_nov if screening_nov is not None else 0.0)
    )
    conf = (
        1.0
        if is_reference
        else (novelty_confidence if novelty_confidence is not None else 1.0)
    )
    novelty_gate = 1.0
    if perf_lr is not None:
        # Floor at 0.3: novel architectures that haven't converged still get
        # 30% novelty credit. Without this floor, the gate creates a degenerate
        # fitness landscape where novelty can never overcome a loss deficit.
        novelty_gate = min(1.0, 0.3 + 0.7 * max(0.0, (0.9 - perf_lr) / 0.6))
    _raw_novelty_pts = 40.0 * eff_nov * conf * novelty_gate
    # Cap structural-only novelty: if post-investigation fingerprint has not
    # been completed, or CKA is invalid, novelty contribution is capped.
    # Calibrated against c9c7075e741a8790: structural_novelty=0.381 should
    # produce ~5.7 pts, not ~32 pts from fake CKA inflation.
    _novelty_cap = float(
        kwargs.get("novelty_structural_only_cap", STRUCTURAL_ONLY_NOVELTY_CAP)
    )
    _fp_completed = bool(kwargs.get("fingerprint_completed_post_investigation"))
    _novelty_valid = bool(kwargs.get("novelty_valid_for_promotion"))
    if is_reference:
        score += _raw_novelty_pts
    elif _fp_completed and _novelty_valid:
        # Full CKA + behavioral blend — uncapped
        score += _raw_novelty_pts
    else:
        # Graduated scaling: partial probe completion earns partial credit.
        _analyses_succeeded = int(kwargs.get("analyses_succeeded", 0))
        _graduated = _raw_novelty_pts * (0.4 + 0.6 * (_analyses_succeeded / 4))
        score += min(_graduated, 40.0)
    _track("novelty", _s0)

    # 3. Efficiency & Scaling Utility
    _s0 = score
    if scaling_param_efficiency is not None:
        eff_above_1 = max(0.0, scaling_param_efficiency - 1.0)
        score += 25.0 * math.sqrt(eff_above_1)
        if scaling_param_efficiency >= 5.0:
            score += 30.0
        if scaling_param_efficiency >= 10.0:
            score += 20.0
    _track("scaling_efficiency", _s0)

    # Compound efficiency bonus: routing + sparse + high efficiency
    _s0 = score
    if (
        n_routing_ops is not None
        and n_routing_ops >= 1
        and n_sparse_ops is not None
        and n_sparse_ops >= 1
        and scaling_param_efficiency is not None
        and scaling_param_efficiency >= 3.0
    ):
        score += 15.0
    _track("compound_efficiency", _s0)

    # Efficiency amplifier gate: reward extreme efficiency
    if scaling_param_efficiency is not None and scaling_param_efficiency > 3.0:
        amplifier = 1.0 + 0.05 * min(5, scaling_param_efficiency - 3.0)
        score *= amplifier

    _s0 = score
    if routing_savings is not None:
        score += 50.0 * routing_savings
        # Routing overhead penalty: penalize wasteful routing that adds
        # complexity without savings AND without improving loss.
        if routing_savings < 0.05:
            best_lr = (
                val_baseline
                if val_baseline is not None
                else (
                    val_lr
                    if val_lr is not None
                    else (inv_lr if inv_lr is not None else screening_lr)
                )
            )
            if best_lr is not None and best_lr > INSUFFICIENT_LEARNING_LR:
                waste = 0.05 - routing_savings
                score -= 30.0 * waste  # max penalty ~1.5 at savings=0
    _track("routing_savings", _s0)

    # Routing efficiency bonus: reward models that route effectively
    _s0 = score
    if routing_fast_fraction is not None and routing_fast_fraction > 0.05:
        # Up to 20pts for routing a significant fraction of tokens to fast paths
        score += 20.0 * min(1.0, routing_fast_fraction)
        # Balance bonus: up to 10pts for well-distributed routing
        if routing_balance_score is not None and routing_balance_score > 0.5:
            score += 10.0 * routing_balance_score
        # Multiplicative efficiency bonus when routing saves compute AND loss is good
        if scaling_param_efficiency is not None and scaling_param_efficiency > 1.0:
            routing_eff = 1.0 + routing_fast_fraction * 0.5
            score *= min(1.3, routing_eff)  # cap at 30% boost
    _track("routing_efficiency", _s0)

    _s0 = score
    if compression_ratio is not None:
        comp_score = 20.0 * max(0, 1.0 - (compression_ratio / 1.0))
        if quant_quality_per_byte is not None:
            comp_score += 10.0 * max(0, quant_quality_per_byte)
        score += comp_score
    _track("compression", _s0)

    _s0 = score
    if ncd_score is not None:
        score += 15.0 * max(0, 1.0 - ncd_score)
    _track("ncd", _s0)

    # 3b. Structural complexity bonus
    _s0 = score
    if n_routing_ops is not None and n_routing_ops > 0:
        score += min(15.0, n_routing_ops * 5.0)
    _track("routing_ops", _s0)

    # 3c. Sparsity bonus (max 30pts: 20 structural + 10 activation)
    _s0 = score
    if n_sparse_ops is not None and n_sparse_ops > 0:
        score += min(20.0, n_sparse_ops * 6.0)
    if activation_sparsity is not None and activation_sparsity > 0.3:
        score += 10.0 * min(1.0, (activation_sparsity - 0.3) / 0.5)
    _track("sparsity", _s0)

    # 3d. MoE quality bonus (max ~25pts)
    _s0 = score
    if n_moe_ops is not None and n_moe_ops > 0:
        moe_base = min(10.0, n_moe_ops * 5.0)
        if routing_expert_count is not None and routing_expert_count > 1:
            expert_mult = min(1.5, 1.0 + math.log2(routing_expert_count) / 6.0)
            moe_base *= expert_mult
        if routing_confidence_mean is not None and routing_confidence_mean > 0.5:
            moe_base *= 1.0 + 0.3 * (routing_confidence_mean - 0.5)
        if routing_drop_rate is not None and routing_drop_rate > 0.3:
            moe_base *= max(0.5, 1.0 - (routing_drop_rate - 0.3))
        score += moe_base
    _track("moe_quality", _s0)

    # 3e. Adaptive computation bonus (max 25pts)
    _s0 = score
    if recursion_savings is not None and recursion_savings > 0:
        score += 15.0 * min(1.0, recursion_savings / 0.5)
    if depth_savings is not None and depth_savings > 0:
        score += 10.0 * min(1.0, depth_savings / 0.5)
    _track("adaptive_computation", _s0)

    # 4. Robustness & Stability Utility
    _s0 = score
    if spectral_norm is not None:
        score += 10.0 * max(0, 1.0 - (spectral_norm / 20.0))
    if robustness_noise is not None:
        score += 15.0 * max(0, 1.0 - robustness_noise)
    if robustness_score is not None:
        score += 15.0 * robustness_score
    if quant_retention is not None:
        score += 15.0 * max(0, quant_retention - 0.5) / 0.5
    _track("robustness", _s0)

    # 4b. Expanded long-context scoring (total budget 50pts, up from 20)
    _s0 = score
    if long_ctx_score is not None:
        score += 20.0 * long_ctx_score
        if long_ctx_passkey is not None:
            score += 10.0 * long_ctx_passkey
        if long_ctx_multi_hop is not None:
            score += 10.0 * long_ctx_multi_hop
        if long_ctx_scaling is not None:
            score += 5.0 * long_ctx_scaling
        if long_ctx_assoc is not None:
            score += 5.0 * long_ctx_assoc
    if max_viable_seq_len is not None and max_viable_seq_len > 512:
        score += 5.0 * min(4.0, math.log2(max_viable_seq_len / 512))
    _track("long_context", _s0)

    # 5. Generalization Utility (The "Anti-Cheat")
    _s0 = score
    wikitext_perplexity = kwargs.get("wikitext_perplexity")
    # Use peak_ppl (best PPL at any trajectory checkpoint) when available;
    # fall back to single-point wikitext_perplexity for entries without
    # trajectory data.
    effective_ppl = peak_ppl if peak_ppl is not None else wikitext_perplexity
    if wikitext_score is not None:
        score += 35.0 * max(0.0, min(1.0, wikitext_score))
    # Trajectory capability bonus: reward models whose peak_ppl beats
    # reference ceiling.  Up to +20 pts for peak_ppl approaching 1.0.
    # Only applies when trajectory data exists (peak_ppl is not None).
    # Score formula: log(vocab/ppl)/log(vocab), vocab=32000.
    if peak_ppl is not None and peak_ppl > 0:
        _vocab = 32000
        _peak_score = max(0.0, math.log(_vocab / peak_ppl) / math.log(_vocab))
        score += 20.0 * min(1.0, _peak_score)
    # Generalization stability bonus: models that never diverge within
    # their eval budget get a bonus.  NULL steps_to_divergence = stable.
    if peak_ppl is not None and steps_to_divergence is None:
        score += 10.0
    if effective_ppl is not None:
        if effective_ppl > 1000000:
            if decompose:
                _bd["generalization"] = 0.0
                _bd["_disqualified"] = True  # type: ignore[assignment]
                return {"composite_score": 0.0, "breakdown": _bd}
            return 0.0
        if effective_ppl > _WIKITEXT_REF_PPL_CEILING:
            over_ref = min(
                2.0,
                (effective_ppl - _WIKITEXT_REF_PPL_CEILING)
                / max(_WIKITEXT_REF_PPL_CEILING, 1e-6),
            )
            score -= 20.0 * over_ref
        if effective_ppl > 1000:
            score -= 50.0 * math.log10(effective_ppl / 1000.0)
    _track("generalization", _s0)

    # 5b. Investigation robustness — positive signal + reliability penalty.
    _s0 = score
    # Positive: gracefully degrading reward for investigation robustness.
    # +10 for fully robust (1.0), +3.3 for 1/3 pass rate, +0 for untested.
    if inv_robust is not None and inv_robust > 0:
        score += 10.0 * inv_robust
    # Negative: penalty for investigation failure WITHOUT strong real-token
    # evidence.  Models that fail investigation but beat references on
    # WikiText should not be penalised as hard.
    inv_failed = investigation_passed is False and validation_passed is not True
    if inv_failed:
        # Cap penalty for models with frontier-competitive WikiText scores
        has_frontier_evidence = (
            wikitext_score is not None and wikitext_score >= _WIKITEXT_REF_SCORE_FLOOR
        )
        if inv_robust is not None and inv_robust < 0.5 and not has_frontier_evidence:
            score -= 25.0 * min(1.0, (0.5 - inv_robust) / 0.5)
        if wikitext_score is not None and wikitext_score < _WIKITEXT_REF_SCORE_FLOOR:
            score -= 20.0 * min(
                1.0,
                (_WIKITEXT_REF_SCORE_FLOOR - wikitext_score)
                / max(_WIKITEXT_REF_SCORE_FLOOR, 1e-6),
            )
        elif wikitext_score is None and wikitext_perplexity is None:
            score -= 8.0
    _track("investigation_reliability", _s0)

    # 6. Numerical Integrity (Spectral Floor)
    _s0 = score
    if spectral_norm is not None and spectral_norm < SPECTRAL_NORM_FLOOR:
        score -= 40.0
    _track("spectral_floor_penalty", _s0)

    # 7. Penalties
    _s0 = score
    if val_std is not None and val_std > 0.1:
        score -= 50.0 * min(2.0, val_std / 0.5)
    if is_unstable:
        score -= 40.0
    if entropy is not None and entropy > 0.95:
        score -= 5.0 * (entropy - 0.95)
    _track("penalties", _s0)

    # Scaling gate: mild penalty for sub-baseline efficiency (non-reference).
    # Floor at 0.5 prevents crushing novel architectures that haven't been
    # profiled yet. The old floor of 0.1 could destroy 90% of a score.
    if (
        not is_reference
        and scaling_param_efficiency is not None
        and scaling_param_efficiency < 1.0
    ):
        pre_gate = score
        gate = max(0.5, scaling_param_efficiency)
        score *= gate
        if _bd is not None:
            _bd["scaling_gate_multiplier"] = gate
            _bd["scaling_gate_reduction"] = score - pre_gate

    final = max(0.0, score)

    # Apply insufficient-learning cap: prevents inflated scores from
    # novelty/routing/scaling when the model didn't actually learn.
    if _insufficient_learning_cap is not None:
        final = min(final, _insufficient_learning_cap)
        if _bd is not None:
            _bd["insufficient_learning_cap"] = _insufficient_learning_cap

    # Replication confidence dampening: scores from < 3 replicated runs
    # are dampened to prevent lucky single-run outliers from dominating.
    # sqrt(n/3) ramps: n=1 → 0.577, n=2 → 0.816, n=3 → 1.0.
    if not is_reference and _repl_n > 0 and _repl_n < 3:
        repl_confidence = math.sqrt(_repl_n / 3.0)
        pre_repl = final
        final *= repl_confidence
        if _bd is not None:
            _bd["replication_confidence"] = repl_confidence
            _bd["replication_dampening"] = final - pre_repl
    elif _bd is not None and _repl_n >= 3:
        _bd["replication_confidence"] = 1.0

    # Lucky-outlier penalty: if best run is much better than mean,
    # the best run was likely noise. Penalize the gap.
    if (
        not is_reference
        and replication_best_vs_mean_gap is not None
        and _repl_n >= 2
        and replication_best_vs_mean_gap > 0.1
    ):
        outlier_penalty = min(20.0, (replication_best_vs_mean_gap - 0.1) * 200.0)
        final = max(0.0, final - outlier_penalty)
        if _bd is not None:
            _bd["outlier_penalty"] = -outlier_penalty

    if decompose:
        return {"composite_score": final, "breakdown": _bd}
    return final


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
# v7 scoring: 13-component composite with perplexity anchors + additive
# See BASELINE_SCORING_DO_NOT_DELETE.md for full design rationale.
# ---------------------------------------------------------------------------

# Frontier reference averages (measured 2026-03-23 on wiki103)
_V7_FRONTIER_PPL_1000 = 10.0  # screening anchor
_V7_FRONTIER_PPL_2500 = 8.6  # investigation anchor
_V7_FRONTIER_PPL_10000 = 5.6  # validation anchor
_V7_FRONTIER_PARAM_EFF = (
    1.09  # (frontier_ppl/model_ppl) * (frontier_params/model_params)
)
_V7_FRONTIER_LEARN_EFF = 1.13  # ppl@500 / ppl@1000
_V7_FRONTIER_LONG_CTX = 0.375  # average long context score
_V7_FRONTIER_AVG_PARAMS = 28_561_920  # average params across 4 refs
_V7_FRONTIER_CONVERGENCE = 1.30  # avg ppl@100 / ppl@500 across 4 refs


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


def compute_composite_v7(
    # S-curved inputs (perplexity, efficiency)
    ppl_screening: Optional[float] = None,  # wiki103 ppl at 1000 steps
    ppl_investigation: Optional[float] = None,  # wiki103 ppl at 2500 steps
    ppl_validation: Optional[float] = None,  # wiki103 ppl at 10000 steps
    param_count: Optional[float] = None,  # total model parameters
    ppl_at_100: Optional[float] = None,  # for early convergence
    ppl_at_500: Optional[float] = None,  # for learning efficiency + convergence
    ppl_at_1000: Optional[float] = None,  # for learning efficiency
    long_ctx_score: Optional[float] = None,  # combined long context score
    # Additive inputs (routing/compression/sparsity — from v5)
    routing_savings: Optional[float] = None,
    routing_fast_fraction: Optional[float] = None,
    routing_balance_score: Optional[float] = None,
    compression_ratio: Optional[float] = None,
    quant_quality_per_byte: Optional[float] = None,
    n_sparse_ops: Optional[int] = None,
    activation_sparsity: Optional[float] = None,
    recursion_savings: Optional[float] = None,
    depth_savings: Optional[float] = None,
    # Novelty (from v5)
    screening_nov: Optional[float] = None,
    novelty_confidence: Optional[float] = None,
    is_reference: bool = False,
    # NCD
    ncd_score: Optional[float] = None,
    # Robustness (from v5)
    spectral_norm: Optional[float] = None,
    robustness_noise: Optional[float] = None,
    robustness_score: Optional[float] = None,
    quant_retention: Optional[float] = None,
    # Loss ratio fallback (for entries without ppl)
    screening_lr: Optional[float] = None,
    # Speed/throughput
    throughput_tok_s: Optional[float] = None,
    forward_time_ms: Optional[float] = None,
    # Tier context (controls which components are active)
    tier: Optional[str] = None,  # "screening", "investigation", "validation"
    # Control
    decompose: bool = False,
    **kwargs: Any,
) -> Union[float, Dict[str, Any]]:
    """Composite score v7 — 16 components, 590pt max (before param penalty).

    6 S-curved components use frontier reference averages as anchor.
    7 additive components use old v5 formulas (frontier refs score 0).
    1 multiplicative param-size penalty for oversized models.

    See BASELINE_SCORING_DO_NOT_DELETE.md for the full spec.
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

    # Hard gate: screened_out or investigation_failed — model doesn't learn.
    # Zero all performance components — only structural/novelty points survive.
    _inv_failed = tier in ("investigation_failed", "screened_out")

    # ── S-CURVED COMPONENTS ──────────────────────────────────────────

    # 1. Performance (short) — screening, 1000 steps, max 50pts
    perf_short = 0.0
    if not _inv_failed:
        if ppl_screening is not None and ppl_screening > 0:
            ratio = _V7_FRONTIER_PPL_1000 / ppl_screening
            perf_short = 50.0 * _scurve(ratio)
        elif screening_lr is not None:
            # Fallback: estimate from loss_ratio (rough mapping)
            perf_short = 50.0 * _scurve(max(0.01, 1.0 - screening_lr))
    score += perf_short
    _track("perf_short", perf_short)

    # 2. Performance (medium) — investigation, 2500 steps, max 75pts
    perf_med = 0.0
    if not _inv_failed and ppl_investigation is not None and ppl_investigation > 0:
        ratio = _V7_FRONTIER_PPL_2500 / ppl_investigation
        perf_med = 75.0 * _scurve(ratio)
    score += perf_med
    _track("perf_medium", perf_med)

    # 3. Performance (long) — validation, 10000 steps, max 100pts
    perf_long = 0.0
    if not _inv_failed and ppl_validation is not None and ppl_validation > 0:
        ratio = _V7_FRONTIER_PPL_10000 / ppl_validation
        perf_long = 100.0 * _scurve(ratio)
    score += perf_long
    _track("perf_long", perf_long)

    # 4. Parameter efficiency — max 50pts
    param_eff_pts = 0.0
    if (
        not _inv_failed
        and param_count is not None
        and param_count > 0
        and ppl_screening is not None
        and ppl_screening > 0
    ):
        model_eff = (_V7_FRONTIER_PPL_1000 / ppl_screening) * (
            _V7_FRONTIER_AVG_PARAMS / param_count
        )
        ratio = model_eff / _V7_FRONTIER_PARAM_EFF
        param_eff_pts = 50.0 * _scurve(ratio)
    score += param_eff_pts
    _track("param_efficiency", param_eff_pts)

    # 5. Learning efficiency — max 25pts
    learn_eff_pts = 0.0
    if (
        not _inv_failed
        and ppl_at_500 is not None
        and ppl_at_1000 is not None
        and ppl_at_1000 > 0
    ):
        model_conv = ppl_at_500 / ppl_at_1000
        ratio = model_conv / _V7_FRONTIER_LEARN_EFF
        learn_eff_pts = 25.0 * _scurve(ratio)
    score += learn_eff_pts
    _track("learning_efficiency", learn_eff_pts)

    # ── ADDITIVE COMPONENTS (from v5, unchanged) ─────────────────────

    # 6. Routing savings — max 50pts
    routing_pts = 0.0
    if routing_savings is not None:
        routing_pts = 50.0 * routing_savings
    _track("routing_savings", routing_pts)
    score += routing_pts

    # 7. Compression — max 30pts
    comp_pts = 0.0
    if compression_ratio is not None:
        comp_pts = 20.0 * max(0, 1.0 - compression_ratio)
        if quant_quality_per_byte is not None:
            comp_pts += 10.0 * max(0, quant_quality_per_byte)
    comp_pts = min(30.0, comp_pts)
    _track("compression", comp_pts)
    score += comp_pts

    # 8. Activation sparsity — max 30pts (20 structural + 10 activation)
    sparsity_pts = 0.0
    if n_sparse_ops is not None and n_sparse_ops > 0:
        sparsity_pts += min(20.0, n_sparse_ops * 6.0)
    if activation_sparsity is not None and activation_sparsity > 0.3:
        sparsity_pts += 10.0 * min(1.0, (activation_sparsity - 0.3) / 0.5)
    _track("sparsity", sparsity_pts)
    score += sparsity_pts

    # 9. Adaptive computation — max 25pts
    adaptive_pts = 0.0
    if recursion_savings is not None and recursion_savings > 0:
        adaptive_pts += 15.0 * min(1.0, recursion_savings / 0.5)
    if depth_savings is not None and depth_savings > 0:
        adaptive_pts += 10.0 * min(1.0, depth_savings / 0.5)
    _track("adaptive_computation", adaptive_pts)
    score += adaptive_pts

    # 10. Novelty — max 40pts (from v5 logic)
    # References score 0 — they ARE the baseline, not novel.
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
    # Graduated novelty scaling: partial probe completion earns partial credit.
    # analyses_succeeded=4 means full CKA+behavioral → full credit (scale=1.0).
    # analyses_succeeded=0 means structural only → scale=0.4, capped at 15pts.
    _novelty_valid = bool(kwargs.get("novelty_valid_for_promotion"))
    if not is_reference and not _novelty_valid:
        _analyses_succeeded = int(kwargs.get("analyses_succeeded", 0))
        novelty_pts = raw_novelty_pts * (0.4 + 0.6 * (_analyses_succeeded / 4))
        novelty_pts = min(novelty_pts, 40.0)
    else:
        novelty_pts = raw_novelty_pts
    _track("novelty", novelty_pts)
    score += novelty_pts

    # 11. NCD — max 15pts
    # ncd_score is description_length_per_param: lower = more compressible = better
    # None means no data → 0pts (not free max)
    ncd_pts = 0.0
    if ncd_score is not None and ncd_score > 0:
        ncd_pts = 15.0 * max(0, 1.0 - ncd_score)
    _track("ncd", ncd_pts)
    score += ncd_pts

    # 12. Robustness — max 40pts, investigation/validation only
    # These metrics are only measured during investigation runs.
    robust_pts = 0.0
    _is_investigated = (
        tier in ("investigation", "validation", "breakthrough")
        if tier
        else (ppl_investigation is not None)
    )
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

    # 13. Long context — S-curved, max 25pts, validation tier only
    long_ctx_pts = 0.0
    _is_validation = (
        tier in ("validation", "breakthrough") if tier else (ppl_validation is not None)
    )
    if (
        _is_validation
        and long_ctx_score is not None
        and long_ctx_score > 0
        and _V7_FRONTIER_LONG_CTX > 0
    ):
        ratio = long_ctx_score / _V7_FRONTIER_LONG_CTX
        long_ctx_pts = 25.0 * _scurve(ratio)
    _track("long_context", long_ctx_pts)
    score += long_ctx_pts

    # 14. Early convergence — max 10pts
    # ppl@100 / ppl@500 ratio, S-curved against frontier avg of 1.30
    convergence_pts = 0.0
    if (
        not _inv_failed
        and ppl_at_100 is not None
        and ppl_at_500 is not None
        and ppl_at_500 > 0
    ):
        conv_ratio = ppl_at_100 / ppl_at_500
        convergence_pts = 10.0 * _scurve(conv_ratio / _V7_FRONTIER_CONVERGENCE)
    _track("early_convergence", convergence_pts)
    score += convergence_pts

    # 15. Speed/Latency — max 25pts
    # Compares throughput or latency against GPT-2 reference.
    # A model 5x slower than GPT-2 at same quality is not a win.
    speed_pts = 0.0
    if not _inv_failed:
        if throughput_tok_s is not None and throughput_tok_s > 0:
            ratio = throughput_tok_s / _GPT2_REF["throughput_tok_s"]
            speed_pts = 25.0 * _scurve(ratio)
        elif forward_time_ms is not None and forward_time_ms > 0:
            ratio = _GPT2_REF["forward_time_ms"] / forward_time_ms
            speed_pts = 25.0 * _scurve(ratio)
    _track("speed", speed_pts)
    score += speed_pts

    # 16. Param-size penalty — multiplicative, penalizes oversized models.
    # A 3x-over-budget model gets ~15% score reduction; prevents capacity-
    # driven wins where big models score high purely from having more params.
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
