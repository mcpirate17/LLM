"""Leaderboard scoring functions — pure arithmetic, no I/O.

All functions are static / module-level. Candidates for future Cython port.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

from .leaderboard_schema import SCORE_COLUMN_MAP

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
) -> Optional[Dict[str, float]]:
    """Geometric mean of per-dimension ratios vs GPT-2.

    All ratios >1.0 = better than GPT-2. Requires at least 3 of 6
    dimensions to return a result (graceful with missing data).
    Returns dict with per-dimension ratios and ``geomean``, or None.
    """
    ref = _GPT2_REF
    ratios: Dict[str, float] = {}

    if loss_ratio is not None and loss_ratio > 0:
        ratios["x_quality"] = ref["loss_ratio"] / loss_ratio
    if param_count is not None and param_count > 0:
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


def build_score_kwargs(
    conn: Any,
    notebook: Any,
    result_id: str,
    d: Dict[str, Any],
    is_reference: bool,
    **extra: Any,
) -> Dict[str, Any]:
    """Build the kwargs dict for ``compute_composite_score`` from a row dict.

    Centralises the column->parameter mapping so callers stay DRY.
    ``conn`` is a sqlite3 connection, ``notebook`` provides helper methods.
    """
    pr = conn.execute(
        "SELECT novelty_confidence, loss_improvement_rate, graph_json "
        "FROM program_results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    nov_conf = d.get("novelty_confidence")
    if nov_conf is None and pr:
        nov_conf = pr["novelty_confidence"]

    lir = d.get("loss_improvement_rate")
    if lir is None and pr:
        lir = pr["loss_improvement_rate"]

    structural_counts = notebook._graph_structural_counts(
        result_id,
        graph_json=pr["graph_json"] if pr else None,
    )

    kw: Dict[str, Any] = {
        param: d.get(col) for col, param in SCORE_COLUMN_MAP.items()
    }
    # Fields not in SCORE_COLUMN_MAP (require special handling).
    kw["novelty_confidence"] = nov_conf
    kw["scaling_param_efficiency"] = (
        d.get("scaling_param_efficiency") or d.get("efficiency_multiple")
    )
    kw["is_reference"] = is_reference
    kw["loss_improvement_rate"] = lir
    kw["n_routing_ops"] = structural_counts.get("routing")
    kw["n_sparse_ops"] = structural_counts.get("sparse")
    kw["n_moe_ops"] = structural_counts.get("moe")
    # Routing efficiency metrics stored in program_results metadata
    if pr:
        _pr_row = dict(pr) if hasattr(pr, "keys") else {}
        kw.setdefault("routing_fast_fraction", _pr_row.get("routing_fast_fraction"))
        kw.setdefault("routing_balance_score", _pr_row.get("routing_balance_score"))
    kw.update(extra)
    return kw


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
    wikitext_score: Optional[float] = None,
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
    if perf_lr is not None:
        perf_norm = max(0.0, min(1.0, 1.0 - perf_lr))
        score += 100.0 * (perf_norm ** 1.6) * perf_confidence
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

    # 2. Novelty Utility
    _s0 = score
    eff_nov = 1.0 if is_reference else (screening_nov if screening_nov is not None else 0.0)
    conf = 1.0 if is_reference else (novelty_confidence if novelty_confidence is not None else 1.0)
    novelty_gate = 1.0
    if perf_lr is not None:
        # Floor at 0.3: novel architectures that haven't converged still get
        # 30% novelty credit. Without this floor, the gate creates a degenerate
        # fitness landscape where novelty can never overcome a loss deficit.
        novelty_gate = min(1.0, 0.3 + 0.7 * max(0.0, (0.9 - perf_lr) / 0.6))
    score += 40.0 * eff_nov * conf * novelty_gate
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
    if (n_routing_ops is not None and n_routing_ops >= 1
            and n_sparse_ops is not None and n_sparse_ops >= 1
            and scaling_param_efficiency is not None
            and scaling_param_efficiency >= 3.0):
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
            best_lr = val_baseline if val_baseline is not None else (
                val_lr if val_lr is not None else (
                    inv_lr if inv_lr is not None else screening_lr
                )
            )
            if best_lr is not None and best_lr > 0.95:
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
    if wikitext_score is not None:
        score += 35.0 * max(0.0, min(1.0, wikitext_score))
    if wikitext_perplexity is not None:
        if wikitext_perplexity > 1000000:
            if decompose:
                _bd["generalization"] = 0.0
                _bd["_disqualified"] = True  # type: ignore[assignment]
                return {"composite_score": 0.0, "breakdown": _bd}
            return 0.0
        if wikitext_perplexity > _WIKITEXT_REF_PPL_CEILING:
            over_ref = min(
                2.0,
                (wikitext_perplexity - _WIKITEXT_REF_PPL_CEILING)
                / max(_WIKITEXT_REF_PPL_CEILING, 1e-6),
            )
            score -= 20.0 * over_ref
        if wikitext_perplexity > 1000:
            score -= 50.0 * math.log10(wikitext_perplexity / 1000.0)
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
            wikitext_score is not None
            and wikitext_score >= _WIKITEXT_REF_SCORE_FLOOR
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
    if spectral_norm is not None and spectral_norm < 0.01:
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
    if not is_reference and scaling_param_efficiency is not None and scaling_param_efficiency < 1.0:
        pre_gate = score
        gate = max(0.5, scaling_param_efficiency)
        score *= gate
        if _bd is not None:
            _bd["scaling_gate_multiplier"] = gate
            _bd["scaling_gate_reduction"] = score - pre_gate

    final = max(0.0, score)
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
