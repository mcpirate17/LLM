"""Leaderboard scoring functions — pure arithmetic, no I/O.

All functions are static / module-level. Candidates for future Cython port.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union


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
    """Build kwargs for ``compute_composite_v6`` from a leaderboard row.

    Queries program_results for fields not on the leaderboard row.
    ``conn`` is a sqlite3 connection, ``notebook`` provides helper methods.
    """
    pr = conn.execute(
        "SELECT novelty_confidence, loss_improvement_rate, final_loss, "
        "param_count, n_train_steps, behavioral_novelty, structural_novelty, "
        "fp_cka_vs_transformer "
        "FROM program_results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    pr_dict = dict(pr) if pr else {}

    tags = str(d.get("tags") or "")
    is_wiki_tik = "tiktoken_native" in tags and "wikitext103" in tags

    kw: Dict[str, Any] = {
        # Performance anchors
        "wikitext_perplexity": d.get("wikitext_perplexity"),
        "final_loss": pr_dict.get("final_loss"),
        "is_wikitext_tiktoken": is_wiki_tik,
        # Loss ratios (for hard learning gate)
        "screening_lr": d.get("screening_loss_ratio"),
        "inv_lr": d.get("investigation_loss_ratio"),
        "val_lr": d.get("validation_loss_ratio"),
        "val_baseline": d.get("validation_baseline_ratio"),
        "val_std": d.get("validation_multi_seed_std"),
        "loss_ratio": pr_dict.get("loss_ratio")
        if pr_dict
        else d.get("screening_loss_ratio"),
        # Robustness
        "inv_robust": d.get("investigation_robustness"),
        "spectral_norm": d.get("fp_jacobian_spectral_norm"),
        "investigation_passed": d.get("investigation_passed"),
        "validation_passed": d.get("validation_passed"),
        # Novelty
        "screening_nov": d.get("screening_novelty"),
        "novelty_confidence": d.get("novelty_confidence")
        or pr_dict.get("novelty_confidence"),
        "behavioral_novelty": pr_dict.get("behavioral_novelty"),
        "structural_novelty": pr_dict.get("structural_novelty"),
        "cka_reference_quality": (
            pr_dict.get("fp_cka_vs_transformer") is not None
            and pr_dict.get("fp_cka_vs_transformer", 0) > 0
        ),
        # Convergence & efficiency
        "loss_improvement_rate": pr_dict.get("loss_improvement_rate"),
        "param_count": pr_dict.get("param_count"),
        "n_train_steps": pr_dict.get("n_train_steps"),
        # Reference
        "is_reference": is_reference,
        # Normalization anchor
        "gpt2_raw_anchor": 95.0,
    }
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
    if perf_lr is not None and perf_lr > 0.95:
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

    # 2. Novelty Utility
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

    if decompose:
        return {"composite_score": final, "breakdown": _bd}
    return final


# ---------------------------------------------------------------------------
# v5 scoring: simplified 100-point scale calibrated against GPT-2 WikiText-103
# ---------------------------------------------------------------------------

# Ground truth from 8-experiment ablation series (WikiText-103, tiktoken cl100k)
_V5_GPT2_PARAMS = 28_000_000  # 4-layer 256d GPT-2 from ablation experiments
_V5_GPT2_IMPROVEMENT_RATE = 0.116  # 2K→6K rate on WikiText-103 (smoothed)


def compute_composite_v5(
    screening_lr: Optional[float] = None,
    screening_nov: Optional[float] = None,
    inv_lr: Optional[float] = None,
    inv_robust: Optional[float] = None,
    val_lr: Optional[float] = None,
    val_baseline: Optional[float] = None,
    val_std: Optional[float] = None,
    novelty_confidence: Optional[float] = None,
    is_reference: bool = False,
    loss_improvement_rate: Optional[float] = None,
    param_count: Optional[float] = None,
    wikitext_perplexity: Optional[float] = None,
    wikitext_score: Optional[float] = None,
    spectral_norm: Optional[float] = None,
    cka_reference_quality: Optional[bool] = None,
    behavioral_novelty: Optional[float] = None,
    structural_novelty: Optional[float] = None,
    investigation_passed: Optional[bool] = None,
    validation_passed: Optional[bool] = None,
    is_verified: bool = False,
    gap_vs_gpt2: Optional[float] = None,
    decompose: bool = False,
    **kwargs: Any,
) -> Union[float, Dict[str, Any]]:
    """Composite score v5 — 100-point scale calibrated against GPT-2.

    Invariant: no architecture scores above GPT-2 unless it is verified
    (tiktoken_native + wikitext103 corpus) with a negative gap.

    Budget:
      Primary performance:  55 pts
      Convergence quality:  20 pts
      Novelty:              10 pts (capped at 8)
      Efficiency:           10 pts
      Robustness:            5 pts
    """
    _bd: Optional[Dict[str, float]] = {} if decompose else None

    # Determine best available loss_ratio for learning gate
    _best_lr: Optional[float] = None
    for _lr_candidate in (val_baseline, val_lr, inv_lr, screening_lr):
        if _lr_candidate is not None:
            _best_lr = _lr_candidate
            break

    # HARD GATE: model that didn't learn cannot score high.
    # Thresholds match v4 — see comment there for normalized loss_ratio semantics.
    _insufficient_learning_cap_v5: Optional[float] = None
    if _best_lr is not None and _best_lr > 0.95:
        _insufficient_learning_cap_v5 = 10.0
    elif _best_lr is not None and _best_lr > 0.9:
        _insufficient_learning_cap_v5 = 20.0

    # --- 1. Primary performance (55%) ---
    # For verified entries with gap_vs_gpt2: use gap directly.
    # gap < 0 = beats GPT-2; map to score where gap=-0.15 → 1.0, gap=+0.5 → 0.0
    if is_verified and gap_vs_gpt2 is not None:
        # Normalized: -0.15 or better → 1.0, +0.50 → 0.0
        perf_norm = max(0.0, min(1.0, (0.50 - gap_vs_gpt2) / 0.65))
        loss_score = perf_norm * 1.0  # Full confidence for verified
    else:
        # Fallback: use loss_ratio from pipeline evaluation
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

        loss_score = 0.0
        if perf_lr is not None:
            # Normalize: 0.0 loss_ratio = perfect, 1.0 = random
            perf_norm = max(0.0, min(1.0, 1.0 - perf_lr))
            loss_score = perf_norm**1.3 * perf_confidence

        # Verification discount
        if wikitext_perplexity is not None:
            loss_score *= 0.80  # tiktoken but maybe wrong corpus
        else:
            loss_score *= 0.60  # byte-era

    primary = loss_score * 55.0
    if _bd is not None:
        _bd["primary"] = primary

    # --- 2. Convergence quality (20%) ---
    if loss_improvement_rate is not None and loss_improvement_rate > 0:
        rate_score = min(1.0, loss_improvement_rate / _V5_GPT2_IMPROVEMENT_RATE)
    else:
        rate_score = 0.5  # Unknown — neutral
    convergence = rate_score * 20.0
    if _bd is not None:
        _bd["convergence"] = convergence

    # --- 3. Novelty (10%, capped at 8 points) ---
    if cka_reference_quality and behavioral_novelty is not None:
        nov = behavioral_novelty
    elif structural_novelty is not None:
        nov = structural_novelty * 0.5  # Structural only, half weight
    elif screening_nov is not None:
        conf = novelty_confidence if novelty_confidence is not None else 0.5
        nov = screening_nov * conf
    else:
        nov = 0.0
    novelty = min(8.0, nov * 10.0)
    if _bd is not None:
        _bd["novelty"] = novelty

    # --- 4. Efficiency (10%) ---
    if param_count is not None and param_count > 0:
        param_eff = min(1.2, _V5_GPT2_PARAMS / param_count)
    else:
        param_eff = 1.0  # Unknown — neutral
    efficiency = param_eff * 10.0
    if _bd is not None:
        _bd["efficiency"] = efficiency

    # --- 5. Robustness (5%) ---
    robustness = 0.0
    if validation_passed and inv_robust is not None:
        robustness = inv_robust * 5.0
    if _bd is not None:
        _bd["robustness"] = robustness

    # --- Penalties ---
    penalty = 0.0
    if val_std is not None and val_std > 0.1:
        penalty += min(10.0, 25.0 * val_std)
    if spectral_norm is not None and spectral_norm < 0.01:
        penalty += 10.0
    if _bd is not None:
        _bd["penalty"] = -penalty

    raw = primary + convergence + novelty + efficiency + robustness - penalty

    # --- GPT-2 invariant ---
    # gpt2_cap is passed in by the rescorer (the actual GPT-2 reference score).
    # If not provided, use a conservative estimate.
    gpt2_cap: Optional[float] = kwargs.get("gpt2_cap")
    if gpt2_cap is None:
        gpt2_cap = 75.0  # Conservative fallback

    if not is_verified and not is_reference:
        raw = min(raw, gpt2_cap - 1.0)
    elif is_verified and gap_vs_gpt2 is not None and gap_vs_gpt2 > 0.0:
        raw = min(raw, gpt2_cap - 1.0)
    # Verified with negative gap: no cap — earned it

    final = max(0.0, raw)

    # Apply insufficient-learning cap (same invariant as v4).
    if _insufficient_learning_cap_v5 is not None:
        final = min(final, _insufficient_learning_cap_v5)
        if _bd is not None:
            _bd["insufficient_learning_cap"] = _insufficient_learning_cap_v5

    if _bd is not None:
        _bd["gpt2_reference_score"] = gpt2_cap
        _bd["capped"] = final < raw
        return {"composite_score": final, "breakdown": _bd}
    return final


# ── v6: Open-ended competitive scoring, GPT-2 = 100 anchor ──

# Two performance anchors — never mix them in the same comparison.
# Anchor 1: rapid WikiText PPL (screening eval, 200 steps)
_V6_GPT2_WIKITEXT_PPL = 18.64  # GPT-2 leaderboard.wikitext_perplexity
# Anchor 2: full training on wikitext103+tiktoken (investigation/validation)
_V6_GPT2_WIKI_TRAIN_LOSS = 6.8  # GPT-2 final_loss on wikitext103+tiktoken ~10K steps
_V6_GPT2_PARAMS = 28_821_248  # GPT-2 4-layer 256d with tiktoken (100277 vocab)
_V6_GPT2_IMPROVEMENT_RATE = (
    0.974  # (initial - final) / initial on wikitext103+tiktoken 7K steps
)
_V6_NO_ANCHOR_CAP = 60.0  # cap when neither anchor applies

# Step gates: screening runs cannot compete with validated architectures.
# validation_loss from screening is measured on the training corpus val split
# (micro_corpus.txt), NOT on WikiText-103 — not comparable to GPT-2's loss.
_V6_MIN_STEPS_FOR_COMPETITIVE = 2000
_V6_SCREENING_HARD_CAP = 40.0
_V6_MIN_STEPS_FOR_VALIDATION = 4000
_V6_INVESTIGATION_CEILING = 85.0


def compute_composite_v6(
    wikitext_perplexity: Optional[float] = None,
    gpt2_wikitext_ppl: Optional[float] = None,
    final_loss: Optional[float] = None,
    is_wikitext_tiktoken: bool = False,
    loss_ratio: Optional[float] = None,
    screening_lr: Optional[float] = None,
    inv_lr: Optional[float] = None,
    val_lr: Optional[float] = None,
    val_baseline: Optional[float] = None,
    val_std: Optional[float] = None,
    inv_robust: Optional[float] = None,
    screening_nov: Optional[float] = None,
    novelty_confidence: Optional[float] = None,
    is_reference: bool = False,
    loss_improvement_rate: Optional[float] = None,
    param_count: Optional[float] = None,
    behavioral_novelty: Optional[float] = None,
    structural_novelty: Optional[float] = None,
    cka_reference_quality: Optional[bool] = None,
    investigation_passed: Optional[bool] = None,
    validation_passed: Optional[bool] = None,
    spectral_norm: Optional[float] = None,
    n_train_steps: Optional[int] = None,
    decompose: bool = False,
    **kwargs: Any,
) -> Union[float, Dict[str, Any]]:
    """Composite score v6 — open-ended scale, GPT-2 = 100.0 anchor.

    Three performance anchors (never mixed):
      Anchor 1: wikitext_perplexity available →
                gpt2_wikitext_ppl / model_wikitext_ppl
      Anchor 2: wikitext103+tiktoken trained, n_steps >= 2000 →
                6.8 / model_final_loss
      Anchor 3: neither → score capped at 60

    Steps gate: n_train_steps < 2000 → capped at 40.

    Budget:
      Performance (60%): anchor-dependent ratio
      Convergence (20%): improvement rate vs GPT-2
      Efficiency  (10%): param/compute efficiency vs GPT-2
      Novelty      (5%): behavioral novelty bonus
      Robustness   (5%): validation + multi-seed
    """
    _bd: Optional[Dict[str, float]] = {} if decompose else None

    gpt2_ppl = gpt2_wikitext_ppl or _V6_GPT2_WIKITEXT_PPL

    # HARD GATE: model that didn't learn
    # Use the LOWEST (best) loss_ratio across all tiers.
    # val_baseline can be high even for good models (e.g. Var H = 0.98) if
    # it was measured on a different corpus than investigation_loss_ratio.
    _best_lr = None
    for _lr_candidate in (val_baseline, val_lr, inv_lr, screening_lr, loss_ratio):
        if _lr_candidate is not None:
            if _best_lr is None or _lr_candidate < _best_lr:
                _best_lr = _lr_candidate

    _insufficient_learning_cap: Optional[float] = None
    if _best_lr is not None and _best_lr > 0.95:
        _insufficient_learning_cap = 10.0
    elif _best_lr is not None and _best_lr > 0.9:
        _insufficient_learning_cap = 20.0

    # --- 1. Performance (60%) — three anchors, never mixed ---
    n_steps = n_train_steps or 0
    _has_ppl = wikitext_perplexity is not None and wikitext_perplexity > 0
    _has_wiki_train = (
        is_wikitext_tiktoken
        and final_loss is not None
        and final_loss > 0
        and n_steps >= 2000
    )

    if _has_ppl:
        # Anchor 1: rapid WikiText PPL (screening eval)
        performance_ratio = gpt2_ppl / wikitext_perplexity
        _anchor_used = "wikitext_ppl"
    elif _has_wiki_train:
        # Anchor 2: full training on wikitext103+tiktoken, >=2000 steps
        performance_ratio = _V6_GPT2_WIKI_TRAIN_LOSS / final_loss
        _anchor_used = "wiki_train_loss"
    else:
        # Anchor 3: no comparable metric
        performance_ratio = 0.5
        _anchor_used = "none"

    performance_score = performance_ratio * 60.0
    if _bd is not None:
        _bd["performance"] = performance_score
        _bd["performance_ratio"] = performance_ratio
        _bd["anchor_used"] = _anchor_used  # type: ignore[assignment]
        if _has_ppl:
            _bd["gpt2_wikitext_ppl"] = gpt2_ppl
            _bd["model_wikitext_ppl"] = wikitext_perplexity  # type: ignore[assignment]
        elif _has_wiki_train:
            _bd["gpt2_wiki_train_loss"] = _V6_GPT2_WIKI_TRAIN_LOSS
            _bd["model_final_loss"] = final_loss  # type: ignore[assignment]

    # --- 2. Convergence rate (20%) ---
    # Neutral: DB loss_improvement_rate is inconsistent across entries
    # (some store 2K→6K rate, others store (init-final)/init).
    # Until the metric is standardized, give all entries the baseline 20pts.
    # Performance (60%) is the real differentiator.
    convergence_score = 20.0
    if _bd is not None:
        _bd["convergence"] = convergence_score

    # --- 3. Efficiency (10%) ---
    if param_count is not None and param_count > 0:
        param_ratio = _V6_GPT2_PARAMS / param_count  # >1.0 = more efficient
        param_ratio = min(param_ratio, 2.0)  # cap at 2x
    else:
        param_ratio = 1.0  # unknown — neutral
    efficiency_score = param_ratio * 10.0
    if _bd is not None:
        _bd["efficiency"] = efficiency_score

    # --- 4. Novelty (5%, additive bonus only) ---
    # References get 0 novelty — they ARE the known baselines.
    # Novelty rewards discovering something NEW.
    if is_reference:
        nov = 0.0
    elif cka_reference_quality and behavioral_novelty is not None:
        nov = behavioral_novelty
    elif structural_novelty is not None:
        nov = structural_novelty * 0.5
    elif screening_nov is not None:
        conf = novelty_confidence if novelty_confidence is not None else 0.5
        nov = screening_nov * conf
    else:
        nov = 0.0
    novelty_score = min(1.0, nov) * 5.0
    if _bd is not None:
        _bd["novelty"] = novelty_score

    # --- 5. Robustness (5%) ---
    robustness_score = 0.0
    if validation_passed:
        robustness_score += 2.5
    if inv_robust is not None and inv_robust > 0:
        robustness_score += min(2.5, inv_robust * 2.5)
    if _bd is not None:
        _bd["robustness"] = robustness_score

    # --- Penalties ---
    penalty = 0.0
    if val_std is not None and val_std > 0.1:
        penalty += min(10.0, 25.0 * val_std)
    if spectral_norm is not None and spectral_norm < 0.01:
        penalty += 5.0
    if _bd is not None:
        _bd["penalty"] = -penalty

    raw = (
        performance_score
        + convergence_score
        + efficiency_score
        + novelty_score
        + robustness_score
        - penalty
    )

    # --- Normalize so GPT-2 = 100.0 ---
    # gpt2_raw_anchor is GPT-2's own raw score, computed by the rescorer
    # from GPT-2's actual metrics. Default 95 is a conservative estimate.
    gpt2_baseline_raw: float = kwargs.get("gpt2_raw_anchor") or 95.0
    composite = raw * (100.0 / gpt2_baseline_raw)

    # Apply insufficient-learning cap
    if _insufficient_learning_cap is not None:
        composite = min(composite, _insufficient_learning_cap)
        if _bd is not None:
            _bd["insufficient_learning_cap"] = _insufficient_learning_cap

    composite = max(0.0, composite)

    # Step gate: screening runs with insufficient training cannot compete.
    # A 500-step run on the training corpus val split is not comparable to
    # GPT-2 trained on WikiText-103 for 10K+ steps.
    n_steps = n_train_steps or 0
    if not is_reference and n_steps > 0:
        if n_steps < _V6_MIN_STEPS_FOR_COMPETITIVE:
            step_fraction = n_steps / _V6_MIN_STEPS_FOR_COMPETITIVE
            composite = min(composite * step_fraction, _V6_SCREENING_HARD_CAP)
            if _bd is not None:
                _bd["step_gate"] = True
                _bd["step_fraction"] = step_fraction
                _bd["step_hard_cap"] = _V6_SCREENING_HARD_CAP
        elif validation_passed and n_steps < _V6_MIN_STEPS_FOR_VALIDATION:
            composite = min(composite, _V6_INVESTIGATION_CEILING)
            if _bd is not None:
                _bd["validation_step_gate"] = True
                _bd["investigation_ceiling"] = _V6_INVESTIGATION_CEILING

    # No anchor → cap at 60
    if _anchor_used == "none" and not is_reference:
        composite = min(composite, _V6_NO_ANCHOR_CAP)
        if _bd is not None:
            _bd["no_anchor_cap"] = _V6_NO_ANCHOR_CAP

    if _bd is not None:
        _bd["gpt2_wikitext_ppl_anchor"] = gpt2_ppl
        _bd["gpt2_baseline_raw"] = gpt2_baseline_raw
        _bd["n_train_steps"] = float(n_steps)
        return {"composite_score": composite, "breakdown": _bd}
    return composite


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
