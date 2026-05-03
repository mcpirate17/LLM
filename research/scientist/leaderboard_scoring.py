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
    "graph_fingerprint, param_count, n_train_steps, behavioral_novelty, structural_novelty, "
    "fp_cka_vs_transformer, wikitext_perplexity, wikitext_score, "
    "wikitext_ppl_200, wikitext_ppl_500, wikitext_eval_steps, "
    "routing_savings_ratio, compression_ratio, activation_sparsity_score, "
    "depth_savings_ratio, recursion_depth_ratio, "
    "fp_jacobian_spectral_norm, validation_robustness_score, "
    "ncd_description_length_per_param, novelty_valid_for_promotion, "
    "fingerprint_json, hellaswag_acc, hellaswag_metric_version, "
    "hellaswag_tokenizer_mode, hellaswag_tiktoken_encoding, "
    "ar_auc, ar_final_acc, ar_timed_out, "
    "ar_above_chance, induction_auc, binding_auc, blimp_overall_accuracy, "
    "tinystories_score, cross_task_score, diagnostic_score, "
    "fp_gromov_delta, fp_hierarchy_fitness, "
    "induction_v2_investigation_auc, induction_v2_investigation_max_gap_acc, "
    "induction_v2_investigation_protocol_version, binding_v2_investigation_auc, "
    "binding_v2_investigation_max_distance_acc, "
    "binding_v2_investigation_protocol_version, "
    # v9 trajectory metrics — read by compute_composite_v10's capability
    # tier (cap_erf_density, cap_id_collapse, cap_erf_decay, cap_logit_margin)
    # and aux trajectory tier (aux_erf_variance, aux_icld). Without these
    # selected here, _pr_dict_to_score_kwargs would silently pass None for
    # all four and v10 would zero-out the entire 175pt capability tier
    # (minus the 75pt induction/binding/ar block).
    "fp_jacobian_erf_density, fp_jacobian_erf_variance, "
    "fp_jacobian_erf_decay_slope, fp_id_collapse_rate, "
    "fp_logit_margin_velocity, fp_icld_velocity, "
    # Long-context retrieval probes — read by the v12 non-attention bypass
    # in _v12_non_loss_sequence_signal_count. Without these, the bypass
    # only sees HellaSwag and BLiMP and SSM-class architectures cannot
    # accumulate the 2 sequence signals needed to clear the 360 ceiling.
    "robustness_long_ctx_passkey_score, robustness_long_ctx_multi_hop_score, "
    "robustness_long_ctx_scaling_score, robustness_long_ctx_combined_score, "
    # Controlled-language probe ladder (v14): nano-scale BLiMP/HellaSwag
    # replacement at three difficulty tiers. Read by the v14 composite
    # scorer for the 45pt controlled_lang tier.
    "controlled_lang_metric_version, "
    "controlled_lang_s05_sa_score, controlled_lang_s05_nb_order_acc, "
    "controlled_lang_s05_nb_score, "
    "controlled_lang_s10_sa_score, controlled_lang_s10_nb_order_acc, "
    "controlled_lang_s10_nb_score, "
    "controlled_lang_inv_sa_score, controlled_lang_inv_nb_order_acc, "
    "controlled_lang_inv_nb_score, "
    # routing_collapse_score is a misnomer: it is actually a routing-health
    # score in [0,1] (higher = healthier). Selected here so consumers can
    # build a routing-quality subscore. Not yet used by composite scoring.
    "routing_collapse_score, tokenizer_mode, "
    # screening_wikitext_metric_version is the *reliable* indicator that
    # wikitext_perplexity is in BPE units (vs the byte-era stale rows).
    # tokenizer_mode is unreliable — it was set to 'tiktoken' on some
    # byte-era rows and missing on many legitimate ones.  v11's tokenizer
    # integrity penalty reads metric_version, not tokenizer_mode.
    "screening_wikitext_metric_version"
)


def _pr_dict_to_score_kwargs(
    pr_dict: Dict[str, Any],
    d: Dict[str, Any],
    is_reference: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build kwargs for ``compute_composite`` from a pr_dict + leaderboard row.

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
    # Threshold was 1000 when training ran 1000 steps; current eval pipeline
    # runs 750 steps so every row tripped the >= 1000 gate and learning_efficiency
    # silently stayed 0 for everyone. Lowered to 500 to capture the 750-step regime.
    # NOTE: wikitext_ppl_200/500 are still byte-era stale (BPE backfill only
    # refreshed wikitext_perplexity), so the ratio is currently mismatched
    # tokenizations and learn_eff produces ~0 across the cohort. Re-anchoring
    # learn_eff after a BPE trajectory backfill is the proper fix.
    ppl_at_1000 = ppl_final if (eval_steps is not None and eval_steps >= 500) else None

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
        "long_ctx_score": d.get("robustness_long_ctx_combined_score")
        or d.get("robustness_long_ctx_score"),
        # Routing health (passed through; composite scoring may use this in
        # a future revision — not yet wired into the points formula).
        "routing_health_score": pr_dict.get("routing_collapse_score"),
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
        # v2 investigation-tier probes — overrides induction_auc/binding_auc
        # in the composite when present. `or`-chained against `d` so we read
        # the leaderboard row for rows where the program_results row is
        # missing the column (pre-backfill).
        "induction_v2_inv_auc": pr_dict.get("induction_v2_investigation_auc")
        or d.get("induction_v2_investigation_auc"),
        "binding_v2_inv_auc": pr_dict.get("binding_v2_investigation_auc")
        or d.get("binding_v2_investigation_auc"),
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
        # v9 trajectory metrics — feed compute_composite_v10's capability
        # and aux trajectory tiers. Without these, the v10 dispatcher
        # silently zeros 4×25pt capability components and 2×10pt aux
        # components (~120pts of headroom on a frontier model).
        "fp_jacobian_erf_density": pr_dict.get("fp_jacobian_erf_density"),
        "fp_jacobian_erf_variance": pr_dict.get("fp_jacobian_erf_variance"),
        "fp_jacobian_erf_decay_slope": pr_dict.get("fp_jacobian_erf_decay_slope"),
        "fp_id_collapse_rate": pr_dict.get("fp_id_collapse_rate"),
        "fp_logit_margin_velocity": pr_dict.get("fp_logit_margin_velocity"),
        "fp_icld_velocity": pr_dict.get("fp_icld_velocity"),
        # Long-context retrieval probes — passed through under the names the
        # v12 non-attention bypass already falls back to. Source from
        # program_results first, then leaderboard row (`d`).
        "robustness_long_ctx_passkey_score": (
            pr_dict.get("robustness_long_ctx_passkey_score")
            or d.get("robustness_long_ctx_passkey_score")
        ),
        "robustness_long_ctx_multi_hop_score": (
            pr_dict.get("robustness_long_ctx_multi_hop_score")
            or d.get("robustness_long_ctx_multi_hop_score")
        ),
        "robustness_long_ctx_scaling_score": (
            pr_dict.get("robustness_long_ctx_scaling_score")
            or d.get("robustness_long_ctx_scaling_score")
        ),
        "robustness_long_ctx_combined_score": (
            pr_dict.get("robustness_long_ctx_combined_score")
            or d.get("robustness_long_ctx_combined_score")
        ),
        # In-context loss decay rate, exposed to the bypass under the name
        # it expects (`icld_score`). fp_icld_velocity is signed: negative
        # means loss is declining (good). The bypass treats higher = better,
        # so we report the magnitude of the negative side; positive velocity
        # (loss going up) maps to 0.
        "icld_score": (
            -float(pr_dict["fp_icld_velocity"])
            if pr_dict.get("fp_icld_velocity") is not None
            and float(pr_dict["fp_icld_velocity"]) < 0
            else 0.0
        ),
        # Template name from program_graph_features — needed by the v12
        # non-attention family detector. Joined in by build_score_kwargs and
        # prefetch_program_results; absent for callers that bypass those
        # helpers (the detector then falls back to its other haystack keys).
        "template_name": pr_dict.get("template_name") or d.get("template_name"),
        # Controlled-language probe ladder (v14): pass through the full
        # recorded probe payload. The scorer consumes SA + NB order today;
        # dashboard consumers also render NB score. Absent on un-probed rows
        # (None scores -> 0pts via _scurve_higher_better).
        "controlled_lang_s05_sa_score": pr_dict.get("controlled_lang_s05_sa_score"),
        "controlled_lang_s05_nb_order_acc": pr_dict.get(
            "controlled_lang_s05_nb_order_acc"
        ),
        "controlled_lang_s05_nb_score": pr_dict.get("controlled_lang_s05_nb_score"),
        "controlled_lang_s10_sa_score": pr_dict.get("controlled_lang_s10_sa_score"),
        "controlled_lang_s10_nb_order_acc": pr_dict.get(
            "controlled_lang_s10_nb_order_acc"
        ),
        "controlled_lang_s10_nb_score": pr_dict.get("controlled_lang_s10_nb_score"),
        "controlled_lang_inv_sa_score": pr_dict.get("controlled_lang_inv_sa_score"),
        "controlled_lang_inv_nb_order_acc": pr_dict.get(
            "controlled_lang_inv_nb_order_acc"
        ),
        "controlled_lang_inv_nb_score": pr_dict.get("controlled_lang_inv_nb_score"),
        "tokenizer_mode": pr_dict.get("tokenizer_mode") or d.get("tokenizer_mode"),
        "screening_wikitext_metric_version": (
            pr_dict.get("screening_wikitext_metric_version")
            or d.get("screening_wikitext_metric_version")
        ),
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
    """Build kwargs for ``compute_composite`` from a leaderboard row.

    Queries program_results for fields not on the leaderboard row.
    ``conn`` is a sqlite3 connection, ``notebook`` provides helper methods.
    """
    pr = conn.execute(
        f"SELECT {_PR_SELECT_COLS} FROM program_results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    pr_dict = dict(pr) if pr else {}
    pgf = conn.execute(
        "SELECT template_name FROM program_graph_features WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    if pgf and pgf["template_name"]:
        pr_dict["template_name"] = pgf["template_name"]
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
        # Augment with template_name from program_graph_features so the v12
        # non-attention family detector can identify SSM-class architectures.
        pgf_rows = conn.execute(
            f"SELECT result_id, template_name FROM program_graph_features "
            f"WHERE result_id IN ({placeholders})",
            chunk,
        ).fetchall()
        for row in pgf_rows:
            rid = row["result_id"]
            if rid in out and row["template_name"]:
                out[rid]["template_name"] = row["template_name"]
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
    effective_ar_auc: Optional[float],
    induction_auc: Optional[float],
    binding_auc_val: Optional[float],
) -> tuple[float, Dict[str, float]]:
    """Robustness (40pts), long context (25pts), binding probes, BLiMP (40pts)."""
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
        if effective_ar_auc is not None:
            _bc += 0.4 * effective_ar_auc
            _bc_n += 1
        if induction_auc is not None:
            _bc += 0.3 * induction_auc
            _bc_n += 1
        if binding_auc_val is not None:
            _bc += 0.3 * binding_auc_val
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


def _apply_scoring_penalties(
    score: float,
    *,
    inv_failed: bool,
    param_count: Optional[float],
    induction_auc: Optional[float],
    binding_auc_val: Optional[float],
    effective_ar_auc: Optional[float],
    ar_above_chance: Optional[bool],
    cfg: Optional[Dict[str, float]] = None,
) -> tuple[float, float, float]:
    """Apply binding soft gate and param-size penalties.

    Returns (score, binding_pen, param_pen).

    When ``cfg`` provides ``binding_all_below_penalty`` and/or
    ``binding_composite_boost``, those override the default thresholds-based
    multipliers. This lets v8.1 tighten the penalty from 0.80 → 0.50 and add
    a +15% boost for graphs that actually bind, without disturbing the v7/v8
    scoring behavior for historical rows.
    """
    cfg = cfg or {}
    _induction_below = (
        induction_auc is not None and induction_auc < BINDING_INDUCTION_SOFT_GATE
    )
    _binding_below = (
        binding_auc_val is not None and binding_auc_val < BINDING_BINDING_AUC_SOFT_GATE
    )
    _ar_below = (
        effective_ar_auc is not None
        and effective_ar_auc < BINDING_AR_SOFT_GATE
        and not ar_above_chance
    )
    _signals = sum(
        [
            induction_auc is not None,
            binding_auc_val is not None,
            effective_ar_auc is not None,
        ]
    )
    _all_below = _signals >= 2 and all(
        [
            _induction_below or induction_auc is None,
            _binding_below or binding_auc_val is None,
            _ar_below or effective_ar_auc is None,
        ]
    )
    binding_penalty = 1.0
    if not inv_failed and _signals >= 2 and _all_below:
        binding_penalty = float(
            cfg.get("binding_all_below_penalty", BINDING_LOCAL_ONLY_PENALTY)
        )
        score *= binding_penalty

    # v8.1 optional boost: reward graphs that clear a binding_composite floor.
    # binding_composite matches the understanding gate convention:
    #   0.4 * ar + 0.3 * induction + 0.3 * binding
    boost_mult = float(cfg.get("binding_composite_boost", 1.0))
    boost_floor = float(cfg.get("binding_composite_boost_floor", 0.0))
    if not inv_failed and boost_mult > 1.0 and boost_floor > 0.0:
        _ind = float(induction_auc) if induction_auc is not None else 0.0
        _bind = float(binding_auc_val) if binding_auc_val is not None else 0.0
        _ar = float(effective_ar_auc) if effective_ar_auc is not None else 0.0
        _composite = 0.4 * _ar + 0.3 * _ind + 0.3 * _bind
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
    # v2 investigation-tier probes (override induction_auc/binding_auc when
    # present; pre-backfill rows leave these None and fall back to v1)
    induction_v2_inv_auc: Optional[float] = None,
    binding_v2_inv_auc: Optional[float] = None,
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
    _effective_ar_auc = None if ar_timed_out else ar_auc
    # v2 investigation probes override v1 when present. Rows pre-backfill
    # have None v2 values and keep scoring against v1 for continuity.
    _effective_induction_auc = (
        induction_v2_inv_auc if induction_v2_inv_auc is not None else induction_auc
    )
    _effective_binding_auc = (
        binding_v2_inv_auc if binding_v2_inv_auc is not None else binding_auc
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
        effective_ar_auc=_effective_ar_auc,
        induction_auc=_effective_induction_auc,
        binding_auc_val=_effective_binding_auc,
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
        induction_auc=_effective_induction_auc,
        binding_auc_val=_effective_binding_auc,
        effective_ar_auc=_effective_ar_auc,
        ar_above_chance=ar_above_chance,
        cfg=cfg,
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


# v7/v8/v8.1 retired 2026-05-03 — they were superseded by v10+ (which is
# composed into v14, the active formula). _compute_composite_generic stays
# because v10 still calls it for the loss/efficiency/novelty/understanding
# tiers; the v7/v8/v8.1 dispatch wrappers were dead code paths.


# v9 retired 2026-05-03 — its 50/50 v8.1+gemini blend was superseded by
# v10's unified S-curve capability tier. The 4 trajectory signals (erf_density,
# id_collapse_rate, erf_decay_slope, logit_margin_velocity) live on in v10's
# `_score_capability_tier_v10` instead.

# Hard gates applied at the pre-investigation gate (notebook_references
# .get_investigation_eligible). Below these floors the architecture cannot
# route information and shouldn't reach investigation regardless of
# composite score. Tuned from smoke data: SSMs at init have erf_density≈0.3
# so the floor must be below that to avoid gating Mamba/RWKV-class graphs.
# Kept across the v9 retirement (2026-05-03) because notebook_references
# imports them by name.
GEMINI_HARD_GATE_ERF_DENSITY = 0.20
GEMINI_HARD_GATE_ERF_VARIANCE = 800.0


# ---------------------------------------------------------------------------
# v10 — three equal buckets (capability / loss / understanding), all S-curves,
# binding multiplicative gates retired (2026-04-25).
# ---------------------------------------------------------------------------
# Three structural changes from v9:
#   1. Binding/induction/AR are unrolled from one 85pt S-curve into three
#      independent 25pt S-curves, sitting alongside four trajectory metrics
#      (erf_density, id_collapse_rate, erf_decay_slope, logit_margin_velocity)
#      at equal 25pt weight — all seven form a 175pt capability tier.
#   2. Loss tier (perf_short/medium/long + learn_eff + early_conv) and
#      understanding tier (blimp + tinystories + cross_task + diagnostic +
#      hellaswag + hierarchy) are sized to ~175pts each so capability,
#      loss, and understanding contribute roughly equally.
#   3. binding_all_below_penalty (×0.50) and binding_composite_boost (×1.15)
#      are retired. With understanding properly weighted at 175pts a
#      ppl-only graph already loses on the additive scoreboard; the
#      multiplicative gate was redundant and penalized SSMs (Mamba-class)
#      that bind weakly by architecture but are otherwise capable.
#
# Anchors come from the 2026-04-25 distribution analysis on the 16k+ rows
# that have at least one Gemini metric populated (see lab_notebook query
# in the v10 design discussion). Lower-is-better metrics (id_collapse_rate,
# erf_decay_slope, icld_velocity) are negated before S-curve normalization.
_V10_CONFIG: Dict[str, float] = {
    # Frontier reference values (formerly _FRONTIER_COMMON, inlined 2026-05-03
    # when v7/v8/v8.1/v9 were retired — only param_eff/learn_eff/long_ctx/
    # avg_params/convergence/binding survive; ppl + blimp are overridden below).
    "param_eff": 1.09,
    "learn_eff": 1.13,
    "long_ctx": 0.375,
    "avg_params": 28_561_920,
    "convergence": 1.30,
    "binding": 0.15,
    # Loss tier (~175pts total): unchanged from v8 except learn_eff trimmed
    "w_perf_short": 35.0,
    "w_perf_medium": 50.0,
    "w_perf_long": 65.0,
    "w_param_eff": 30.0,
    "w_learn_eff": 15.0,
    # Capability tier disabled in the legacy rollup; new unrolled scorer
    # handles ar/induction/binding individually below.
    "w_binding": 0.0,
    # Understanding tier (~175pts total): hellaswag halved, diagnostic
    # trimmed, hierarchy bumped — keeps tier total at parity with capability
    # without leaving any single metric overweighted.
    "w_blimp": 35.0,
    "w_tinystories": 30.0,
    "w_cross_task": 30.0,
    "w_diagnostic": 40.0,
    "w_hellaswag": 15.0,
    "w_hierarchy": 25.0,
    # Capability tier (~175pts total): 7 × 25pts each, all S-curved.
    "w_cap_ar": 25.0,
    "w_cap_induction": 25.0,
    "w_cap_binding": 25.0,
    "w_cap_erf_density": 25.0,
    "w_cap_id_collapse": 25.0,
    "w_cap_erf_decay": 25.0,
    "w_cap_logit_margin": 25.0,
    # Aux trajectory tier (lower-ρ, kept for ML signal): 10pts each.
    "w_aux_erf_variance": 10.0,
    "w_aux_icld": 10.0,
    # Anchors recalibrated 2026-04-26 to the cohort median of each metric.
    # Methodology: anchor = median of populated rows in the cohort that the
    # component scores at. S-curve evaluates ratio=anchor/value (lower-better)
    # or value/anchor (higher-better); ratio=1 → score 0.5. So a row at the
    # cohort median earns half of the component's weight; above-median earns
    # more, below-median less. Replaces ad-hoc / aspirational anchors that
    # were either unreachable (most rows scored zero) or arbitrary.
    # Cohorts: PPL anchors use the appropriate stage cohort; understanding,
    # capability, and aux anchors use val+bt (where they fire).
    "ppl_1000": 744.0,  # median screening wikitext_perplexity (n=1871)
    "ppl_2500": 823.0,  # median investigation wikitext_perplexity (n=159)
    "ppl_10000": 754.0,  # median val+bt wikitext_perplexity (n=2453)
    "blimp": 0.525,  # median val+bt blimp_overall_accuracy (n=2452)
    "hellaswag": 0.225,  # median val+bt hellaswag_acc (n=2453)
    "tinystories": 0.542,  # median val+bt tinystories_score (n=2452)
    "cross_task": 0.279,  # median val+bt cross_task_score (n=2441)
    "diagnostic": 0.004,  # median val+bt diagnostic_score (n=2447)
    "hierarchy": 0.848,  # median val+bt fp_hierarchy_fitness (n=2283)
    "cap_ar_anchor": 0.002,  # median val+bt ar_auc (n=2076)
    "cap_induction_anchor": 0.006,  # median val+bt induction_auc v1 (n=2163)
    "cap_binding_anchor": 0.004,  # median val+bt binding_auc v1 (n=405)
    "cap_erf_density_anchor": 0.047,  # median val+bt fp_jacobian_erf_density
    "cap_id_collapse_anchor": 0.010,  # |median| of negative fp_id_collapse_rate (n_neg=1498)
    "cap_erf_decay_anchor": 0.050,  # |median| of negative fp_jacobian_erf_decay_slope (n_neg=2311)
    "cap_logit_margin_anchor": 0.003,  # median val+bt fp_logit_margin_velocity (n=2284)
    "aux_erf_variance_anchor": 14280.0,  # median val+bt fp_jacobian_erf_variance (n=2452)
    "aux_icld_anchor": 0.017,  # |median| of negative fp_icld_velocity (n_neg=2412)
    # Multiplicative gates retired.
    "binding_all_below_penalty": 1.0,
    "binding_composite_boost": 1.0,
    "binding_composite_boost_floor": 0.0,
    # Score-stability (CV) penalty — applied only at validation/breakthrough.
    # Per-tier multiplier = max(floor, 1 - lambda * tier_CV).
    # CV = std/|mean| across runs of the same graph_fingerprint.  Tiers
    # without enough runs (n<2) get penalty=1.0 (no penalty until we
    # have signal).  Loss tier carries the most points so it gets the
    # strongest lambda.
    "cv_lambda_loss": 0.5,
    "cv_lambda_und": 0.3,
    "cv_lambda_cap": 0.4,
    "cv_penalty_floor": 0.6,
}

# ---------------------------------------------------------------------------
# v11 — Breakthrough alignment (2026-04-26).
# ---------------------------------------------------------------------------
# Structural changes from v10:
#   1. High-ceiling capability anchors: cap_binding_anchor (0.004 -> 0.500)
#      and cap_induction_anchor (0.006 -> 0.300).
#   2. Higher capability weights: cap_binding and cap_induction (25 -> 50).
#   3. Breakthrough multiplier: 1.2x boost to Understanding tier if
#      binding_auc > 0.8 AND induction_auc > 0.3.
#   4. Tokenizer Integrity: 0.1x total multiplier if tokenizer_mode != 'tiktoken'.
_V11_CONFIG: Dict[str, float] = {
    **_V10_CONFIG,
    # 2026-05-02 audit (tasks/scoring_audit_2026-05-02.md):
    # BLiMP doesn't move with training at our 750-step eval scale. b0c38826
    # at champion mode (12 layers × 10K steps, final_loss=0.000) scored
    # BLiMP=0.514 vs baseline 0.502 — a 1.2pp gain inside the noise band
    # (cohort std=0.013). Cohort ρ vs n_train_steps = +0.028. The metric
    # is correctly detecting that nano-LLMs uniformly cannot do BLiMP at
    # our budget. Drop weight 35 → 5; redirect to induction/binding which
    # DO move with training and DO discriminate (ρ vs composite +0.42 / +0.19).
    "w_blimp": 5.0,
    "w_cap_induction": 65.0,
    "w_cap_binding": 65.0,
    "cap_induction_anchor": 0.300,
    "cap_binding_anchor": 0.500,
}


_V11_CAP_INDUCTION_MULTIPLIER = (
    _V11_CONFIG["w_cap_induction"] / _V10_CONFIG["w_cap_induction"]
)
_V11_CAP_BINDING_MULTIPLIER = (
    _V11_CONFIG["w_cap_binding"] / _V10_CONFIG["w_cap_binding"]
)


# ---------------------------------------------------------------------------
# v14 — Tier-progressive controlled-language probe ladder (2026-05-02)
# ---------------------------------------------------------------------------
# Replaces real BLiMP/HellaSwag noise at screening tier with a calibrated
# nano-scale association probe. Three difficulty tiers credit progressively
# more points as the test gets harder.
#
# Calibration data (top-30 leaderboard cohort, n=30, see
# research/reports/nano_probe_audit_top30.json):
#   v120/40 (S0.5): synthetic_assoc median 1.00, 67% saturate → basic floor
#   v200/40 (S1.0): synthetic_assoc median 1.00, 47% saturate → real diff
#   v300/40 (Inv):  synthetic_assoc median 0.94, 13% saturate → sharp
#
# Anchors set at the cohort median per tier so half the cohort earns half
# the weight. nano_blimp.order_grammaticality_acc has the richest dynamic
# range (cohort std ~0.30) and is the primary nano_blimp signal.
#
# Real BLiMP weight stays at 5pt floor (validated by champion-mode test:
# 12L/10K-step training moves BLiMP only 1.2pp inside the 0.013 noise band).
# Real HellaSwag weight drops 15→5 (cohort spread 0.030, ρ vs composite
# +0.088 — near-noise at screening eval scale).
_V14_CONFIG: Dict[str, float] = {
    **_V11_CONFIG,
    # Real-benchmark weight reductions (insufficient signal at nano scale)
    "w_hellaswag": 5.0,
    # Controlled-language ladder weights — progressive credit by tier
    "w_cl_s05_sa": 3.0,
    "w_cl_s05_order": 2.0,
    "w_cl_s10_sa": 9.0,
    "w_cl_s10_order": 6.0,
    "w_cl_inv_sa": 15.0,
    "w_cl_inv_order": 10.0,
    # Anchors = cohort median at each tier (S-curve gives half-credit at median)
    "cl_s05_sa_anchor": 1.000,
    "cl_s05_order_anchor": 0.605,
    "cl_s10_sa_anchor": 1.000,
    "cl_s10_order_anchor": 0.456,
    "cl_inv_sa_anchor": 0.940,
    "cl_inv_order_anchor": 0.420,
}

# Median-relative component anchors are useful for ranking noisy cohorts, but
# champion range needs at least one reproduced trust signal: genuinely low BPE
# PPL, above-chance understanding, or non-local binding plus induction.
_V11_TRUST_CEILING = 360.0
_V11_TRUST_PPL_FLOOR = 150.0
_V11_TRUST_HELLASWAG_FLOOR = 0.25
_V11_TRUST_BLIMP_FLOOR = 0.55
_V11_TRUST_INDUCTION_FLOOR = 0.05
_V11_TRUST_BINDING_FLOOR = 0.20


def _v11_trust_ceiling(
    score: float,
    bd: Dict[str, Any],
    kw: Dict[str, Any],
) -> float:
    """Cap untrusted validation-tier candidates below champion range."""
    if score <= _V11_TRUST_CEILING or bool(kw.get("is_reference")):
        return score

    tier = str(kw.get("tier") or "").strip().lower()
    if tier not in {"investigation", "validation", "breakthrough"}:
        return score

    has_reproduced_low_loss = any(
        p is not None and float(p) > 0.0 and float(p) <= _V11_TRUST_PPL_FLOOR
        for p in (
            kw.get("ppl_validation"),
            kw.get("ppl_investigation"),
            kw.get("ppl_screening"),
        )
    )
    has_understanding = any(
        v is not None and float(v) > _V11_TRUST_HELLASWAG_FLOOR
        for v in (
            kw.get("hellaswag_acc_validation"),
            kw.get("hellaswag_acc_investigation"),
            kw.get("hellaswag_acc_screening"),
        )
    )
    blimp = kw.get("blimp_accuracy")
    if blimp is not None and float(blimp) >= _V11_TRUST_BLIMP_FLOOR:
        has_understanding = True

    eff_ind = (
        kw.get("induction_v2_inv_auc")
        if kw.get("induction_v2_inv_auc") is not None
        else kw.get("induction_auc")
    )
    eff_bind = (
        kw.get("binding_v2_inv_auc")
        if kw.get("binding_v2_inv_auc") is not None
        else kw.get("binding_auc")
    )
    has_nonlocal_binding = (
        eff_ind is not None
        and eff_bind is not None
        and float(eff_ind) >= _V11_TRUST_INDUCTION_FLOOR
        and float(eff_bind) >= _V11_TRUST_BINDING_FLOOR
    )

    if has_reproduced_low_loss or has_understanding or has_nonlocal_binding:
        return score

    bd["_v11_trust_ceiling"] = _V11_TRUST_CEILING
    bd["_v11_trust_low_loss"] = has_reproduced_low_loss
    bd["_v11_trust_understanding"] = has_understanding
    bd["_v11_trust_nonlocal_binding"] = has_nonlocal_binding
    return _V11_TRUST_CEILING


# v11 tokenizer-integrity penalty (graded).  ``screening_wikitext_metric_version``
# is the reliable signal that PPL is in BPE units; ``tokenizer_mode`` is
# unreliable (was set to 'tiktoken' on some byte-era rows AND missing on
# many legitimate ones).
#
#   * 'bpe_eval_v1'         → 1.00  (good, full credit)
#   * 'screening_wikitext_v1' → 0.10  (definitively byte-era, hammer)
#   * NULL / empty / other  → 0.70  (unknown — soft uncertainty discount)
_V11_TOKENIZER_PENALTY_BPE = 1.0
_V11_TOKENIZER_PENALTY_BYTE = 0.1
_V11_TOKENIZER_PENALTY_UNKNOWN = 0.7


def _v11_tokenizer_penalty(metric_version: Optional[str]) -> float:
    if metric_version is None:
        return _V11_TOKENIZER_PENALTY_UNKNOWN
    mv = str(metric_version).strip().lower()
    if mv == "bpe_eval_v1":
        return _V11_TOKENIZER_PENALTY_BPE
    if mv == "screening_wikitext_v1":
        return _V11_TOKENIZER_PENALTY_BYTE
    return _V11_TOKENIZER_PENALTY_UNKNOWN


def compute_composite_v11(
    *,
    decompose: bool = False,
    **kw: Any,
) -> Union[float, Dict[str, Any]]:
    """Composite v11 — Breakthrough-first alignment.

    Builds on v10 with high-ceiling capability weights for binding /
    induction, a breakthrough multiplier for logic-probes, and a graded
    tokenizer-integrity penalty.

    Implementation note: ``compute_composite_v10`` hardcodes
    ``_V10_CONFIG`` internally, so ``_V11_CONFIG``'s cap-tier weight
    bumps cannot take effect through the v10 call directly.  We
    recompose them here by multiplying ``cap_induction`` /
    ``cap_binding`` in the breakdown by the v11/v10 weight ratios.
    Anchors are NOT rescaled here — that would require re-evaluating
    ``_score_capability_tier_v10``, and the larger weights already
    deliver the intended "frontier-archs win bigger" effect on top of
    v10's anchors.
    """
    # Use v10 as the structural base.  Inside v10 the CV penalty (if any)
    # has already been applied to loss/und/cap subscores per tier.
    result = compute_composite_v10(decompose=True, **kw)
    score = float(result["composite_score"])
    bd = result.get("breakdown") or {}

    # 1. v11 capability rescale: lift cap_induction / cap_binding to the
    #    v11 weights (50pts each) by multiplying the v10 contribution.
    cap_ind_old = float(bd.get("cap_induction", 0.0) or 0.0)
    cap_bind_old = float(bd.get("cap_binding", 0.0) or 0.0)
    cap_ind_new = cap_ind_old * _V11_CAP_INDUCTION_MULTIPLIER
    cap_bind_new = cap_bind_old * _V11_CAP_BINDING_MULTIPLIER
    score += (cap_ind_new - cap_ind_old) + (cap_bind_new - cap_bind_old)
    bd["cap_induction"] = cap_ind_new
    bd["cap_binding"] = cap_bind_new

    # 2. Breakthrough multiplier — 1.2× understanding tier when both
    #    induction and binding clear their gates.  Uses v2 probes when
    #    populated, falls back to v1.
    eff_ind = (
        kw.get("induction_v2_inv_auc")
        if kw.get("induction_v2_inv_auc") is not None
        else kw.get("induction_auc")
    )
    eff_bind = (
        kw.get("binding_v2_inv_auc")
        if kw.get("binding_v2_inv_auc") is not None
        else kw.get("binding_auc")
    )
    is_breakthrough = (
        eff_ind is not None
        and eff_ind > 0.3
        and eff_bind is not None
        and eff_bind > 0.8
    )
    if is_breakthrough:
        und_sum = sum(float(bd.get(k, 0.0) or 0.0) for k in _UND_TIER_BD_KEYS)
        boost = und_sum * 0.2
        score += boost
        bd["_v11_breakthrough_boost"] = boost
        for k in _UND_TIER_BD_KEYS:
            if k in bd and bd[k]:
                bd[k] = float(bd[k]) * 1.2

    # 3. Tokenizer integrity penalty (graded by metric_version).
    metric_version = kw.get("screening_wikitext_metric_version")
    tok_pen = _v11_tokenizer_penalty(metric_version)
    if tok_pen < 1.0:
        score *= tok_pen
        bd["_v11_tokenizer_penalty"] = tok_pen
        bd["_v11_tokenizer_penalty_metric_version"] = metric_version

    # 4. Trust ceiling — a candidate cannot be a champion on efficiency and
    # median-relative side metrics alone.
    score = _v11_trust_ceiling(score, bd, kw)

    if decompose:
        result["composite_score"] = score
        result["breakdown"] = bd
        return result
    return score


_V12_LOSS_COMPONENT_FACTORS = {
    "perf_short": 30.0 / 35.0,
    "perf_medium": 40.0 / 50.0,
    "perf_long": 55.0 / 65.0,
    "param_efficiency": 20.0 / 30.0,
    "learning_efficiency": 10.0 / 15.0,
    "speed": 15.0 / 25.0,
    "early_convergence": 5.0 / 10.0,
}
_V12_LOSS_BUDGET_MAX = 175.0
_V12_CHAMPION_ELIGIBILITY_CEILING = 360.0
_V12_INDUCTION_QUALIFIED = 0.05
_V12_STRONG_INDUCTION = 0.30
_V12_BINDING_QUALIFIED = 0.20
_V12_STRONG_BINDING = 0.50


def _v12_effective_signal(
    kw: Dict[str, Any],
    preferred: str,
    fallback: str,
) -> Optional[float]:
    value = kw.get(preferred)
    if value is None:
        value = kw.get(fallback)
    if value is None:
        return None
    return float(value)


def _v12_is_non_attention_exception_family(kw: Dict[str, Any]) -> bool:
    if bool(kw.get("non_attention_model")):
        return True
    haystack = " ".join(
        str(kw.get(key) or "").lower()
        for key in (
            "architecture_family",
            "model_family",
            "mechanism",
            "model_source",
            "op_names",
            "graph_ops",
            # template_name is the most populated identifier in practice —
            # the others are typically None unless the caller specifically
            # set them. Templates like `latent_attn_ssm_hybrid`,
            # `local_attn_ssm_hybrid`, `codex_ssm_*`, `spiking_*` all carry
            # the family signal in their name.
            "template_name",
        )
    )
    return any(
        token in haystack
        for token in (
            "mamba",
            "ssm",
            "state_space",
            "selective_scan",
            "rwkv",
            "recurrent",
        )
    )


def _v12_has_reproduced_bpe_loss(kw: Dict[str, Any]) -> bool:
    metric_version = (
        str(kw.get("screening_wikitext_metric_version") or "").strip().lower()
    )
    if metric_version != "bpe_eval_v1":
        return False
    return any(
        p is not None and float(p) > 0.0 and float(p) <= _V11_TRUST_PPL_FLOOR
        for p in (
            kw.get("ppl_validation"),
            kw.get("ppl_investigation"),
            kw.get("ppl_screening"),
        )
    )


# Per-signal bypass thresholds. AUC-style signals (passkey, multi_hop,
# scaling, combined, selective_copy) keep the 0.20 floor because they
# live on a [0,1] scale where chance is near 0 — 0.20 means "20pts above
# noise". ICLD is on a different scale: it's a loss-decline rate in
# nats/step. Cohort distribution (n=18,644 program_results rows) of
# |fp_icld_velocity|: p50=0.016, p75=0.026, p90=0.034, p99=0.043,
# max=0.160. The original 0.20 threshold qualified zero rows in the
# entire database. Setting ICLD threshold to 0.030 (≈p85 of the
# cohort) keeps it stringent — only the top ~15% of learners trigger
# the bypass signal — while making it physically reachable.
_V12_BYPASS_SIGNAL_THRESHOLDS: Dict[str, float] = {
    "selective_copy": 0.20,
    "long_ctx_passkey": 0.20,
    "long_ctx_multi_hop": 0.20,
    "long_ctx_scaling": 0.20,
    "long_ctx_combined": 0.20,
    "icld": 0.030,
}


def _v12_non_loss_sequence_signal_count(kw: Dict[str, Any]) -> int:
    thr = _V12_BYPASS_SIGNAL_THRESHOLDS
    pairs = (
        ("selective_copy", kw.get("selective_copy_score")),
        (
            "long_ctx_passkey",
            kw.get("long_ctx_passkey_score")
            or kw.get("robustness_long_ctx_passkey_score"),
        ),
        (
            "long_ctx_multi_hop",
            kw.get("long_ctx_multi_hop_score")
            or kw.get("robustness_long_ctx_multi_hop_score"),
        ),
        (
            "long_ctx_scaling",
            kw.get("long_ctx_scaling_score")
            or kw.get("robustness_long_ctx_scaling_score"),
        ),
        (
            "long_ctx_combined",
            kw.get("long_ctx_combined_score")
            or kw.get("robustness_long_ctx_combined_score"),
        ),
        ("icld", kw.get("icld_score") or kw.get("trajectory_learning_score")),
    )
    count = sum(
        1 for name, value in pairs if value is not None and float(value) >= thr[name]
    )

    hellaswag = max(
        float(v)
        for v in (
            kw.get("hellaswag_acc_validation") or 0.0,
            kw.get("hellaswag_acc_investigation") or 0.0,
            kw.get("hellaswag_acc_screening") or 0.0,
        )
    )
    if hellaswag >= _V11_TRUST_HELLASWAG_FLOOR:
        count += 1

    blimp = kw.get("blimp_accuracy")
    if blimp is not None and float(blimp) >= _V11_TRUST_BLIMP_FLOOR:
        count += 1
    return count


def _v12_champion_eligibility_gate(
    score: float,
    bd: Dict[str, Any],
    kw: Dict[str, Any],
) -> float:
    if score <= _V12_CHAMPION_ELIGIBILITY_CEILING or bool(kw.get("is_reference")):
        return score

    tier = str(kw.get("tier") or "").strip().lower()
    if tier not in {"investigation", "validation", "breakthrough"}:
        return score

    eff_ind = _v12_effective_signal(kw, "induction_v2_inv_auc", "induction_auc")
    eff_bind = _v12_effective_signal(kw, "binding_v2_inv_auc", "binding_auc")
    induction_qualified = eff_ind is not None and eff_ind >= _V12_INDUCTION_QUALIFIED
    strong_induction = eff_ind is not None and eff_ind >= _V12_STRONG_INDUCTION
    binding_qualified = eff_bind is not None and eff_bind >= _V12_BINDING_QUALIFIED
    strong_binding = eff_bind is not None and eff_bind >= _V12_STRONG_BINDING
    sequence_signal_count = _v12_non_loss_sequence_signal_count(kw)
    exception_allowed = (
        _v12_is_non_attention_exception_family(kw)
        and _v12_has_reproduced_bpe_loss(kw)
        and sequence_signal_count >= 2
    )

    if induction_qualified or exception_allowed:
        bd["_v12_champion_induction_qualified"] = induction_qualified
        bd["_v12_champion_binding_qualified"] = binding_qualified
        bd["_v12_champion_exception_allowed"] = exception_allowed
        bd["_v12_champion_sequence_signal_count"] = sequence_signal_count
        bd["_v12_champion_strong_induction"] = strong_induction
        bd["_v12_champion_strong_binding"] = strong_binding
        return score

    bd["_v12_champion_eligibility_ceiling"] = _V12_CHAMPION_ELIGIBILITY_CEILING
    bd["_v12_champion_induction_qualified"] = induction_qualified
    bd["_v12_champion_binding_qualified"] = binding_qualified
    bd["_v12_champion_exception_allowed"] = exception_allowed
    bd["_v12_champion_sequence_signal_count"] = sequence_signal_count
    bd["_v12_champion_strong_induction"] = strong_induction
    bd["_v12_champion_strong_binding"] = strong_binding
    return _V12_CHAMPION_ELIGIBILITY_CEILING


def compute_composite_v12(
    *,
    decompose: bool = False,
    **kw: Any,
) -> Union[float, Dict[str, Any]]:
    """Composite v12 dry-run: v11 plus loss-budget rebalance and champion gate."""
    result = compute_composite_v11(decompose=True, **kw)
    score = float(result["composite_score"])
    bd = result.get("breakdown") or {}

    loss_before = sum(float(bd.get(key, 0.0) or 0.0) for key in _LOSS_TIER_BD_KEYS)
    for key, factor in _V12_LOSS_COMPONENT_FACTORS.items():
        if key not in bd or not bd[key]:
            continue
        old_value = float(bd[key])
        new_value = old_value * factor
        bd[key] = new_value
        score += new_value - old_value
    loss_after = sum(float(bd.get(key, 0.0) or 0.0) for key in _LOSS_TIER_BD_KEYS)
    bd["_v12_loss_budget_before"] = loss_before
    bd["_v12_loss_budget_after"] = loss_after
    bd["_v12_loss_budget_max"] = _V12_LOSS_BUDGET_MAX
    if "_v10_base_v8style_total" in bd:
        bd["_v10_base_v8style_total"] = max(
            0.0,
            float(bd.get("_v10_base_v8style_total") or 0.0)
            + (loss_after - loss_before),
        )

    score = _v12_champion_eligibility_gate(score, bd, kw)

    if decompose:
        result["composite_score"] = score
        result["breakdown"] = bd
        return result
    return score


_CL_TIER_BD_KEYS = (
    "cl_s05_sa",
    "cl_s05_order",
    "cl_s10_sa",
    "cl_s10_order",
    "cl_inv_sa",
    "cl_inv_order",
)


def _score_controlled_lang_tier(
    cfg: Dict[str, float],
    *,
    inv_failed: bool,
    kw: Dict[str, Any],
) -> tuple[float, Dict[str, float]]:
    """Score the controlled-language probe ladder. Each tier credits
    progressively more points: S0.5 (5pt) → S1.0 (15pt) → Investigation
    (25pt) = 45pt total. Anchors set at cohort medians per tier so half
    the cohort earns half the weight. inv_failed rows zero out the tier."""
    bd: Dict[str, float] = {k: 0.0 for k in _CL_TIER_BD_KEYS}
    if inv_failed:
        return 0.0, bd
    pairs = (
        ("cl_s05_sa", kw.get("controlled_lang_s05_sa_score"), cfg["cl_s05_sa_anchor"]),
        (
            "cl_s05_order",
            kw.get("controlled_lang_s05_nb_order_acc"),
            cfg["cl_s05_order_anchor"],
        ),
        ("cl_s10_sa", kw.get("controlled_lang_s10_sa_score"), cfg["cl_s10_sa_anchor"]),
        (
            "cl_s10_order",
            kw.get("controlled_lang_s10_nb_order_acc"),
            cfg["cl_s10_order_anchor"],
        ),
        ("cl_inv_sa", kw.get("controlled_lang_inv_sa_score"), cfg["cl_inv_sa_anchor"]),
        (
            "cl_inv_order",
            kw.get("controlled_lang_inv_nb_order_acc"),
            cfg["cl_inv_order_anchor"],
        ),
    )
    total = 0.0
    for key, value, anchor in pairs:
        weight = float(cfg.get(f"w_{key}", 0.0))
        pts = weight * _scurve_higher_better(value, anchor)
        bd[key] = pts
        total += pts
    return total, bd


def compute_composite_v14(
    *,
    decompose: bool = False,
    **kw: Any,
) -> Union[float, Dict[str, Any]]:
    """Composite v14: v12 + tier-progressive controlled-language ladder.

    Adds 45pt of nano-scale BLiMP/HellaSwag replacement signal across
    three difficulty tiers (S0.5/S1.0/Investigation). Real BLiMP stays
    at 5pt floor; real HellaSwag drops 15→5 (per cohort audit showing
    near-noise discrimination at our 750-step eval scale).
    """
    base = compute_composite_v12(decompose=True, **kw)
    score = float(base["composite_score"])
    bd = base.get("breakdown") or {}

    tier = kw.get("tier")
    inv_failed = tier in ("investigation_failed", "screened_out")
    cl_pts, cl_bd = _score_controlled_lang_tier(
        _V14_CONFIG, inv_failed=inv_failed, kw=kw
    )
    score += cl_pts
    bd.update(cl_bd)
    bd["_v14_controlled_lang_total"] = cl_pts

    if decompose:
        base["composite_score"] = score
        base["breakdown"] = bd
        return base
    return score


_LOSS_TIER_BD_KEYS = (
    "perf_short",
    "perf_medium",
    "perf_long",
    "param_efficiency",
    "learning_efficiency",
    "early_convergence",
    "speed",
)
_UND_TIER_BD_KEYS = (
    "blimp",
    "tinystories",
    "cross_task",
    "diagnostic",
    "hellaswag",
    "hierarchy",
)


def _cv_penalty_multiplier(
    cv: Optional[float],
    lam: float,
    floor: float,
) -> float:
    """1.0 if cv is None or non-positive; else max(floor, 1 - lambda * cv)."""
    if cv is None:
        return 1.0
    try:
        cv_f = float(cv)
    except (TypeError, ValueError):
        return 1.0
    if cv_f <= 0.0:
        return 1.0
    return max(floor, 1.0 - lam * cv_f)


def _scurve_lower_better(value: Optional[float], anchor: float) -> float:
    """S-curve where MORE NEGATIVE is better (id_collapse, erf_decay, icld).

    Negates the value (so negative inputs become positive ratios) then
    runs the standard _scurve. Anchor is the absolute value of the
    frontier-equivalent (e.g. 0.01 for id_collapse means a frontier rate
    of -0.01). Returns 0.0 for missing/positive values (positive =
    collapsing, which is the bad direction).
    """
    if value is None:
        return 0.0
    flipped = -float(value)
    if flipped <= 0.0:
        return 0.0
    return _scurve(flipped / max(anchor, 1e-9))


def _scurve_higher_better(value: Optional[float], anchor: float) -> float:
    """S-curve where higher is better (erf_density, logit_margin, etc.)."""
    if value is None or value <= 0.0:
        return 0.0
    return _scurve(value / max(anchor, 1e-9))


def _score_capability_tier_v10(
    cfg: Dict[str, float],
    *,
    inv_failed: bool,
    effective_ar_auc: Optional[float],
    effective_induction_auc: Optional[float],
    effective_binding_auc: Optional[float],
    erf_density: Optional[float],
    id_collapse_rate: Optional[float],
    erf_decay_slope: Optional[float],
    logit_margin_velocity: Optional[float],
) -> tuple[float, Dict[str, float]]:
    """v10 capability tier — 7 metrics × 25pts, each S-curved independently."""
    bd: Dict[str, float] = {}
    total = 0.0

    if inv_failed:
        for k in (
            "cap_ar",
            "cap_induction",
            "cap_binding",
            "cap_erf_density",
            "cap_id_collapse",
            "cap_erf_decay",
            "cap_logit_margin",
        ):
            bd[k] = 0.0
        return 0.0, bd

    pairs = (
        ("cap_ar", _scurve_higher_better(effective_ar_auc, cfg["cap_ar_anchor"])),
        (
            "cap_induction",
            _scurve_higher_better(effective_induction_auc, cfg["cap_induction_anchor"]),
        ),
        (
            "cap_binding",
            _scurve_higher_better(effective_binding_auc, cfg["cap_binding_anchor"]),
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
        pts = cfg[f"w_{key}"] * frac
        bd[key] = pts
        total += pts
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

    ar_timed_out = kw.get("ar_timed_out")
    effective_ar = None if ar_timed_out else kw.get("ar_auc")
    eff_ind = (
        kw.get("induction_v2_inv_auc")
        if kw.get("induction_v2_inv_auc") is not None
        else kw.get("induction_auc")
    )
    eff_bind = (
        kw.get("binding_v2_inv_auc")
        if kw.get("binding_v2_inv_auc") is not None
        else kw.get("binding_auc")
    )

    cap_pts, cap_bd = _score_capability_tier_v10(
        cfg,
        inv_failed=inv_failed,
        effective_ar_auc=effective_ar,
        effective_induction_auc=eff_ind,
        effective_binding_auc=eff_bind,
        erf_density=kw.get("fp_jacobian_erf_density"),
        id_collapse_rate=kw.get("fp_id_collapse_rate"),
        erf_decay_slope=kw.get("fp_jacobian_erf_decay_slope"),
        logit_margin_velocity=kw.get("fp_logit_margin_velocity"),
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


def composite_score_ceiling(version: str | None = None) -> float:
    """Return the theoretical maximum composite score under the active formula.

    Derived from the v14 weight config so UI scale ceilings stay in sync
    with the scorer. ``version`` is accepted for backwards compatibility
    with callers from the multi-version era; it is ignored.
    """
    del version  # Single-version era; argument retained for API stability.
    cfg = _V14_CONFIG
    base_max = (
        cfg["w_perf_short"]
        + cfg["w_perf_medium"]
        + cfg["w_perf_long"]
        + cfg["w_param_eff"]
        + cfg["w_learn_eff"]
        + 50.0  # routing_savings
        + 30.0  # compression
        + 30.0  # sparsity
        + 25.0  # adaptive_computation
        + 40.0  # novelty
        + 15.0  # ncd
        + 40.0  # robustness
        + 25.0  # long_context
        + cfg["w_binding"]
        + cfg.get("w_blimp", 40.0)
        + cfg["w_tinystories"]
        + cfg["w_cross_task"]
        + cfg["w_diagnostic"]
        + cfg["w_hellaswag"]
        + cfg["w_hierarchy"]
        + 25.0  # speed
        + 10.0  # early_convergence
        + cfg["w_cap_ar"]
        + cfg["w_cap_induction"]
        + cfg["w_cap_binding"]
        + cfg["w_cap_erf_density"]
        + cfg["w_cap_id_collapse"]
        + cfg["w_cap_erf_decay"]
        + cfg["w_cap_logit_margin"]
        + cfg["w_aux_erf_variance"]
        + cfg["w_aux_icld"]
        # Controlled-language ladder (v14 addition)
        + cfg["w_cl_s05_sa"]
        + cfg["w_cl_s05_order"]
        + cfg["w_cl_s10_sa"]
        + cfg["w_cl_s10_order"]
        + cfg["w_cl_inv_sa"]
        + cfg["w_cl_inv_order"]
    )
    return float(base_max)


# ---------------------------------------------------------------------------
# Active scoring formula
# ---------------------------------------------------------------------------
# Single canonical formula (v14, locked 2026-05-03). The prior multi-version
# dispatcher (v7-v14) is gone — every rescore overwrote the version stamp
# anyway, so version pinning was illusory provenance. Stage 2 of this refactor
# will move the weights/anchors out of code into ``research/scoring_config.yaml``
# so tuning becomes a config edit, not a code patch.
ACTIVE_SCORING_VERSION: str = "v14"


def get_scoring_version() -> str:
    """Return the active scoring formula identifier (constant: ``v14``)."""
    return ACTIVE_SCORING_VERSION


def compute_composite(
    *, decompose: bool = False, **kw: Any
) -> Union[float, Dict[str, Any]]:
    """Compute the leaderboard composite score under the active (v14) formula."""
    return compute_composite_v14(decompose=decompose, **kw)
