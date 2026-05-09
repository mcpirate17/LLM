"""Score-kwargs builders.

Translates raw program_results rows + leaderboard rows into the kwargs dict
consumed by ``compute_composite``. Pure logic — SQL touches the connection
caller-side, no I/O scheduling here.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence


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
    "ar_legacy_auc, ar_legacy_final_acc, ar_legacy_timed_out, "
    "ar_legacy_above_chance, ar_gate_score, induction_screening_auc, binding_screening_auc, blimp_overall_accuracy, "
    "tinystories_score, cross_task_score, diagnostic_score, "
    "fp_gromov_delta, fp_hierarchy_fitness, "
    "induction_intermediate_auc, induction_intermediate_max_gap_acc, "
    "induction_intermediate_protocol_version, binding_intermediate_auc, "
    "binding_intermediate_max_distance_acc, "
    "binding_intermediate_protocol_version, "
    "champion_tiny_model_protocol_version, champion_steps_to_floor, "
    "champion_floor_ppl, champion_floor_loss, champion_floor_loss_std, "
    "champion_baseline_result_id, champion_baseline_layers, "
    "champion_baseline_protocol_version, induction_validation_auc, "
    "induction_validation_gap_accuracy_cv, induction_validation_protocol_version, "
    "champion_ar_validation_score, ar_validation_metric_version, "
    "ar_validation_held_pair_acc, "
    "ar_validation_held_class_acc, ar_validation_steps_to_floor, "
    "ar_validation_rank_score, ar_validation_status, "
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
    # Language-control probe ladder (v14): nano-scale BLiMP/HellaSwag
    # replacement at three difficulty tiers. Read by the v14 composite
    # scorer for the 45pt language_control tier.
    "language_control_metric_version, "
    "language_control_s05_sentence_assoc_score, language_control_s05_binding_order_acc, "
    "language_control_s05_binding_score, "
    "language_control_s10_sentence_assoc_score, language_control_s10_binding_order_acc, "
    "language_control_s10_binding_score, "
    "language_control_investigation_sentence_assoc_score, language_control_investigation_binding_order_acc, "
    "language_control_investigation_binding_score, "
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
        # ar_legacy_auc retained for legacy display only; weight is zero in
        # binding_screening_composite — ar_gate_score is the active signal.
        "ar_legacy_auc": pr_dict.get("ar_legacy_auc") or d.get("ar_legacy_auc"),
        "ar_legacy_timed_out": (
            bool(pr_dict.get("ar_legacy_timed_out"))
            if pr_dict.get("ar_legacy_timed_out") is not None
            else None
        ),
        "ar_legacy_above_chance": (
            bool(pr_dict.get("ar_legacy_above_chance"))
            if pr_dict.get("ar_legacy_above_chance") is not None
            else None
        ),
        "ar_gate_score": pr_dict.get("ar_gate_score") or d.get("ar_gate_score"),
        "induction_screening_auc": pr_dict.get("induction_screening_auc")
        or d.get("induction_screening_auc"),
        "binding_screening_auc": pr_dict.get("binding_screening_auc")
        or d.get("binding_screening_auc"),
        # v2 investigation-tier probes — overrides induction_screening_auc/binding_screening_auc
        # in the composite when present. `or`-chained against `d` so we read
        # the leaderboard row for rows where the program_results row is
        # missing the column (pre-backfill).
        "induction_intermediate_inv_auc": pr_dict.get("induction_intermediate_auc")
        or d.get("induction_intermediate_auc"),
        "binding_intermediate_inv_auc": pr_dict.get("binding_intermediate_auc")
        or d.get("binding_intermediate_auc"),
        # BLiMP
        "blimp_accuracy": pr_dict.get("blimp_overall_accuracy")
        or d.get("blimp_overall_accuracy"),
        # HellaSwag commonsense reasoning
        "hellaswag_acc_screening": pr_dict.get("hellaswag_acc")
        or d.get("hellaswag_acc"),
        "hellaswag_acc_investigation": (
            d.get("hellaswag_acc") if tier in _inv_tiers else None
        ),
        "hellaswag_acc_validation": (
            d.get("hellaswag_acc") if tier in _val_tiers else None
        ),
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
        # Language-control probe ladder (v14): pass through the full
        # recorded probe payload. The scorer consumes SA + NB order today;
        # dashboard consumers also render NB score. Absent on un-probed rows
        # (None scores -> 0pts via _scurve_higher_better).
        "language_control_s05_sentence_assoc_score": pr_dict.get(
            "language_control_s05_sentence_assoc_score"
        ),
        "language_control_s05_binding_order_acc": pr_dict.get(
            "language_control_s05_binding_order_acc"
        ),
        "language_control_s05_binding_score": pr_dict.get(
            "language_control_s05_binding_score"
        ),
        "language_control_s10_sentence_assoc_score": pr_dict.get(
            "language_control_s10_sentence_assoc_score"
        ),
        "language_control_s10_binding_order_acc": pr_dict.get(
            "language_control_s10_binding_order_acc"
        ),
        "language_control_s10_binding_score": pr_dict.get(
            "language_control_s10_binding_score"
        ),
        "language_control_investigation_sentence_assoc_score": pr_dict.get(
            "language_control_investigation_sentence_assoc_score"
        ),
        "language_control_investigation_binding_order_acc": pr_dict.get(
            "language_control_investigation_binding_order_acc"
        ),
        "language_control_investigation_binding_score": pr_dict.get(
            "language_control_investigation_binding_score"
        ),
        "tokenizer_mode": pr_dict.get("tokenizer_mode") or d.get("tokenizer_mode"),
        "screening_wikitext_metric_version": (
            pr_dict.get("screening_wikitext_metric_version")
            or d.get("screening_wikitext_metric_version")
        ),
    }
    for name in (
        "champion_tiny_model_protocol_version",
        "champion_tiny_model_protocol",
        "use_champion_tiny_model_score",
        "diverged",
        "training_diverged",
        "divergence_detected",
        "champion_diverged",
        "training_status",
        "champion_status",
        "persistence_failed",
        "champion_persistence_failed",
        "result_persistence_failed",
        "checkpoint_available",
        "champion_checkpoint_available",
        "missing_checkpoint",
        "champion_missing_checkpoint",
        "checkpoint_path",
        "champion_checkpoint_path",
        "champion_steps_to_floor",
        "champion_baseline_steps_to_floor",
        "gpt2_steps_to_floor",
        "baseline_steps_to_floor",
        "champion_floor_ppl",
        "floor_ppl",
        "champion_baseline_floor_ppl",
        "gpt2_floor_ppl",
        "baseline_floor_ppl",
        "champion_floor_loss",
        "floor_loss",
        "champion_baseline_floor_loss",
        "gpt2_floor_loss",
        "baseline_floor_loss",
        "champion_floor_loss_std",
        "champion_baseline_floor_loss_std",
        "gpt2_floor_loss_std",
        "baseline_floor_loss_std",
        "induction_validation_auc",
        "induction_validation_gap_accuracy_cv",
        "induction_validation_protocol_version",
        "binding_intermediate_auc",
        "champion_long_ctx_combined_score",
        "long_ctx_combined_score",
        "champion_baseline_long_ctx_combined_score",
        "gpt2_long_context_baseline",
        "baseline_long_ctx_combined_score",
        "ar_validation_held_pair_acc",
        "ar_validation_held_class_acc",
        "ar_validation_rank_score",
        "ar_validation_learning_speed_score",
        "ar_validation_steps_to_floor",
        "champion_ar_validation_score",
        "ar_validation_metric_version",
        "ar_validation_status",
        "champion_baseline_ar_validation_steps_to_floor",
        "gpt2_ar_validation_steps_to_floor",
        "baseline_ar_validation_steps_to_floor",
    ):
        value = pr_dict.get(name)
        if value is None:
            value = d.get(name)
        if value is not None:
            kw[name] = value
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
