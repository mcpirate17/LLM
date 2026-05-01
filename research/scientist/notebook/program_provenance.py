from __future__ import annotations

"""Focused provenance inference helpers for program results."""

import json
from collections import Counter
from typing import Any, Callable, Dict

from ._shared import sanitize_for_db


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalized_fraction(value: Any, default: float) -> float:
    try:
        return round(float(value if value is not None else default), 6)
    except (TypeError, ValueError):
        return default


_EXPERIMENT_PROVENANCE_KEYS = (
    "data_mode",
    "corpus_path",
    "corpus_format",
    "corpus_text_key",
    "corpus_train_fraction",
    "corpus_val_fraction",
    "corpus_max_chars",
    "tokenizer_mode",
    "tiktoken_encoding",
    "vocab_size",
    "hf_dataset",
    "hf_subset",
    "hf_split",
    "hf_text_key",
    "hydra_dataset",
    "hydra_data_dir",
)

_EXPERIMENT_CONTEXT_KEYS = (
    "model_source",
    "mode",
    "threshold",
    "device",
    "model_dim",
    "n_layers",
    "s1_steps",
    "rapid_steps",
    "graphs_per_op",
    "n_graphs_weighted",
    "backfill_template",
    "backfill_phase",
    "backfill_weight_mode",
    "backfill_n_programs",
)

_DEFAULT_SCREENING_WIKITEXT_METRIC_VERSION = "screening_wikitext_v1"


def merge_experiment_provenance_kwargs(
    kwargs: Dict[str, Any],
    experiment_config: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Backfill provenance-relevant fields from experiment config when missing."""
    if not experiment_config:
        return dict(kwargs)
    merged = dict(kwargs)
    for key in _EXPERIMENT_PROVENANCE_KEYS + _EXPERIMENT_CONTEXT_KEYS:
        if merged.get(key) in (None, "") and experiment_config.get(key) not in (
            None,
            "",
        ):
            merged[key] = experiment_config.get(key)
    return merged


def with_inferred_metric_versions(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Recover canonical metric versions when the measurement exists but the version was dropped."""
    merged = dict(kwargs)
    has_screening_wikitext = any(
        merged.get(key) is not None
        for key in (
            "wikitext_perplexity",
            "wikitext_score",
            "wikitext_pre_perplexity",
            "wikitext_ppl_improvement",
            "screening_wikitext_status",
        )
    )
    if has_screening_wikitext and not _clean_text(
        merged.get("screening_wikitext_metric_version")
    ):
        merged["screening_wikitext_metric_version"] = (
            _DEFAULT_SCREENING_WIKITEXT_METRIC_VERSION
        )
    return merged


def infer_comparability_gap_details(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = with_inferred_metric_versions(kwargs)
    derived = derive_provenance_fields(kwargs)
    gaps: list[str] = []
    metric_versions = {
        "screening_wikitext_metric_version": _clean_text(
            kwargs.get("screening_wikitext_metric_version")
        ),
        "induction_probe_metric_version": _clean_text(
            kwargs.get("induction_probe_metric_version")
        ),
        "novelty_reference_version": _clean_text(
            kwargs.get("novelty_reference_version")
        ),
    }
    if not derived.get("corpus_id"):
        gaps.append("missing_corpus_id")
    if not derived.get("tokenizer_id"):
        gaps.append("missing_tokenizer_id")
    if not derived.get("split_id"):
        gaps.append("missing_split_id")
    if not any(metric_versions.values()):
        gaps.append("missing_metric_version")
    if gaps:
        reason = gaps[0]
    else:
        reason = "comparable"
    return sanitize_for_db(
        {
            "comparability_reason": reason,
            "comparability_gaps": gaps,
        }
    )


def _has_complete_candidate_provenance(kwargs: Dict[str, Any]) -> bool:
    kwargs = with_inferred_metric_versions(kwargs)
    provenance_complete = kwargs.get("provenance_complete")
    if provenance_complete is True:
        return True
    derived = derive_provenance_fields(kwargs)
    return bool(derived.get("provenance_complete"))


def derive_provenance_fields(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Derive durable corpus/tokenizer/split identifiers from run config fields."""
    kwargs = with_inferred_metric_versions(kwargs)
    data_mode = normalize_text(kwargs.get("data_mode"))
    tokenizer_mode = normalize_text(kwargs.get("tokenizer_mode"))
    vocab_size = kwargs.get("vocab_size")
    tiktoken_encoding = _clean_text(kwargs.get("tiktoken_encoding"))

    tokenizer_id = _clean_text(kwargs.get("tokenizer_id"))
    tokenizer_version = _clean_text(kwargs.get("tokenizer_version"))
    if tokenizer_mode == "tiktoken":
        tokenizer_version = tiktoken_encoding or "default"
        tokenizer_id = f"tiktoken:{tokenizer_version}"
    elif tokenizer_mode:
        suffix = f":vocab{int(vocab_size)}" if vocab_size not in (None, "") else ""
        tokenizer_id = f"{tokenizer_mode}{suffix}"
        tokenizer_version = (
            f"vocab_size={int(vocab_size)}" if vocab_size not in (None, "") else ""
        )

    corpus_id = _clean_text(kwargs.get("corpus_id"))
    corpus_version = _clean_text(kwargs.get("corpus_version"))
    split_id = _clean_text(kwargs.get("split_id"))
    split_policy = _clean_text(kwargs.get("split_policy_version"))
    data_signature = _clean_text(kwargs.get("data_signature"))

    if data_mode == "corpus":
        corpus_path = _clean_text(kwargs.get("corpus_path"))
        corpus_format = normalize_text(kwargs.get("corpus_format")) or "auto"
        text_key = _clean_text(kwargs.get("corpus_text_key")) or "text"
        max_chars = int(kwargs.get("corpus_max_chars") or 0)
        train_fraction = _normalized_fraction(kwargs.get("corpus_train_fraction"), 0.9)
        val_fraction = _normalized_fraction(kwargs.get("corpus_val_fraction"), 0.1)
        corpus_id = f"file:{corpus_path}" if corpus_path else "file:missing"
        corpus_version = (
            f"fmt={corpus_format};text_key={text_key};max_chars={max_chars}"
        )
        split_policy = "fractional_holdout_v1"
        split_id = f"train={train_fraction:.3f};val={val_fraction:.3f}"
        data_signature = (
            f"{corpus_id}|{corpus_version}|{split_id}|tokenizer={tokenizer_id}"
        )
    elif data_mode == "huggingface":
        hf_dataset = _clean_text(kwargs.get("hf_dataset"))
        hf_subset = _clean_text(kwargs.get("hf_subset"))
        hf_split = _clean_text(kwargs.get("hf_split")) or "train"
        hf_text_key = _clean_text(kwargs.get("hf_text_key")) or "text"
        max_chars = int(kwargs.get("corpus_max_chars") or 0)
        corpus_id = f"hf:{hf_dataset}:{hf_subset or 'default'}"
        corpus_version = (
            f"split={hf_split};text_key={hf_text_key};max_chars={max_chars}"
        )
        split_policy = "hf_declared_split_v1"
        split_id = hf_split
        data_signature = f"{corpus_id}|{corpus_version}|tokenizer={tokenizer_id}"
    elif data_mode == "hydra":
        hydra_dataset = _clean_text(kwargs.get("hydra_dataset")) or "unknown"
        hydra_root = _clean_text(kwargs.get("hydra_data_dir"))
        corpus_id = f"hydra:{hydra_dataset}"
        corpus_version = f"data_dir={hydra_root}"
        split_policy = "hydra_runtime_v1"
        split_id = "runtime"
        data_signature = f"{corpus_id}|{corpus_version}|tokenizer={tokenizer_id}"
    elif data_mode == "random":
        corpus_id = "synthetic:random_tokens"
        corpus_version = "uniform_random_v1"
        split_policy = "synthetic_runtime_v1"
        split_id = "runtime"
        data_signature = f"{corpus_id}|tokenizer={tokenizer_id}"

    provenance_complete = bool(
        corpus_id
        and tokenizer_id
        and split_id
        and (
            kwargs.get("screening_wikitext_metric_version")
            or kwargs.get("induction_probe_metric_version")
            or kwargs.get("novelty_reference_version")
        )
    )
    if kwargs.get("provenance_complete") is True:
        provenance_complete = True

    return sanitize_for_db(
        {
            "corpus_id": corpus_id,
            "corpus_version": corpus_version,
            "split_id": split_id,
            "split_policy_version": split_policy,
            "tokenizer_id": tokenizer_id,
            "tokenizer_version": tokenizer_version,
            "data_signature": data_signature,
            "provenance_complete": provenance_complete,
        }
    )


def parse_graph_json(graph_json: Any) -> Dict[str, Any]:
    if isinstance(graph_json, dict):
        return graph_json
    if isinstance(graph_json, str) and graph_json.strip():
        try:
            loaded = json.loads(graph_json)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        if isinstance(loaded, dict):
            return loaded
    return {}


def extract_graph_provenance(graph_json: Any) -> Dict[str, Any]:
    graph = parse_graph_json(graph_json)
    if not graph:
        return {}

    raw_nodes = graph.get("nodes") or {}
    if isinstance(raw_nodes, list):
        nodes = raw_nodes
    elif isinstance(raw_nodes, dict):
        nodes = list(raw_nodes.values())
    else:
        nodes = []

    op_names = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        op_name = (
            node.get("op_name") or node.get("component_type") or node.get("op") or ""
        )
        op_names.append(str(op_name).split("/")[-1].strip().lower())

    op_counter = Counter(op for op in op_names if op and op != "input")
    metadata = graph.get("metadata") or {}
    templates_used = metadata.get("templates_used") or []
    lineage = metadata.get("lineage") or {}

    attention_ops = {
        "softmax_attention",
        "linear_attention",
        "graph_attention",
        "local_window_attn",
        "diff_attention",
        "latent_attention_compressor",
    }
    routing_ops = {
        "route_topk",
        "route_lanes",
        "route_recursion",
        "moe_topk",
        "moe_2expert",
        "sparse_bottleneck_moe",
        "learned_token_gate",
        "confidence_token_gate",
        "hybrid_sparse_router",
        "hybrid_token_gate",
        "default_path",
        "lane_conditioned_block",
        "signal_conditioned_compression",
        "difficulty_blend_3way",
        "depth_weighted_proj",
        "score_depth_blend",
        "adaptive_rank_gate",
        "dual_compression_blend",
    }
    norm_ops = {"layernorm", "rmsnorm", "group_norm", "batch_norm"}
    sparse_ops = {
        "nm_sparse_linear",
        "block_sparse_linear",
        "semi_structured_2_4_linear",
        "sparse_threshold",
        "depth_token_mask",
    }
    moe_ops = {"moe_topk", "moe_2expert", "sparse_bottleneck_moe", "hetero_moe"}

    family = "dense"
    if any(op in op_counter for op in moe_ops):
        family = "moe"
    elif any(op in op_counter for op in routing_ops):
        family = "routing"
    elif any(op in op_counter for op in sparse_ops):
        family = "sparse"
    elif any(op in op_counter for op in attention_ops):
        family = "attention"

    return sanitize_for_db(
        {
            "graph_family": family,
            "graph_n_ops_derived": int(sum(op_counter.values())),
            "graph_n_unique_ops_derived": int(len(op_counter)),
            "graph_has_attention": bool(any(op in op_counter for op in attention_ops)),
            "graph_has_routing": bool(any(op in op_counter for op in routing_ops)),
            "graph_has_moe": bool(any(op in op_counter for op in moe_ops)),
            "graph_has_sparse": bool(any(op in op_counter for op in sparse_ops)),
            "graph_has_norm": bool(any(op in op_counter for op in norm_ops)),
            "graph_has_residual": bool("add" in op_counter),
            "graph_templates_used": list(templates_used)
            if isinstance(templates_used, list)
            else [],
            "graph_template_count": int(len(templates_used))
            if isinstance(templates_used, list)
            else 0,
            "graph_has_lineage_parent": bool(
                isinstance(lineage, dict)
                and (
                    lineage.get("parent")
                    or lineage.get("parents")
                    or lineage.get("seed_fingerprint")
                )
            ),
            "graph_top_ops": [op for op, _ in op_counter.most_common(8)],
        }
    )


def infer_result_cohort(
    kwargs: Dict[str, Any],
    *,
    experiment_type_for_id: Callable[[Any], str],
) -> str:
    source = normalize_text(kwargs.get("model_source"))
    experiment_type = normalize_text(
        kwargs.get("experiment_type")
        or experiment_type_for_id(kwargs.get("experiment_id"))
    )
    if experiment_type in {"backfill", "exact_graph_replay", "forced_exploration"}:
        return "backfill"
    if experiment_type == "designer":
        return "designer"
    if experiment_type == "reference":
        return "reference"
    if not source:
        if experiment_type in {
            "synthesis",
            "novelty",
            "evolution",
            "ablation",
            "validation",
            "investigation",
        }:
            return "search"
        return "legacy_unlabeled"
    if source == "reference":
        return "reference"
    if "backfill" in source or "backpopulate" in source or "replay" in source:
        return "backfill"
    if source.startswith("designer"):
        return "designer"
    if source == "forced_exploration":
        return "backfill"
    if source in {
        "graph_synthesis",
        "novelty",
        "evolution",
        "morphological_box",
        "mixed",
        "grammar",
        "fingerprint_refine",
        "ablation",
    }:
        return "search"
    return "runtime"


def infer_trust_label(kwargs: Dict[str, Any], result_cohort: str) -> str:
    if result_cohort == "reference":
        return "reference"
    if result_cohort == "backfill":
        return "backfill_observation"
    if result_cohort == "legacy_unlabeled":
        return "legacy_unlabeled"
    if result_cohort == "designer":
        return "exploratory"
    # The universal metric-completeness guard
    # (notebook/program_writes.py:_enforce_s1_metric_completeness) requires
    # every stage1_passed=True row to carry the 7 post-S1 probe metrics, so
    # a passing row always qualifies as candidate_grade. The 'candidate_
    # screening' branch was for historical loss-only S1 rows; production
    # cannot produce those any more (and historical rows keep their stored
    # label since this function only fires on writes). 1565 historical rows
    # with trust_label='candidate_screening' remain valid for reads via
    # trust_policy.TRUSTED_TRUST_LABELS and notebook_dashboard filters.
    if kwargs.get("stage1_passed") in (1, True):
        return "candidate_grade"
    return "runtime_observation"


def infer_comparability_label(
    kwargs: Dict[str, Any],
    result_cohort: str,
    trust_label: str,
) -> str:
    kwargs = with_inferred_metric_versions(kwargs)
    if result_cohort == "legacy_unlabeled":
        return "legacy_noncomparable"
    if result_cohort == "backfill":
        return "reconstructed_init_variant"
    if trust_label == "reference":
        return "reference_comparable"
    has_data_provenance = any(
        kwargs.get(key)
        for key in (
            "tokenizer_mode",
            "tokenizer_id",
            "corpus_path",
            "corpus_id",
            "split_id",
            "screening_wikitext_metric_version",
            "induction_probe_metric_version",
            "novelty_reference_version",
        )
    )
    if trust_label == "candidate_grade" and _has_complete_candidate_provenance(kwargs):
        return "candidate_comparable"
    return "partial" if has_data_provenance else "noncomparable"


def infer_evaluation_protocol_version(
    kwargs: Dict[str, Any],
    result_cohort: str,
    trust_label: str,
) -> str:
    if result_cohort == "backfill":
        return "backfill_replay_v1"
    if result_cohort == "designer":
        return "designer_bridge_v1"
    if trust_label == "candidate_grade":
        return "candidate_grade_v1"
    if result_cohort == "reference":
        return "reference_v1"
    return "runtime_observation_v1"


def infer_init_regime(kwargs: Dict[str, Any], result_cohort: str) -> str:
    if result_cohort == "backfill":
        return "reconstructed_fresh_init"
    if result_cohort == "reference":
        return "reference_control"
    data_mode = normalize_text(kwargs.get("data_mode"))
    return "random_token_train" if data_mode == "random" else "runtime_default"


def infer_usage_eligibility(
    kwargs: Dict[str, Any],
    *,
    result_cohort: str,
    trust_label: str,
    comparability_label: str,
) -> Dict[str, Any]:
    kwargs = with_inferred_metric_versions(kwargs)
    derived = derive_provenance_fields(kwargs)
    promotion_provenance_complete = _has_complete_candidate_provenance(kwargs)
    screening_data_complete = bool(
        derived.get("corpus_id")
        and derived.get("tokenizer_id")
        and derived.get("split_id")
    )
    stage0_passed = bool(kwargs.get("stage0_passed"))
    stage05_passed = bool(kwargs.get("stage05_passed"))
    stage1_passed = bool(kwargs.get("stage1_passed"))

    promotion_eligible = trust_label in {"reference", "candidate_grade"} and (
        comparability_label in {"reference_comparable", "candidate_comparable"}
    )
    if promotion_eligible:
        promotion_reason = "trusted_comparable_candidate"
    elif trust_label == "candidate_grade":
        promotion_reason = "candidate_grade_missing_comparability"
    else:
        promotion_reason = "non_promotable_cohort"

    screening_role = "excluded"
    screening_eligible = False
    screening_reason = "not_screening_trainable"
    if (
        trust_label == "candidate_grade"
        and comparability_label == "candidate_comparable"
    ):
        screening_eligible = True
        screening_role = "positive"
        screening_reason = "trusted_candidate_positive"
    elif (
        trust_label == "runtime_observation"
        and result_cohort == "search"
        and stage0_passed
        and stage05_passed
        and not stage1_passed
        and screening_data_complete
    ):
        screening_eligible = True
        screening_role = "negative"
        screening_reason = "runtime_search_negative_with_complete_provenance"

    return sanitize_for_db(
        {
            "eligible_for_promotion": promotion_eligible,
            "promotion_eligibility_reason": promotion_reason,
            "eligible_for_screening_model_training": screening_eligible,
            "screening_model_training_role": screening_role,
            "screening_model_training_reason": screening_reason,
            "screening_training_data_complete": screening_data_complete,
            "promotion_provenance_complete": promotion_provenance_complete,
        }
    )


def build_data_provenance(
    kwargs: Dict[str, Any],
    *,
    experiment_type_for_id: Callable[[Any], str],
    result_cohort: str,
    trust_label: str,
    comparability_label: str,
    evaluation_protocol_version: str,
    init_regime: str,
) -> str:
    kwargs = with_inferred_metric_versions(kwargs)
    derived = derive_provenance_fields(kwargs)
    comparability_details = infer_comparability_gap_details(kwargs)
    eligibility = infer_usage_eligibility(
        kwargs,
        result_cohort=result_cohort,
        trust_label=trust_label,
        comparability_label=comparability_label,
    )
    payload = sanitize_for_db(
        {
            "model_source": kwargs.get("model_source"),
            "experiment_type": kwargs.get("experiment_type")
            or experiment_type_for_id(kwargs.get("experiment_id")),
            "result_cohort": result_cohort,
            "trust_label": trust_label,
            "comparability_label": comparability_label,
            "comparability_reason": comparability_details.get("comparability_reason"),
            "comparability_gaps": comparability_details.get("comparability_gaps"),
            "evaluation_protocol_version": evaluation_protocol_version,
            "init_regime": init_regime,
            "data_mode": kwargs.get("data_mode"),
            "tokenizer_mode": kwargs.get("tokenizer_mode"),
            "tokenizer_id": derived.get("tokenizer_id"),
            "tokenizer_version": derived.get("tokenizer_version"),
            "corpus_path": kwargs.get("corpus_path"),
            "corpus_id": derived.get("corpus_id"),
            "corpus_version": derived.get("corpus_version"),
            "split_id": derived.get("split_id"),
            "split_policy_version": derived.get("split_policy_version"),
            "data_signature": derived.get("data_signature"),
            "provenance_complete": derived.get("provenance_complete"),
            "corpus_format": kwargs.get("corpus_format"),
            "corpus_text_key": kwargs.get("corpus_text_key"),
            "corpus_train_fraction": kwargs.get("corpus_train_fraction"),
            "corpus_val_fraction": kwargs.get("corpus_val_fraction"),
            "corpus_max_chars": kwargs.get("corpus_max_chars"),
            "tiktoken_encoding": kwargs.get("tiktoken_encoding"),
            "hf_dataset": kwargs.get("hf_dataset"),
            "hf_subset": kwargs.get("hf_subset"),
            "hf_split": kwargs.get("hf_split"),
            "hf_text_key": kwargs.get("hf_text_key"),
            "hydra_dataset": kwargs.get("hydra_dataset"),
            "hydra_data_dir": kwargs.get("hydra_data_dir"),
            "screening_wikitext_metric_version": kwargs.get(
                "screening_wikitext_metric_version"
            ),
            "induction_probe_metric_version": kwargs.get(
                "induction_probe_metric_version"
            ),
            "novelty_reference_version": kwargs.get("novelty_reference_version"),
            "novelty_scoring_policy_version": kwargs.get(
                "novelty_scoring_policy_version"
            ),
            "evaluation_stage": kwargs.get("evaluation_stage"),
            "capability_tier": kwargs.get("capability_tier"),
            "experiment_mode": kwargs.get("mode"),
            "exploration_threshold": kwargs.get("threshold"),
            "execution_device": kwargs.get("device"),
            "model_dim": kwargs.get("model_dim"),
            "n_layers": kwargs.get("n_layers"),
            "s1_steps": kwargs.get("s1_steps"),
            "rapid_steps": kwargs.get("rapid_steps"),
            "graphs_per_op": kwargs.get("graphs_per_op"),
            "n_graphs_weighted": kwargs.get("n_graphs_weighted"),
            "backfill_template": kwargs.get("backfill_template"),
            "backfill_phase": kwargs.get("backfill_phase"),
            "backfill_weight_mode": kwargs.get("backfill_weight_mode"),
            "backfill_n_programs": kwargs.get("backfill_n_programs"),
            "eligible_for_promotion": eligibility.get("eligible_for_promotion"),
            "promotion_eligibility_reason": eligibility.get(
                "promotion_eligibility_reason"
            ),
            "eligible_for_screening_model_training": eligibility.get(
                "eligible_for_screening_model_training"
            ),
            "screening_model_training_role": eligibility.get(
                "screening_model_training_role"
            ),
            "screening_model_training_reason": eligibility.get(
                "screening_model_training_reason"
            ),
            "graph": extract_graph_provenance(kwargs.get("graph_json")),
        }
    )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
