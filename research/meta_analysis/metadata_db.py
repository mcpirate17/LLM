"""Build a separate SQLite meta-analysis database.

This module reads the lab notebook in read-only mode and writes derived,
wide template/slot property tables to a standalone database. It deliberately
does not migrate or mutate ``lab_notebook.db``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from research.defaults import LAB_NOTEBOOK_DB

from .descriptive_properties import (
    ALTERNATIVE_MATH_CANDIDATE_COLUMNS,
    OP_PROPERTY_COLUMNS,
    OP_PROPERTY_VERSION,
    PROPERTY_VERSION,
    SLOT_PROPERTY_COLUMNS,
    TEMPLATE_PROPERTY_COLUMNS,
    alternative_math_candidate_properties,
    canonical_slot_key,
    op_descriptive_properties,
    slot_descriptive_properties,
    template_descriptive_properties,
)


DEFAULT_META_ANALYSIS_DB = "research/meta_analysis.db"
DEFAULT_PROFILING_DB = "research/profiling/component_profiles.db"

_OP_PROFILE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("op_name", "TEXT"),
    ("registry", "TEXT"),
    ("category", "TEXT"),
    ("n_inputs", "INTEGER"),
    ("shape_rule", "TEXT"),
    ("has_params", "INTEGER"),
    ("algebraic_space", "TEXT"),
    ("param_count", "INTEGER"),
    ("output_mean", "REAL"),
    ("output_std", "REAL"),
    ("output_min", "REAL"),
    ("output_max", "REAL"),
    ("output_has_nan", "INTEGER"),
    ("output_has_inf", "INTEGER"),
    ("output_kurtosis", "REAL"),
    ("grad_norm", "REAL"),
    ("grad_max", "REAL"),
    ("grad_min", "REAL"),
    ("grad_has_nan", "INTEGER"),
    ("grad_has_zero", "INTEGER"),
    ("grad_vanishing", "INTEGER"),
    ("grad_exploding", "INTEGER"),
    ("jacobian_spectral_norm", "REAL"),
    ("jacobian_condition_num", "REAL"),
    ("lipschitz_estimate", "REAL"),
    ("forward_time_us", "REAL"),
    ("backward_time_us", "REAL"),
    ("peak_memory_bytes", "INTEGER"),
    ("flops_estimate", "INTEGER"),
    ("error", "TEXT"),
    ("profiled_at", "REAL"),
)
_PAIR_PROFILE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("op_a", "TEXT"),
    ("op_b", "TEXT"),
    ("composition", "TEXT"),
    ("shape_compatible", "INTEGER"),
    ("algebraic_compatible", "INTEGER"),
    ("output_mean", "REAL"),
    ("output_std", "REAL"),
    ("output_min", "REAL"),
    ("output_max", "REAL"),
    ("output_has_nan", "INTEGER"),
    ("output_has_inf", "INTEGER"),
    ("output_kurtosis", "REAL"),
    ("grad_norm", "REAL"),
    ("grad_max", "REAL"),
    ("grad_min", "REAL"),
    ("grad_has_nan", "INTEGER"),
    ("grad_has_zero", "INTEGER"),
    ("grad_vanishing", "INTEGER"),
    ("grad_exploding", "INTEGER"),
    ("jacobian_spectral_norm", "REAL"),
    ("jacobian_condition_num", "REAL"),
    ("lipschitz_estimate", "REAL"),
    ("forward_time_us", "REAL"),
    ("backward_time_us", "REAL"),
    ("peak_memory_bytes", "INTEGER"),
    ("flops_estimate", "INTEGER"),
    ("stability_delta", "REAL"),
    ("distribution_shift", "REAL"),
    ("speed_overhead", "REAL"),
    ("error", "TEXT"),
    ("profiled_at", "REAL"),
)
_TRIPLET_PROFILE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("op_a", "TEXT"),
    ("op_b", "TEXT"),
    ("op_c", "TEXT"),
    ("output_std", "REAL"),
    ("output_has_nan", "INTEGER"),
    ("grad_norm", "REAL"),
    ("grad_has_nan", "INTEGER"),
    ("grad_vanishing", "INTEGER"),
    ("grad_exploding", "INTEGER"),
    ("lipschitz_estimate", "REAL"),
    ("forward_time_us", "REAL"),
    ("pair_ab_predicted_stable", "INTEGER"),
    ("pair_bc_predicted_stable", "INTEGER"),
    ("triplet_stable", "INTEGER"),
    ("diverges_from_pair_prediction", "INTEGER"),
    ("error", "TEXT"),
    ("profiled_at", "REAL"),
)

_PROFILE_PROVENANCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("profile_source_db_path", "TEXT"),
    ("profile_source_mtime", "REAL"),
)

_OUTCOME_COLUMNS = (
    "result_id",
    "experiment_id",
    "timestamp",
    "graph_fingerprint",
    "stage0_passed",
    "stage05_passed",
    "stage1_passed",
    "loss_ratio",
    "validation_loss_ratio",
    "discovery_loss_ratio",
    "novelty_score",
    "novelty_confidence",
    "induction_auc",
    "induction_v2_investigation_auc",
    "induction_v2_investigation_max_gap_acc",
    "induction_v2_investigation_gap_accuracies_json",
    "induction_v2_investigation_steps_trained",
    "induction_v2_investigation_status",
    "induction_v2_investigation_elapsed_ms",
    "induction_v2_investigation_protocol_version",
    "binding_auc",
    "binding_auc_curriculum",
    "binding_v2_investigation_auc",
    "binding_v2_investigation_max_gap_acc",
    "binding_v2_investigation_gap_accuracies_json",
    "binding_v2_investigation_steps_trained",
    "binding_v2_investigation_status",
    "binding_v2_investigation_elapsed_ms",
    "binding_v2_investigation_protocol_version",
    "ar_auc",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "wikitext_perplexity",
    "wikitext_score",
    "tinystories_perplexity",
    "tinystories_score",
    "cross_task_score",
    "param_count",
    "graph_depth",
    "graph_n_ops",
    "graph_n_unique_ops",
    "graph_uses_math_spaces",
    "graph_uses_frequency_domain",
    "compression_ratio",
    "routing_savings_ratio",
    "activation_sparsity_score",
    "dead_neuron_ratio",
    "routing_collapse_score",
    "trust_label",
    "comparability_label",
    "evaluation_protocol_version",
    "model_source",
    "error_type",
    "stage_at_death",
    "composite_score",
    "screening_wikitext_status",
    "screening_wikitext_metric_version",
    "screening_wikitext_variant",
    "wikitext_ppl_200",
    "wikitext_ppl_500",
    "wikitext_improvement_ratio",
    "wikitext_eval_steps",
    "routing_fast_lane_applied",
    "routing_fast_lane_status",
    "routing_fast_lane_metric_version",
    "routing_fast_lane_perplexity",
    "routing_fast_lane_score",
    "routing_fast_lane_pre_perplexity",
    "routing_fast_lane_ppl_improvement",
    "routing_fast_lane_slope",
    "routing_fast_lane_slope_consistent",
    "controlled_lang_metric_version",
    "controlled_lang_s05_sa_score",
    "controlled_lang_s05_nb_order_acc",
    "controlled_lang_s05_nb_score",
    "controlled_lang_s10_sa_score",
    "controlled_lang_s10_nb_order_acc",
    "controlled_lang_s10_nb_score",
    "controlled_lang_inv_sa_score",
    "controlled_lang_inv_nb_order_acc",
    "controlled_lang_inv_nb_score",
    "failure_op",
    "failure_details_json",
    "semantic_warnings_json",
)
_OBS_OUTCOME_COLUMNS = tuple(col for col in _OUTCOME_COLUMNS if col != "result_id")
_TEXT_OUTCOME_COLUMNS = frozenset(
    {
        "result_id",
        "experiment_id",
        "graph_fingerprint",
        "trust_label",
        "comparability_label",
        "evaluation_protocol_version",
        "model_source",
        "error_type",
        "stage_at_death",
        "screening_wikitext_status",
        "screening_wikitext_metric_version",
        "screening_wikitext_variant",
        "routing_fast_lane_status",
        "routing_fast_lane_metric_version",
        "controlled_lang_metric_version",
        "failure_op",
        "failure_details_json",
        "semantic_warnings_json",
        "induction_v2_investigation_gap_accuracies_json",
        "induction_v2_investigation_status",
        "induction_v2_investigation_protocol_version",
        "binding_v2_investigation_gap_accuracies_json",
        "binding_v2_investigation_status",
        "binding_v2_investigation_protocol_version",
    }
)
_DERIVED_GRAPH_COLUMNS = (
    "motif_count",
    "non_norm_motif_count",
    "norm_motif_count",
    "norm_dominance",
    "has_attention_motif",
    "has_ssm_motif",
    "has_conv_motif",
    "has_recurrent_motif",
    "has_routing_motif",
    "has_compression_motif",
    "has_effective_positional_mixer",
    "mixer_after_compression",
    "motif_thinness_score",
    "frequency_collapse_risk",
)
_DERIVED_GRAPH_SAMPLE = {
    "motif_count": 0,
    "non_norm_motif_count": 0,
    "norm_motif_count": 0,
    "norm_dominance": 0.0,
    "has_attention_motif": 0,
    "has_ssm_motif": 0,
    "has_conv_motif": 0,
    "has_recurrent_motif": 0,
    "has_routing_motif": 0,
    "has_compression_motif": 0,
    "has_effective_positional_mixer": 0,
    "mixer_after_compression": 0,
    "motif_thinness_score": 0.0,
    "frequency_collapse_risk": 0.0,
}
_CANDIDATE_VALUE_COLUMNS = tuple(
    col for col in ALTERNATIVE_MATH_CANDIDATE_COLUMNS if col != "candidate_name"
)
_NORM_MOTIF_TOKENS = ("norm", "rms", "layernorm")
_ATTENTION_MOTIF_TOKENS = ("attn", "attention")
_SSM_MOTIF_TOKENS = ("ssm", "scan", "state_space", "mamba", "retention")
_CONV_MOTIF_TOKENS = ("conv", "hyena", "window")
_RECURRENT_MOTIF_TOKENS = ("recurrent", "rwkv", "recursion", "delta")
_ROUTING_MOTIF_TOKENS = ("route", "routing", "router", "gate", "gated", "topk")
_COMPRESSION_MOTIF_TOKENS = (
    "compress",
    "compression",
    "merge",
    "bottleneck",
    "sparse",
)


@dataclass(slots=True)
class BuildSummary:
    source_db: str
    output_db: str
    profiling_db: str
    n_program_rows: int
    n_template_catalog_rows: int
    n_slot_catalog_rows: int
    n_op_catalog_rows: int
    n_alternative_candidate_rows: int
    n_op_profile_rows: int
    n_pair_profile_rows: int
    n_triplet_profile_rows: int
    n_eval_metric_rows: int
    n_external_component_prior_rows: int
    n_graph_profile_observation_rows: int
    n_template_observation_rows: int
    n_slot_observation_rows: int
    n_op_observation_rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_db": self.source_db,
            "output_db": self.output_db,
            "profiling_db": self.profiling_db,
            "n_program_rows": self.n_program_rows,
            "n_template_catalog_rows": self.n_template_catalog_rows,
            "n_slot_catalog_rows": self.n_slot_catalog_rows,
            "n_op_catalog_rows": self.n_op_catalog_rows,
            "n_alternative_candidate_rows": self.n_alternative_candidate_rows,
            "n_op_profile_rows": self.n_op_profile_rows,
            "n_pair_profile_rows": self.n_pair_profile_rows,
            "n_triplet_profile_rows": self.n_triplet_profile_rows,
            "n_eval_metric_rows": self.n_eval_metric_rows,
            "n_external_component_prior_rows": self.n_external_component_prior_rows,
            "n_graph_profile_observation_rows": self.n_graph_profile_observation_rows,
            "n_template_observation_rows": self.n_template_observation_rows,
            "n_slot_observation_rows": self.n_slot_observation_rows,
            "n_op_observation_rows": self.n_op_observation_rows,
        }


def _connect_source_readonly(path: str | os.PathLike[str]) -> sqlite3.Connection:
    src = Path(path)
    uri = f"file:{src}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _connect_output(
    path: str | os.PathLike[str], *, replace: bool
) -> sqlite3.Connection:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if replace and dst.exists():
        dst.unlink()
    conn = sqlite3.connect(str(dst), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _json_loads(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (list, dict)):
        return raw
    try:
        loaded = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default
    return loaded if loaded is not None else default


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_type(value: Any) -> str:
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"


def _property_defs(sample: dict[str, Any]) -> str:
    return ",\n    ".join(
        f"{_quote(key)} {_sql_type(value)}" for key, value in sample.items()
    )


def _value_for_db(value: Any) -> Any:
    if isinstance(value, (str, int, float)) or value is None:
        return value
    return _json_dumps(value)


def _as_motif_list(raw: Any) -> list[str]:
    loaded = _json_loads(raw, [])
    if isinstance(loaded, list):
        return [str(item).strip() for item in loaded if str(item).strip()]
    if isinstance(loaded, dict):
        motifs: list[str] = []
        for value in loaded.values():
            if isinstance(value, list):
                motifs.extend(str(item).strip() for item in value if str(item).strip())
            elif str(value).strip():
                motifs.append(str(value).strip())
        return motifs
    return []


def _motif_has_any(motifs: list[str], tokens: tuple[str, ...]) -> bool:
    return any(any(token in motif.lower() for token in tokens) for motif in motifs)


def _derived_graph_payload(row: sqlite3.Row) -> dict[str, Any]:
    row_keys = row.keys()
    motifs = _as_motif_list(row["motifs_json"] if "motifs_json" in row_keys else None)
    motif_count = len(motifs)
    norm_count = sum(
        1
        for motif in motifs
        if any(token in motif.lower() for token in _NORM_MOTIF_TOKENS)
    )
    non_norm_count = max(motif_count - norm_count, 0)
    has_attention = _motif_has_any(motifs, _ATTENTION_MOTIF_TOKENS)
    has_ssm = _motif_has_any(motifs, _SSM_MOTIF_TOKENS)
    has_conv = _motif_has_any(motifs, _CONV_MOTIF_TOKENS)
    has_recurrent = _motif_has_any(motifs, _RECURRENT_MOTIF_TOKENS)
    has_routing = _motif_has_any(motifs, _ROUTING_MOTIF_TOKENS)
    has_compression = _motif_has_any(motifs, _COMPRESSION_MOTIF_TOKENS)
    has_positional = has_attention or has_ssm or has_conv or has_recurrent
    norm_dominance = norm_count / motif_count if motif_count else 0.0
    motif_thinness = 1.0 - min(1.0, non_norm_count / 4.0)
    frequency_risk = (
        (0.45 * motif_thinness)
        + (0.25 if has_compression else 0.0)
        + (0.20 if not has_positional else 0.0)
        + (0.10 * norm_dominance)
        - (0.10 if has_routing else 0.0)
    )
    return {
        "motif_count": motif_count,
        "non_norm_motif_count": non_norm_count,
        "norm_motif_count": norm_count,
        "norm_dominance": round(norm_dominance, 4),
        "has_attention_motif": int(has_attention),
        "has_ssm_motif": int(has_ssm),
        "has_conv_motif": int(has_conv),
        "has_recurrent_motif": int(has_recurrent),
        "has_routing_motif": int(has_routing),
        "has_compression_motif": int(has_compression),
        "has_effective_positional_mixer": int(has_positional and non_norm_count >= 2),
        "mixer_after_compression": int(has_compression and has_positional),
        "motif_thinness_score": round(max(0.0, min(1.0, motif_thinness)), 4),
        "frequency_collapse_risk": round(max(0.0, min(1.0, frequency_risk)), 4),
    }


def _insert_sql(table: str, columns: Iterable[str]) -> str:
    cols = list(columns)
    placeholders = ", ".join("?" for _ in cols)
    quoted_cols = ", ".join(_quote(col) for col in cols)
    return f"INSERT OR REPLACE INTO {_quote(table)} ({quoted_cols}) VALUES ({placeholders})"


_EVAL_METRIC_ROWS: tuple[dict[str, Any], ...] = (
    {
        "metric_name": "stage1_passed",
        "metric_family": "screening",
        "metric_scale": "binary",
        "data_type": "boolean",
        "metric_direction": "higher_is_better",
        "known_bias": "Coarse S1 pass label; depends on screening budget and protocol.",
        "compute_cost_class": "low",
        "reliability_status": "primary_screening_label",
        "primary_use": "gate_target",
        "source_columns_json": ["stage1_passed", "loss_ratio"],
        "active_for_priors": 1,
    },
    {
        "metric_name": "wikitext",
        "metric_family": "language_modeling",
        "metric_scale": "perplexity_and_score",
        "data_type": "float",
        "metric_direction": "lower_ppl_higher_score",
        "known_bias": "Small-model byte/BPE setup favors local language modeling over controlled binding.",
        "compute_cost_class": "medium",
        "reliability_status": "primary_lm_signal",
        "primary_use": "quality_rank",
        "source_columns_json": [
            "wikitext_perplexity",
            "wikitext_score",
            "wikitext_ppl_200",
            "wikitext_ppl_500",
            "wikitext_improvement_ratio",
        ],
        "active_for_priors": 1,
    },
    {
        "metric_name": "screening_wikitext",
        "metric_family": "language_modeling",
        "metric_scale": "fast_perplexity",
        "data_type": "float",
        "metric_direction": "lower_ppl_higher_score",
        "known_bias": "Fast lane can be noisy and should not hard-gate without corroboration.",
        "compute_cost_class": "low",
        "reliability_status": "triage_signal",
        "primary_use": "early_screening",
        "source_columns_json": [
            "screening_wikitext_status",
            "screening_wikitext_metric_version",
            "screening_wikitext_variant",
            "routing_fast_lane_score",
            "routing_fast_lane_ppl_improvement",
        ],
        "active_for_priors": 1,
    },
    {
        "metric_name": "tinystories",
        "metric_family": "language_modeling",
        "metric_scale": "perplexity_and_score",
        "data_type": "float",
        "metric_direction": "lower_ppl_higher_score",
        "known_bias": "Narrative/simple syntax signal; not a substitute for binding probes.",
        "compute_cost_class": "medium",
        "reliability_status": "secondary_lm_signal",
        "primary_use": "quality_rank",
        "source_columns_json": ["tinystories_perplexity", "tinystories_score"],
        "active_for_priors": 1,
    },
    {
        "metric_name": "hellaswag",
        "metric_family": "commonsense",
        "metric_scale": "accuracy",
        "data_type": "float",
        "metric_direction": "higher_is_better",
        "known_bias": "Small samples and tokenizer mode can dominate early signals.",
        "compute_cost_class": "high",
        "reliability_status": "expensive_secondary_signal",
        "primary_use": "generalization_check",
        "source_columns_json": [
            "hellaswag_acc",
            "hellaswag_status",
            "hellaswag_metric_version",
        ],
        "active_for_priors": 1,
    },
    {
        "metric_name": "blimp",
        "metric_family": "syntax",
        "metric_scale": "accuracy",
        "data_type": "float",
        "metric_direction": "higher_is_better",
        "known_bias": "Can reward narrow syntactic preferences and cached subset effects.",
        "compute_cost_class": "high",
        "reliability_status": "expensive_secondary_signal",
        "primary_use": "syntax_check",
        "source_columns_json": [
            "blimp_overall_accuracy",
            "blimp_status",
            "blimp_n_subtasks",
        ],
        "active_for_priors": 1,
    },
    {
        "metric_name": "induction_binding_ar",
        "metric_family": "mechanistic_probe",
        "metric_scale": "auc_accuracy",
        "data_type": "float",
        "metric_direction": "higher_is_better",
        "known_bias": "Synthetic probe sensitivity depends on train steps and protocol version.",
        "compute_cost_class": "medium",
        "reliability_status": "capability_probe",
        "primary_use": "mechanistic_capability",
        "source_columns_json": [
            "induction_auc",
            "induction_v2_investigation_auc",
            "binding_auc",
            "binding_auc_curriculum",
            "binding_v2_investigation_auc",
            "ar_auc",
        ],
        "active_for_priors": 1,
    },
    {
        "metric_name": "controlled_language_nanobind",
        "metric_family": "controlled_language",
        "metric_scale": "score_accuracy_failure",
        "data_type": "mixed",
        "metric_direction": "higher_score_lower_failure",
        "known_bias": "Targeted NanoBind failures can disagree with broad perplexity quality.",
        "compute_cost_class": "medium",
        "reliability_status": "primary_failure_signal",
        "primary_use": "routing_compression_failure_analysis",
        "source_columns_json": [
            "controlled_lang_s05_sa_score",
            "controlled_lang_s05_nb_order_acc",
            "controlled_lang_s05_nb_score",
            "controlled_lang_s10_sa_score",
            "controlled_lang_inv_sa_score",
            "failure_op",
            "failure_details_json",
        ],
        "active_for_priors": 1,
    },
    {
        "metric_name": "routing_fast_lane",
        "metric_family": "routing",
        "metric_scale": "score_improvement_slope",
        "data_type": "mixed",
        "metric_direction": "higher_is_better",
        "known_bias": "Only applies to routed graphs and can confound router quality with base LM quality.",
        "compute_cost_class": "low",
        "reliability_status": "triage_signal",
        "primary_use": "routing_efficiency",
        "source_columns_json": [
            "routing_fast_lane_applied",
            "routing_fast_lane_score",
            "routing_fast_lane_ppl_improvement",
            "routing_fast_lane_slope",
            "routing_fast_lane_slope_consistent",
        ],
        "active_for_priors": 1,
    },
    {
        "metric_name": "permutation_composition",
        "metric_family": "synthetic_composition",
        "metric_scale": "accuracy_score",
        "data_type": "float",
        "metric_direction": "higher_is_better",
        "known_bias": "Synthetic algorithmic signal; use with language metrics before design promotion.",
        "compute_cost_class": "medium",
        "reliability_status": "candidate_probe",
        "primary_use": "composition_generalization",
        "source_columns_json": [
            "permutation_composition_score",
            "permutation_composition_extrapolation_acc",
            "permutation_composition_metric_version",
        ],
        "active_for_priors": 0,
    },
    {
        "metric_name": "composite_score",
        "metric_family": "aggregate",
        "metric_scale": "score",
        "data_type": "float",
        "metric_direction": "higher_is_better",
        "known_bias": "Depends on leaderboard weighting and available probe coverage.",
        "compute_cost_class": "none",
        "reliability_status": "derived_summary",
        "primary_use": "ranking",
        "source_columns_json": ["composite_score", "cross_task_score"],
        "active_for_priors": 1,
    },
)

_EXTERNAL_COMPONENT_PRIOR_ROWS: tuple[dict[str, Any], ...] = (
    {
        "external_family": "ssm_mamba_selective_scan",
        "mapped_ops_json": [
            "selective_scan",
            "state_space",
            "gated_delta",
            "conv1d_seq",
        ],
        "mapped_templates_json": ["state_space", "scan", "mamba", "delta"],
        "expected_strength": "linear-time sequence mixing with long-context memory",
        "expected_risk": "gradient explosion and unstable gates without normalization",
        "hardware_note": "scan kernels benefit from fused/native implementations",
        "tags_json": ["math", "routing", "efficient_sequence", "state_space"],
        "confidence": 0.85,
        "source_ref": "Mamba/selective SSM family",
    },
    {
        "external_family": "rwkv_time_mixing",
        "mapped_ops_json": ["rwkv_time_mixing", "rwkv_channel"],
        "mapped_templates_json": ["rwkv", "recurrent", "time_mixing"],
        "expected_strength": "recurrent attention-like mixing with small inference state",
        "expected_risk": "channel/time balance and initialization sensitivity",
        "hardware_note": "good streaming profile; backward can be specialized",
        "tags_json": ["routing", "efficient_sequence", "recurrent"],
        "confidence": 0.75,
        "source_ref": "RWKV architecture family",
    },
    {
        "external_family": "long_filter_hyena_conv",
        "mapped_ops_json": [
            "conv1d_seq",
            "conv_only",
            "chebyshev_spectral_mix",
            "spectral_filter",
        ],
        "mapped_templates_json": ["conv", "hyena", "spectral", "frequency"],
        "expected_strength": "cheap local/global filtering and frequency-basis compression",
        "expected_risk": "over-smoothing and frequency collapse when paired with token merge",
        "hardware_note": "convolution/spectral kernels should be benchmarked by sequence length",
        "tags_json": ["compression", "math", "frequency", "efficient_sequence"],
        "confidence": 0.70,
        "source_ref": "Hyena/long convolution family",
    },
    {
        "external_family": "attention_gqa_mqa_rope",
        "mapped_ops_json": [
            "softmax_attention",
            "linear_attention",
            "diff_attention",
            "rope_rotate",
            "local_window_attn",
        ],
        "mapped_templates_json": ["attention", "local_window", "rope"],
        "expected_strength": "robust positional/content mixer and strong baseline for small models",
        "expected_risk": "quadratic cost or approximation drift in compressed variants",
        "hardware_note": "attention kernels are hardware-sensitive; prefer fused paths",
        "tags_json": ["routing", "math", "positional", "baseline"],
        "confidence": 0.90,
        "source_ref": "Transformer attention variants",
    },
    {
        "external_family": "moe_topk_sparse_routing",
        "mapped_ops_json": [
            "moe_topk",
            "route_topk",
            "topk_gate",
            "relu_gated_moe",
            "sparse_bottleneck_moe",
        ],
        "mapped_templates_json": ["moe", "topk", "router", "sparse"],
        "expected_strength": "conditional capacity and sparse compute",
        "expected_risk": "routing collapse, nondifferentiable gates, load imbalance",
        "hardware_note": "small models may lose efficiency to dispatch overhead",
        "tags_json": ["routing", "sparse_learning", "compression"],
        "confidence": 0.82,
        "source_ref": "Sparse MoE/top-k routing family",
    },
    {
        "external_family": "token_merge_compression",
        "mapped_ops_json": [
            "token_merge",
            "adjacent_token_merge",
            "latent_attention_compressor",
            "low_rank_proj",
            "bottleneck_proj",
        ],
        "mapped_templates_json": ["merge", "compress", "bottleneck", "latent"],
        "expected_strength": "sequence/representation compression for cheaper downstream mixing",
        "expected_risk": "NanoBind and frequency collapse when no positional mixer follows",
        "hardware_note": "benefit depends on actual token reduction before expensive ops",
        "tags_json": ["compression", "routing", "efficient_sequence"],
        "confidence": 0.88,
        "source_ref": "Token merging and low-rank compression families",
    },
    {
        "external_family": "structured_sparse_linear",
        "mapped_ops_json": [
            "block_sparse_linear",
            "nm_sparse_linear",
            "semi_structured_2_4_linear",
            "ternary_projection",
        ],
        "mapped_templates_json": ["sparse_linear", "2_4", "ternary", "block_sparse"],
        "expected_strength": "parameter and compute reduction with hardware-friendly sparsity",
        "expected_risk": "capacity loss and brittle training at very small widths",
        "hardware_note": "N:M sparsity only helps on supported kernels/hardware",
        "tags_json": ["compression", "sparse_learning", "hardware"],
        "confidence": 0.78,
        "source_ref": "Structured sparsity and quantized linear layers",
    },
    {
        "external_family": "normalization_stability",
        "mapped_ops_json": ["rmsnorm", "layernorm", "norm_last"],
        "mapped_templates_json": ["norm", "rms", "layernorm", "stabilize"],
        "expected_strength": "stabilizes deep/risky compositions and improves trainability",
        "expected_risk": "norm dominance can thin useful motifs if overused",
        "hardware_note": "cheap relative to mixing; fusion can remove overhead",
        "tags_json": ["stability", "math"],
        "confidence": 0.92,
        "source_ref": "RMSNorm/LayerNorm stabilization",
    },
    {
        "external_family": "alternative_math_spaces",
        "mapped_ops_json": [
            "tropical_attention",
            "tropical_router",
            "tropical_moe",
            "hyperbolic_norm",
            "hyp_linear",
            "clifford_attention",
            "geometric_product",
            "padic_gate",
        ],
        "mapped_templates_json": ["tropical", "hyperbolic", "clifford", "padic"],
        "expected_strength": "new inductive biases and routing/composition math spaces",
        "expected_risk": "numerical instability and weak kernel maturity",
        "hardware_note": "profile before promotion; many ops need native kernels",
        "tags_json": ["math", "routing", "new_mathspace"],
        "confidence": 0.68,
        "source_ref": "Alternative algebraic/geometric model components",
    },
)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(
        row[1] == column for row in conn.execute(f"PRAGMA table_info({_quote(table)})")
    )


def _ensure_table_columns(
    conn: sqlite3.Connection,
    table: str,
    columns: dict[str, str],
) -> None:
    existing = {
        str(row[1]) for row in conn.execute(f"PRAGMA table_info({_quote(table)})")
    }
    for column, sql_type in columns.items():
        if column not in existing:
            conn.execute(
                f"ALTER TABLE {_quote(table)} ADD COLUMN {_quote(column)} {sql_type}"
            )


def _select_expr(conn: sqlite3.Connection, table: str, column: str) -> str:
    return (
        f"pr.{_quote(column)}"
        if _column_exists(conn, table, column)
        else f"NULL AS {_quote(column)}"
    )


def _fetch_program_rows(src: sqlite3.Connection) -> list[sqlite3.Row]:
    gf_join = (
        "LEFT JOIN program_graph_features gf ON gf.result_id = pr.result_id"
        if _table_exists(src, "program_graph_features")
        else ""
    )
    lb_join = (
        "LEFT JOIN leaderboard l ON l.result_id = pr.result_id"
        if _table_exists(src, "leaderboard")
        else ""
    )
    graph_feature_exprs = (
        "gf.template_name AS gf_template_name, "
        "gf.templates_json AS templates_json, "
        "gf.motifs_json AS motifs_json, "
        "gf.slot_usage_json AS slot_usage_json"
        if gf_join
        else "NULL AS gf_template_name, NULL AS templates_json, NULL AS motifs_json, NULL AS slot_usage_json"
    )
    composite_expr = (
        "l.composite_score AS composite_score"
        if lb_join and _column_exists(src, "leaderboard", "composite_score")
        else "NULL AS composite_score"
    )
    outcome_exprs = [
        _select_expr(src, "program_results", col)
        for col in _OUTCOME_COLUMNS
        if col != "composite_score"
    ]
    sql = f"""
        SELECT
            {", ".join(outcome_exprs)},
            {composite_expr},
            pr.graph_json AS graph_json,
            {graph_feature_exprs}
        FROM program_results pr
        {gf_join}
        {lb_join}
        WHERE COALESCE(pr.graph_json, '') NOT IN ('', '{{}}')
    """
    return src.execute(sql).fetchall()


def _fetch_program_op_rows(src: sqlite3.Connection) -> list[sqlite3.Row]:
    if not _table_exists(src, "program_graph_ops"):
        return []
    gf_join = (
        "LEFT JOIN program_graph_features gf ON gf.result_id = pr.result_id"
        if _table_exists(src, "program_graph_features")
        else ""
    )
    lb_join = (
        "LEFT JOIN leaderboard l ON l.result_id = pr.result_id"
        if _table_exists(src, "leaderboard")
        else ""
    )
    graph_feature_exprs = (
        "gf.motifs_json AS motifs_json" if gf_join else "NULL AS motifs_json"
    )
    composite_expr = (
        "l.composite_score AS composite_score"
        if lb_join and _column_exists(src, "leaderboard", "composite_score")
        else "NULL AS composite_score"
    )
    outcome_exprs = [
        _select_expr(src, "program_results", col)
        for col in _OUTCOME_COLUMNS
        if col != "composite_score"
    ]
    sql = f"""
        SELECT
            go.op_name,
            {", ".join(outcome_exprs)},
            {composite_expr},
            {graph_feature_exprs}
        FROM program_graph_ops go
        JOIN program_results pr ON pr.result_id = go.result_id
        {gf_join}
        {lb_join}
    """
    return src.execute(sql).fetchall()


def _fetch_op_stats_rows(src: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    if not _table_exists(src, "op_stats"):
        return {}
    rows = src.execute("SELECT * FROM op_stats").fetchall()
    return {str(row["op_name"]): dict(row) for row in rows if row["op_name"]}


def _connect_optional_readonly(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _column_defs(
    columns: Iterable[tuple[str, str]],
    *,
    extra_columns: Iterable[tuple[str, str]] = (),
) -> str:
    return ",\n    ".join(
        f"{_quote(name)} {sql_type}"
        for name, sql_type in (*tuple(columns), *tuple(extra_columns))
    )


def _copy_profile_table(
    *,
    src: sqlite3.Connection | None,
    dst: sqlite3.Connection,
    source_path: Path,
    source_mtime: float | None,
    source_table: str,
    target_table: str,
    columns: tuple[tuple[str, str], ...],
) -> int:
    if src is None or not _table_exists(src, source_table):
        return 0
    col_names = [name for name, _sql_type_name in columns]
    rows = src.execute(
        f"SELECT {', '.join(_quote(name) for name in col_names)} FROM {_quote(source_table)}"
    ).fetchall()
    insert_cols = [*col_names, "profile_source_db_path", "profile_source_mtime"]
    insert_sql = _insert_sql(target_table, insert_cols)
    payload = [
        (
            *(row[name] for name in col_names),
            str(source_path),
            source_mtime,
        )
        for row in rows
    ]
    if payload:
        dst.executemany(insert_sql, payload)
    return len(payload)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _num(value: Any, default: float = 0.0) -> float:
    parsed = _float_or_none(value)
    return default if parsed is None else parsed


def _int_flag(value: Any) -> int:
    return int(bool(_num(value, 0.0)))


def _load_profile_maps(
    src: sqlite3.Connection | None,
) -> tuple[
    dict[str, sqlite3.Row],
    dict[tuple[str, str], sqlite3.Row],
    dict[tuple[str, str, str], sqlite3.Row],
]:
    if src is None:
        return {}, {}, {}
    op_profiles: dict[str, sqlite3.Row] = {}
    pair_profiles: dict[tuple[str, str], sqlite3.Row] = {}
    triplet_profiles: dict[tuple[str, str, str], sqlite3.Row] = {}
    if _table_exists(src, "op_profiles"):
        for row in src.execute(
            """
            SELECT * FROM op_profiles
            ORDER BY CASE WHEN error IS NULL THEN 0 ELSE 1 END, registry ASC
            """
        ):
            op_profiles.setdefault(str(row["op_name"]), row)
    if _table_exists(src, "pair_profiles"):
        for row in src.execute(
            """
            SELECT * FROM pair_profiles
            WHERE composition = 'sequential'
            ORDER BY CASE WHEN error IS NULL THEN 0 ELSE 1 END
            """
        ):
            pair_profiles.setdefault((str(row["op_a"]), str(row["op_b"])), row)
    if _table_exists(src, "triplet_profiles"):
        for row in src.execute(
            "SELECT * FROM triplet_profiles ORDER BY CASE WHEN error IS NULL THEN 0 ELSE 1 END"
        ):
            triplet_profiles.setdefault(
                (str(row["op_a"]), str(row["op_b"]), str(row["op_c"])),
                row,
            )
    return op_profiles, pair_profiles, triplet_profiles


def _graph_nodes_edges(graph_json: Any) -> tuple[list[str], list[tuple[str, str]]]:
    graph = _json_loads(graph_json, {})
    if not isinstance(graph, dict):
        return [], []
    raw_nodes = graph.get("nodes")
    if isinstance(raw_nodes, dict):
        node_items = []
        for key, value in raw_nodes.items():
            if isinstance(value, dict):
                node_id = value.get("id", key)
                node_items.append((str(node_id), value))
    elif isinstance(raw_nodes, list):
        node_items = [
            (str(node.get("id", index)), node)
            for index, node in enumerate(raw_nodes)
            if isinstance(node, dict)
        ]
    else:
        return [], []

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        raw_id = item[0]
        try:
            return (0, f"{int(raw_id):012d}")
        except (TypeError, ValueError):
            return (1, raw_id)

    node_items.sort(key=sort_key)
    id_to_op = {
        node_id: str(node.get("op_name") or node.get("component_type") or "").strip()
        for node_id, node in node_items
    }
    ops = [op for _node_id, op in id_to_op.items() if op]
    edges: list[tuple[str, str]] = []
    for node_id, node in node_items:
        child_op = id_to_op.get(node_id, "")
        if not child_op:
            continue
        raw_inputs = node.get("input_ids") or node.get("inputs") or []
        if isinstance(raw_inputs, dict):
            raw_inputs = raw_inputs.values()
        for raw_parent in raw_inputs:
            parent_id = str(raw_parent)
            parent_op = id_to_op.get(parent_id, "")
            if parent_op:
                edges.append((parent_op, child_op))
    return ops, edges


def _graph_profile_payload(
    row: sqlite3.Row,
    op_profiles: dict[str, sqlite3.Row],
    pair_profiles: dict[tuple[str, str], sqlite3.Row],
    triplet_profiles: dict[tuple[str, str, str], sqlite3.Row],
) -> dict[str, Any]:
    ops, edges = _graph_nodes_edges(row["graph_json"])
    known = [op_profiles[op] for op in ops if op in op_profiles]
    known_count = len(known)
    op_count = len(ops)
    missing_count = max(op_count - known_count, 0)
    coverage = known_count / op_count if op_count else 0.0

    fwd_values = [_num(profile["forward_time_us"]) for profile in known]
    bwd_values = [_num(profile["backward_time_us"]) for profile in known]
    max_fwd = max(fwd_values) if fwd_values else 0.0
    slowest = ""
    if known:
        slowest_profile = max(
            known, key=lambda profile: _num(profile["forward_time_us"])
        )
        slowest = str(slowest_profile["op_name"] or "")

    pair_rows = [pair_profiles[edge] for edge in edges if edge in pair_profiles]
    triplet_keys: list[tuple[str, str, str]] = []
    by_parent: dict[str, list[str]] = {}
    for parent, child in edges:
        by_parent.setdefault(parent, []).append(child)
    for left, middle in edges:
        for right in by_parent.get(middle, []):
            triplet_keys.append((left, middle, right))
    triplet_rows = [
        triplet_profiles[key] for key in triplet_keys if key in triplet_profiles
    ]

    def pair_unstable(profile: sqlite3.Row) -> bool:
        return bool(
            profile["error"]
            or _int_flag(profile["output_has_nan"])
            or _int_flag(profile["grad_has_nan"])
            or _int_flag(profile["grad_vanishing"])
        )

    def triplet_unstable(profile: sqlite3.Row) -> bool:
        return bool(
            profile["error"]
            or _int_flag(profile["output_has_nan"])
            or _int_flag(profile["grad_has_nan"])
            or _int_flag(profile["grad_vanishing"])
            or not _int_flag(profile["triplet_stable"])
        )

    return {
        "result_id": row["result_id"],
        "graph_fingerprint": row["graph_fingerprint"],
        "graph_profile_op_count": op_count,
        "profile_known_op_count": known_count,
        "profile_missing_op_count": missing_count,
        "profile_coverage_rate": round(coverage, 6),
        "profile_total_forward_time_us": round(sum(fwd_values), 6),
        "profile_total_backward_time_us": round(sum(bwd_values), 6),
        "profile_mean_forward_time_us": round(sum(fwd_values) / known_count, 6)
        if known_count
        else 0.0,
        "profile_max_forward_time_us": round(max_fwd, 6),
        "profile_slowest_op_name": slowest,
        "profile_total_peak_memory_bytes": int(
            sum(_num(profile["peak_memory_bytes"]) for profile in known)
        ),
        "profile_total_flops_estimate": int(
            sum(_num(profile["flops_estimate"]) for profile in known)
        ),
        "profile_max_lipschitz_estimate": round(
            max(
                (_num(profile["lipschitz_estimate"]) for profile in known), default=0.0
            ),
            6,
        ),
        "profile_max_jacobian_condition_num": round(
            max(
                (_num(profile["jacobian_condition_num"]) for profile in known),
                default=0.0,
            ),
            6,
        ),
        "profile_grad_vanishing_op_count": sum(
            _int_flag(profile["grad_vanishing"]) for profile in known
        ),
        "profile_grad_exploding_op_count": sum(
            _int_flag(profile["grad_exploding"]) for profile in known
        ),
        "profile_output_nan_op_count": sum(
            _int_flag(profile["output_has_nan"]) for profile in known
        ),
        "profile_pair_count": len(pair_rows),
        "profile_pair_unstable_count": sum(
            pair_unstable(profile) for profile in pair_rows
        ),
        "profile_pair_grad_exploding_count": sum(
            _int_flag(profile["grad_exploding"]) for profile in pair_rows
        ),
        "profile_pair_max_lipschitz_estimate": round(
            max(
                (_num(profile["lipschitz_estimate"]) for profile in pair_rows),
                default=0.0,
            ),
            6,
        ),
        "profile_triplet_count": len(triplet_rows),
        "profile_triplet_unstable_count": sum(
            triplet_unstable(profile) for profile in triplet_rows
        ),
        "profile_triplet_divergent_count": sum(
            _int_flag(profile["diverges_from_pair_prediction"])
            for profile in triplet_rows
        ),
        "profile_triplet_grad_vanishing_count": sum(
            _int_flag(profile["grad_vanishing"]) for profile in triplet_rows
        ),
    }


def _metadata_from_graph_json(
    graph_json: Any,
) -> tuple[list[str], list[dict[str, Any]]]:
    graph = _json_loads(graph_json, {})
    metadata = graph.get("metadata") if isinstance(graph, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    templates = metadata.get("templates_used") or []
    slots = metadata.get("template_slot_usage") or []
    return (
        [str(item) for item in templates if item is not None],
        [item for item in slots if isinstance(item, dict)],
    )


def _row_templates(row: sqlite3.Row) -> list[str]:
    templates = _json_loads(row["templates_json"], [])
    if not isinstance(templates, list) or not templates:
        templates, _slots = _metadata_from_graph_json(row["graph_json"])
    out = [str(item).strip() for item in templates if str(item).strip()]
    if not out and row["gf_template_name"]:
        out = [str(row["gf_template_name"]).strip()]
    return out


def _row_slots(row: sqlite3.Row) -> list[dict[str, Any]]:
    slots = _json_loads(row["slot_usage_json"], [])
    if not isinstance(slots, list) or not slots:
        _templates, slots = _metadata_from_graph_json(row["graph_json"])
    return [slot for slot in slots if isinstance(slot, dict)]


def _infer_active_template_names() -> set[str]:
    try:
        from research.synthesis.templates import TEMPLATES
    except Exception:
        return set()
    return {str(name) for name in TEMPLATES}


def _collect_slot_counts(
    rows: list[sqlite3.Row],
    active_templates: set[str],
) -> dict[str, int]:
    counts = {name: 0 for name in active_templates}
    for row in rows:
        for template in _row_templates(row):
            counts.setdefault(template, 0)
        for slot in _row_slots(row):
            template_name = str(slot.get("template_name") or "").strip()
            if not template_name:
                slot_key = canonical_slot_key(str(slot.get("slot_key") or ""))
                template_name = slot_key.split(".", 1)[0] if slot_key else "unknown"
            slot_index = int(slot.get("slot_index") or 0)
            counts[template_name] = max(counts.get(template_name, 0), slot_index + 1)
    return counts


def _create_schema(dst: sqlite3.Connection) -> None:
    template_sample = template_descriptive_properties("sample_router", slot_count=2)
    slot_sample = slot_descriptive_properties(
        "sample.slot0",
        template_name="sample",
        slot_index=0,
        slot_count=1,
        slot_classes=["role:router", "attention"],
    )
    op_sample = op_descriptive_properties(
        "sample_lambda_map",
        metadata={
            "category": "functional",
            "n_inputs": 1,
            "shape_rule": "identity",
            "description": "Lambda calculus style map/compose candidate",
            "has_params": False,
            "preserves_gradient": True,
            "numerically_risky": False,
            "binding_range_class": "none",
        },
    )
    candidate_sample = alternative_math_candidate_properties("lambda_calculus")
    template_defs = _property_defs(template_sample)
    slot_defs = _property_defs(slot_sample)
    op_defs = _property_defs(op_sample)
    candidate_defs = _property_defs(
        {key: candidate_sample[key] for key in _CANDIDATE_VALUE_COLUMNS}
    )
    outcome_defs = ",\n    ".join(
        f"{_quote(col)} {'TEXT' if col in _TEXT_OUTCOME_COLUMNS else 'REAL'}"
        for col in _OBS_OUTCOME_COLUMNS
    )
    derived_defs = _property_defs(_DERIVED_GRAPH_SAMPLE)
    dst.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS meta_builds (
            build_id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            source_db_path TEXT NOT NULL,
            source_db_mtime REAL,
            profile_source_db_path TEXT,
            profile_source_db_mtime REAL,
            source_program_count INTEGER NOT NULL,
            property_version TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS template_property_catalog (
            template_name TEXT PRIMARY KEY,
            slot_count INTEGER NOT NULL,
            observed_count INTEGER NOT NULL DEFAULT 0,
            property_version TEXT NOT NULL,
            {template_defs}
        );

        CREATE TABLE IF NOT EXISTS slot_property_catalog (
            slot_key TEXT PRIMARY KEY,
            template_name TEXT NOT NULL,
            slot_index INTEGER NOT NULL,
            slot_classes_json TEXT NOT NULL,
            observed_count INTEGER NOT NULL DEFAULT 0,
            property_version TEXT NOT NULL,
            {slot_defs}
        );

        CREATE TABLE IF NOT EXISTS op_property_catalog (
            op_name TEXT PRIMARY KEY,
            observed_count INTEGER NOT NULL DEFAULT 0,
            eval_count INTEGER NOT NULL DEFAULT 0,
            s1_pass_count INTEGER NOT NULL DEFAULT 0,
            mean_loss REAL,
            min_loss REAL,
            std_loss REAL,
            mean_novelty REAL,
            math_space_rate REAL,
            property_version TEXT NOT NULL,
            {op_defs}
        );

        CREATE TABLE IF NOT EXISTS op_observations (
            result_id TEXT NOT NULL,
            op_name TEXT NOT NULL,
            {outcome_defs},
            {derived_defs},
            property_version TEXT NOT NULL,
            {op_defs},
            PRIMARY KEY (result_id, op_name)
        );

        CREATE TABLE IF NOT EXISTS alternative_math_candidate_catalog (
            candidate_name TEXT PRIMARY KEY,
            property_version TEXT NOT NULL,
            {candidate_defs}
        );

        CREATE TABLE IF NOT EXISTS op_profile_catalog (
            {_column_defs(_OP_PROFILE_COLUMNS, extra_columns=_PROFILE_PROVENANCE_COLUMNS)},
            PRIMARY KEY (op_name, registry)
        );

        CREATE TABLE IF NOT EXISTS op_pair_profile_catalog (
            {_column_defs(_PAIR_PROFILE_COLUMNS, extra_columns=_PROFILE_PROVENANCE_COLUMNS)},
            PRIMARY KEY (op_a, op_b, composition)
        );

        CREATE TABLE IF NOT EXISTS op_triplet_profile_catalog (
            {_column_defs(_TRIPLET_PROFILE_COLUMNS, extra_columns=_PROFILE_PROVENANCE_COLUMNS)},
            PRIMARY KEY (op_a, op_b, op_c)
        );

        CREATE TABLE IF NOT EXISTS eval_metric_catalog (
            metric_name TEXT PRIMARY KEY,
            metric_family TEXT NOT NULL,
            metric_scale TEXT NOT NULL,
            data_type TEXT NOT NULL,
            metric_direction TEXT NOT NULL,
            known_bias TEXT NOT NULL,
            compute_cost_class TEXT NOT NULL,
            reliability_status TEXT NOT NULL,
            primary_use TEXT NOT NULL,
            source_columns_json TEXT NOT NULL,
            active_for_priors INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS external_component_prior_catalog (
            external_family TEXT PRIMARY KEY,
            mapped_ops_json TEXT NOT NULL,
            mapped_templates_json TEXT NOT NULL,
            expected_strength TEXT NOT NULL,
            expected_risk TEXT NOT NULL,
            hardware_note TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            source_ref TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS graph_profile_observations (
            result_id TEXT PRIMARY KEY,
            graph_fingerprint TEXT,
            graph_profile_op_count INTEGER NOT NULL,
            profile_known_op_count INTEGER NOT NULL,
            profile_missing_op_count INTEGER NOT NULL,
            profile_coverage_rate REAL NOT NULL,
            profile_total_forward_time_us REAL NOT NULL,
            profile_total_backward_time_us REAL NOT NULL,
            profile_mean_forward_time_us REAL NOT NULL,
            profile_max_forward_time_us REAL NOT NULL,
            profile_slowest_op_name TEXT NOT NULL,
            profile_total_peak_memory_bytes INTEGER NOT NULL,
            profile_total_flops_estimate INTEGER NOT NULL,
            profile_max_lipschitz_estimate REAL NOT NULL,
            profile_max_jacobian_condition_num REAL NOT NULL,
            profile_grad_vanishing_op_count INTEGER NOT NULL,
            profile_grad_exploding_op_count INTEGER NOT NULL,
            profile_output_nan_op_count INTEGER NOT NULL,
            profile_pair_count INTEGER NOT NULL,
            profile_pair_unstable_count INTEGER NOT NULL,
            profile_pair_grad_exploding_count INTEGER NOT NULL,
            profile_pair_max_lipschitz_estimate REAL NOT NULL,
            profile_triplet_count INTEGER NOT NULL,
            profile_triplet_unstable_count INTEGER NOT NULL,
            profile_triplet_divergent_count INTEGER NOT NULL,
            profile_triplet_grad_vanishing_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS template_observations (
            result_id TEXT NOT NULL,
            template_name TEXT NOT NULL,
            {outcome_defs},
            {derived_defs},
            slot_count INTEGER NOT NULL,
            property_version TEXT NOT NULL,
            {template_defs},
            PRIMARY KEY (result_id, template_name)
        );

        CREATE TABLE IF NOT EXISTS slot_observations (
            result_id TEXT NOT NULL,
            slot_key TEXT NOT NULL,
            template_name TEXT NOT NULL,
            slot_index INTEGER NOT NULL,
            selected_motif TEXT,
            selected_motif_class TEXT,
            wildcard INTEGER NOT NULL DEFAULT 0,
            slot_classes_json TEXT NOT NULL,
            {outcome_defs},
            {derived_defs},
            slot_count INTEGER NOT NULL,
            property_version TEXT NOT NULL,
            {template_defs},
            {slot_defs},
            PRIMARY KEY (result_id, slot_key)
        );

        CREATE INDEX IF NOT EXISTS idx_template_observations_template ON template_observations(template_name);
        CREATE INDEX IF NOT EXISTS idx_template_observations_fp ON template_observations(graph_fingerprint);
        CREATE INDEX IF NOT EXISTS idx_slot_observations_slot ON slot_observations(slot_key);
        CREATE INDEX IF NOT EXISTS idx_slot_observations_template ON slot_observations(template_name);
        CREATE INDEX IF NOT EXISTS idx_slot_observations_fp ON slot_observations(graph_fingerprint);
        CREATE INDEX IF NOT EXISTS idx_op_observations_op ON op_observations(op_name);
        CREATE INDEX IF NOT EXISTS idx_op_observations_fp ON op_observations(graph_fingerprint);
        CREATE INDEX IF NOT EXISTS idx_graph_profile_observations_fp ON graph_profile_observations(graph_fingerprint);
        CREATE INDEX IF NOT EXISTS idx_op_profile_catalog_op ON op_profile_catalog(op_name);
        CREATE INDEX IF NOT EXISTS idx_op_pair_profile_catalog_ops ON op_pair_profile_catalog(op_a, op_b);
        CREATE INDEX IF NOT EXISTS idx_op_triplet_profile_catalog_ops ON op_triplet_profile_catalog(op_a, op_b, op_c);
        """
    )
    _ensure_table_columns(
        dst,
        "meta_builds",
        {
            "profile_source_db_path": "TEXT",
            "profile_source_db_mtime": "REAL",
        },
    )
    obs_column_types = {
        col: ("TEXT" if col in _TEXT_OUTCOME_COLUMNS else "REAL")
        for col in _OBS_OUTCOME_COLUMNS
    }
    obs_column_types.update(
        {col: _sql_type(value) for col, value in _DERIVED_GRAPH_SAMPLE.items()}
    )
    for table in ("op_observations", "template_observations", "slot_observations"):
        _ensure_table_columns(dst, table, obs_column_types)


def _reset_materialized_tables(dst: sqlite3.Connection) -> None:
    for table in (
        "template_property_catalog",
        "slot_property_catalog",
        "template_observations",
        "slot_observations",
        "op_property_catalog",
        "op_observations",
        "alternative_math_candidate_catalog",
        "op_profile_catalog",
        "op_pair_profile_catalog",
        "op_triplet_profile_catalog",
        "eval_metric_catalog",
        "external_component_prior_catalog",
        "graph_profile_observations",
    ):
        dst.execute(f"DELETE FROM {_quote(table)}")


def _outcome_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {col: row[col] for col in _OUTCOME_COLUMNS}


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()
    return int(row[0] if row else 0)


def _primitive_metadata_by_name() -> dict[str, dict[str, Any]]:
    try:
        from research.synthesis.primitives import PRIMITIVE_REGISTRY
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, op in PRIMITIVE_REGISTRY.items():
        category = getattr(op, "category", None)
        algebraic_type = getattr(op, "algebraic_type", None)
        out[str(name)] = {
            "category": category.value if hasattr(category, "value") else str(category),
            "n_inputs": int(getattr(op, "n_inputs", 0) or 0),
            "shape_rule": str(getattr(op, "shape_rule", "") or ""),
            "has_params": bool(getattr(op, "has_params", False)),
            "param_formula": str(getattr(op, "param_formula", "0") or "0"),
            "preserves_gradient": bool(getattr(op, "preserves_gradient", True)),
            "numerically_risky": bool(getattr(op, "numerically_risky", False)),
            "description": str(getattr(op, "description", "") or ""),
            "standalone": bool(getattr(op, "standalone", True)),
            "byte_safe": bool(getattr(op, "byte_safe", True)),
            "min_layer_depth": int(getattr(op, "min_layer_depth", 0) or 0),
            "algebraic_space": str(getattr(op, "algebraic_space", "") or ""),
            "binding_range_class": str(
                getattr(op, "binding_range_class", "none") or "none"
            ),
            "algebraic_input_constraint": str(
                getattr(algebraic_type, "input_constraint", "") or ""
            ),
            "algebraic_output_guarantee": str(
                getattr(algebraic_type, "output_guarantee", "") or ""
            ),
        }
    return out


def build_meta_analysis_db(
    *,
    source_db: str | os.PathLike[str] = LAB_NOTEBOOK_DB,
    output_db: str | os.PathLike[str] = DEFAULT_META_ANALYSIS_DB,
    profiling_db: str | os.PathLike[str] = DEFAULT_PROFILING_DB,
    replace: bool = True,
) -> BuildSummary:
    """Materialize a separate template/slot meta-analysis SQLite database."""

    source_path = Path(source_db)
    profile_path = Path(profiling_db)
    src = _connect_source_readonly(source_path)
    try:
        rows = _fetch_program_rows(src)
        op_rows = _fetch_program_op_rows(src)
        op_stats = _fetch_op_stats_rows(src)
    finally:
        src.close()

    active_templates = _infer_active_template_names()
    slot_counts = _collect_slot_counts(rows, active_templates)
    template_observed: dict[str, int] = {name: 0 for name in slot_counts}
    slot_observed: dict[str, int] = {}
    slot_examples: dict[str, dict[str, Any]] = {}

    for row in rows:
        for template_name in _row_templates(row):
            template_observed[template_name] = (
                template_observed.get(template_name, 0) + 1
            )
        for slot in _row_slots(row):
            slot_key = canonical_slot_key(
                str(
                    slot.get("slot_key_canonical")
                    or slot.get("slot_key")
                    or f"{slot.get('template_name', 'unknown')}.slot{slot.get('slot_index', 0)}"
                )
            )
            slot_observed[slot_key] = slot_observed.get(slot_key, 0) + 1
            slot_examples.setdefault(slot_key, slot)

    primitive_metadata = _primitive_metadata_by_name()
    observed_ops: dict[str, int] = {}
    for row in op_rows:
        op_name = str(row["op_name"] or "").strip()
        if op_name:
            observed_ops[op_name] = observed_ops.get(op_name, 0) + 1
    all_op_names = sorted(set(primitive_metadata) | set(observed_ops) | set(op_stats))
    profile_source_mtime = (
        profile_path.stat().st_mtime if profile_path.exists() else None
    )
    profile_src = _connect_optional_readonly(profile_path)
    op_profiles, pair_profiles, triplet_profiles = _load_profile_maps(profile_src)

    dst = _connect_output(output_db, replace=replace)
    try:
        _create_schema(dst)
        _reset_materialized_tables(dst)
        now = time.time()
        source_mtime = source_path.stat().st_mtime if source_path.exists() else None
        build_id = f"meta_{time.strftime('%Y%m%dT%H%M%S', time.gmtime(now))}"
        dst.execute(
            """
            INSERT OR REPLACE INTO meta_builds
                (build_id, created_at, source_db_path, source_db_mtime,
                 profile_source_db_path, profile_source_db_mtime,
                 source_program_count, property_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                build_id,
                now,
                str(source_path),
                source_mtime,
                str(profile_path),
                profile_source_mtime,
                len(rows),
                PROPERTY_VERSION,
            ),
        )

        template_catalog_cols = (
            "template_name",
            "slot_count",
            "observed_count",
            "property_version",
            *TEMPLATE_PROPERTY_COLUMNS,
        )
        template_catalog_sql = _insert_sql(
            "template_property_catalog", template_catalog_cols
        )
        for template_name in sorted(slot_counts):
            slot_count = slot_counts.get(template_name, 0)
            props = template_descriptive_properties(
                template_name, slot_count=slot_count
            )
            dst.execute(
                template_catalog_sql,
                (
                    template_name,
                    slot_count,
                    template_observed.get(template_name, 0),
                    PROPERTY_VERSION,
                    *(_value_for_db(props[col]) for col in TEMPLATE_PROPERTY_COLUMNS),
                ),
            )

        slot_catalog_cols = (
            "slot_key",
            "template_name",
            "slot_index",
            "slot_classes_json",
            "observed_count",
            "property_version",
            *SLOT_PROPERTY_COLUMNS,
        )
        slot_catalog_sql = _insert_sql("slot_property_catalog", slot_catalog_cols)
        for slot_key in sorted(slot_examples):
            slot = slot_examples[slot_key]
            template_name = str(slot.get("template_name") or slot_key.split(".", 1)[0])
            slot_index = int(slot.get("slot_index") or 0)
            slot_classes = [str(item) for item in (slot.get("slot_classes") or [])]
            slot_count = slot_counts.get(template_name, 0)
            props = slot_descriptive_properties(
                slot_key,
                template_name=template_name,
                slot_index=slot_index,
                slot_count=slot_count,
                slot_classes=slot_classes,
            )
            dst.execute(
                slot_catalog_sql,
                (
                    slot_key,
                    template_name,
                    slot_index,
                    _json_dumps(slot_classes),
                    slot_observed.get(slot_key, 0),
                    PROPERTY_VERSION,
                    *(_value_for_db(props[col]) for col in SLOT_PROPERTY_COLUMNS),
                ),
            )

        op_catalog_cols = (
            "op_name",
            "observed_count",
            "eval_count",
            "s1_pass_count",
            "mean_loss",
            "min_loss",
            "std_loss",
            "mean_novelty",
            "math_space_rate",
            "property_version",
            *OP_PROPERTY_COLUMNS,
        )
        op_catalog_sql = _insert_sql("op_property_catalog", op_catalog_cols)
        for op_name in all_op_names:
            stats = op_stats.get(op_name, {})
            props = op_descriptive_properties(
                op_name,
                metadata=primitive_metadata.get(op_name, {}),
            )
            dst.execute(
                op_catalog_sql,
                (
                    op_name,
                    observed_ops.get(op_name, 0),
                    int(stats.get("eval_count") or 0),
                    int(stats.get("s1_pass_count") or 0),
                    stats.get("mean_loss"),
                    stats.get("min_loss"),
                    stats.get("std_loss"),
                    stats.get("mean_novelty"),
                    stats.get("math_space_rate"),
                    OP_PROPERTY_VERSION,
                    *(_value_for_db(props[col]) for col in OP_PROPERTY_COLUMNS),
                ),
            )

        candidate_cols = (
            "candidate_name",
            "property_version",
            *_CANDIDATE_VALUE_COLUMNS,
        )
        candidate_sql = _insert_sql(
            "alternative_math_candidate_catalog", candidate_cols
        )
        for candidate_name in (
            "lambda_calculus",
            "combinatory_logic",
            "category_theory",
            "boolean_algebra",
        ):
            props = alternative_math_candidate_properties(candidate_name)
            dst.execute(
                candidate_sql,
                (
                    candidate_name,
                    OP_PROPERTY_VERSION,
                    *(_value_for_db(props[col]) for col in _CANDIDATE_VALUE_COLUMNS),
                ),
            )

        n_op_profile_rows = _copy_profile_table(
            src=profile_src,
            dst=dst,
            source_path=profile_path,
            source_mtime=profile_source_mtime,
            source_table="op_profiles",
            target_table="op_profile_catalog",
            columns=_OP_PROFILE_COLUMNS,
        )
        n_pair_profile_rows = _copy_profile_table(
            src=profile_src,
            dst=dst,
            source_path=profile_path,
            source_mtime=profile_source_mtime,
            source_table="pair_profiles",
            target_table="op_pair_profile_catalog",
            columns=_PAIR_PROFILE_COLUMNS,
        )
        n_triplet_profile_rows = _copy_profile_table(
            src=profile_src,
            dst=dst,
            source_path=profile_path,
            source_mtime=profile_source_mtime,
            source_table="triplet_profiles",
            target_table="op_triplet_profile_catalog",
            columns=_TRIPLET_PROFILE_COLUMNS,
        )

        metric_cols = (
            "metric_name",
            "metric_family",
            "metric_scale",
            "data_type",
            "metric_direction",
            "known_bias",
            "compute_cost_class",
            "reliability_status",
            "primary_use",
            "source_columns_json",
            "active_for_priors",
        )
        metric_sql = _insert_sql("eval_metric_catalog", metric_cols)
        for metric in _EVAL_METRIC_ROWS:
            dst.execute(
                metric_sql,
                tuple(
                    _json_dumps(metric[col])
                    if col == "source_columns_json"
                    else metric[col]
                    for col in metric_cols
                ),
            )

        external_cols = (
            "external_family",
            "mapped_ops_json",
            "mapped_templates_json",
            "expected_strength",
            "expected_risk",
            "hardware_note",
            "tags_json",
            "confidence",
            "source_ref",
        )
        external_sql = _insert_sql("external_component_prior_catalog", external_cols)
        for prior in _EXTERNAL_COMPONENT_PRIOR_ROWS:
            dst.execute(
                external_sql,
                tuple(
                    _json_dumps(prior[col])
                    if col in {"mapped_ops_json", "mapped_templates_json", "tags_json"}
                    else prior[col]
                    for col in external_cols
                ),
            )

        graph_profile_cols = (
            "result_id",
            "graph_fingerprint",
            "graph_profile_op_count",
            "profile_known_op_count",
            "profile_missing_op_count",
            "profile_coverage_rate",
            "profile_total_forward_time_us",
            "profile_total_backward_time_us",
            "profile_mean_forward_time_us",
            "profile_max_forward_time_us",
            "profile_slowest_op_name",
            "profile_total_peak_memory_bytes",
            "profile_total_flops_estimate",
            "profile_max_lipschitz_estimate",
            "profile_max_jacobian_condition_num",
            "profile_grad_vanishing_op_count",
            "profile_grad_exploding_op_count",
            "profile_output_nan_op_count",
            "profile_pair_count",
            "profile_pair_unstable_count",
            "profile_pair_grad_exploding_count",
            "profile_pair_max_lipschitz_estimate",
            "profile_triplet_count",
            "profile_triplet_unstable_count",
            "profile_triplet_divergent_count",
            "profile_triplet_grad_vanishing_count",
        )
        graph_profile_sql = _insert_sql(
            "graph_profile_observations", graph_profile_cols
        )
        for row in rows:
            payload = _graph_profile_payload(
                row,
                op_profiles,
                pair_profiles,
                triplet_profiles,
            )
            dst.execute(
                graph_profile_sql,
                tuple(_value_for_db(payload[col]) for col in graph_profile_cols),
            )

        template_obs_cols = (
            "result_id",
            "template_name",
            *_OBS_OUTCOME_COLUMNS,
            *_DERIVED_GRAPH_COLUMNS,
            "slot_count",
            "property_version",
            *TEMPLATE_PROPERTY_COLUMNS,
        )
        template_obs_sql = _insert_sql("template_observations", template_obs_cols)
        for row in rows:
            outcomes = _outcome_payload(row)
            derived = _derived_graph_payload(row)
            for template_name in _row_templates(row):
                slot_count = slot_counts.get(template_name, 0)
                props = template_descriptive_properties(
                    template_name, slot_count=slot_count
                )
                dst.execute(
                    template_obs_sql,
                    (
                        row["result_id"],
                        template_name,
                        *(_value_for_db(outcomes[col]) for col in _OBS_OUTCOME_COLUMNS),
                        *(
                            _value_for_db(derived[col])
                            for col in _DERIVED_GRAPH_COLUMNS
                        ),
                        slot_count,
                        PROPERTY_VERSION,
                        *(
                            _value_for_db(props[col])
                            for col in TEMPLATE_PROPERTY_COLUMNS
                        ),
                    ),
                )

        slot_obs_cols = (
            "result_id",
            "slot_key",
            "template_name",
            "slot_index",
            "selected_motif",
            "selected_motif_class",
            "wildcard",
            "slot_classes_json",
            *_OBS_OUTCOME_COLUMNS,
            *_DERIVED_GRAPH_COLUMNS,
            "slot_count",
            "property_version",
            *TEMPLATE_PROPERTY_COLUMNS,
            *SLOT_PROPERTY_COLUMNS,
        )
        slot_obs_sql = _insert_sql("slot_observations", slot_obs_cols)
        for row in rows:
            outcomes = _outcome_payload(row)
            derived = _derived_graph_payload(row)
            for slot in _row_slots(row):
                slot_key = canonical_slot_key(
                    str(
                        slot.get("slot_key_canonical")
                        or slot.get("slot_key")
                        or f"{slot.get('template_name', 'unknown')}.slot{slot.get('slot_index', 0)}"
                    )
                )
                template_name = str(
                    slot.get("template_name") or slot_key.split(".", 1)[0]
                )
                slot_index = int(slot.get("slot_index") or 0)
                slot_classes = [str(item) for item in (slot.get("slot_classes") or [])]
                slot_count = slot_counts.get(template_name, 0)
                template_props = template_descriptive_properties(
                    template_name, slot_count=slot_count
                )
                slot_props = slot_descriptive_properties(
                    slot_key,
                    template_name=template_name,
                    slot_index=slot_index,
                    slot_count=slot_count,
                    slot_classes=slot_classes,
                )
                dst.execute(
                    slot_obs_sql,
                    (
                        row["result_id"],
                        slot_key,
                        template_name,
                        slot_index,
                        slot.get("selected_motif"),
                        slot.get("selected_motif_class"),
                        int(bool(slot.get("wildcard"))),
                        _json_dumps(slot_classes),
                        *(_value_for_db(outcomes[col]) for col in _OBS_OUTCOME_COLUMNS),
                        *(
                            _value_for_db(derived[col])
                            for col in _DERIVED_GRAPH_COLUMNS
                        ),
                        slot_count,
                        PROPERTY_VERSION,
                        *(
                            _value_for_db(template_props[col])
                            for col in TEMPLATE_PROPERTY_COLUMNS
                        ),
                        *(
                            _value_for_db(slot_props[col])
                            for col in SLOT_PROPERTY_COLUMNS
                        ),
                    ),
                )

        op_obs_cols = (
            "result_id",
            "op_name",
            *_OBS_OUTCOME_COLUMNS,
            *_DERIVED_GRAPH_COLUMNS,
            "property_version",
            *OP_PROPERTY_COLUMNS,
        )
        op_obs_sql = _insert_sql("op_observations", op_obs_cols)
        for row in op_rows:
            op_name = str(row["op_name"] or "").strip()
            if not op_name:
                continue
            props = op_descriptive_properties(
                op_name,
                metadata=primitive_metadata.get(op_name, {}),
            )
            outcomes = _outcome_payload(row)
            derived = _derived_graph_payload(row)
            dst.execute(
                op_obs_sql,
                (
                    row["result_id"],
                    op_name,
                    *(_value_for_db(outcomes[col]) for col in _OBS_OUTCOME_COLUMNS),
                    *(_value_for_db(derived[col]) for col in _DERIVED_GRAPH_COLUMNS),
                    OP_PROPERTY_VERSION,
                    *(_value_for_db(props[col]) for col in OP_PROPERTY_COLUMNS),
                ),
            )

        dst.commit()
        return BuildSummary(
            source_db=str(source_path),
            output_db=str(output_db),
            profiling_db=str(profile_path),
            n_program_rows=len(rows),
            n_template_catalog_rows=_count_rows(dst, "template_property_catalog"),
            n_slot_catalog_rows=_count_rows(dst, "slot_property_catalog"),
            n_op_catalog_rows=_count_rows(dst, "op_property_catalog"),
            n_alternative_candidate_rows=_count_rows(
                dst, "alternative_math_candidate_catalog"
            ),
            n_op_profile_rows=n_op_profile_rows,
            n_pair_profile_rows=n_pair_profile_rows,
            n_triplet_profile_rows=n_triplet_profile_rows,
            n_eval_metric_rows=_count_rows(dst, "eval_metric_catalog"),
            n_external_component_prior_rows=_count_rows(
                dst, "external_component_prior_catalog"
            ),
            n_graph_profile_observation_rows=_count_rows(
                dst, "graph_profile_observations"
            ),
            n_template_observation_rows=_count_rows(dst, "template_observations"),
            n_slot_observation_rows=_count_rows(dst, "slot_observations"),
            n_op_observation_rows=_count_rows(dst, "op_observations"),
        )
    finally:
        dst.close()
        if profile_src is not None:
            profile_src.close()
