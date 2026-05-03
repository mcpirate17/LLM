"""Runner helpers — split from _helpers. Re-exported via _helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..json_utils import json_safe
from ._routing_ops import (
    ROUTING_FAST_LANE_OPS as _ROUTING_FAST_LANE_OPS,
    ROUTING_OBSERVED_OPS as _ROUTING_OBSERVED_OPS,
)

logger = logging.getLogger(__name__)
_REFERENCE_TRAJECTORY_PATH = Path("research/eval/reference_trajectories.json")


def compute_seed_metrics(
    seed_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate metrics from multi-seed training results.

    Returns dict with: passed_seeds, loss_ratios, val_loss_ratio,
    multi_seed_std, robustness_score, is_unstable, init_sensitivity_std,
    best_seed.
    """
    passed_seeds = [r for r in seed_results if r.get("passed")]
    loss_ratios = [
        r["loss_ratio"] for r in seed_results if r.get("loss_ratio") is not None
    ]

    val_loss_ratio = sum(loss_ratios) / len(loss_ratios) if loss_ratios else None
    multi_seed_std = 0.0
    robustness_score = 1.0
    is_unstable = False

    if len(loss_ratios) > 1:
        mean_lr = val_loss_ratio
        variance = sum((lr - mean_lr) ** 2 for lr in loss_ratios) / len(loss_ratios)
        multi_seed_std = variance**0.5
        if variance > 0.15:
            is_unstable = True
        if mean_lr > 1e-6:
            robustness_score = max(0.0, 1.0 - (multi_seed_std / mean_lr))

    # Init sensitivity: std between default and xavier seeds
    init_sensitivity_std = None
    default_losses: List[float] = []
    xavier_losses: List[float] = []
    for r in seed_results:
        lr = r.get("loss_ratio")
        if lr is None:
            continue
        scheme = r.get("init_scheme")
        if scheme == "default":
            default_losses.append(lr)
        elif scheme == "xavier_uniform":
            xavier_losses.append(lr)
    if default_losses and xavier_losses:
        default_mean = sum(default_losses) / len(default_losses)
        xavier_mean = sum(xavier_losses) / len(xavier_losses)
        init_sensitivity_std = abs(default_mean - xavier_mean)

    # Best seed: lowest final_loss
    best_seed = None
    if loss_ratios:
        best_seed = min(
            (r for r in seed_results if r.get("final_loss") is not None),
            key=lambda r: r["final_loss"],
            default=None,
        )

    return {
        "passed_seeds": passed_seeds,
        "loss_ratios": loss_ratios,
        "val_loss_ratio": val_loss_ratio,
        "multi_seed_std": multi_seed_std,
        "robustness_score": robustness_score,
        "is_unstable": is_unstable,
        "init_sensitivity_std": init_sensitivity_std,
        "best_seed": best_seed,
    }


def screening_wikitext_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract persisted screening WikiText fields from a result dict."""
    fields: Dict[str, Any] = {}
    for key in (
        "wikitext_perplexity",
        "wikitext_score",
        "wikitext_pre_perplexity",
        "wikitext_ppl_improvement",
        "screening_wikitext_status",
        "screening_wikitext_metric_version",
    ):
        value = row.get(key)
        if value is not None:
            fields[key] = value

    budget = row.get("screening_wikitext_budget")
    if budget:
        fields["screening_wikitext_budget_json"] = json.dumps(
            json_safe(budget),
            sort_keys=True,
            separators=(",", ":"),
        )

    variant = row.get("variant")
    if variant is not None:
        fields["screening_wikitext_variant"] = variant

    elapsed = row.get("elapsed_ms")
    if elapsed is not None:
        fields["screening_wikitext_elapsed_ms"] = elapsed

    return fields


def screening_probe_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract persisted screening/probe telemetry from a result dict."""
    fields: Dict[str, Any] = {}
    for key in (
        "rapid_screening_passed",
        "rapid_screening_elapsed_ms",
        "rapid_screening_steps_completed",
        "rapid_screening_max_steps",
        "rapid_screening_degraded",
        "rapid_screening_kill_reason",
        "rapid_screening_kill_step",
        "rapid_screening_kill_metric",
        "rapid_screening_gpu_minutes_saved",
        "ar_auc",
        "ar_final_acc",
        "ar_timed_out",
        "ar_above_chance",
        "induction_auc",
        "induction_probe_train_steps",
        "induction_probe_eval_examples",
        "induction_probe_batch_size",
        "induction_probe_elapsed_ms",
        "induction_probe_metric_version",
        "induction_probe_speed_mode",
        "induction_probe_pool_size",
        "binding_auc",
        "binding_auc_curriculum",
        "binding_probe_eval_examples",
        "binding_probe_elapsed_ms",
        "binding_probe_curriculum_steps",
        "binding_probe_curriculum_elapsed_ms",
        "binding_probe_curriculum_protocol_version",
        "binding_composite",
        "local_only",
        "hellaswag_acc",
        "hellaswag_status",
        "hellaswag_n_examples",
        "hellaswag_metric_version",
        "hellaswag_tokenizer_mode",
        "hellaswag_tiktoken_encoding",
        "screening_hellaswag_correct",
        "screening_hellaswag_total",
        "screening_hellaswag_elapsed_ms",
        # BLiMP linguistic acceptability (v8 scoring component)
        "blimp_overall_accuracy",
        "blimp_n_subtasks",
        "blimp_status",
        "blimp_elapsed_ms",
        # v8 understanding metrics
        "tinystories_score",
        "cross_task_score",
        "diagnostic_score",
        "permutation_composition_score",
        "permutation_composition_train_chain_acc",
        "permutation_composition_extrapolation_acc",
        "permutation_composition_n_items",
        "permutation_composition_train_chain_len",
        "permutation_composition_eval_chain_len",
        "permutation_composition_train_steps",
        "permutation_composition_chance",
        "permutation_composition_elapsed_ms",
        "permutation_composition_status",
        "permutation_composition_metric_version",
        "train_budget_steps",
    ):
        value = row.get(key)
        if value is not None:
            fields[key] = value

    rapid_degraded_reasons = row.get("rapid_screening_degraded_reasons")
    if rapid_degraded_reasons:
        fields["rapid_screening_degraded_reasons_json"] = json.dumps(
            json_safe(rapid_degraded_reasons),
            sort_keys=True,
            separators=(",", ":"),
        )

    rapid_metrics = row.get("rapid_screening_metrics")
    if rapid_metrics:
        fields["rapid_screening_metrics_json"] = json.dumps(
            json_safe(rapid_metrics),
            sort_keys=True,
            separators=(",", ":"),
        )

    induction_gap_accuracies = row.get("induction_gap_accuracies")
    if induction_gap_accuracies:
        fields["induction_gap_accuracies_json"] = json.dumps(
            json_safe(induction_gap_accuracies),
            sort_keys=True,
            separators=(",", ":"),
        )

    induction_gaps = row.get("induction_probe_gaps")
    if induction_gaps:
        fields["induction_probe_gaps_json"] = json.dumps(
            json_safe(induction_gaps),
            sort_keys=True,
            separators=(",", ":"),
        )

    binding_distance_accuracies = row.get("binding_distance_accuracies")
    if binding_distance_accuracies:
        fields["binding_distance_accuracies_json"] = json.dumps(
            json_safe(binding_distance_accuracies),
            sort_keys=True,
            separators=(",", ":"),
        )

    binding_distance_accuracies_curriculum = row.get(
        "binding_distance_accuracies_curriculum"
    )
    if binding_distance_accuracies_curriculum:
        fields["binding_distance_accuracies_curriculum_json"] = json.dumps(
            json_safe(binding_distance_accuracies_curriculum),
            sort_keys=True,
            separators=(",", ":"),
        )

    binding_distances = row.get("binding_probe_distances")
    if binding_distances:
        fields["binding_probe_distances_json"] = json.dumps(
            json_safe(binding_distances),
            sort_keys=True,
            separators=(",", ":"),
        )

    blimp_subtasks = row.get("blimp_subtask_accuracies")
    if blimp_subtasks:
        fields["blimp_subtask_accuracies_json"] = json.dumps(
            json_safe(blimp_subtasks),
            sort_keys=True,
            separators=(",", ":"),
        )

    return fields


def routing_fast_lane_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract persisted routing fast-lane fields from a result dict."""
    fields: Dict[str, Any] = {}
    for key in (
        "routing_fast_lane_applied",
        "routing_fast_lane_status",
        "routing_fast_lane_metric_version",
        "routing_fast_lane_perplexity",
        "routing_fast_lane_score",
        "routing_fast_lane_pre_perplexity",
        "routing_fast_lane_ppl_improvement",
        "routing_fast_lane_elapsed_ms",
        "routing_fast_lane_slope",
        "routing_fast_lane_slope_consistent",
    ):
        value = row.get(key)
        if value is not None:
            fields[key] = value

    budget = row.get("routing_fast_lane_budget")
    if budget:
        fields["routing_fast_lane_budget_json"] = json.dumps(
            json_safe(budget),
            sort_keys=True,
            separators=(",", ":"),
        )

    routing_ops = row.get("routing_fast_lane_routing_ops")
    if routing_ops:
        fields["routing_fast_lane_routing_ops_json"] = json.dumps(
            sorted({str(op) for op in routing_ops if op}),
            sort_keys=True,
            separators=(",", ":"),
        )

    return fields


def graph_routing_ops(graph: Any) -> List[str]:
    """Return sorted routing-related ops present in a graph-like object."""
    return _graph_matching_ops(graph, _ROUTING_FAST_LANE_OPS)


def graph_observed_routing_ops(graph: Any) -> List[str]:
    """Return sorted routing/specialization ops for human-facing observability."""
    return _graph_matching_ops(graph, _ROUTING_OBSERVED_OPS)


def _graph_matching_ops(graph: Any, allowed_ops: frozenset[str]) -> List[str]:
    """Return sorted ops from *allowed_ops* present in a graph-like object."""
    nodes = getattr(graph, "nodes", None)
    ops: Set[str] = set()
    if isinstance(nodes, dict):
        for node in nodes.values():
            op_name = getattr(node, "op_name", None)
            if op_name in allowed_ops:
                ops.add(str(op_name))
    elif isinstance(graph, dict):
        raw_nodes = graph.get("nodes")
        if isinstance(raw_nodes, dict):
            iterable = raw_nodes.values()
        elif isinstance(raw_nodes, list):
            iterable = raw_nodes
        else:
            iterable = []
        for node in iterable:
            if not isinstance(node, dict):
                continue
            op_name = node.get("op_name")
            if op_name in allowed_ops:
                ops.add(str(op_name))
    return sorted(ops)


def trajectory_probe_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract persisted trajectory-probe fields from a benchmark result dict."""
    fields: Dict[str, Any] = {}
    for key in (
        "wikitext_ppl_200",
        "wikitext_ppl_500",
        "wikitext_improvement_ratio",
        "wikitext_eval_steps",
    ):
        value = row.get(key)
        if value is not None:
            fields[key] = value

    if row.get("wikitext_improvement_ratio") is not None:
        fields["wikitext_ppl_improvement_ratio"] = row["wikitext_improvement_ratio"]
    if row.get("eval_budget_steps") is not None:
        fields["eval_budget_steps"] = row["eval_budget_steps"]
    if row.get("evaluation_stage"):
        fields["evaluation_stage"] = row["evaluation_stage"]
    if row.get("capability_tier"):
        fields["capability_tier"] = row["capability_tier"]
    return fields


# Every column written by ``TrajectoryMetricsResult.to_column_dict()`` —
# the canonical list of v9 fingerprint columns on program_results.
# Sourced once so callers (screening recorder, investigation updater,
# backfill scripts) can't drift apart.
_V9_TRAJECTORY_COLUMNS: Tuple[str, ...] = (
    "fp_metric_phase",
    "fp_jacobian_spectral_norm",
    "fp_jacobian_effective_rank",
    "fp_sensitivity_uniformity",
    "fp_spec_norm_status",
    "fp_jacobian_erf_density",
    "fp_jacobian_erf_variance",
    "fp_jacobian_erf_decay_slope",
    "fp_jacobian_erf_last_norm",
    "fp_jacobian_erf_first_norm",
    "fp_jacobian_erf_status",
    "fp_jacobian_erf_elapsed_ms",
    "fp_icld_velocity",
    "fp_icld_early_loss",
    "fp_icld_late_loss",
    "fp_icld_delta_loss",
    "fp_icld_seq_len",
    "fp_icld_status",
    "fp_icld_elapsed_ms",
    "fp_logit_margin_velocity",
    "fp_logit_margin_initial",
    "fp_logit_margin_final",
    "fp_logit_margin_delta",
    "fp_logit_margin_n_steps",
    "fp_logit_margin_status",
    "fp_logit_margin_elapsed_ms",
    "fp_id_pr_early",
    "fp_id_pr_late",
    "fp_id_norm_early",
    "fp_id_norm_late",
    "fp_id_step_early",
    "fp_id_step_late",
    "fp_id_collapse_rate",
    "fp_id_collapse_rate_normalized",
    "fp_id_collapse_status",
    "fp_id_collapse_elapsed_ms",
)


def v9_trajectory_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract v9 trajectory-metric columns from a benchmark result dict.

    Pairs with ``compute_trajectory_metrics(...).to_column_dict()``: every
    column landing in that dict has a matching ``program_results`` schema
    entry. The investigation path uses this to thread its
    ``metric_phase="investigation_full"`` measurements through the
    ``UPDATE program_results SET …`` write, since investigation reuses the
    screening row instead of inserting a fresh one. Returns only keys
    whose value is not None to preserve any earlier-phase data when
    investigation didn't measure that particular metric.
    """
    return {
        col: row[col]
        for col in _V9_TRAJECTORY_COLUMNS
        if col in row and row[col] is not None
    }


_S1_SCALAR_COLUMNS: Tuple[str, ...] = (
    "initial_loss",
    "min_loss",
    "loss_improvement_rate",
    "avg_step_time_ms",
    "total_train_time_ms",
    "max_grad_norm",
    "mean_grad_norm",
    "grad_norm_std",
    "n_train_steps",
    "final_lr",
    "validation_loss",
    "validation_loss_ratio",
    "generalization_gap",
    "discovery_loss",
    "discovery_loss_ratio",
    "train_budget_steps",
    "param_count",
    "throughput_tok_s",
)


_BEHAVIORAL_FINGERPRINT_COLUMNS: Tuple[str, ...] = (
    "fp_interaction_locality",
    "fp_interaction_sparsity",
    "fp_interaction_symmetry",
    "fp_interaction_hierarchy",
    "fp_intrinsic_dim",
    "fp_isotropy",
    "fp_rank_ratio",
    "fp_jacobian_spectral_norm",
    "fp_jacobian_effective_rank",
    "fp_sensitivity_uniformity",
    "fp_cka_vs_transformer",
    "fp_cka_vs_ssm",
    "fp_cka_vs_conv",
    "fp_hierarchy_fitness",
    "fp_gromov_delta",
)


# Columns that MUST be present on a row marked stage1_passed=True for it to
# count as a complete post-S1 observation. The Causal Ablation Diagnostics
# page treats anything missing one of these as "loss-only" and refuses to
# present it as known-good/known-bad evidence.
S1_REQUIRED_POST_METRIC_COLUMNS: Tuple[str, ...] = (
    "wikitext_perplexity",
    "wikitext_score",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "induction_auc",
    "binding_auc",
    "binding_composite",
    "ar_auc",
    "fp_jacobian_erf_density",
    "fp_icld_delta_loss",
    "fp_logit_margin_delta",
)


def _behavioral_fingerprint_kwargs(
    s1: Dict[str, Any],
) -> Dict[str, Any]:
    """Reconstruct BehavioralFingerprint columns + fingerprint_json from s1.

    The S1 worker returns a ``_behavioral_fingerprint`` dict (the canonical
    intermediate representation between the worker and persistence). Both
    the regular pipeline (dashboard_orchestrator._build_novelty_kwargs) and
    the ablation pipeline must turn that into the same set of program_results
    columns. Centralizing it here makes drift impossible.
    """
    fp_dict = s1.get("_behavioral_fingerprint")
    if not isinstance(fp_dict, dict):
        return {}
    try:
        from ...eval.fingerprint import BehavioralFingerprint
    except ImportError:
        return {}
    fp = BehavioralFingerprint()
    for k, v in fp_dict.items():
        if hasattr(fp, k):
            setattr(fp, k, v)
    out: Dict[str, Any] = {
        "fingerprint_json": json.dumps(json_safe(fp.to_dict())),
    }
    for col in _BEHAVIORAL_FINGERPRINT_COLUMNS:
        if hasattr(fp, col[len("fp_") :]):
            out[col] = getattr(fp, col[len("fp_") :], None)
    # Drop None entries so the persistence layer's COALESCE-style upsert
    # doesn't blow away earlier-phase values.
    return {k: v for k, v in out.items() if v is not None}


def program_result_kwargs_from_s1(
    s1: Dict[str, Any],
    *,
    model_source: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical adapter from ``_micro_train`` output → ``record_program_result`` kwargs.

    Single source of truth used by every persistence path (regular synthesis,
    ablation runner, ablation backfill) so a metric added in one place lands
    in every place. Returns a flat dict of column-name → value that can be
    expanded with ``**`` into ``record_program_result`` or
    ``merge_program_result_patch``.

    Includes:
      - S1 scalar metrics (loss/perf/grad/throughput/etc.)
      - WikiText screening fields
      - HellaSwag, BLiMP, induction, binding, AR probe fields
      - Routing fast-lane fields
      - v9 trajectory/fingerprint fields (ERF, ICLD, logit margin, ID collapse)
      - pruning_*  (anything starting with that prefix on the s1 dict)
      - perf_report_json, kernel_timings_json, starvation_report_json
      - Reconstructed BehavioralFingerprint columns + fingerprint_json
      - ``model_source`` and final_loss/loss_ratio/error fields

    Always passes through ``error_type``/``error_message`` from s1 even on a
    failed S1 — caller decides whether to drop them. Caller is responsible
    for adding stage gate flags (stage0/05/1_passed), graph_fingerprint,
    graph_json, experiment_id, and any campaign-specific provenance labels
    (intentional_rerun_reason, trust_label, etc).
    """
    if not isinstance(s1, dict):
        s1 = {}

    out: Dict[str, Any] = {"model_source": model_source}

    if "final_loss" in s1:
        out["final_loss"] = s1.get("final_loss")
    if "loss_ratio" in s1:
        out["loss_ratio"] = s1.get("loss_ratio")
    if s1.get("error_type") is not None:
        out["error_type"] = s1.get("error_type")
    if s1.get("error") is not None:
        out["error_message"] = s1.get("error")

    for key in _S1_SCALAR_COLUMNS:
        value = s1.get(key)
        if value is not None:
            out[key] = value

    out.update(screening_wikitext_fields(s1))
    out.update(screening_probe_fields(s1))
    out.update(routing_fast_lane_fields(s1))
    out.update(v9_trajectory_fields(s1))

    for key, value in s1.items():
        if isinstance(key, str) and key.startswith("pruning_") and value is not None:
            out[key] = value

    perf = s1.get("perf_report")
    if perf is not None:
        out["perf_report_json"] = json.dumps(json_safe(perf), sort_keys=True)
    kernels = s1.get("kernel_timings_ms")
    if kernels is not None:
        out["kernel_timings_json"] = json.dumps(json_safe(kernels), sort_keys=True)
    starvation = s1.get("starvation_report")
    if starvation is not None:
        out["starvation_report_json"] = json.dumps(
            json_safe(starvation), sort_keys=True
        )

    out.update(_behavioral_fingerprint_kwargs(s1))

    if extra:
        out.update({k: v for k, v in extra.items() if v is not None})

    return out


def s1_post_metric_completeness(row: Dict[str, Any]) -> Dict[str, Any]:
    """Audit a program_result row for post-S1 metric completeness.

    Returns a dict suitable for diagnostics/tests with:
      - missing: list of S1_REQUIRED_POST_METRIC_COLUMNS that are None
      - present: list of columns that are populated
      - is_complete: True iff every required column is non-None
      - coverage: fraction (0..1) of required columns present
    """
    missing = [c for c in S1_REQUIRED_POST_METRIC_COLUMNS if row.get(c) is None]
    present = [c for c in S1_REQUIRED_POST_METRIC_COLUMNS if row.get(c) is not None]
    total = len(S1_REQUIRED_POST_METRIC_COLUMNS)
    return {
        "missing": missing,
        "present": present,
        "is_complete": not missing,
        "coverage": (total - len(missing)) / total if total else 1.0,
    }


def _load_best_reference_probe_ppl(step: int) -> Optional[float]:
    """Return the best cached reference PPL at the requested checkpoint."""
    try:
        payload = json.loads(_REFERENCE_TRAJECTORY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, dict):
        return None
    best = None
    step_key = str(step)
    for trajectory in trajectories.values():
        if not isinstance(trajectory, dict):
            continue
        checkpoints = trajectory.get("checkpoints")
        if not isinstance(checkpoints, dict):
            continue
        point = checkpoints.get(step) or checkpoints.get(step_key)
        if not isinstance(point, dict):
            continue
        try:
            ppl = float(point.get("ppl"))
        except (TypeError, ValueError):
            continue
        if best is None or ppl < best:
            best = ppl
    return best


def _trajectory_probe_capability_tier(
    ppl_500: Optional[float],
    improvement_ratio: Optional[float],
    threshold: float,
) -> str:
    """Classify probe outcome for downstream escalation and UI."""
    if ppl_500 is not None:
        best_ref_ppl = _load_best_reference_probe_ppl(500)
        if best_ref_ppl is not None and ppl_500 <= best_ref_ppl * 1.2:
            return "frontier_signal"
        if best_ref_ppl is not None and ppl_500 <= best_ref_ppl * 1.5:
            return "near_frontier"
    if improvement_ratio is not None and improvement_ratio >= threshold:
        return "slow_burn"
    return "routine"


def apply_adaptive_grad_clip(model: Any, current_clip: float) -> float:
    """Return the effective grad clip norm, respecting model's recommendation.

    Math-space models recommend higher clip values (5.0 vs default 1.0).
    """
    model_clip = getattr(model, "recommended_grad_clip", None)
    if model_clip is not None and model_clip > current_clip:
        return model_clip
    return current_clip


def _native_proactive_gating(graph) -> Dict[str, Any]:
    """
    Perform high-performance DAG validation and proactive gating using aria_core.
    Identifies stability risks and toxic motifs before compilation.
    """
    try:
        import aria_core
        from ...synthesis.primitives import OPCODE_MAP

        # 1. Map node IDs to 0..N-1 for C++ interop
        nodes = list(graph.nodes.values())
        id_map = {node.id: i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)

        # 2. Extract edges
        edges = []
        for node in nodes:
            for iid in node.input_ids:
                if iid in id_map:
                    edges.append([id_map[iid], id_map[node.id]])

        # 3. Extract op_codes
        op_codes = []
        for node in nodes:
            op_codes.append(OPCODE_MAP.get(node.op_name, -1))

        # 4. Call native engine
        return aria_core.proactive_gating(n_nodes, edges, op_codes)
    except (ImportError, RuntimeError, KeyError, TypeError) as e:
        strict = False
        try:
            metadata = getattr(graph, "metadata", None) or {}
            strict = bool(
                metadata.get("strict_candidate_gating")
                or metadata.get("candidate_grade_strict")
            )
        except Exception:
            strict = False
        if not strict:
            import os

            strict = os.getenv("ARIA_STRICT_CANDIDATE_GATING", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        logger.debug("Native proactive gating failed: %s", e)
        if strict:
            return {
                "passed": False,
                "reason": "native_gating_unavailable",
                "error": str(e),
            }
        return {"passed": True, "reason": "native_gating_error", "error": str(e)}


def _native_runner_progress_report() -> Dict[str, Any]:
    try:
        from ..native.telemetry import native_runner_capability_report

        return native_runner_capability_report()
    except (ImportError, RuntimeError, OSError) as exc:
        return {
            "enabled": False,
            "strict": False,
            "designer_runtime_available": False,
            "status": f"native_runner_report_error:{exc}",
            "supported_ops": [],
            "unsupported_ops": [],
            "approximate_mappings": {},
            "semantic_warnings": [],
            "semantic_warning_count": 0,
            "mapping_source": "",
        }


def _rebuild_graph_with_overrides(
    candidate_graph, overrides: Dict[int, Dict[str, Any]]
):
    """Rebuild a graph with targeted node op/config overrides."""
    rebuilt = type(candidate_graph)(candidate_graph.model_dim)
    id_map: Dict[int, int] = {}
    topo = candidate_graph.topological_order()
    for old_id in topo:
        node = candidate_graph.nodes[old_id]
        if node.is_input:
            id_map[old_id] = rebuilt.add_input()
            continue
        override = overrides.get(old_id, {})
        op_name = override.get("op_name", node.op_name)
        config = override.get("config", node.config)
        new_inputs = [id_map[i] for i in node.input_ids]
        try:
            new_id = rebuilt.add_op(op_name, new_inputs, config=config)
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            logger.debug("Graph rebuild add_op failed: %s", e)
            return None
        id_map[old_id] = new_id

    if candidate_graph.output_node is None:
        return None
    out_old = candidate_graph.output_node.id
    out_new = id_map.get(out_old)
    if out_new is None:
        return None
    try:
        rebuilt.set_output(out_new)
    except (ValueError, KeyError, RuntimeError) as e:
        logger.debug("Graph rebuild set_output failed: %s", e)
        return None
    rebuilt.metadata = dict(getattr(candidate_graph, "metadata", {}) or {})
    return rebuilt


def propose_ablation_suite(candidate_graph, hypothesis) -> List[Any]:
    """Generate counterfactual ablations by replacing suspected components."""
    from ...synthesis.primitives import get_primitive, list_primitives

    if candidate_graph is None:
        return []
    hyp = str(hypothesis or "").lower()
    ops = list_primitives()
    replacement_by_signature: Dict[Tuple[int, str], List[str]] = {}
    for op in ops:
        key = (op.n_inputs, op.shape_rule)
        replacement_by_signature.setdefault(key, []).append(op.name)
    for key in replacement_by_signature:
        replacement_by_signature[key] = sorted(set(replacement_by_signature[key]))

    target_nodes: List[int] = []
    for nid in candidate_graph.topological_order():
        node = candidate_graph.nodes[nid]
        if node.is_input:
            continue
        try:
            prim = get_primitive(node.op_name)
            category = prim.category.value
        except (KeyError, ValueError) as e:
            logger.debug("get_primitive failed for %s: %s", node.op_name, e)
            category = ""
        if node.op_name in hyp or category in hyp:
            target_nodes.append(nid)
        elif ("math space" in hyp or "math_space" in hyp) and category == "math_space":
            target_nodes.append(nid)

    if not target_nodes:
        non_input = [
            nid
            for nid in candidate_graph.topological_order()
            if not candidate_graph.nodes[nid].is_input
        ]
        target_nodes = non_input[-2:] if len(non_input) >= 2 else non_input

    ablations: List[Any] = []
    seen: Set[str] = set()
    for nid in target_nodes[:4]:
        node = candidate_graph.nodes[nid]
        try:
            prim = get_primitive(node.op_name)
        except (KeyError, ValueError):
            continue
        key = (prim.n_inputs, prim.shape_rule)
        candidates = [
            name
            for name in replacement_by_signature.get(key, [])
            if name != node.op_name
        ]
        if not candidates:
            continue

        # Prefer a non-identical family replacement to produce a meaningful counterfactual.
        replacement = candidates[0]
        for name in candidates:
            try:
                if get_primitive(name).category != prim.category:
                    replacement = name
                    break
            except (KeyError, ValueError):
                continue
        rebuilt = _rebuild_graph_with_overrides(
            candidate_graph,
            {nid: {"op_name": replacement, "config": dict(node.config or {})}},
        )
        if rebuilt is None:
            continue
        try:
            fp = rebuilt.fingerprint()
        except (ValueError, RuntimeError):
            continue
        if fp in seen:
            continue
        seen.add(fp)
        ablations.append(rebuilt)
        if len(ablations) >= 4:
            break

    return ablations
