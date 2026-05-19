"""Results analysis mixin: graph metrics, sandbox metrics, architecture telemetry."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..json_utils import json_safe
from ..notebook import LabNotebook
from ..notebook.notebook_programs import DuplicateFingerprintError
from ...synthesis.serializer import graph_to_json

logger = logging.getLogger(__name__)


def _accumulate_expert_counts(
    running: Optional[torch.Tensor], incoming: torch.Tensor
) -> torch.Tensor:
    """Sum routing expert-count tensors even when layers use different expert widths."""
    incoming = incoming.detach()
    if running is None:
        return incoming.clone()
    if running.shape == incoming.shape:
        return running + incoming

    max_len = max(int(running.numel()), int(incoming.numel()))
    device = running.device
    dtype = running.dtype
    merged = torch.zeros(max_len, device=device, dtype=dtype)
    merged[: running.numel()] += running
    merged[: incoming.numel()] += incoming.to(device=device, dtype=dtype)
    return merged


@dataclass
class _ArchitectureTelemetry:
    telemetry_rows: List[Dict[str, Any]] = field(default_factory=list)
    total_calls: int = 0
    total_fallback_calls: int = 0
    kernel_fallback_calls: int = 0
    density_sum: float = 0.0
    density_last_values: List[float] = field(default_factory=list)
    nm_compliant: int = 0
    nm_total: int = 0
    sparse_active_params_estimate: float = 0.0
    rt_tokens_total: int = 0
    rt_tokens_processed: int = 0
    rt_entropy_sum: float = 0.0
    rt_confidence_sum: float = 0.0
    rt_confidence_sq_sum: float = 0.0
    rt_confidence_count: int = 0
    rt_count: int = 0
    rt_expert_counts: Optional[torch.Tensor] = None
    rt_lane_histogram: Optional[torch.Tensor] = None
    rt_keep_count: int = 0
    rt_drop_count: int = 0
    rt_default_path_count: int = 0
    rt_sparse_span_count: int = 0
    rt_sparse_span_width_sum: float = 0.0
    rt_sparse_span_width_count: int = 0
    rt_sparse_span_coverage_tokens: int = 0
    at_savings_sum: float = 0.0
    at_depth_sum: float = 0.0
    at_count: int = 0
    recursion_savings_sum: float = 0.0
    recursion_depth_sum: float = 0.0
    recursion_count: int = 0
    recursion_max_depth_sum: float = 0.0


def _architecture_layers(model: nn.Module) -> List[Any]:
    try:
        layers = list(getattr(model, "layers", []) or [])
    except (TypeError, RuntimeError):
        layers = []
    if layers:
        return layers
    try:
        topo = getattr(model, "topology", None)
        blocks = getattr(topo, "blocks", None) if topo is not None else None
        return list(blocks) if blocks is not None else []
    except (TypeError, RuntimeError):
        return []


def _model_routing_mode(model: nn.Module) -> Optional[Any]:
    spec = getattr(model, "spec", None)
    choices = getattr(spec, "choices", None) if spec is not None else None
    return choices.get("compute_routing") if isinstance(choices, dict) else None


def _iter_layer_ops(layer: Any) -> List[Any]:
    ops = getattr(layer, "ops", None)
    if ops is None:
        return []
    if isinstance(ops, dict):
        return list(ops.values())
    try:
        return list(ops)
    except TypeError:
        return []


def _record_routing_telemetry(acc: _ArchitectureTelemetry, rt: Any) -> None:
    if not isinstance(rt, dict):
        return
    acc.rt_tokens_total += rt.get("tokens_total", 0)
    acc.rt_tokens_processed += rt.get("tokens_processed", 0)
    acc.rt_entropy_sum += rt.get("entropy_sum", 0.0)
    acc.rt_confidence_sum += rt.get("confidence_sum", 0.0)
    acc.rt_confidence_sq_sum += rt.get("confidence_sq_sum", 0.0)
    acc.rt_confidence_count += rt.get("confidence_count", 0)
    acc.rt_count += rt.get("count", 0)
    expert_counts = rt.get("expert_counts")
    if isinstance(expert_counts, torch.Tensor):
        acc.rt_expert_counts = _accumulate_expert_counts(
            acc.rt_expert_counts, expert_counts
        )
    acc.rt_keep_count += int(rt.get("keep_count", 0) or 0)
    acc.rt_drop_count += int(rt.get("drop_count", 0) or 0)
    acc.rt_default_path_count += int(rt.get("default_path_count", 0) or 0)
    acc.rt_sparse_span_count += int(rt.get("sparse_span_count", 0) or 0)
    acc.rt_sparse_span_width_sum += float(rt.get("sparse_span_width_sum", 0.0) or 0.0)
    acc.rt_sparse_span_width_count += int(rt.get("sparse_span_width_count", 0) or 0)
    acc.rt_sparse_span_coverage_tokens += int(
        rt.get("sparse_span_coverage_tokens", 0) or 0
    )
    lane_histogram = rt.get("lane_histogram")
    if isinstance(lane_histogram, torch.Tensor):
        acc.rt_lane_histogram = _accumulate_expert_counts(
            acc.rt_lane_histogram, lane_histogram
        )


def _record_adaptive_telemetry(
    acc: _ArchitectureTelemetry, telemetry: Any, routing: Optional[Any] = None
) -> None:
    if not isinstance(telemetry, dict):
        return
    count = telemetry.get("count", 0)
    savings_sum = telemetry.get("savings_sum", 0.0)
    depth_sum = telemetry.get("depth_sum", 0.0)
    acc.at_savings_sum += savings_sum
    acc.at_depth_sum += depth_sum
    acc.at_count += count
    if routing is not None and routing.__class__.__name__ == "AdaptiveRecursionRouting":
        acc.recursion_savings_sum += savings_sum
        acc.recursion_depth_sum += depth_sum
        acc.recursion_count += count
        acc.recursion_max_depth_sum += float(getattr(routing, "max_depth", 0)) * count


def _record_sparse_telemetry(acc: _ArchitectureTelemetry, compiled_op: Any) -> None:
    sparse_telemetry = getattr(compiled_op, "sparse_telemetry", None)
    if not sparse_telemetry:
        return
    weight_params = (
        float(compiled_op.weight.numel()) if hasattr(compiled_op, "weight") else 0.0
    )
    for op_name, stats in sparse_telemetry.items():
        calls = int(stats.get("calls", 0) or 0)
        last_density = float(stats.get("last_density", 1.0) or 1.0)
        acc.total_calls += calls
        acc.total_fallback_calls += int(stats.get("fallback_calls", 0) or 0)
        acc.density_sum += float(stats.get("density_sum", 0.0) or 0.0)
        acc.density_last_values.append(last_density)
        if stats.get("last_fallback_reason") == "kernel_unavailable":
            acc.kernel_fallback_calls += int(stats.get("fallback_calls", 0) or 0)
        if op_name in ("nm_sparse_linear", "semi_structured_2_4_linear"):
            acc.nm_total += 1
            acc.nm_compliant += 1 if last_density <= 0.51 else 0
        if weight_params > 0.0:
            density_for_params = (
                float(stats.get("density_sum", 0.0)) / calls
                if calls > 0
                else last_density
            )
            acc.sparse_active_params_estimate += weight_params * density_for_params
        acc.telemetry_rows.append(
            {"op_name": op_name, "calls": calls, "last_density": last_density}
        )


def _total_weight_params(layers: List[Any]) -> float:
    total = 0.0
    for layer in layers:
        for op in _iter_layer_ops(layer):
            if hasattr(op, "weight"):
                total += float(getattr(op, "weight", torch.empty(0)).numel())
    return total


def _finalize_sparse_metrics(
    metrics: Dict[str, Any], acc: _ArchitectureTelemetry, layers: List[Any]
) -> None:
    if acc.total_calls <= 0:
        return
    metrics["sparse_density_mean"] = acc.density_sum / max(acc.total_calls, 1)
    metrics["sparse_density_last"] = sum(acc.density_last_values) / max(
        len(acc.density_last_values), 1
    )
    metrics["sparse_fallback_calls"] = acc.total_fallback_calls
    metrics["sparse_kernel_fallback_calls"] = acc.kernel_fallback_calls
    metrics["sparse_active_params_estimate"] = int(
        max(0.0, acc.sparse_active_params_estimate)
    )
    metrics["sparse_telemetry_json"] = json.dumps(json_safe(acc.telemetry_rows))
    if acc.nm_total > 0:
        metrics["sparse_nm_compliance"] = acc.nm_compliant / acc.nm_total
    if acc.sparse_active_params_estimate > 0:
        total_weight_params = _total_weight_params(layers)
        if total_weight_params > 0:
            metrics["compression_ratio"] = (
                acc.sparse_active_params_estimate / total_weight_params
            )


_ROUTING_OP_NAMES = {
    "moe_2expert",
    "moe_topk",
    "topk_gate",
    "depth_token_mask",
    "confidence_token_gate",
    "depth_weighted_proj",
    "token_merging",
    "adjacent_token_merge",
    "learned_token_gate",
    "cheap_verify_blend",
    "route_topk",
    "route_lanes",
    "route_recursion",
}


def _infer_compiled_routing_mode(layers: List[Any]) -> Optional[str]:
    for layer in layers:
        for compiled_op in _iter_layer_ops(layer):
            op_obj = getattr(compiled_op, "op", None)
            op_name = getattr(op_obj, "name", "") if op_obj else ""
            if op_name in _ROUTING_OP_NAMES:
                return op_name
    return None


def _finalize_routing_metrics(
    metrics: Dict[str, Any], acc: _ArchitectureTelemetry, routing_mode: Optional[Any]
) -> None:
    if acc.rt_count <= 0:
        return
    metrics["routing_tokens_total"] = acc.rt_tokens_total
    metrics["routing_tokens_processed"] = acc.rt_tokens_processed
    metrics["routing_tokens_skipped"] = max(
        0, acc.rt_tokens_total - acc.rt_tokens_processed
    )
    metrics["routing_utilization_entropy"] = acc.rt_entropy_sum / acc.rt_count
    if acc.rt_tokens_total > 0:
        skipped_ratio = max(0.0, 1.0 - (acc.rt_tokens_processed / acc.rt_tokens_total))
        metrics["routing_drop_rate"] = skipped_ratio
        metrics["routing_savings_ratio"] = skipped_ratio
    if acc.rt_confidence_count > 0:
        conf_mean = acc.rt_confidence_sum / acc.rt_confidence_count
        conf_var = max(
            0.0,
            (acc.rt_confidence_sq_sum / acc.rt_confidence_count)
            - conf_mean * conf_mean,
        )
        metrics["routing_confidence_mean"] = conf_mean
        metrics["routing_confidence_std"] = conf_var**0.5
    if acc.rt_expert_counts is not None:
        metrics["routing_expert_count"] = int(len(acc.rt_expert_counts))
        metrics["routing_expert_utilization_json"] = json.dumps(
            acc.rt_expert_counts.cpu().tolist()
        )
    if acc.rt_keep_count or acc.rt_drop_count:
        denom = max(1, acc.rt_keep_count + acc.rt_drop_count)
        metrics["routing_keep_ratio"] = acc.rt_keep_count / denom
        metrics["routing_drop_ratio"] = acc.rt_drop_count / denom
    if acc.rt_default_path_count > 0 and acc.rt_tokens_total > 0:
        metrics["routing_default_path_fraction"] = (
            acc.rt_default_path_count / acc.rt_tokens_total
        )
    if acc.rt_sparse_span_count > 0:
        metrics["routing_sparse_span_count"] = acc.rt_sparse_span_count
    if acc.rt_sparse_span_width_count > 0:
        metrics["routing_sparse_span_width_mean"] = (
            acc.rt_sparse_span_width_sum / acc.rt_sparse_span_width_count
        )
    if acc.rt_sparse_span_coverage_tokens > 0 and acc.rt_tokens_total > 0:
        metrics["routing_sparse_span_coverage"] = (
            acc.rt_sparse_span_coverage_tokens / acc.rt_tokens_total
        )
    if acc.rt_lane_histogram is not None:
        metrics["routing_lane_utilization_json"] = json.dumps(
            acc.rt_lane_histogram.cpu().tolist()
        )
    if routing_mode:
        metrics["routing_mode"] = routing_mode


def _has_entropy_gate(layers: List[Any]) -> bool:
    for layer in layers:
        for compiled_op in _iter_layer_ops(layer):
            op_obj = getattr(compiled_op, "op", None)
            op_name = getattr(op_obj, "name", "") if op_obj else ""
            if op_name == "token_entropy":
                return True
    return False


def _finalize_adaptive_metrics(
    metrics: Dict[str, Any], acc: _ArchitectureTelemetry, layers: List[Any]
) -> None:
    if acc.at_count > 0:
        metrics["depth_savings_ratio"] = acc.at_savings_sum / acc.at_count
        if acc.at_depth_sum > 0:
            metrics["effective_depth_ratio"] = (
                acc.at_depth_sum / (acc.at_count * len(layers))
                if len(layers) > 0
                else 1.0
            )
    if acc.recursion_count > 0:
        metrics["recursion_savings_ratio"] = (
            acc.recursion_savings_sum / acc.recursion_count
        )
        if acc.recursion_depth_sum > 0:
            avg_max_depth = (
                acc.recursion_max_depth_sum / acc.recursion_count
                if acc.recursion_max_depth_sum > 0
                else None
            )
            if avg_max_depth and avg_max_depth > 0:
                metrics["recursion_depth_ratio"] = acc.recursion_depth_sum / (
                    acc.recursion_count * avg_max_depth
                )


class _ResultsAnalysisMixin:
    """Graph/sandbox metric extraction and post-evaluation callbacks."""

    __slots__ = ()

    def _check_intra_experiment_dedup(self, nb, exp_id: str, fingerprint: str) -> bool:
        """Return True if this (exp_id, fingerprint) pair has already been recorded.

        Lazily loads the per-experiment fingerprint set from the DB on first
        call for a given experiment, then maintains it in memory. Used by
        evolution / novelty / synthesis paths to short-circuit duplicate
        ``record_program_result`` calls — a regression caught downstream
        (slice 3b of the dedup-governance plan).
        """
        if not exp_id or not fingerprint:
            return False
        cache = getattr(self, "_evaluated_fp_by_experiment", None)
        if cache is None:
            cache = {}
            self._evaluated_fp_by_experiment = cache
        seen = cache.get(exp_id)
        if seen is None:
            seen = {
                row[0]
                for row in nb.conn.execute(
                    "SELECT graph_fingerprint FROM program_results_compat "
                    "WHERE experiment_id = ?",
                    (exp_id,),
                ).fetchall()
                if row[0]
            }
            cache[exp_id] = seen
        if fingerprint in seen:
            return True
        seen.add(fingerprint)
        return False

    def _on_program_evaluated(
        self,
        graph,
        fitness,
        sandbox_result,
        s1_result,
        eval_counters,
        nb,
        exp_id,
        model_source="evolution",
        behavioral_fingerprint=None,
        debug: bool = False,
    ):
        """Unified callback for recording results and updating counters during search."""
        eval_counters["total"] += 1
        if fitness > 0:
            eval_counters["s0"] += 1
        if fitness > 0.2:
            eval_counters["s1"] += 1

        try:
            graph_metrics = self._extract_graph_metrics(graph)

            # Extract sandbox metrics if available
            if sandbox_result:
                graph_metrics.update(self._extract_sandbox_metrics(sandbox_result))

            # Extract S1 and architecture telemetry if available
            if s1_result:
                # Basic training metrics
                for k in (
                    "initial_loss",
                    "final_loss",
                    "min_loss",
                    "throughput",
                    "avg_step_time_ms",
                    "total_train_time_ms",
                    "validation_loss",
                    "validation_loss_ratio",
                    "generalization_gap",
                    "discovery_loss",
                    "discovery_loss_ratio",
                ):
                    if k in s1_result:
                        graph_metrics[k] = s1_result[k]
                self._merge_s1_telemetry(graph_metrics, s1_result)

            # Extract behavioral fingerprint metrics if available
            fp_novelty = None
            fp_confidence = 0.2
            if behavioral_fingerprint is not None:
                fp = behavioral_fingerprint
                # Compute proper novelty score relative to existing DB entries,
                # not just fp.novelty_score which is distance-from-references
                try:
                    from ...eval.metrics import novelty_score as compute_novelty

                    nov = compute_novelty(graph, fingerprint=fp)
                    fp_novelty = float(nov.overall_novelty)
                    fp_confidence = float(nov.novelty_confidence)
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.debug("Novelty score computation failed: %s", e)
                    fp_novelty = fp.novelty_score if fp.novelty_score > 0 else None
                fp_fields = {
                    "fingerprint_json": json.dumps(json_safe(fp.to_dict())),
                    "fp_interaction_locality": fp.interaction_locality,
                    "fp_interaction_sparsity": fp.interaction_sparsity,
                    "fp_interaction_symmetry": fp.interaction_symmetry,
                    "fp_interaction_hierarchy": fp.interaction_hierarchy,
                    "fp_intrinsic_dim": fp.intrinsic_dim,
                    "fp_isotropy": fp.isotropy,
                    "fp_rank_ratio": fp.rank_ratio,
                    "fp_jacobian_spectral_norm": fp.jacobian_spectral_norm,
                    "fp_jacobian_effective_rank": fp.jacobian_effective_rank,
                    "fp_sensitivity_uniformity": fp.sensitivity_uniformity,
                    "fp_cka_vs_transformer": fp.cka_vs_transformer,
                    "fp_cka_vs_ssm": fp.cka_vs_ssm,
                    "fp_cka_vs_conv": fp.cka_vs_conv,
                    "fp_hierarchy_fitness": fp.hierarchy_fitness,
                    "fp_gromov_delta": fp.gromov_delta,
                    "cka_source": fp.cka_source,
                    "cka_artifact_version": fp.cka_artifact_version,
                    "cka_probe_protocol_hash": fp.cka_probe_protocol_hash,
                    "cka_reference_quality": fp.cka_reference_quality,
                    "novelty_valid_for_promotion": fp.novelty_valid_for_promotion,
                    "novelty_validity_reason": fp.novelty_validity_reason,
                    "novelty_reference_version": fp.novelty_reference_version,
                }
                graph_metrics.update(fp_fields)

            # Prefer actual loss_ratio from training over synthetic 1.0-fitness
            _actual_lr = None
            _fl = None
            if s1_result:
                _fl = s1_result.get("final_loss")
                _il = s1_result.get("initial_loss")
                if _fl is not None and _il is not None and _il > 0:
                    _actual_lr = _fl / _il
            recorded_lr = (
                _actual_lr
                if _actual_lr is not None
                else (1.0 - fitness if fitness > 0 else None)
            )
            # Remove final_loss from graph_metrics to avoid duplicate kwarg
            graph_metrics.pop("final_loss", None)

            # S1 pass: fitness > 0.2 AND training gate passed.
            # The fitness function caps gate-failed programs at 0.19, so
            # fitness > 0.2 already implies gate passed.  But be explicit:
            # the proxy fitness threshold must align with actual S1 semantics.
            _s1_passed = fitness > 0.2 and (
                s1_result is not None and s1_result.get("passed", False)
            )

            # S0/S0.5 pass: if s1_result exists the model was compiled and
            # trained — it reached S1, so S0 and S0.5 are definitionally passed.
            # Using `fitness > 0` was wrong: fitness=0 when S1 training fails
            # (baseline gate, convergence failure) or the fitness_fn throws,
            # but none of those mean compilation failed.
            _s0_passed = (
                s1_result is not None or graph_metrics.get("stability_score", 0) > 0
            )
            _s05_passed = _s0_passed

            if debug:
                logger.info(
                    "DEBUG record: fp=%s fitness=%.3f s0=%s s1=%s lr=%s bfp=%s",
                    graph.fingerprint()[:16],
                    fitness,
                    _s0_passed,
                    _s1_passed,
                    f"{recorded_lr:.4f}" if recorded_lr is not None else "None",
                    behavioral_fingerprint is not None,
                )

            # Slice 3b: per-experiment fingerprint dedup. The synthesis-time
            # gate at execution_experiment_phase3._dedup_graph_candidates
            # should have prevented us from re-evaluating an already-known
            # fingerprint. If we hit a duplicate here we still skip the
            # write — both to honor the future UNIQUE(graph_fingerprint,
            # experiment_id) index and to surface the regression in logs.
            graph_fp = graph.fingerprint()
            if self._check_intra_experiment_dedup(nb, exp_id, graph_fp):
                eval_counters["skipped_intra_experiment_dedup"] = (
                    eval_counters.get("skipped_intra_experiment_dedup", 0) + 1
                )
                logger.warning(
                    "Dedup regression: fp=%s already recorded under exp=%s — skipping duplicate write (model_source=%s)",
                    graph_fp[:16],
                    str(exp_id)[:12],
                    model_source,
                )
                return

            rid = nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=graph.fingerprint(),
                graph_json=graph_to_json(graph),
                stage1_passed=_s1_passed,
                stage0_passed=_s0_passed,
                stage05_passed=_s05_passed,
                loss_ratio=recorded_lr,
                final_loss=_fl,
                novelty_score=fp_novelty,
                novelty_confidence=fp_confidence,
                stage_at_death=(
                    "survived" if _s1_passed else ("stage1" if _s0_passed else "stage0")
                ),
                model_source=model_source,
                bypass_quality_gate=debug,
                **graph_metrics,
            )
            if debug and not rid:
                logger.warning(
                    "DEBUG: record_program_result returned empty for fp=%s (quality gate still blocked?)",
                    graph.fingerprint()[:16],
                )
            if fitness > 0.2 and rid:
                from ._helpers import _upsert_screening_entry

                _upsert_screening_entry(
                    nb,
                    {
                        "result_id": rid,
                        "model_source": model_source,
                        "graph_fingerprint": graph.fingerprint(),
                        "loss_ratio": recorded_lr,
                        **{
                            k: graph_metrics.get(k)
                            for k in (
                                "fp_jacobian_spectral_norm",
                                "routing_savings_ratio",
                                "activation_sparsity_score",
                                "depth_savings_ratio",
                                "compression_ratio",
                            )
                        },
                    },
                )
        except DuplicateFingerprintError as e:
            eval_counters["skipped_cross_experiment_dedup"] = (
                eval_counters.get("skipped_cross_experiment_dedup", 0) + 1
            )
            logger.warning(
                "Cross-experiment dedup blocked fp=%s during %s eval under exp=%s; existing rid=%s exp=%s",
                e.fingerprint[:16],
                model_source,
                str(exp_id)[:12],
                e.existing_result_id,
                str(e.existing_experiment_id)[:12]
                if e.existing_experiment_id
                else None,
            )
        except (
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            sqlite3.OperationalError,
        ) as e:
            if debug:
                logger.exception(
                    "DEBUG: Failed to record program result for fp=%s",
                    graph.fingerprint()[:16],
                )
            else:
                logger.debug("Failed to record program result: %s", e)

    def _analyze_results(
        self, results: Dict, exp_id: str, nb: LabNotebook, context: str = ""
    ) -> List[str]:
        """Analyze experiment results and record insights for dashboard display.

        Insights are recorded to the DB for dashboard visibility but are NOT
        fed into Aria's persona state or decision-making.  The current insight
        confidence model is not grounded in real predictive accuracy and was
        causing Aria to make counterproductive decisions (excluding good ops,
        shrinking architecture limits).  A future overhaul will replace this
        with Bayesian confidence that improves with data.
        """
        # Try data-driven analytics first
        try:
            from ..analytics import ExperimentAnalytics

            analytics = ExperimentAnalytics(nb)
            structured = analytics.compute_insights()

            recorded = []
            for ins in structured:
                content = ins if isinstance(ins, str) else ins.get("content", "")
                category = (
                    ins.get("category", "pattern")
                    if isinstance(ins, dict)
                    else "pattern"
                )
                confidence = (
                    ins.get("confidence", 0.7) if isinstance(ins, dict) else 0.7
                )
                insight_type = (
                    ins.get("insight_type") if isinstance(ins, dict) else None
                )
                subject_key = ins.get("subject_key") if isinstance(ins, dict) else None
                semantic_key = (
                    ins.get("semantic_key") if isinstance(ins, dict) else None
                )

                nb.record_insight(
                    category,
                    content,
                    exp_id,
                    confidence=confidence,
                    insight_type=insight_type,
                    subject_key=subject_key,
                    semantic_key=semantic_key,
                )
                # Insights are display-only — not fed to Aria's decision state
                recorded.append(content)
            return recorded
        except (ImportError, RuntimeError, sqlite3.OperationalError) as e:
            logger.debug(
                "Data-driven analytics failed, falling back to rule-based: %s", e
            )

        # Fall back to rule-based
        return self._rule_based_insights(results, exp_id, nb)

    def _extract_graph_metrics(self, graph) -> Dict:
        """Extract structural metrics from a computation graph."""
        metrics = {}
        metrics["graph_n_ops"] = graph.n_ops()
        metrics["graph_depth"] = graph.depth()
        metrics["graph_n_params_estimate"] = graph.n_params_estimate()
        metrics["graph_has_gradient_path"] = graph.has_gradient_path()

        # Edge count
        n_edges = sum(len(n.input_ids) for n in graph.nodes.values())
        metrics["graph_n_edges"] = n_edges

        # Unique ops and category histogram
        ops_used = set()
        cat_counts: Dict[str, int] = {}
        uses_math = False
        uses_freq = False
        for node in graph.nodes.values():
            if node.is_input:
                continue
            ops_used.add(node.op_name)
            try:
                from ...synthesis.primitives import get_primitive

                op = get_primitive(node.op_name)
                cat = op.category.value
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                if cat == "math_space":
                    uses_math = True
                if cat == "frequency":
                    uses_freq = True
            except (KeyError, ValueError, ImportError):
                pass

        metrics["graph_n_unique_ops"] = len(ops_used)
        metrics["graph_category_histogram"] = json.dumps(cat_counts)
        metrics["graph_uses_math_spaces"] = uses_math
        metrics["graph_uses_frequency_domain"] = uses_freq

        # Z7: Sparsity Ledger
        sparse_ops = {
            "block_sparse_linear",
            "nm_sparse_linear",
            "semi_structured_2_4_linear",
        }
        dense_ops = {
            "linear_proj",
            "linear_proj_down",
            "linear_proj_up",
            "fused_linear_gelu",
        }
        n_sparse = sum(1 for node in graph.nodes.values() if node.op_name in sparse_ops)
        n_dense = sum(1 for node in graph.nodes.values() if node.op_name in dense_ops)
        total_param_ops = n_sparse + n_dense
        metrics["sparsity_ratio"] = (
            n_sparse / total_param_ops if total_param_ops > 0 else 0.0
        )

        return metrics

    def _extract_sandbox_metrics(self, sandbox_result) -> Dict:
        """Extract ALL fields from a SandboxResult."""
        metrics = {}
        metrics["compile_time_ms"] = sandbox_result.compile_time_ms
        metrics["forward_time_ms"] = sandbox_result.forward_time_ms
        metrics["backward_time_ms"] = sandbox_result.backward_time_ms
        metrics["peak_memory_mb"] = sandbox_result.peak_memory_mb
        metrics["grad_norm"] = sandbox_result.grad_norm
        metrics["stability_score"] = sandbox_result.stability_score
        metrics["extreme_input_passed"] = sandbox_result.extreme_input_passed
        metrics["random_input_passed"] = sandbox_result.random_input_passed
        metrics["has_nan_output"] = sandbox_result.has_nan_output
        metrics["has_inf_output"] = sandbox_result.has_inf_output
        metrics["has_nan_grad"] = sandbox_result.has_nan_grad
        metrics["has_zero_grad"] = sandbox_result.has_zero_grad
        metrics["error_type"] = sandbox_result.error_type
        metrics["error_message"] = sandbox_result.error

        # Activation Sparsity & Heatmaps
        activation_sparsity = getattr(sandbox_result, "activation_sparsity", None)
        dead_neuron_count = getattr(sandbox_result, "dead_neuron_count", None)
        sparsity_report = getattr(sandbox_result, "sparsity_report", None)
        if activation_sparsity is not None:
            metrics["sparsity_ratio"] = activation_sparsity
        if dead_neuron_count is not None:
            metrics["dead_neuron_count"] = dead_neuron_count
        if sparsity_report:
            metrics["sparsity_report_json"] = json.dumps(json_safe(sparsity_report))

        # Parse output_range "[min, max]" string
        if sandbox_result.output_range:
            try:
                parts = sandbox_result.output_range.strip("[]").split(",")
                metrics["output_range_min"] = float(parts[0].strip())
                metrics["output_range_max"] = float(parts[1].strip())
            except (ValueError, IndexError):
                pass

        return metrics

    def _extract_architecture_telemetry(self, model: Optional[nn.Module]) -> Dict:
        """Extract sparse, routing, and adaptive telemetry from compiled layer ops."""
        return self._extract_architecture_telemetry_impl(model)

    def _extract_architecture_telemetry_impl(self, model: Optional[nn.Module]) -> Dict:
        """Extract sparse, routing, and adaptive telemetry from compiled layer ops."""
        if model is None:
            return {}

        metrics: Dict[str, Any] = {}
        layers = _architecture_layers(model)
        routing_mode = _model_routing_mode(model)
        if routing_mode:
            metrics["routing_mode"] = routing_mode
        acc = _ArchitectureTelemetry()
        for layer in layers:
            routing = getattr(layer, "routing", None)
            if routing is not None:
                _record_routing_telemetry(
                    acc, getattr(routing, "routing_telemetry", None)
                )
                _record_adaptive_telemetry(
                    acc, getattr(routing, "adaptive_telemetry", None), routing
                )
            for compiled_op in _iter_layer_ops(layer):
                _record_sparse_telemetry(acc, compiled_op)
                _record_routing_telemetry(
                    acc, getattr(compiled_op, "routing_telemetry", None)
                )
                _record_adaptive_telemetry(
                    acc, getattr(compiled_op, "adaptive_telemetry", None)
                )

        _finalize_sparse_metrics(metrics, acc, layers)
        if not routing_mode and acc.rt_count > 0:
            routing_mode = _infer_compiled_routing_mode(layers)
        if acc.rt_count > 0 and not routing_mode:
            routing_mode = "routed"
        _finalize_routing_metrics(metrics, acc, routing_mode)
        if _has_entropy_gate(layers):
            metrics["has_entropy_gate"] = True
            if "routing_utilization_entropy" not in metrics:
                metrics["routing_utilization_entropy"] = None
        _finalize_adaptive_metrics(metrics, acc, layers)
        return metrics

    @staticmethod
    def _merge_s1_telemetry(
        program_metrics: Dict[str, Any], s1_result: Dict[str, Any]
    ) -> None:
        telemetry_keys = (
            "routing_mode",
            "routing_tokens_total",
            "routing_tokens_processed",
            "routing_tokens_skipped",
            "routing_drop_rate",
            "routing_utilization_entropy",
            "routing_capacity_overflow_count",
            "routing_confidence_mean",
            "routing_confidence_std",
            "routing_expert_utilization_json",
            "routing_expert_count",
            "routing_savings_ratio",
            "compression_ratio",
            "depth_savings_ratio",
            "effective_depth_ratio",
            "recursion_savings_ratio",
            "recursion_depth_ratio",
            "has_entropy_gate",
            "entropy_gate_trajectory_json",
            "routing_collapse_score",
        )
        for key in telemetry_keys:
            if key in s1_result and s1_result.get(key) is not None:
                program_metrics[key] = s1_result.get(key)
