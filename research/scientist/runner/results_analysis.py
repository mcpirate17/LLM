"""Results analysis mixin: graph metrics, sandbox metrics, architecture telemetry."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..notebook import LabNotebook
from ...synthesis.serializer import graph_to_json

logger = logging.getLogger(__name__)


class _ResultsAnalysisMixin:
    """Graph/sandbox metric extraction and post-evaluation callbacks."""

    __slots__ = ()

    def _on_program_evaluated(self, graph, fitness, sandbox_result, s1_result,
                              eval_counters, nb, exp_id, model_source="evolution",
                              behavioral_fingerprint=None):
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
                for k in ("initial_loss", "final_loss", "min_loss", "throughput",
                          "avg_step_time_ms", "total_train_time_ms",
                          "validation_loss", "validation_loss_ratio", "generalization_gap",
                          "discovery_loss", "discovery_loss_ratio"):
                    if k in s1_result: graph_metrics[k] = s1_result[k]
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
                except Exception:
                    fp_novelty = fp.novelty_score if fp.novelty_score > 0 else None
                fp_fields = {
                    "fingerprint_json": json.dumps(fp.to_dict()),
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

            rid = nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=graph.fingerprint(),
                graph_json=graph_to_json(graph),
                stage1_passed=fitness > 0.2,
                stage0_passed=fitness > 0,
                stage05_passed=fitness > 0,
                loss_ratio=1.0 - fitness if fitness > 0 else None,
                novelty_score=fp_novelty,
                novelty_confidence=fp_confidence,
                stage_at_death="survived" if fitness > 0.2 else "stage1",
                model_source=model_source,
                **graph_metrics,
            )
            if fitness > 0.2 and rid:
                from ._helpers import _upsert_screening_entry
                _upsert_screening_entry(nb, {
                    "result_id": rid,
                    "model_source": model_source,
                    "graph_fingerprint": graph.fingerprint(),
                    "loss_ratio": 1.0 - fitness if fitness > 0 else None,
                    **{k: graph_metrics.get(k) for k in (
                        "fp_jacobian_spectral_norm", "routing_savings_ratio",
                        "activation_sparsity_score", "depth_savings_ratio",
                        "compression_ratio",
                    )},
                })
        except Exception as e:
            logger.debug("Failed to record program result: %s", e)

    def _analyze_results(self, results: Dict, exp_id: str,
                         nb: LabNotebook, context: str = "") -> List[str]:
        """Analyze experiment results and generate insights."""
        # Try data-driven analytics first
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            structured = analytics.compute_insights()

            recorded = []
            for ins in structured:
                content = ins if isinstance(ins, str) else ins.get("content", "")
                category = ins.get("category", "pattern") if isinstance(ins, dict) else "pattern"
                confidence = ins.get("confidence", 0.7) if isinstance(ins, dict) else 0.7
                insight_type = ins.get("insight_type") if isinstance(ins, dict) else None
                subject_key = ins.get("subject_key") if isinstance(ins, dict) else None
                semantic_key = ins.get("semantic_key") if isinstance(ins, dict) else None

                nb.record_insight(
                    category,
                    content,
                    exp_id,
                    confidence=confidence,
                    insight_type=insight_type,
                    subject_key=subject_key,
                    semantic_key=semantic_key,
                )
                self.aria.add_insight(content)
                recorded.append(content)
            return recorded
        except Exception:
            pass

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
            except (KeyError, Exception):
                pass

        metrics["graph_n_unique_ops"] = len(ops_used)
        metrics["graph_category_histogram"] = json.dumps(cat_counts)
        metrics["graph_uses_math_spaces"] = uses_math
        metrics["graph_uses_frequency_domain"] = uses_freq

        # Z7: Sparsity Ledger
        sparse_ops = {"block_sparse_linear", "nm_sparse_linear", "semi_structured_2_4_linear"}
        dense_ops = {"linear_proj", "linear_proj_down", "linear_proj_up", "fused_linear_gelu"}
        n_sparse = sum(1 for node in graph.nodes.values() if node.op_name in sparse_ops)
        n_dense = sum(1 for node in graph.nodes.values() if node.op_name in dense_ops)
        total_param_ops = n_sparse + n_dense
        metrics["sparsity_ratio"] = n_sparse / total_param_ops if total_param_ops > 0 else 0.0

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
            metrics["sparsity_report_json"] = json.dumps(sparsity_report)

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
        if model is None:
            return {}

        metrics: Dict[str, Any] = {}
        try:
            layers = list(getattr(model, "layers", []) or [])
        except Exception:
            layers = []
        if not layers:
            try:
                topo = getattr(model, "topology", None)
                blocks = getattr(topo, "blocks", None) if topo is not None else None
                if blocks is not None:
                    layers = list(blocks)
            except Exception:
                pass
        routing_mode = None
        spec = getattr(model, "spec", None)
        if spec is not None:
            choices = getattr(spec, "choices", None)
            if isinstance(choices, dict):
                routing_mode = choices.get("compute_routing")
        if routing_mode:
            metrics["routing_mode"] = routing_mode

        # 1. Sparse Telemetry
        telemetry_rows: List[Dict[str, Any]] = []
        total_calls = 0
        total_fallback_calls = 0
        kernel_fallback_calls = 0
        density_sum = 0.0
        density_last_values: List[float] = []
        nm_compliant = 0
        nm_total = 0
        sparse_active_params_estimate = 0.0

        # 2. Routing Telemetry (MoE)
        rt_tokens_total = 0
        rt_tokens_processed = 0
        rt_entropy_sum = 0.0
        rt_count = 0
        rt_expert_counts: Optional[torch.Tensor] = None

        # 3. Adaptive Telemetry (MoD/MoR)
        at_savings_sum = 0.0
        at_depth_sum = 0.0
        at_count = 0
        recursion_savings_sum = 0.0
        recursion_depth_sum = 0.0
        recursion_count = 0
        recursion_max_depth_sum = 0.0

        for layer in layers:
            # Check for routing/adaptive telemetry on the layer/routing itself (arch_builder style)
            routing = getattr(layer, "routing", None)
            if routing is not None:
                # Routing (MoE)
                rt = getattr(routing, "routing_telemetry", None)
                if isinstance(rt, dict):
                    rt_tokens_total += rt.get("tokens_total", 0)
                    rt_tokens_processed += rt.get("tokens_processed", 0)
                    rt_entropy_sum += rt.get("entropy_sum", 0.0)
                    rt_count += rt.get("count", 0)
                    ec = rt.get("expert_counts")
                    if isinstance(ec, torch.Tensor):
                        if rt_expert_counts is None: rt_expert_counts = ec.clone()
                        else: rt_expert_counts += ec

                # Adaptive (MoD/MoR)
                at = getattr(routing, "adaptive_telemetry", None)
                if isinstance(at, dict):
                    at_savings_sum += at.get("savings_sum", 0.0)
                    at_depth_sum += at.get("depth_sum", 0.0)
                    at_count += at.get("count", 0)
                    if routing.__class__.__name__ == "AdaptiveRecursionRouting":
                        recursion_savings_sum += at.get("savings_sum", 0.0)
                        recursion_depth_sum += at.get("depth_sum", 0.0)
                        recursion_count += at.get("count", 0)
                        recursion_max_depth_sum += float(getattr(routing, "max_depth", 0)) * at.get("count", 0)

            # Check for op-level telemetry (compiler style)
            ops = getattr(layer, "ops", None)
            if ops is None:
                continue
            op_values = None
            if isinstance(ops, dict):
                op_values = list(ops.values())
            else:
                try:
                    op_values = list(ops)
                except Exception:
                    # Guard against non-iterable op containers
                    continue
            for compiled_op in op_values:
                # Sparse
                sparse_telemetry = getattr(compiled_op, "sparse_telemetry", None)
                if sparse_telemetry:
                    has_weight = hasattr(compiled_op, "weight")
                    weight_params = float(compiled_op.weight.numel()) if has_weight else 0.0
                    for op_name, stats in sparse_telemetry.items():
                        calls = int(stats.get("calls", 0) or 0)
                        total_calls += calls
                        total_fallback_calls += int(stats.get("fallback_calls", 0) or 0)
                        density_sum += float(stats.get("density_sum", 0.0) or 0.0)
                        last_density = float(stats.get("last_density", 1.0) or 1.0)
                        density_last_values.append(last_density)
                        if stats.get("last_fallback_reason") == "kernel_unavailable":
                            kernel_fallback_calls += int(stats.get("fallback_calls", 0) or 0)
                        if op_name in ("nm_sparse_linear", "semi_structured_2_4_linear"):
                            nm_total += 1
                            if last_density <= 0.51: nm_compliant += 1
                        if weight_params > 0.0:
                            density_for_params = (float(stats.get("density_sum", 0.0)) / calls) if calls > 0 else last_density
                            sparse_active_params_estimate += weight_params * density_for_params
                        telemetry_rows.append({"op_name": op_name, "calls": calls, "last_density": last_density})

                # Routing (MoE)
                rt = getattr(compiled_op, "routing_telemetry", None)
                if isinstance(rt, dict):
                    rt_tokens_total += rt.get("tokens_total", 0)
                    rt_tokens_processed += rt.get("tokens_processed", 0)
                    rt_entropy_sum += rt.get("entropy_sum", 0.0)
                    rt_count += rt.get("count", 0)
                    ec = rt.get("expert_counts")
                    if isinstance(ec, torch.Tensor):
                        if rt_expert_counts is None: rt_expert_counts = ec.clone()
                        else: rt_expert_counts += ec

                # Adaptive
                at = getattr(compiled_op, "adaptive_telemetry", None)
                if isinstance(at, dict):
                    at_savings_sum += at.get("savings_sum", 0.0)
                    at_depth_sum += at.get("depth_sum", 0.0)
                    at_count += at.get("count", 0)

        # Finalize Sparse
        if total_calls > 0:
            metrics["sparse_density_mean"] = density_sum / max(total_calls, 1)
            metrics["sparse_density_last"] = sum(density_last_values) / max(len(density_last_values), 1)
            metrics["sparse_fallback_calls"] = total_fallback_calls
            metrics["sparse_kernel_fallback_calls"] = kernel_fallback_calls
            metrics["sparse_active_params_estimate"] = int(max(0.0, sparse_active_params_estimate))
            metrics["sparse_telemetry_json"] = json.dumps(telemetry_rows)
            if nm_total > 0: metrics["sparse_nm_compliance"] = nm_compliant / nm_total
            # Compression ratio = effective params / dense params
            if sparse_active_params_estimate > 0:
                total_weight_params = 0.0
                for layer in layers:
                    ops = getattr(layer, "ops", None)
                    if ops is None:
                        continue
                    if isinstance(ops, dict):
                        op_values = ops.values()
                    else:
                        try:
                            op_values = list(ops)
                        except Exception:
                            continue
                    for op in op_values:
                        if hasattr(op, "weight"):
                            total_weight_params += float(getattr(op, "weight", torch.empty(0)).numel())
                if total_weight_params > 0:
                    metrics["compression_ratio"] = sparse_active_params_estimate / total_weight_params

        # Infer routing_mode from compiled ops if not already set
        if not routing_mode and rt_count > 0:
            for layer in layers:
                ops = getattr(layer, "ops", None)
                if ops is None:
                    continue
                if isinstance(ops, dict):
                    op_values = list(ops.values())
                else:
                    try:
                        op_values = list(ops)
                    except Exception:
                        continue
                for compiled_op in op_values:
                    op_obj = getattr(compiled_op, "op", None)
                    op_name = getattr(op_obj, "name", "") if op_obj else ""
                    if op_name == "moe_2expert":
                        routing_mode = "moe_2expert"
                        break
                    elif op_name == "moe_topk":
                        routing_mode = "moe_topk"
                        break
                    elif op_name == "topk_gate":
                        routing_mode = "topk_gate"
                        break
                    elif op_name in {
                        "mod_topk", "early_exit", "adaptive_recursion",
                        "token_merging", "token_merge", "cascade",
                        "speculative", "route_topk", "route_lanes", "route_recursion",
                    }:
                        routing_mode = op_name
                        break
                if routing_mode:
                    break
            if routing_mode:
                metrics["routing_mode"] = routing_mode
        if rt_count > 0 and not routing_mode:
            routing_mode = "routed"
            metrics["routing_mode"] = routing_mode

        # Finalize Routing
        if rt_count > 0:
            metrics["routing_tokens_total"] = rt_tokens_total
            metrics["routing_tokens_processed"] = rt_tokens_processed
            metrics["routing_utilization_entropy"] = rt_entropy_sum / rt_count
            if rt_tokens_total > 0:
                metrics["routing_drop_rate"] = max(0.0, 1.0 - (rt_tokens_processed / rt_tokens_total))
                # savings = fraction of tokens NOT processed (routed away)
                metrics["routing_savings_ratio"] = max(0.0, 1.0 - (rt_tokens_processed / rt_tokens_total))
            if rt_expert_counts is not None:
                metrics["routing_expert_count"] = int(len(rt_expert_counts))
                metrics["routing_expert_utilization_json"] = json.dumps(rt_expert_counts.cpu().tolist())

        # Finalize Adaptive
        if at_count > 0:
            metrics["depth_savings_ratio"] = at_savings_sum / at_count
            if at_depth_sum > 0:
                metrics["effective_depth_ratio"] = at_depth_sum / (at_count * len(layers)) if len(layers) > 0 else 1.0
        if recursion_count > 0:
            metrics["recursion_savings_ratio"] = recursion_savings_sum / recursion_count
            if recursion_depth_sum > 0:
                avg_max_depth = recursion_max_depth_sum / recursion_count if recursion_max_depth_sum > 0 else None
                if avg_max_depth and avg_max_depth > 0:
                    metrics["recursion_depth_ratio"] = recursion_depth_sum / (recursion_count * avg_max_depth)

        return metrics

    @staticmethod
    def _merge_s1_telemetry(program_metrics: Dict[str, Any], s1_result: Dict[str, Any]) -> None:
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
        )
        for key in telemetry_keys:
            if key in s1_result and s1_result.get(key) is not None:
                program_metrics[key] = s1_result.get(key)
