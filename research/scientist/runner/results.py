"""
Experiment Runner

The autonomous experiment execution engine. Aria uses this to:
1. Generate batches of synthesized programs
2. Evaluate them through the funnel
3. Record results in the lab notebook
4. Analyze patterns and formulate new hypotheses
5. Adjust strategy based on outcomes

Supports background execution controlled from the dashboard.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from ..native_runner import reset_native_runner_telemetry
from ...synthesis.serializer import graph_to_json
from ...synthesis.primitives import get_primitive, PROTECTED_OPS
from ..notebook import LabNotebook, ExperimentEntry
from ..evidence import (
    build_evidence_pack,
)
from ..preregistration import (
    HypothesisPreregistration,
    PreregistrationError,
    validate_preregistration,
)
from ..llm.context import (
    build_knowledge_extraction_context,
    build_campaign_formulation_context,
)
from ..llm.decision import NextExperimentDecisionPlanner

import logging
logger = logging.getLogger(__name__)

from ._types import RunConfig
from .results_auto_escalate_phase7 import _ResultsAutoEscalatePhase7Mixin

class _ResultsMixin(_ResultsAutoEscalatePhase7Mixin):
    """Auto-escalation, scoring, analysis, recommendations."""

    def _auto_escalate(self, results: Dict, config: RunConfig,
                       nb: LabNotebook, phase: str = "screening"):
        """Auto-escalate candidates through the research pipeline.

        Called after screening or investigation completes.
        """
        if phase == "screening" or phase == "experiment":
            self._auto_escalate_screening(results, config, nb)
        elif phase == "investigation":
            self._auto_escalate_investigation(results, config, nb)

    def _on_program_evaluated(self, graph, fitness, sandbox_result, s1_result, 
                              eval_counters, nb, exp_id, model_source="evolution"):
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

            nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=graph.fingerprint(),
                graph_json=graph_to_json(graph),
                stage1_passed=fitness > 0.2,
                stage0_passed=fitness > 0,
                stage05_passed=fitness > 0,
                loss_ratio=1.0 - fitness if fitness > 0 else None,
                novelty_score=None,
                novelty_confidence=0.2,
                stage_at_death="survived" if fitness > 0.2 else "stage1",
                model_source=model_source,
                **graph_metrics,
            )
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
                metrics["routing_savings_ratio"] = rt_tokens_processed / rt_tokens_total
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

    def _auto_recommend(self, results: Dict, config: RunConfig,
                        hypothesis: str, nb: LabNotebook):
        """Auto-generate a recommendation after experiment completion and APPLY it."""
        try:
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            _analytics = self._gather_analytics_data(nb)
            op_rates = _analytics.get("op_success_rates")
            comp_cov = _analytics.get("compression_coverage")
            heuristic = self.aria.suggest_experiment(
                context, op_success_rates=op_rates,
                compression_coverage=comp_cov) or {}
            summary_payload = self._build_next_experiment_summary(nb, results)
            planner = NextExperimentDecisionPlanner.from_run_config(config)
            plan = planner.propose_plan(
                summary_payload,
                current_cost_dollars=float(self.aria.total_cost or 0.0),
                fallback_plan=heuristic,
            )
            suggestion = {
                "mode": plan.get("mode", heuristic.get("mode", "synthesis")),
                "reasoning": plan.get("reasoning", heuristic.get("reasoning", "")),
                "confidence": float(plan.get("confidence", heuristic.get("confidence", 0.5)) or 0.5),
                "config": plan.get("config", heuristic.get("config", {})),
                "planner": plan.get("planner", {}),
                "guardrails": plan.get("guardrails", {}),
                "summary_excerpt": plan.get("summary_excerpt", {}),
            }
            if suggestion:
                evidence_pack = build_evidence_pack(
                    nb,
                    analytics=None,
                    recommendation=suggestion,
                    decision_type="experiment_recommendation",
                )
                suggestion["evidence_pack"] = evidence_pack
                with self._lock:
                    self._last_recommendation = suggestion
                self._emit_event("aria_recommendation", {
                    "mode": suggestion.get("mode"),
                    "reasoning": suggestion.get("reasoning", ""),
                    "confidence": suggestion.get("confidence", 0),
                    "config": suggestion.get("config", {}),
                    "planner": suggestion.get("planner", {}),
                    "evidence_pack": evidence_pack,
                })
                # Store as notebook entry
                nb.add_entry(ExperimentEntry(
                    entry_type="decision",
                    title="Aria's Next Experiment Recommendation",
                    content=suggestion.get("reasoning", ""),
                    metadata={
                        "mode": suggestion.get("mode"),
                        "confidence": suggestion.get("confidence", 0),
                        "suggested_config": suggestion.get("config", {}),
                        "planner": suggestion.get("planner", {}),
                        "guardrails": suggestion.get("guardrails", {}),
                        "summary_payload": summary_payload,
                        "evidence_pack": evidence_pack,
                    },
                ))
                nb.record_decision(
                    campaign_id=self._active_campaign_id,
                    decision_type="next_experiment_plan",
                    subject=f"experiment:{summary_payload.get('recent_experiment_id') or 'latest'}",
                    rationale=suggestion.get("reasoning", ""),
                    alternatives=[{
                        "heuristic_fallback": heuristic,
                    }],
                    evidence_pack={
                        "mode": suggestion.get("mode"),
                        "confidence": suggestion.get("confidence", 0),
                        "config": suggestion.get("config", {}),
                        "planner": suggestion.get("planner", {}),
                        "guardrails": suggestion.get("guardrails", {}),
                        "summary_payload": summary_payload,
                    },
                )
                # PROACTIVE: Apply suggested config/grammar changes immediately
                self._apply_recommendation(suggestion, nb)
        except Exception as e:
            logger.debug(f"Auto-recommendation failed: {e}")

    def _apply_recommendation(self, suggestion: Dict, nb: LabNotebook):
        """Proactively apply Aria's recommended config and grammar changes.

        Also detects code-level issues in reasoning and spawns repair agents.
        """
        if not suggestion.get("evidence_pack"):
            logger.warning("Skipping recommendation application: missing Evidence Pack.")
            return
        confidence = suggestion.get("confidence", 0)
        reasoning = str(suggestion.get("reasoning") or "")

        # Detect code-level issues in reasoning and spawn agent
        if confidence >= 0.3 and reasoning:
            self._maybe_spawn_agent_from_reasoning(reasoning, nb)

        if confidence < 0.4:
            return  # Low confidence — don't auto-apply config

        suggested_config = suggestion.get("config") or {}
        if not suggested_config:
            return

        # Categorize suggested keys into bins
        GRAMMAR_WEIGHT_KEYS = {"math_space_weight"}
        CATEGORY_WEIGHT_KEY = "category_weights"
        CONFIG_OVERRIDE_KEYS = {
            "n_programs", "model_dim", "max_depth", "max_ops",
            "model_source", "morph_focus_sparse",
            "use_synthesized_training", "novelty_weight",
            "selection_family_bonus_weight", "refinement_top_k",
            "refinement_generations", "refinement_budget_programs",
            "grammar_split_prob", "grammar_merge_prob",
            "grammar_risky_op_prob", "grammar_freq_domain_prob",
            "structured_sparsity_bias", "residual_prob",
            "optimizer_preference",
        }

        # Sanity clamps for numeric config values
        CLAMP_RANGES: Dict[str, Tuple[float, float]] = {
            "grammar_split_prob": (0.0, 1.0),
            "grammar_merge_prob": (0.0, 1.0),
            "grammar_risky_op_prob": (0.0, 1.0),
            "grammar_freq_domain_prob": (0.0, 1.0),
            "structured_sparsity_bias": (0.0, 1.0),
            "residual_prob": (0.0, 1.0),
            "n_programs": (4, 500),
            "max_depth": (2, 30),
            "max_ops": (3, 40),
            "model_dim": (32, 1024),
        }
        GRAMMAR_WEIGHT_CLAMP = (0.1, 10.0)  # category weights & math_space_weight
        OP_WEIGHT_CLAMP = (0.01, 10.0)

        def _clamp(val: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, val))

        grammar_overrides = {}
        config_overrides = {}
        for k, v in suggested_config.items():
            if k in GRAMMAR_WEIGHT_KEYS:
                if isinstance(v, (int, float)):
                    grammar_overrides[k] = _clamp(float(v), *GRAMMAR_WEIGHT_CLAMP)
            elif k == CATEGORY_WEIGHT_KEY and isinstance(v, dict):
                # Category weights dict → merge into grammar weight overrides
                for cat_name, weight in v.items():
                    if isinstance(weight, (int, float)):
                        grammar_overrides[cat_name] = _clamp(float(weight), *GRAMMAR_WEIGHT_CLAMP)
            elif k == "excluded_ops" and isinstance(v, list):
                new_excluded = {str(op) for op in v if isinstance(op, str)}
                # Strip protected ops — they must never be hard-excluded
                protected_stripped = new_excluded & PROTECTED_OPS
                new_excluded -= PROTECTED_OPS
                if protected_stripped:
                    logger.info("Blocked Aria from excluding protected ops: %s", sorted(protected_stripped))
                if new_excluded:
                    self._excluded_ops_overrides |= new_excluded
                    nb.log_learning_event(
                        "auto_excluded_ops",
                        f"Aria excluded ops: {sorted(new_excluded)}",
                        excluded_ops=sorted(new_excluded),
                    )
                    logger.info("Aria auto-excluded ops: %s", sorted(new_excluded))
            elif k == "op_weights" and isinstance(v, dict):
                new_op_weights = {
                    str(op): _clamp(float(w), *OP_WEIGHT_CLAMP)
                    for op, w in v.items()
                    if isinstance(op, str) and isinstance(w, (int, float))
                }
                if new_op_weights:
                    self._op_weights_overrides.update(new_op_weights)
                    nb.log_learning_event(
                        "auto_op_weights",
                        f"Aria adjusted op weights: {new_op_weights}",
                        op_weights=new_op_weights,
                    )
                    logger.info("Aria auto-applied op weights: %s", new_op_weights)
            elif k in CONFIG_OVERRIDE_KEYS:
                if k in CLAMP_RANGES and isinstance(v, (int, float)):
                    lo, hi = CLAMP_RANGES[k]
                    v = type(v)(max(lo, min(hi, v)))
                config_overrides[k] = v

        if grammar_overrides:
            self._grammar_weight_overrides.update(grammar_overrides)
            nb.log_learning_event(
                "auto_grammar_adjusted",
                f"Aria proactively adjusted grammar weights: {grammar_overrides}",
                weights=grammar_overrides,
            )
            logger.info("Aria auto-applied grammar overrides: %s", grammar_overrides)

        if config_overrides:
            self._last_chat_config_overrides = {
                **(self._last_chat_config_overrides or {}),
                **config_overrides,
            }
            nb.log_learning_event(
                "auto_config_adjusted",
                f"Aria proactively adjusted config: {config_overrides}",
                changes=config_overrides,
            )
            logger.info("Aria auto-applied config overrides: %s", config_overrides)

    def _maybe_auto_report(self, config: RunConfig, nb: LabNotebook,
                            reason: str = "session_end"):
        """Auto-generate and store a research report."""
        if not config.auto_report:
            return

        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            report_data = {
                "summary": nb.get_dashboard_summary(),
                "top_programs": nb.get_top_programs(20, sort_by="loss_ratio"),
                "recent_experiments": nb.get_recent_experiments(100),
                "op_success_rates": analytics.op_success_rates(),
                "structural_correlations": analytics.structural_correlations(),
                "failure_patterns": analytics.failure_patterns(),
                "top_op_combinations": analytics.top_op_combinations(10),
                "efficiency_frontier": analytics.efficiency_frontier(),
                "efficiency_frontier_3d": analytics.efficiency_frontier_3d(),
                "grammar_weights": analytics.compute_grammar_weights() or {},
                "default_weights": analytics.get_current_grammar_weights(),
            }

            narrative = self.aria.generate_report_narrative(report_data)

            nb.add_entry(ExperimentEntry(
                entry_type="report",
                title=f"Research Report ({reason})",
                content=narrative,
                metadata={
                    "trigger": reason,
                    "total_experiments": report_data["summary"].get("total_experiments", 0),
                    "stage1_survivors": report_data["summary"].get("stage1_survivors", 0),
                },
            ))

            # Save as markdown file for human/LLM consumption
            nb.save_report_markdown(narrative, reason, report_data["summary"])

            self._emit_event("auto_report_generated", {
                "reason": reason,
                "narrative_length": len(narrative),
                "summary": report_data["summary"],
            })

            logger.info(f"Auto-report generated ({reason}): {len(narrative)} chars")
        except Exception as e:
            logger.warning(f"Auto-report generation failed: {e}")

    def _maybe_auto_scale_up(self, results: Dict, config: RunConfig,
                              nb: LabNotebook):
        """Check if we should auto-trigger scale-up after an experiment.

        Criteria:
        1. auto_scale_up is enabled in config
        2. Enough S1 survivors (>= auto_scale_up_min_survivors)
        3. Survivors have sufficient novelty (>= auto_scale_up_min_novelty avg)
        4. Not already a scale_up experiment (avoid recursion)
        5. No experiment currently running
        """
        if not config.auto_scale_up:
            return
        if config.scale_up:
            return  # don't chain scale-ups

        survivors = results.get("survivors", [])
        s1_count = results.get("stage1_passed", 0)

        if s1_count < config.auto_scale_up_min_survivors:
            return

        # Check novelty
        if survivors:
            valid_survivors = [
                s for s in survivors if s.get("novelty_valid_for_promotion", True)
            ]
            if not valid_survivors:
                return
            avg_novelty = (
                sum(s.get("novelty", 0) for s in valid_survivors)
                / len(valid_survivors)
            )
            if avg_novelty < config.auto_scale_up_min_novelty:
                return

        # Select top programs by loss ratio
        top_programs = nb.get_top_programs(
            config.auto_scale_up_top_n, sort_by="loss_ratio")
        result_ids = [
            p["result_id"] for p in top_programs
            if p.get("stage1_passed")
        ][:config.auto_scale_up_top_n]

        if not result_ids:
            return

        logger.info(
            f"Auto-scale-up triggered: {len(result_ids)} programs qualify "
            f"(s1={s1_count}, survivors={len(survivors)})"
        )

        # Store the intent — can't start immediately since thread is still
        # running. Schedule via a flag the main thread can pick up.
        self._pending_scale_up = {
            "result_ids": result_ids,
            "config": config,
            "hypothesis": (
                f"Auto-scale-up: validating top {len(result_ids)} performers "
                f"at {config.scale_up_steps} steps to confirm they work at scale."
            ),
        }
        evidence_pack = build_evidence_pack(
            nb,
            analytics=None,
            recommendation={"mode": "scale_up"},
            decision_type="auto_scale_up",
        )
        self._pending_scale_up["evidence_pack"] = evidence_pack

        self._emit_event("auto_scale_up_queued", {
            "result_ids": result_ids,
            "n_programs": len(result_ids),
            "reason": f"{s1_count} S1 survivors with avg novelty >= {config.auto_scale_up_min_novelty}",
            "evidence_pack": evidence_pack,
        })

        nb.add_entry(ExperimentEntry(
            entry_type="decision",
            title="Auto-Scale-Up Triggered",
            content=(
                f"Automatically queuing scale-up validation for {len(result_ids)} "
                f"top performers. Criteria met: {s1_count} S1 survivors."
            ),
            metadata={"result_ids": result_ids, "evidence_pack": evidence_pack},
        ))

    def _run_pending_scale_up(self):
        """Launch pending auto-scale-up, auto-investigation, or auto-validation."""
        # Check investigation first (higher priority)
        self._run_pending_investigation()
        if self.is_running:
            return

        # Then validation
        self._run_pending_validation()
        if self.is_running:
            return

        # Then scale-up
        pending = getattr(self, "_pending_scale_up", None)
        if pending is None:
            return
        self._pending_scale_up = None

        if self.is_running:
            return  # something else started, skip

        try:
            self.start_scale_up(
                result_ids=pending["result_ids"],
                config=pending["config"],
                hypothesis=pending["hypothesis"],
            )
        except Exception as e:
            logger.warning(f"Failed to launch auto-scale-up: {e}")

    # ── Model Source Abstraction ──

    def _maybe_evaluate_campaign(self, config: RunConfig, nb: LabNotebook) -> None:
        """Evaluate campaign success criteria after an experiment.

        Auto-completes the campaign if criteria are met or the campaign is
        stale (10+ experiments with no criteria passing).  When a campaign
        completes, a successor campaign is formulated based on pipeline state.
        """
        if not config.enable_campaigns or not self._active_campaign_id:
            return

        try:
            evaluation = nb.evaluate_campaign_criteria(self._active_campaign_id)

            if not evaluation["all_met"] and not evaluation["stale"]:
                return  # still in progress

            campaign = nb.get_campaign(self._active_campaign_id)
            if not campaign or campaign.get("status") != "active":
                return

            # --- Complete the campaign ---
            if evaluation["all_met"]:
                reason = "criteria_met"
                findings = (
                    f"All {evaluation['n_criteria']} success criteria met. "
                    f"{evaluation['n_passing']} criteria passing."
                )
            else:
                reason = "stale"
                findings = (
                    f"Campaign stale after {len(nb.get_campaign_experiments(self._active_campaign_id))} "
                    f"experiments: {evaluation['n_at_risk']} criteria at risk, "
                    f"{evaluation['n_passing']} passing."
                )

            nb.update_campaign(
                self._active_campaign_id,
                status="completed",
                completed_at=time.time(),
                completion_reason=reason,
                findings_summary=findings,
            )

            self._emit_event("campaign_completed", {
                "campaign_id": self._active_campaign_id,
                "title": campaign.get("title", ""),
                "reason": reason,
                "findings": findings,
            })
            logger.info(
                f"Campaign completed ({reason}): "
                f"{campaign.get('title', '')} ({self._active_campaign_id})"
            )

            # --- Formulate successor campaign ---
            completed_id = self._active_campaign_id
            self._active_campaign_id = None

            # Determine next focus from pipeline state
            leaderboard_rows = nb.conn.execute(
                "SELECT tier, COUNT(*) as cnt FROM leaderboard GROUP BY tier"
            ).fetchall()
            tiers = {r["tier"]: r["cnt"] for r in leaderboard_rows}

            recent = nb.get_recent_experiments(10)
            knowledge = nb.get_knowledge()
            all_campaigns = nb.conn.execute(
                "SELECT * FROM campaigns ORDER BY timestamp DESC LIMIT 5"
            ).fetchall()
            previous = [dict(r) for r in all_campaigns]

            # Build context that includes pipeline state for Aria
            from ..llm.context import build_campaign_formulation_context
            context = build_campaign_formulation_context(
                recent_experiments=recent,
                knowledge=knowledge,
                previous_campaigns=previous,
            )
            pipeline_hint = (
                f"\n\nPipeline state: "
                f"{tiers.get('screening', 0)} screening, "
                f"{tiers.get('investigation', 0)} investigation, "
                f"{tiers.get('validation', 0)} validation, "
                f"{tiers.get('breakthrough', 0)} breakthrough. "
            )
            if reason == "criteria_met":
                pipeline_hint += (
                    "Previous campaign succeeded — evolve to a more ambitious "
                    "objective (deeper investigation, validation, or scale-up)."
                )
            else:
                pipeline_hint += (
                    "Previous campaign stalled — pivot to a different approach "
                    "(novelty search, different architecture families, or "
                    "relaxed criteria)."
                )

            camp_data = self.aria.formulate_campaign(
                context=context + pipeline_hint
            )

            # Rule-based fallback: evolve based on pipeline state
            if camp_data["title"] == "Architecture Discovery Campaign":
                camp_data = self._pipeline_driven_campaign(tiers, reason)

            successor_id = nb.create_campaign(
                title=camp_data["title"],
                objective=camp_data["objective"],
                success_criteria=camp_data["success_criteria"],
                parent_id=completed_id,
            )

            # Link successor to completed campaign
            nb.update_campaign(
                completed_id,
                successor_campaign_id=successor_id,
            )

            self._active_campaign_id = successor_id
            self._emit_event("campaign_created", {
                "campaign_id": successor_id,
                "title": camp_data["title"],
                "objective": camp_data["objective"],
                "predecessor": completed_id,
            })
            logger.info(
                f"Successor campaign: {camp_data['title']} ({successor_id}) "
                f"→ replacing {completed_id}"
            )

        except Exception as e:
            logger.debug(f"Campaign evaluation failed: {e}")

    def _maybe_extract_knowledge(self, config: RunConfig, nb: LabNotebook,
                                  n_experiments: int) -> None:
        """Extract knowledge every N experiments."""
        if not config.enable_campaigns:
            return
        if n_experiments <= 0 or n_experiments % config.knowledge_extraction_interval != 0:
            return

        try:
            allowed_categories = {
                "principle",
                "anti_pattern",
                "sweet_spot",
                "correlation",
                "tool_insight",
            }

            def _normalize_category(raw: str) -> str:
                value = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
                aliases = {
                    "anti_pattern": "anti_pattern",
                    "anti_patterns": "anti_pattern",
                    "antipattern": "anti_pattern",
                    "anti-pattern": "anti_pattern",
                    "principles": "principle",
                    "sweetspot": "sweet_spot",
                    "sweet_spot": "sweet_spot",
                    "tool": "tool_insight",
                    "toolinsight": "tool_insight",
                    "tool_insights": "tool_insight",
                }
                value = aliases.get(value, value)
                return value if value in allowed_categories else "principle"

            def _canonical_text(raw: str) -> str:
                text = " ".join(str(raw or "").split()).strip().lower()
                text = re.sub(r"\b\d+(?:\.\d+)?%?\b", "#", text)
                text = re.sub(r"[^a-z0-9#\s]+", " ", text)
                return re.sub(r"\s+", " ", text).strip()

            stopwords = {
                "the", "and", "for", "that", "with", "this", "from", "into", "when", "then", "than", "were",
                "been", "have", "has", "had", "are", "was", "show", "shows", "showed", "over", "under",
                "across", "between", "using", "use", "used", "high", "low", "very", "more", "less", "near",
                "around", "recent", "experiments", "experiment", "result", "results", "indicate", "indicates",
                "suggest", "suggests", "mode", "patterns", "pattern", "architecture", "architectures",
            }

            def _tokenize_semantic(raw: str) -> Set[str]:
                canonical = _canonical_text(raw)
                return {
                    tok for tok in canonical.split()
                    if len(tok) > 3 and tok not in stopwords
                }

            def _is_semantic_duplicate(tokens: Set[str], existing_tokens: Set[str]) -> bool:
                if not tokens or not existing_tokens:
                    return False
                inter = len(tokens & existing_tokens)
                if inter < 5:
                    return False
                union = len(tokens | existing_tokens)
                return bool(union) and (inter / union) >= 0.18

            def _is_low_value_entry(title: str, content: str) -> bool:
                title_clean = " ".join(str(title or "").split()).strip()
                content_clean = " ".join(str(content or "").split()).strip()
                title_l = title_clean.lower()
                content_l = content_clean.lower()

                if len(title_clean) < 12 or len(content_clean) < 40:
                    return True
                if "..." in title_clean or "..." in content_clean:
                    return True
                if "1-2 sentences" in content_l or "i will now synthesize" in content_l:
                    return True
                if title_l.startswith("recent experiments show ") or title_l.startswith("all recent experiments show "):
                    return True
                if title_l.startswith("recent synthesis") and "failure" in title_l:
                    return True
                if "[principle/" in title_l or "hybrid? no" in title_l:
                    return True
                if "$" in content_clean or "\\approx" in content_l:
                    return True

                mechanism_tokens = (
                    "depth", "residual", "inverse", "log ", "frequency", "math_space",
                    "parameter", "parallel", "routing", "s1", "loss", "novelty", "baseline",
                )
                action_tokens = (
                    "improve", "improves", "degrade", "degrades", "fail", "fails", "underperform",
                    "correlate", "correlates", "correlation", "predict", "predicts",
                    "optimal", "requires", "avoid", "boost", "increase", "reduce",
                    "enhance", "enhances", "outperform", "outperforms", "suggests", "indicates",
                )
                has_mechanism = any(tok in content_l or tok in title_l for tok in mechanism_tokens)
                has_action = any(tok in content_l for tok in action_tokens)
                has_numeric = bool(re.search(r"\d", content_clean))
                return not (has_mechanism and (has_action or has_numeric))

            recent = nb.get_recent_experiments(config.knowledge_extraction_interval)
            resolved = []
            if self._active_campaign_id:
                all_hyps = nb.get_campaign_hypotheses(self._active_campaign_id)
                resolved = [h for h in all_hyps
                           if h.get("status") in ("confirmed", "refuted")]

            context = build_knowledge_extraction_context(recent, resolved)
            entries = self.aria.extract_knowledge(recent, resolved, context=context)

            existing_entries = nb.get_knowledge()
            existing_by_title: Dict[str, str] = {}
            existing_by_content: Dict[str, str] = {}
            existing_by_semantic: Dict[str, List[Tuple[str, Set[str]]]] = {}
            for row in existing_entries:
                eid = str(row.get("entry_id") or "")
                if not eid:
                    continue
                existing_by_title[_canonical_text(row.get("title") or "")] = eid
                existing_by_content[_canonical_text(row.get("content") or "")] = eid
                category = _normalize_category(str(row.get("category") or "principle"))
                tokens = _tokenize_semantic(f"{row.get('title') or ''} {row.get('content') or ''}")
                if tokens:
                    existing_by_semantic.setdefault(category, []).append((eid, tokens))

            accepted = 0
            skipped_low_value = 0
            deduped = 0

            for entry in entries:
                raw_title = str(entry.get("title") or "").strip()
                raw_content = str(entry.get("content") or "").strip()
                if _is_low_value_entry(raw_title, raw_content):
                    skipped_low_value += 1
                    continue

                category = _normalize_category(entry.get("category", "principle"))
                confidence = float(entry.get("confidence", 0.5) or 0.5)
                confidence = max(0.45, min(0.95, confidence))
                title = " ".join(raw_title.split())
                content = " ".join(raw_content.split())

                title_key = _canonical_text(title)
                content_key = _canonical_text(content)

                existing_entry_id = (
                    existing_by_title.get(title_key)
                    or existing_by_content.get(content_key)
                )
                if not existing_entry_id:
                    semantic_tokens = _tokenize_semantic(f"{title} {content}")
                    for eid, seen_tokens in existing_by_semantic.get(category, []):
                        if _is_semantic_duplicate(semantic_tokens, seen_tokens):
                            existing_entry_id = eid
                            break
                if existing_entry_id:
                    nb.validate_knowledge(existing_entry_id)
                    deduped += 1
                    continue

                evidence = [
                    str(e.get("experiment_id", "")).strip()
                    for e in recent[:5]
                    if str(e.get("experiment_id", "")).strip()
                ]
                new_entry_id = nb.add_knowledge(
                    category=category,
                    title=title,
                    content=content,
                    evidence=evidence,
                    confidence=confidence,
                )
                existing_by_title[title_key] = new_entry_id
                existing_by_content[content_key] = new_entry_id
                semantic_tokens = _tokenize_semantic(f"{title} {content}")
                if semantic_tokens:
                    existing_by_semantic.setdefault(category, []).append((new_entry_id, semantic_tokens))
                accepted += 1

            if entries:
                self._emit_event("knowledge_extracted", {
                    "n_entries": accepted,
                    "categories": list(set(e.get("category", "") for e in entries)),
                    "n_deduped": deduped,
                    "n_skipped_low_value": skipped_low_value,
                })
                logger.info(
                    "Knowledge extracted: accepted=%d deduped=%d skipped_low_value=%d raw=%d",
                    accepted, deduped, skipped_low_value, len(entries),
                )
        except Exception as e:
            logger.debug(f"Knowledge extraction failed: {e}")

    def _ensure_campaign(self, config: RunConfig, nb: LabNotebook) -> Optional[str]:
        """Ensure an active campaign exists. Create one if needed."""
        if not config.enable_campaigns:
            return None

        # Check for existing active campaign
        active = nb.get_active_campaigns()
        if active:
            self._active_campaign_id = active[0]["campaign_id"]
            return self._active_campaign_id

        # Create new campaign via Aria
        recent = nb.get_recent_experiments(10)
        knowledge = nb.get_knowledge()
        all_campaigns = nb.conn.execute(
            "SELECT * FROM campaigns ORDER BY timestamp DESC LIMIT 5"
        ).fetchall()
        previous = [dict(r) for r in all_campaigns]

        context = build_campaign_formulation_context(
            recent_experiments=recent,
            knowledge=knowledge,
            previous_campaigns=previous,
        )
        camp_data = self.aria.formulate_campaign(context=context)
        post_hoc_note = (
            "\n\n[POST-HOC] Success criteria were formulated after reviewing "
            "recent experiment outcomes; treat claims as exploratory until "
            "prospective criteria are pre-registered."
        )
        campaign_id = nb.create_campaign(
            title=camp_data["title"],
            objective=camp_data["objective"],
            success_criteria=f"{camp_data['success_criteria']}{post_hoc_note}",
        )
        self._active_campaign_id = campaign_id
        self._emit_event("campaign_created", {
            "campaign_id": campaign_id,
            "title": camp_data["title"],
            "objective": camp_data["objective"],
        })
        logger.info(f"Campaign created: {camp_data['title']} ({campaign_id})")
        return campaign_id

    @staticmethod
    def _pipeline_driven_campaign(tiers: dict, reason: str) -> dict:
        """Deterministic campaign formulation based on pipeline state."""
        screening = tiers.get("screening", 0)
        investigation = tiers.get("investigation", 0)
        validation = tiers.get("validation", 0)
        breakthrough = tiers.get("breakthrough", 0)

        if breakthrough > 0:
            return {
                "title": "Scale-Up & Generalization",
                "objective": (
                    f"Validate {breakthrough} breakthrough architecture(s) at "
                    f"larger scale (512+ dim, longer sequences) and on diverse "
                    f"data distributions to confirm generalization."
                ),
                "success_criteria": (
                    "Breakthrough architecture maintains loss_ratio < 0.5 at "
                    "model_dim=512; OOD generalization >= 0.67; "
                    "Reproducible across 5+ random seeds with std <= 0.03"
                ),
            }
        elif validation > 0:
            return {
                "title": "Validation & Robustness",
                "objective": (
                    f"Complete multi-seed validation for {validation} candidate(s) "
                    f"and identify which architectures are robust enough for "
                    f"breakthrough consideration."
                ),
                "success_criteria": (
                    "At least 1 candidate passes validation with multi-seed "
                    "std <= 0.03 and baseline_ratio < 0.90; "
                    "Go/no-go decision recorded for each candidate"
                ),
            }
        elif investigation > 0 or screening > 0:
            total_candidates = investigation + screening
            return {
                "title": "Deep Investigation",
                "objective": (
                    f"Investigate {total_candidates} screening/investigation "
                    f"candidate(s) with extended training to identify which "
                    f"architectures warrant full validation."
                ),
                "success_criteria": (
                    "At least 1 candidate passes investigation with "
                    "loss_ratio < 0.6 and robustness > 0.7; "
                    "Clear go/no-go decision for each investigated candidate"
                ),
            }
        elif reason == "stale":
            return {
                "title": "Novelty Exploration",
                "objective": (
                    "Escape the current search region using evolution and "
                    "novelty search to discover fundamentally different "
                    "architecture patterns."
                ),
                "success_criteria": (
                    "Find 3+ architectures with loss_ratio < 0.5 and "
                    "novelty_score > 0.5; Stage-1 survival rate > 5%"
                ),
            }
        else:
            return {
                "title": "Architecture Discovery",
                "objective": (
                    "Discover novel computation patterns by exploring diverse "
                    "op combinations, math spaces, and weight storage techniques."
                ),
                "success_criteria": (
                    "Find 3+ architectures with loss_ratio < 0.5; "
                    "Stage-1 survival rate > 3%; "
                    "At least 1 go/no-go decision recorded"
                ),
            }

    def _ensure_preregistration(
        self,
        nb: LabNotebook,
        experiment_type: str,
        config: Dict[str, Any],
        hypothesis: Optional[str],
        preregistration: Optional[Dict[str, Any]] = None,
        exploratory: bool = False,
        created_by: str = "runner",
    ) -> str:
        require_prereg = bool(config.get("require_preregistration", True))
        auto_preregister = bool(config.get("auto_preregister", True))
        payload = preregistration
        if payload is None and auto_preregister:
            payload = self._build_default_preregistration(
                experiment_type=experiment_type,
                config=config,
                hypothesis=hypothesis,
                exploratory=exploratory,
            )
        if require_prereg and payload is None:
            raise PreregistrationError(
                "Experiment blocked: preregistration required but missing."
            )
        if payload is None:
            raise PreregistrationError(
                "Experiment blocked: preregistration payload unavailable."
            )
        validate_preregistration(payload)
        return nb.create_preregistration(
            experiment_type=experiment_type,
            preregistration=payload,
            created_by=created_by,
        )

    def _build_default_preregistration(
        self,
        experiment_type: str,
        config: Dict[str, Any],
        hypothesis: Optional[str],
        exploratory: bool = False,
    ) -> Dict[str, Any]:
        statement = str(hypothesis or f"{experiment_type} batch will improve prioritized objectives.")
        primary_metrics = ["loss_ratio", "stage1_passed"]
        if experiment_type in {"novelty", "evolution"}:
            primary_metrics = ["novelty_score", "stage1_passed"]
        if experiment_type in {"validation", "scale_up"}:
            primary_metrics = ["baseline_loss_ratio", "loss_ratio", "novelty_confidence"]

        prereg = HypothesisPreregistration(
            hypothesis={
                "statement": statement,
                "variables": {
                    "independent": ["architecture_family", "op_composition", "training_recipe"],
                    "dependent": primary_metrics + ["throughput_tok_s", "stability_score"],
                    "controls": ["model_dim", "n_layers", "stage1_steps", "batch_size"],
                },
                "expected_direction": {
                    "loss_ratio": "decrease",
                    "novelty_score": "increase",
                    "throughput_tok_s": "increase",
                    "stability_score": "increase",
                },
                "success_criteria": {
                    "stage1_passed_min": 1,
                    "best_loss_ratio_max": 0.95,
                    "novelty_confidence_min": 0.5,
                },
            },
            analysis_plan={
                "primary_metrics": primary_metrics,
                "secondary_metrics": [
                    "compile_time_ms",
                    "grad_norm_std",
                    "throughput_tok_s",
                    "flops_per_token",
                    "novelty_confidence",
                ],
                "thresholds": {
                    "loss_ratio": {"operator": "<", "value": 1.0},
                    "novelty_confidence": {"operator": ">=", "value": 0.5},
                    "stability_score": {"operator": ">=", "value": 0.5},
                },
                "baseline_comparison": {
                    "method": "relative_loss_ratio",
                    "source": "TransformerBaseline.compare",
                    "delta_operator": "<",
                    "delta_value": 1.0,
                },
            },
            falsification_conditions=[
                "No candidate passes Stage1.",
                "Best loss_ratio does not beat baseline threshold.",
                "Novelty only appears with heuristic fallback and no justification.",
            ],
            confounders_checklist=[
                {"name": "unstable_seed_behavior", "checked": False},
                {"name": "fallback_novelty_mode", "checked": False},
                {"name": "noisy_throughput", "checked": False},
                {"name": "compile_instability", "checked": False},
            ],
            exploratory=exploratory,
        ).to_dict()
        prereg["analysis_plan"]["config_snapshot"] = {
            "n_programs": config.get("n_programs"),
            "stage1_steps": config.get("stage1_steps"),
            "model_dim": config.get("model_dim"),
            "n_layers": config.get("n_layers"),
        }
        return prereg

    def _start_preregistered_experiment(
        self,
        nb: LabNotebook,
        experiment_type: str,
        config: Dict[str, Any],
        hypothesis: Optional[str] = None,
        research_question: Optional[str] = None,
        hypothesis_metadata: Optional[Dict[str, Any]] = None,
        preregistration: Optional[Dict[str, Any]] = None,
        exploratory: bool = False,
        created_by: str = "runner",
    ) -> str:
        prereg_id = self._ensure_preregistration(
            nb=nb,
            experiment_type=experiment_type,
            config=config,
            hypothesis=hypothesis,
            preregistration=preregistration,
            exploratory=exploratory,
            created_by=created_by,
        )
        meta = dict(hypothesis_metadata or {})
        meta["preregistration_id"] = prereg_id
        
        # Z17: Reset global native-runner counters between experiments
        reset_native_runner_telemetry()
        
        return nb.start_experiment(
            experiment_type=experiment_type,
            config=config,
            hypothesis=hypothesis,
            research_question=research_question,
            hypothesis_metadata=meta,
            preregistration_id=prereg_id,
            require_preregistration=bool(config.get("require_preregistration", True)),
        )

    def _check_stale_screening_candidates(self, nb: LabNotebook, config: RunConfig):
        """Force investigation if top screening models beat references but are uninvestigated."""
        try:
            # Compare screening_loss_ratio directly against best reference
            # screening_loss_ratio (tier-neutral metric, not discounted composite).
            best_ref_lr = nb.conn.execute(
                "SELECT MIN(l.screening_loss_ratio) FROM leaderboard l"
                " WHERE COALESCE(l.is_reference, 0) = 1"
                " AND l.screening_loss_ratio IS NOT NULL"
            ).fetchone()[0]
            if best_ref_lr is None:
                return None
            stale = nb.conn.execute(
                """SELECT l.result_id FROM leaderboard l
                   WHERE l.tier = 'screening' AND l.screening_passed = 1
                     AND COALESCE(l.is_reference, 0) = 0
                     AND l.screening_loss_ratio IS NOT NULL
                     AND l.screening_loss_ratio <= ?
                     AND l.investigation_loss_ratio IS NULL
                   ORDER BY l.screening_loss_ratio ASC LIMIT ?""",
                (best_ref_lr, config.auto_investigate_top_n)
            ).fetchall()
            if stale:
                result_ids = [r["result_id"] for r in stale]
                logger.info("Stale screening check: %d models beat best reference loss_ratio (%.4f) but are uninvestigated",
                            len(result_ids), best_ref_lr)
                return result_ids
        except Exception as e:
            logger.warning("Stale screening check failed: %s", e)
        return None

    def _resolve_novelty_promotion_validity(
        self,
        config: RunConfig,
        valid_for_promotion: bool,
        reason: str,
    ) -> Tuple[bool, str, bool]:
        """Apply explicit override policy for heuristic novelty promotions."""
        valid = bool(valid_for_promotion)
        resolved_reason = str(reason or "unknown")
        requires_justification = not valid
        if valid:
            return True, resolved_reason, False
        if config.allow_heuristic_novelty_promotion and str(config.heuristic_novelty_justification or "").strip():
            return True, f"override:{resolved_reason}", True
        return False, resolved_reason, requires_justification
