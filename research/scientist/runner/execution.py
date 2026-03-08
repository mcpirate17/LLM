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

import gc
import hashlib
import json
import copy
import math
import os
import queue
import random
import re
import shlex
import threading
import time
import traceback
import uuid
import functools
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...synthesis.grammar import GrammarConfig, generate_layer_graph, batch_generate
from ..native_runner import (
    compile_model_native_first as compile_model,
    record_native_abi_parity_result,
    reset_native_runner_telemetry,
)
from ...synthesis.validator import validate_graph
from ...synthesis.serializer import graph_to_json, graph_from_json, graph_summary
from ...synthesis.primitives import get_primitive, list_primitives, PROTECTED_OPS
from ...eval.sandbox import safe_eval
from ...eval.metrics import novelty_score
from ...eval.flops import estimate_flops
from ...eval.baseline import TransformerBaseline
from ...eval.fingerprint import compute_fingerprint, BehavioralFingerprint
from ...eval.diagnostic_tasks import run_diagnostic_suite
from ...eval.perf_budget import evaluate_perf_budget_gate
from ...eval.pruning import apply_one_shot_pruning, estimate_lm_ce_loss
from ...training.training_program import synthesize_training_program, synthesize_training_program_batch
from ...training.data_pipeline import CorpusConfig, CorpusTokenBatcher
from ...training.checkpointing import CheckpointManager
from ...orchestrator.executor import WorkerPoolOrchestrator
from ..persona import Aria, get_aria
from ..notebook import LabNotebook, ExperimentEntry
from ..evidence import (
    build_evidence_pack,
    validate_selection_decision_log,
)
from ..preregistration import (
    HypothesisPreregistration,
    PreregistrationError,
    validate_preregistration,
)
from ...healer import CodeHealer
from ...healer.core import HealerTaskSpec
from ..llm.context import (build_rich_context, build_investigation_context,
                          build_validation_context, build_mode_selection_context,
                          build_hypothesis_context, build_go_no_go_context,
                          build_knowledge_extraction_context,
                          build_campaign_formulation_context,
                          build_manual_start_fallback_context)
from ..llm.decision import NextExperimentDecisionPlanner
from ..shared_utils import resolve_device

import logging
logger = logging.getLogger(__name__)

from ._types import RunConfig, LiveProgress, _LIVE_LOSS_CURVE_MAX_POINTS, _TRAINING_STEP_SSE_EVERY

from ._helpers import _native_proactive_gating


@dataclass(slots=True)
class ModelCandidate:
    """Unified representation of a candidate model from any source."""
    source: str  # "graph_synthesis" or "morphological_box"
    model: nn.Module
    description: str
    # Source-specific data
    graph: Optional[Any] = None
    graph_json: Optional[str] = None
    arch_spec: Optional[Any] = None  # ArchSpec
    arch_spec_json: Optional[str] = None
    fingerprint: str = ""




class _ExecutionMixin:
    """Experiment/investigation/validation/scale-up threads, micro-train."""

    def _run_experiment_thread(self, exp_id: str, config: RunConfig,
                                hypothesis: str):
        """Execute a single experiment in background."""
        with self._lock:
            # Z17: Clear any stale progress data from previous runs
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                aria_message=f"{self.aria.NAME}: Starting experiment {exp_id[:8]}...",
            )
            
        nb = self._make_notebook()
        try:
            results = self._execute_experiment(exp_id, config, nb)
            self._persist_applied_grammar_weights(nb, exp_id, results)

            # Build rich context for LLM-enhanced methods
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)

            summary = self.aria.experiment_summary(results, context=context)
            insights = self._analyze_results(results, exp_id, nb, context=context)

            # Store LLM analysis if available
            llm_analysis = self.aria.analyze_results(results, context=context)

            # Validate hypothesis
            try:
                validation = self.aria.validate_hypothesis(hypothesis, results, context)
                if validation:
                    nb.add_entry(ExperimentEntry(
                        entry_type="analysis",
                        title="Hypothesis Validation",
                        content=validation.get("explanation", ""),
                        experiment_id=exp_id,
                        metadata={"validated": validation.get("validated", False)},
                    ))
            except Exception as e:
                logger.warning("Hypothesis validation logging failed: %s", e)

            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=insights,
                llm_analysis=llm_analysis,
            )

            # Update op success rates and failure signatures after experiment
            nb.update_op_success_rates(exp_id)
            s0_op_counts = results.pop("_s0_op_counts", None)
            if s0_op_counts:
                nb.merge_op_failure_counts(s0_op_counts)
            nb.strip_graph_json_for_failures(exp_id)
            nb.update_failure_signatures(exp_id)

            # Periodic op rehabilitation: test excluded ops in isolation
            try:
                total_exp = nb.conn.execute(
                    "SELECT COUNT(*) FROM experiments"
                ).fetchone()[0]
                if total_exp % 10 == 0:
                    from ...eval.op_rehab import rehabilitate_ops
                    rehab_results = rehabilitate_ops(nb, model_dim=config.model_dim)
                    if rehab_results:
                        logger.info(
                            "Op rehabilitation passed %d ops: %s",
                            len(rehab_results), ", ".join(rehab_results),
                        )
            except Exception as e:
                logger.warning("Op rehabilitation failed: %s", e)

            # Save effective weights + S1 outcome for EMA continuity
            applied_w = results.get("applied_grammar_weights")
            total = results.get("total", 0)
            if applied_w and total > 0:
                s1_rate = results.get("stage1_passed", 0) / total
                nb.save_effective_weights(applied_w, s1_rate, exp_id)

            # Auto-recommend next experiment
            self._auto_recommend(results, config, hypothesis, nb)

            # Auto-escalation pipeline (investigation/validation)
            results["experiment_id"] = exp_id
            self._auto_escalate(results, config, nb, phase="screening")

            # Auto-scale-up if criteria met (legacy, kept for backward compat)
            self._maybe_auto_scale_up(results, config, nb)

            # Auto-report for single experiments
            self._maybe_auto_report(config, nb, reason="experiment_complete")

            with self._lock:
                self._progress.status = "completed"
                self._progress.aria_message = summary.split("\n")[-1] if summary else "Experiment complete."

            self._emit_event("experiment_completed", {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
            })

        except Exception as e:
            error = traceback.format_exc()
            logger.error("Experiment failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Synthesis/experiment failure: {str(e)[:240]}",
                reproduction_steps=["python -m pytest tests/test_integration.py -k \"start_experiment\" -x --tb=short"],
                acceptance_tests=["python -m pytest tests/test_integration.py -k \"start_experiment\" -x --tb=short"],
                trigger_payload={"mode": "synthesis", "error": str(e)},
            )
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
            self._emit_event("experiment_failed", {
                "experiment_id": exp_id,
                "error": str(e),
            })
        finally:
            nb.close()
            # Launch queued auto-scale-up after notebook is closed
            self._run_pending_scale_up()

    def _execute_experiment(self, exp_id: str, config: RunConfig,
                            nb: LabNotebook,
                            use_learned_grammar: bool = True) -> Dict:
        """Core experiment logic shared by single and continuous modes."""
        self._live_training_context = {"exp_id": exp_id, "phase": "synthesis"}
        with self._lock:
            # Z17: Explicitly reset progress object at start of execution
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=config.n_programs,
                aria_message=f"{self.aria.NAME}: Initializing experiment {exp_id[:8]}...",
            )

        results = {
            "total": 0, "stage0_passed": 0, "stage05_passed": 0,
            "stage1_passed": 0, "novel_count": 0,
            "best_loss_ratio": None, "best_novelty_score": None,
            "survivors": [],
            "skipped_proactive_gating": 0,
            "proactive_gating_failures": [],
        }

        grammar_weights = None
        excluded_ops: set = set()
        op_weights: Dict[str, float] = {}
        failure_blocklist: Dict[str, float] = {}
        champion_bias: Dict[str, float] = {}
        analytics = None
        grammar_gate: Optional[Dict[str, Any]] = None
        if use_learned_grammar:
            try:
                from ..analytics import ExperimentAnalytics
                analytics = ExperimentAnalytics(nb)
                last_effective = nb.load_last_effective_weights()
                last_weights = last_effective[0] if last_effective else None
                grammar_weights = analytics.compute_grammar_weights(
                    last_applied=last_weights, alpha=0.6
                )
                if grammar_weights:
                    grammar_gate = self._evaluate_grammar_update_gate(
                        nb=nb,
                        analytics=analytics,
                        config=config,
                    )
                    if not grammar_gate.get("gate_pass"):
                        nb.log_learning_event(
                            "grammar_weights_blocked",
                            f"Blocked grammar weight update for {exp_id}: weak attribution evidence",
                            evidence=json.dumps(grammar_gate, sort_keys=True),
                        )
                        grammar_weights = None
            except Exception as e:
                logger.warning("Failed computing learned grammar weights for %s: %s", exp_id, e)

            # Populate excluded_ops and soft-penalty op_weights from negative results
            # IMPORTANT: Only hard-exclude ops that fail at LEARNING (s0 passes but
            # s1 never does). Ops failing at COMPILATION are compiler bugs, not bad
            # ops — soft-penalize them instead so they can be retried as the compiler
            # improves.  Ops that pass rehabilitation (work in isolation) get a mild
            # penalty instead of exclusion — their failures are placement problems.
            op_weights: Dict[str, float] = {}
            try:
                rehab_cache = nb.get_op_rehabilitation_cache()
                if analytics is not None:
                    neg = analytics.negative_results_synthesis()
                    compilation_failures = []
                    rehabilitated_ops = []
                    for op_info in neg.get("failed_ops", []):
                        if (op_info.get("s1_rate", 1) == 0
                                and op_info.get("n_used", 0) >= 5
                                and op_info.get("confidence", 0) >= 0.7):
                            op_name = op_info["op_name"]
                            rehab = rehab_cache.get(op_name)
                            if rehab and rehab.get("compile_passed") and rehab.get("forward_passed"):
                                # Op works in isolation — placement problem, not op problem
                                op_weights[op_name] = 0.5
                                rehabilitated_ops.append(op_name)
                            elif op_info.get("failure_stage") == "compilation":
                                # Compiler bug — soft-penalize, don't exclude
                                op_weights[op_name] = 0.15
                                compilation_failures.append(op_name)
                            else:
                                if op_name in PROTECTED_OPS:
                                    # Protected op — soft-penalize instead of excluding
                                    op_weights[op_name] = 0.5
                                else:
                                    # Genuine learning failure — hard exclude
                                    excluded_ops.add(op_name)
                    # Soft-penalize weak ops (nonzero but poor S1 rate)
                    for op_info in neg.get("weak_ops", []):
                        op_name = op_info.get("op_name", "")
                        penalty = op_info.get("penalty_weight", 1.0)
                        if op_name and op_name not in excluded_ops:
                            op_weights[op_name] = penalty
                    if excluded_ops:
                        nb.log_learning_event(
                            "excluded_ops_applied",
                            f"Excluded {len(excluded_ops)} ops with 0% S1 rate (learning failures): "
                            f"{', '.join(sorted(excluded_ops))}",
                            excluded_ops=sorted(excluded_ops),
                        )
                    if rehabilitated_ops:
                        nb.log_learning_event(
                            "rehabilitated_ops_softpenalized",
                            f"Soft-penalized {len(rehabilitated_ops)} ops that passed rehabilitation "
                            f"(work in isolation, placement problem): {', '.join(sorted(rehabilitated_ops))}",
                            rehabilitated_ops=sorted(rehabilitated_ops),
                        )
                    if compilation_failures:
                        nb.log_learning_event(
                            "compilation_failures_softpenalized",
                            f"Soft-penalized {len(compilation_failures)} ops failing at compilation "
                            f"(compiler bugs, not excluded): {', '.join(sorted(compilation_failures))}",
                            compilation_failures=sorted(compilation_failures),
                        )
                    if op_weights:
                        nb.log_learning_event(
                            "weak_ops_penalized",
                            f"Soft-penalized {len(op_weights)} weak ops: "
                            f"{', '.join(f'{k}={v:.2f}' for k, v in sorted(op_weights.items()))}",
                            op_weights=op_weights,
                        )
            except Exception as e:
                logger.warning("Failed computing excluded/weak ops for %s: %s", exp_id, e)

            # Load failure-signature blocklist (op-pair bigrams with high fail rate)
            failure_blocklist: Dict[str, float] = {}
            try:
                failure_blocklist = nb.get_failure_signature_blocklist()
                if failure_blocklist:
                    nb.log_learning_event(
                        "failure_signatures_loaded",
                        f"Loaded {len(failure_blocklist)} toxic op-pair patterns",
                        signatures=sorted(failure_blocklist.keys())[:10],
                    )
            except Exception as e:
                logger.warning("Failed loading failure signatures for %s: %s", exp_id, e)

            # Champion bias pass: nudge category weights toward proven winners.
            # This biases the search toward high-performing projection/sparse patterns
            # and known-good structural/sequence motifs without hard-coding op-level picks.
            try:
                if analytics is not None:
                    op_rates = analytics.op_success_rates() or {}
                    if op_rates:
                        winning_ops = {"exp", "selective_scan", "tropical_center"}
                        projection_ops = {"low_rank_proj", "shared_basis_proj", "tied_proj"}
                        sparse_ops = {"nm_sparse_linear", "block_sparse_linear", "semi_structured_2_4_linear"}

                        def _is_reliable(op_name: str, min_used: int = 10, min_s1: float = 0.25) -> bool:
                            info = op_rates.get(op_name) or {}
                            n_used = int(info.get("n_used") or 0)
                            s1_rate = float(info.get("s1_rate") or 0.0)
                            return n_used >= min_used and s1_rate >= min_s1

                        has_winners = any(_is_reliable(op) for op in winning_ops)
                        has_projection = any(_is_reliable(op) for op in projection_ops)
                        has_sparse = any(_is_reliable(op) for op in sparse_ops)

                        if has_winners:
                            champion_bias["structural"] = max(champion_bias.get("structural", 1.0), 1.2)
                            champion_bias["sequence"] = max(champion_bias.get("sequence", 1.0), 1.2)
                        if has_projection:
                            champion_bias["parameterized"] = max(champion_bias.get("parameterized", 1.0), 1.4)
                        if has_sparse:
                            champion_bias["parameterized"] = max(champion_bias.get("parameterized", 1.0), 1.5)
                            # Z7: If sparse ops are reliable, nudge the grammar hard toward them
                            champion_bias["_structured_sparsity_bias"] = 0.8

            except Exception as e:
                logger.warning("Failed computing champion bias for %s: %s", exp_id, e)

        # Merge Aria's overrides into excluded_ops and op_weights
        excluded_ops = excluded_ops | self._excluded_ops_overrides
        op_weights = {**op_weights, **self._op_weights_overrides}
        grammar = self._build_grammar_config(config, excluded_ops=excluded_ops, op_weights=op_weights)
        old_weights = dict(grammar.category_weights)

        if grammar_weights:
            old_weights = dict(grammar.category_weights)
            grammar.category_weights.update(grammar_weights)
            self._log_grammar_weight_application(
                nb,
                exp_id,
                old_weights,
                dict(grammar.category_weights),
                analytics=analytics,
            )
            # Persist for observability
            results["applied_grammar_weights"] = dict(grammar.category_weights)
            if grammar_gate:
                results["grammar_weight_attribution"] = grammar_gate

        if champion_bias:
            before_bias = dict(grammar.category_weights)
            for category, multiplier in champion_bias.items():
                if category == "_structured_sparsity_bias":
                    grammar.structured_sparsity_bias = float(multiplier)
                    continue
                base = float(grammar.category_weights.get(category, 1.0))
                grammar.category_weights[category] = round(max(0.5, min(8.0, base * multiplier)), 2)
            nb.log_learning_event(
                "champion_bias_applied",
                f"Applied champion grammar bias for {exp_id}",
                multipliers=champion_bias,
                old_weights=before_bias,
                new_weights=dict(grammar.category_weights),
            )
            results["applied_grammar_weights"] = dict(grammar.category_weights)

        # Apply chat-driven grammar weight overrides (from Aria actions)
        if self._grammar_weight_overrides:
            grammar.category_weights.update(self._grammar_weight_overrides)
            nb.log_learning_event(
                "chat_grammar_overrides_applied",
                f"Applied chat-driven grammar overrides for {exp_id}",
                overrides=dict(self._grammar_weight_overrides),
                final_weights=dict(grammar.category_weights),
            )
            results["applied_grammar_weights"] = dict(grammar.category_weights)
            # Emit SSE so LiveFeed can show learning events
            source_weights = grammar_weights or {}
            n_changed = sum(1 for k in source_weights
                            if old_weights.get(k) != source_weights[k])
            self._emit_event("learning_event", {
                "event_type": "grammar_weights_applied",
                "experiment_id": exp_id,
                "n_changed": n_changed,
                "description": f"Applied learned grammar weights ({n_changed} categories changed)",
            })
        else:
            grammar.category_weights["math_space"] = config.math_space_weight

        # Hyperbolic promotion: query recent hierarchy fitness from fingerprints
        if analytics is not None:
            try:
                hf = analytics.recent_hierarchy_fitness()
                if hf is not None:
                    grammar._hierarchy_fitness = hf
                    if hf > grammar.hyperbolic_promotion_threshold:
                        logger.info("Hierarchy detected (fitness=%.3f > %.2f): boosting hyperbolic ops",
                                    hf, grammar.hyperbolic_promotion_threshold)
            except Exception:
                pass

        t_start = time.time()

        # Generate graphs
        if config.model_source == "morphological_box":
            # Morphological box evaluation path (arch_builder models, no graph JSON)
            candidates = self._generate_candidates(config, config.n_programs, "morphological_box")
            results["total"] = len(candidates)

            dev = resolve_device(config.device)
            dev_str = str(dev)

            for i, cand in enumerate(candidates):
                if self._stop_event.is_set():
                    break

                with self._lock:
                    self._progress.current_program = i + 1
                    self._progress.current_fingerprint = (cand.fingerprint or "")[:10]
                    self._progress.elapsed_seconds = time.time() - t_start

                model = cand.model
                if model is None:
                    continue

                # Stage 0/0.5
                try:
                    sandbox_result = self._safe_eval_for_stage(
                        model,
                        stage_tag="morph_candidate_screening",
                        batch_size=2,
                        seq_len=min(128, config.max_seq_len),
                        vocab_size=config.vocab_size,
                        device=dev_str,
                    )
                except Exception as e:
                    logger.error("Error evaluating morph candidate %d: %s", i, e)
                    continue

                s0_passed = bool(sandbox_result.passed)
                s05_passed = (sandbox_result.stability_score >= config.stage05_stability_threshold
                              and sandbox_result.causality_passed)
                if s0_passed:
                    results["stage0_passed"] += 1
                    with self._lock: self._progress.stage0_passed += 1
                if s05_passed:
                    results["stage05_passed"] += 1
                    with self._lock: self._progress.stage05_passed += 1

                if not s0_passed or not s05_passed:
                    continue

                # Stage 1 (sync, since we already have a compiled model)
                s1_result = self._micro_train(
                    model,
                    config,
                    dev,
                    seed=self._stable_seed(exp_id, i, "morphology"),
                )
                s1_passed = bool(s1_result.get("passed", False))
                if s1_passed:
                    results["stage1_passed"] += 1
                    with self._lock: self._progress.stage1_passed += 1

                program_metrics: Dict[str, Any] = {}
                try:
                    program_metrics.update(self._extract_sandbox_metrics(sandbox_result))
                except Exception:
                    pass
                try:
                    program_metrics["param_count"] = sandbox_result.param_count
                except Exception:
                    pass

                # Merge S1 metrics
                for k in ("initial_loss", "final_loss", "min_loss", "loss_ratio",
                          "throughput", "avg_step_time_ms", "total_train_time_ms",
                          "validation_loss", "validation_loss_ratio", "generalization_gap",
                          "discovery_loss", "discovery_loss_ratio"):
                    if k in s1_result:
                        program_metrics[k] = s1_result.get(k)
                self._merge_s1_telemetry(program_metrics, s1_result)

                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=cand.fingerprint,
                    graph_json="{}",
                    stage0_passed=s0_passed,
                    stage05_passed=s05_passed,
                    stage1_passed=s1_passed,
                    loss_ratio=s1_result.get("loss_ratio"),
                    final_loss=s1_result.get("final_loss"),
                    model_source="morphological_box",
                    arch_spec_json=cand.arch_spec_json,
                    **program_metrics,
                )

            return results

        if config.model_source == "fingerprint_refine":
            graphs = self._generate_refinement_graphs(exp_id, config, nb, grammar)
        else:
            # Project Hephaestus Phase 4: Adaptive Synthesis
            prior = None
            use_adaptive = False
            if use_learned_grammar and analytics is not None:
                try:
                    frontier = analytics.get_efficiency_frontier()
                    if frontier:
                        from ...synthesis.grammar import EfficiencyPrior
                        prior = EfficiencyPrior(frontier)
                        use_adaptive = True
                        nb.log_learning_event(
                            "adaptive_synthesis_enabled",
                            f"Enabling budget-aware adaptive synthesis for {exp_id}",
                            frontier_size=len(frontier),
                        )
                except Exception as e:
                    logger.warning("Failed to initialize efficiency prior: %s", e)
            
            graphs = batch_generate(
                config.n_programs, 
                grammar, 
                use_adaptive_synthesis=use_adaptive,
                prior=prior
            )
        results["total"] = len(graphs)
        op_distribution = self._compute_generated_op_distribution(graphs)
        if op_distribution:
            results["generated_op_distribution"] = op_distribution
            shift = self._compare_with_previous_synthesis_distribution(
                nb,
                exp_id,
                op_distribution,
            )
            if shift:
                results["generation_distribution_shift"] = shift
                nb.log_learning_event(
                    "architecture_distribution_shift",
                    f"Generated-op distribution shift recorded for synthesis experiment {exp_id}",
                    evidence=json.dumps(shift, sort_keys=True),
                )
            else:
                nb.log_learning_event(
                    "architecture_distribution_snapshot",
                    f"Captured generated-op distribution for synthesis experiment {exp_id}",
                    evidence=json.dumps({"op_distribution": op_distribution}, sort_keys=True),
                )

        with self._lock:
            self._progress.total_programs = len(graphs)
            self._progress.status = "evaluating"

        logger.info(
            "Experiment %s: generated %d graphs (depth=%d, ops=%d, dim=%d, device=%s)",
            exp_id[:8], len(graphs), grammar.max_depth, grammar.max_ops,
            config.model_dim, config.device,
        )

        nb.add_entry(ExperimentEntry(
            entry_type="observation",
            title=f"Generated {len(graphs)} computation graphs",
            content=f"Grammar: depth={grammar.max_depth}, ops={grammar.max_ops}, "
                    f"dim={config.model_dim}, math_space_weight={config.math_space_weight}",
            experiment_id=exp_id,
        ))

        dev = resolve_device(config.device)
        dev_str = str(dev)

        # Z12: Detect available GPUs for distributed search
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(num_gpus)]
            # 2 workers per GPU usually helps overlap data loading
            num_workers = num_gpus * 2
        else:
            devices = ["cpu"]
            num_workers = 1

        # Z12: Multi-node distributed workers
        remote_workers = [
            w.strip() for w in os.environ.get("ARIA_REMOTE_WORKERS", "").split(",")
            if w.strip()
        ]

        # Z6: Initialize asynchronous program orchestrator
        orchestrator = WorkerPoolOrchestrator(
            train_fn=lambda m, c, s, d: self._micro_train_async(m, c, s, d),
            num_workers=num_workers,
            max_queue_size=config.n_programs,
            devices=devices,
            remote_workers=remote_workers
        )
        candidate_batch_size = max(1, min(32, int(math.sqrt(max(1, config.n_programs)))))
        results["candidate_batch_size"] = candidate_batch_size

        time.time()

        # Dedup: load fingerprints already evaluated in previous experiments
        # to avoid wasting compute re-testing identical architectures.
        try:
            _existing_fps = {
                r[0] for r in nb.conn.execute(
                    "SELECT DISTINCT graph_fingerprint FROM program_results"
                ).fetchall() if r[0]
            }
        except Exception:
            _existing_fps = set()

        # Pre-filter known fingerprints and adaptively generate more if needed
        original_count = len(graphs)
        _dedup_max_rounds = 3
        _dedup_target = max(1, int(original_count * 0.5))  # want at least 50% novel
        for _dedup_round in range(_dedup_max_rounds):
            novel = []
            seen_this_batch = set()
            for g in graphs:
                fp = g.fingerprint()
                if fp not in _existing_fps and fp not in seen_this_batch:
                    novel.append(g)
                    seen_this_batch.add(fp)
            graphs = novel
            if len(graphs) >= _dedup_target or config.model_source == "fingerprint_refine":
                break
            # Generate extra graphs to compensate for high dedup rate
            shortfall = original_count - len(graphs)
            if shortfall <= 0:
                break
            extra = batch_generate(min(shortfall * 2, original_count), grammar)
            graphs.extend(extra)
            logger.info(
                "Experiment %s dedup round %d: %d novel / %d generated, "
                "added %d extra candidates",
                exp_id[:8], _dedup_round + 1, len(novel), original_count,
                len(extra),
            )

        # Mark all novel fingerprints as seen for within-run dedup
        for g in graphs:
            _existing_fps.add(g.fingerprint())

        dedup_rate = 1.0 - (len(graphs) / max(original_count, 1))
        results["skipped_dedup"] = original_count - len(graphs)
        results["dedup_rate"] = round(dedup_rate, 3)
        results["dedup_novel_count"] = len(graphs)
        results["dedup_known_fingerprints"] = len(_existing_fps)
        results["total"] = len(graphs)  # update to reflect actual novel count

        if dedup_rate > 0.1:
            logger.info(
                "Experiment %s dedup: %d/%d candidates were duplicates (%.0f%% dedup rate), "
                "%d novel candidates remain, %d known fingerprints in DB",
                exp_id[:8], original_count - len(graphs), original_count,
                dedup_rate * 100, len(graphs), len(_existing_fps),
            )
        if dedup_rate > 0.8:
            logger.warning(
                "Experiment %s: grammar diversity exhaustion — %.0f%% dedup rate. "
                "Consider increasing grammar depth/ops or switching to refinement mode.",
                exp_id[:8], dedup_rate * 100,
            )

        with self._lock:
            self._progress.total_programs = len(graphs)

        # Track ops from S0 failures for op_success_rates (not stored in DB)
        _s0_op_counts: Dict[str, Dict[str, int]] = {}  # op -> {n_used, n_s0, n_s05}

        for i, graph in enumerate(graphs):
            if self._stop_event.is_set():
                break

            fp = graph.fingerprint()
            with self._lock:
                self._progress.current_program = i + 1
                self._progress.current_fingerprint = fp[:10]
                self._progress.elapsed_seconds = time.time() - t_start

            # Real-time dedup: skip if evaluated by another process since experiment start
            if nb.has_fingerprint(fp):
                results.setdefault("skipped_dedup_runtime", 0)
                results["skipped_dedup_runtime"] += 1
                self._emit_event("program_evaluated", {
                    "index": i, "fingerprint": fp[:10],
                    "result": "skipped_dedup",
                })
                continue

            # Pre-screen: skip graphs whose op-pair structure is toxic
            if failure_blocklist:
                bigrams = set()
                for nid, node in graph.nodes.items():
                    if node.is_input:
                        continue
                    for inp_id in node.input_ids:
                        parent = graph.nodes.get(inp_id)
                        if parent and not parent.is_input:
                            bigrams.add(f"{parent.op_name}->{node.op_name}")
                if bigrams:
                    toxic_hits = sum(1 for bg in bigrams if bg in failure_blocklist)
                    toxic_ratio = toxic_hits / len(bigrams)
                    if toxic_ratio >= 0.5:
                        results.setdefault("skipped_toxic", 0)
                        results["skipped_toxic"] += 1
                        self._emit_event("program_evaluated", {
                            "index": i, "fingerprint": graph.fingerprint()[:10],
                            "result": "skipped_toxic",
                            "toxic_ratio": f"{toxic_ratio:.2f}",
                        })
                        continue

            # Collect all metrics for this program
            program_metrics: Dict[str, Any] = {}
            program_metrics.update(self._extract_graph_metrics(graph))

            # Estimate FLOPs
            try:
                flop_est = estimate_flops(graph, seq_len=min(128, config.max_seq_len),
                                          d_model=config.model_dim)
                program_metrics["flops_forward"] = flop_est.flops_forward
                program_metrics["flops_per_param"] = flop_est.flops_per_param
                program_metrics["flops_per_token"] = flop_est.flops_per_token
            except Exception as e:
                logger.debug("FLOP estimate failed for %s: %s", graph.fingerprint()[:10], e)

            # Native Proactive Gating (Project Hephaestus)
            # High-performance stability and toxic motif detection
            try:
                native_gating = _native_proactive_gating(graph)
                if not native_gating.get("passed", True):
                    results.setdefault("skipped_proactive_gating", 0)
                    results["skipped_proactive_gating"] += 1
                    
                    # Update metrics with native data
                    program_metrics["proactive_gating_reason"] = native_gating.get("reason")
                    program_metrics["max_depth"] = native_gating.get("max_depth")
                    program_metrics["n_toxic_motifs"] = native_gating.get("n_toxic_motifs")
                    
                    self._emit_event("program_evaluated", {
                        "index": i, "fingerprint": fp[:10],
                        "result": "skipped_proactive",
                        "reason": native_gating.get("reason"),
                        "max_depth": native_gating.get("max_depth"),
                    })
                    continue
            except Exception as e:
                logger.debug("Native proactive gating failed for %s: %s", fp[:10], e)

            # Validate
            validation = validate_graph(
                graph,
                max_ops=max(1, int(config.max_ops)),
                max_depth=max(1, int(config.max_depth)),
                min_splits=config.min_splits,
            )
            if not validation.valid:
                # Don't store S0 validation failures — they carry no learning
                # signal. Error counts are tracked in results dict and live feed.
                # But DO track op usage for grammar weight adaptation.
                results.setdefault("s0_validation_failures", 0)
                results["s0_validation_failures"] += 1
                for node in graph.nodes.values():
                    if not node.is_input and node.op_name:
                        c = _s0_op_counts.setdefault(node.op_name, {"n_used": 0, "n_s0": 0, "n_s05": 0})
                        c["n_used"] += 1
                self._emit_event("program_evaluated", {
                    "index": i, "fingerprint": fp[:10],
                    "result": "invalid", "error": validation.errors[0] if validation.errors else "",
                })
                continue

            # Compile & Stage 0/0.5
            try:
                # Z13: Defensive pause + GC to stabilize Torch Dynamo context if needed
                if i > 0 and i % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # More aggressive reset every 50 to clear Torch Dynamo cache
                    if i % 50 == 0:
                        try:
                            torch.compiler.reset()
                        except (AttributeError, Exception):
                            pass
                    
                    time.sleep(0.1)

                layer_graphs = [graph] * config.n_layers
                model = compile_model(layer_graphs, vocab_size=config.vocab_size, max_seq_len=config.max_seq_len)
                _eval_timeout = 60 if getattr(config, "_exotic_mode", False) else 30
                sandbox_result = self._safe_eval_for_stage(
                    model,
                    stage_tag="candidate_screening",
                    batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=config.vocab_size,
                    device=dev_str,
                    timeout_seconds=_eval_timeout,
                )
                program_metrics.update(self._extract_sandbox_metrics(sandbox_result))
                program_metrics["param_count"] = sandbox_result.param_count
                
                s0_passed = sandbox_result.passed
                s05_passed = (sandbox_result.stability_score >= config.stage05_stability_threshold
                              and sandbox_result.causality_passed)
                
                if s0_passed:
                    results["stage0_passed"] += 1
                    with self._lock: self._progress.stage0_passed += 1
                if s05_passed:
                    results["stage05_passed"] += 1
                    with self._lock: self._progress.stage05_passed += 1

                if not s0_passed or not s05_passed:
                    # Don't store S0/S0.5 failures — error counts are tracked
                    # in results dict and error_type in the live feed event.
                    # But DO track op usage for grammar weight adaptation.
                    error_type = sandbox_result.error_type or "unknown"
                    results.setdefault("failure_error_types", {})
                    results["failure_error_types"][error_type] = (
                        results["failure_error_types"].get(error_type, 0) + 1
                    )
                    for node in graph.nodes.values():
                        if not node.is_input and node.op_name:
                            c = _s0_op_counts.setdefault(node.op_name, {"n_used": 0, "n_s0": 0, "n_s05": 0})
                            c["n_used"] += 1
                            if s0_passed:
                                c["n_s0"] += 1
                            if s05_passed:
                                c["n_s05"] += 1
                    self._emit_event("program_evaluated", {
                        "index": i, "fingerprint": fp[:10],
                        "result": "fail_s0" if not s0_passed else "fail_s05",
                        "error": (sandbox_result.error or "")[:120] if not s0_passed else None,
                        "error_type": error_type,
                        "stability": f"{sandbox_result.stability_score:.2f}" if s0_passed and not s05_passed else None,
                        "params": sandbox_result.param_count if sandbox_result.param_count else None,
                        "memory_mb": f"{sandbox_result.peak_memory_mb:.1f}" if sandbox_result.peak_memory_mb else None,
                        "has_nan": sandbox_result.has_nan_output or sandbox_result.has_nan_grad or None,
                        "has_inf": sandbox_result.has_inf_output or None,
                    })
                    continue

                # Stage 1: Asynchronous Execution (Z6)
                with self._lock:
                    self._progress.current_stage = "queuing_s1"
                
                orchestrator.submit(
                    index=i,
                    graph=graph,
                    config=config,
                    seed=self._stable_seed(exp_id, i, "screening"),
                    payload={
                        "metrics": program_metrics,
                        "graph": graph,
                        "batch_id": i // candidate_batch_size,
                        "queue_kind": "candidate_screening",
                    },
                    model=model # Reuse compiled model
                )
                
            except Exception as e:
                logger.error("Error evaluating graph %d: %s", i, e)
                # Reset CUDA context if this was a fatal CUDA error
                if torch.cuda.is_available():
                    from ...eval.sandbox import is_cuda_fatal
                    if is_cuda_fatal(e):
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            _probe = torch.zeros(1, device="cuda")
                            del _probe
                            torch.cuda.synchronize()
                            logger.info("CUDA context recovered after fatal error on graph %d", i)
                        except Exception:
                            logger.warning("CUDA context unrecoverable after fatal error on graph %d", i)
                continue
            
            # Periodically process available results to keep the dashboard updated
            self._process_orchestrator_results(orchestrator, nb, exp_id, results, config)

        # Wait for remaining asynchronous Stage 1 evaluations
        with self._lock:
            self._progress.status = "finalizing_evaluations"
            
        while orchestrator.job_queue.unfinished_tasks > 0 or not orchestrator.result_queue.empty():
            if self._stop_event.is_set():
                break
            self._process_orchestrator_results(orchestrator, nb, exp_id, results, config)
            time.sleep(0.5)

        queue_telemetry = orchestrator.get_telemetry()
        orchestrator.shutdown()
        results["queue_telemetry"] = queue_telemetry
        results["perf_report"] = self._build_experiment_perf_report(results, queue_telemetry=queue_telemetry)
        results["perf_budget_gate"] = evaluate_perf_budget_gate(results["perf_report"])
        results.pop("_perf_traces", None)
        results.pop("_gpu_starvation", None)
        results.pop("_kernel_timing", None)
        if _s0_op_counts:
            results["_s0_op_counts"] = _s0_op_counts

        elapsed = time.time() - t_start
        with self._lock:
            self._progress.elapsed_seconds = elapsed
            self._progress.status = "analyzing"
            self._progress.aria_message = self.aria.begin_analysis()

        best = results.get("best_loss_ratio")
        best_str = f", best loss={best:.4f}" if best else ""
        dedup_str = ""
        if results.get("skipped_dedup", 0) > 0:
            dedup_str = f", dedup={results['skipped_dedup']} ({results.get('dedup_rate', 0)*100:.0f}%)"
        logger.info(
            "Experiment %s complete: %d programs → S0=%d → S0.5=%d → S1=%d "
            "(%.1fs)%s%s%s",
            exp_id[:8], results["total"],
            results["stage0_passed"], results["stage05_passed"],
            results["stage1_passed"], elapsed, best_str, dedup_str,
            f", native_gating={results.get('skipped_proactive_gating', 0)}" if results.get('skipped_proactive_gating') else "",
        )

        self._live_training_context = None
        return results

    def _run_investigation_thread(self, exp_id: str, result_ids: List[str],
                                   config: RunConfig, hypothesis: str):
        """Execute investigation phase in background."""
        self._live_training_context = {"exp_id": exp_id, "phase": "investigation"}
        nb = self._make_notebook()
        t_start = time.time()
        ckpt = CheckpointManager(config.checkpoint_dir)

        # Informational: log pre-inv scores for user-triggered investigations
        if config.pre_inv_gate_enabled:
            for rid in result_ids:
                try:
                    row = nb.conn.execute(
                        "SELECT pre_inv_score FROM leaderboard WHERE result_id = ?",
                        (rid,)).fetchone()
                    if row and row[0] is not None:
                        logger.info("Investigation candidate %s pre_inv_score=%.1f",
                                    rid[:8], row[0])
                except Exception:
                    pass

        # Load phase checkpoint to find where we left off
        resume_from_candidate = 0
        ckpt_state = ckpt.load_phase(exp_id, "investigation", -1, 0)
        if ckpt_state:
            resume_from_candidate = ckpt_state.get("candidate_idx", 0)
            logger.info("Resuming investigation from candidate %d", resume_from_candidate)

        try:
            results = {
                "total": len(result_ids), "stage0_passed": 0, "stage05_passed": 0,
                "stage1_passed": 0, "novel_count": 0,
                "best_loss_ratio": None, "best_novelty_score": None,
                "survivors": [], "investigation_results": [],
            }

            dev = resolve_device(config.device)
            dev_str = str(dev)

            inv_config = RunConfig.from_dict(config.to_dict())
            inv_config.stage1_steps = config.investigation_steps
            inv_config.stage1_batch_size = config.investigation_batch_size

            # Fetch all sources at once to avoid N+1 queries
            program_details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
            source_map = {d.get("result_id"): d for d in program_details if d.get("result_id")}

            for prog_idx, source_result_id in enumerate(result_ids):
                if prog_idx < resume_from_candidate:
                    continue
                if self._stop_event.is_set():
                    break

                with self._lock:
                    self._progress.current_program = prog_idx + 1
                    self._progress.status = "investigating"
                    self._progress.aria_message = (
                        f"Investigating {prog_idx + 1}/{len(result_ids)}: "
                        f"{source_result_id[:8]}... "
                        f"({config.n_training_programs} training programs)"
                    )
                    self._progress.elapsed_seconds = time.time() - t_start

                self._emit_event("investigation_progress", {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "starting",
                })

                # Fetch source program
                source = source_map.get(source_result_id)
                if source is None:
                    continue

                # Reconstruct model
                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source") or "graph_synthesis"

                # Generate training programs (queue-level scheduling telemetry)
                training_programs, tp_sched = synthesize_training_program_batch(
                    n_programs=config.n_training_programs,
                    n_steps=config.investigation_steps,
                    max_seq_len=config.max_seq_len,
                    seed_offset=prog_idx * 1000,
                )
                results.setdefault("training_program_scheduling", []).append({
                    "result_id": source_result_id,
                    **tp_sched,
                })

                # Test each (model x training_program) pair
                tp_results = []
                for tp_i, tp in enumerate(training_programs):
                    if self._stop_event.is_set():
                        break

                    # Reconstruct model fresh for each training program
                    try:
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ...morphological_box import ArchSpec
                            from ...arch_builder import build_model, BuildConfig
                            spec_data = self._cached_json_load(arch_spec_json_str)
                            spec = ArchSpec(**spec_data)
                            build_cfg = BuildConfig(
                                dim=config.model_dim,
                                n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.max_seq_len,
                            )
                            model = build_model(spec, build_cfg)
                        elif graph_json_str:
                            graph = graph_from_json(graph_json_str)
                            layer_graphs = [graph] * config.n_layers
                            model = compile_model(
                                layer_graphs,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.max_seq_len,
                            )
                        else:
                            continue
                    except Exception as e:
                        logger.debug(f"Model reconstruction failed: {e}")
                        continue

                    self._emit_event("investigation_progress", {
                        "experiment_id": exp_id,
                        "current": prog_idx + 1,
                        "total": len(result_ids),
                        "source_result_id": source_result_id,
                        "training_program": tp_i + 1,
                        "total_programs": len(training_programs),
                        "status": f"training with {tp.name}",
                    })

                    # Train with this program
                    tp_result = self._train_with_program(
                        model,
                        tp,
                        inv_config,
                        dev,
                        seed=self._stable_seed(exp_id, source_result_id, tp_i, "investigation_inline"),
                    )
                    tp_results.append({
                        "training_program": tp.name,
                        "passed": tp_result.get("passed", False),
                        "loss_ratio": tp_result.get("loss_ratio"),
                        "final_loss": tp_result.get("final_loss"),
                    })

                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                # Skip candidates where no training program could reconstruct the model
                if not tp_results:
                    logger.debug(
                        f"Threaded investigation: skipping {source_result_id[:8]} — "
                        f"model failed to reconstruct for all {len(training_programs)} programs"
                    )
                    continue

                # Compute robustness
                n_passed = sum(1 for r in tp_results if r.get("passed"))
                robustness = n_passed / max(len(tp_results), 1)
                best_tp = min(
                    (r for r in tp_results if r.get("loss_ratio") is not None),
                    key=lambda r: r["loss_ratio"],
                    default=None,
                )
                best_lr = best_tp["loss_ratio"] if best_tp else None
                screening_lr = source.get("loss_ratio")
                lr_multiplier = self._investigation_loss_multiplier(screening_lr, best_lr)
                brittle_risk = (
                    lr_multiplier is not None
                    and lr_multiplier > float(config.investigation_max_loss_ratio_multiplier)
                )

                if n_passed > 0:
                    results["stage1_passed"] += 1
                results["stage0_passed"] += 1
                results["stage05_passed"] += 1

                investigation_entry = {
                    "result_id": source_result_id,
                    "robustness": robustness,
                    "best_loss_ratio": best_lr,
                    "screening_loss_ratio": screening_lr,
                    "baseline_loss_ratio": source.get("baseline_loss_ratio"),
                    "novelty_confidence": source.get("novelty_confidence"),
                    "loss_ratio_multiplier": lr_multiplier,
                    "brittle_risk": brittle_risk,
                    "n_programs_passed": n_passed,
                    "n_programs_tested": len(tp_results),
                    "best_training_program": best_tp.get("training_program") if best_tp else None,
                    "training_program_scheduling_avg_ms": tp_sched.get("scheduling_avg_ms"),
                    "training_program_scheduling_max_ms": tp_sched.get("scheduling_max_ms"),
                }
                results["investigation_results"].append(investigation_entry)

                if best_lr and (results["best_loss_ratio"] is None
                                or best_lr < results["best_loss_ratio"]):
                    results["best_loss_ratio"] = best_lr
                source_novelty = source.get("novelty_score")
                if source_novelty is not None and (
                    results["best_novelty_score"] is None
                    or source_novelty > results["best_novelty_score"]
                ):
                    results["best_novelty_score"] = source_novelty

                # Update leaderboard
                best_tp_json = None
                if best_tp and best_tp.get("training_program"):
                    for tp in training_programs:
                        if tp.name == best_tp["training_program"]:
                            best_tp_json = json.dumps(tp.to_dict())
                            break

                # Brittle risk override: if the investigation LR is good on
                # its own merits (< 0.3), don't let the screening→investigation
                # multiplier veto promotion.  Prevents false positives when
                # screening LR was unrealistically low (e.g. lucky seed).
                investigation_passed = (
                    robustness >= 0.5
                    and (best_lr or 1.0) < 0.5
                    and (not brittle_risk
                         or (best_lr is not None and best_lr < 0.3))
                )

                # Benchmark evals (non-blocking) for investigation thread survivors
                inv_wikitext_ppl = None
                inv_wikitext_score = None
                inv_tinystories_ppl = None
                inv_tinystories_score = None
                if n_passed > 0:
                    eval_seq_len = min(128, config.max_seq_len)
                    try:
                        from ...eval.wikitext_eval import evaluate_wikitext_perplexity
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ...morphological_box import ArchSpec as _AS_wt
                            from ...arch_builder import build_model as _bm_wt, BuildConfig as _BC_wt
                            _spec_wt = _AS_wt(**self._cached_json_load(arch_spec_json_str))
                            _bc_wt = _BC_wt(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size, max_seq_len=config.max_seq_len)
                            wt_model = _bm_wt(_spec_wt, _bc_wt).to(dev)
                        else:
                            wt_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.max_seq_len).to(dev)
                        wt_result = evaluate_wikitext_perplexity(
                            wt_model, config.vocab_size, dev,
                            n_train_steps=200, seq_len=eval_seq_len)
                        inv_wikitext_ppl = wt_result.get("wikitext_perplexity")
                        inv_wikitext_score = wt_result.get("wikitext_score")
                        if inv_wikitext_ppl is not None:
                            logger.info("Investigation WikiText ppl=%.1f score=%.3f",
                                        inv_wikitext_ppl, inv_wikitext_score or 0)
                        del wt_model
                    except Exception as e:
                        logger.debug("Investigation WikiText eval skipped: %s", e)
                    try:
                        from ...eval.tinystories_eval import evaluate_tinystories
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ...morphological_box import ArchSpec as _AS_ts
                            from ...arch_builder import build_model as _bm_ts, BuildConfig as _BC_ts
                            _spec_ts = _AS_ts(**self._cached_json_load(arch_spec_json_str))
                            _bc_ts = _BC_ts(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size, max_seq_len=config.max_seq_len)
                            ts_model = _bm_ts(_spec_ts, _bc_ts).to(dev)
                        else:
                            ts_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.max_seq_len).to(dev)
                        ts_result = evaluate_tinystories(
                            ts_model, config.vocab_size, dev,
                            n_train_steps=200, seq_len=eval_seq_len)
                        inv_tinystories_ppl = ts_result.get("tinystories_perplexity")
                        inv_tinystories_score = ts_result.get("tinystories_score")
                        if inv_tinystories_ppl is not None:
                            logger.info("Investigation TinyStories ppl=%.1f score=%.3f",
                                        inv_tinystories_ppl, inv_tinystories_score or 0)
                        del ts_model
                    except Exception as e:
                        logger.debug("Investigation TinyStories eval skipped: %s", e)

                nb.upsert_leaderboard(
                    result_id=source_result_id,
                    model_source=model_source,
                    architecture_desc=source.get("graph_fingerprint", "")[:40],
                    screening_loss_ratio=source.get("loss_ratio"),
                    screening_novelty=source.get("novelty_score"),
                    screening_passed=True,
                    investigation_loss_ratio=best_lr,
                    investigation_robustness=robustness,
                    investigation_best_training=best_tp_json,
                    investigation_passed=investigation_passed,
                    tier="investigation" if investigation_passed else "screening",
                    novelty_confidence=source.get("novelty_confidence"),
                    fp_jacobian_spectral_norm=source.get("fp_jacobian_spectral_norm"),
                    wikitext_perplexity=inv_wikitext_ppl,
                    wikitext_score=inv_wikitext_score,
                    tinystories_perplexity=inv_tinystories_ppl,
                    tinystories_score=inv_tinystories_score,
                )

                # Record result
                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=source.get("graph_fingerprint", source_result_id),
                    graph_json=graph_json_str or "{}",
                    stage0_passed=True,
                    stage05_passed=True,
                    stage1_passed=n_passed > 0,
                    loss_ratio=best_lr,
                    novelty_score=source.get("novelty_score"),
                    novelty_confidence=source.get("novelty_confidence"),
                    novelty_raw_score=source.get("novelty_raw_score"),
                    novelty_z_score=source.get("novelty_z_score"),
                    novelty_reference_version=source.get("novelty_reference_version"),
                    novelty_valid_for_promotion=source.get("novelty_valid_for_promotion"),
                    novelty_validity_reason=source.get("novelty_validity_reason"),
                    novelty_requires_justification=source.get("novelty_requires_justification"),
                    training_program_json=best_tp_json,
                    model_source=model_source,
                    arch_spec_json=arch_spec_json_str,
                    wikitext_perplexity=inv_wikitext_ppl,
                    wikitext_score=inv_wikitext_score,
                    tinystories_perplexity=inv_tinystories_ppl,
                    tinystories_score=inv_tinystories_score,
                )

                # Save checkpoint after each candidate completes
                try:
                    ckpt.save_phase(
                        experiment_id=exp_id,
                        phase="investigation",
                        candidate_idx=prog_idx + 1,
                        seed_idx=0,
                        model_state_dict={},
                        optimizer_state_dict={},
                        step=0,
                        metrics={"completed_candidate": prog_idx},
                    )
                    # Also save a progress marker at index -1 for resume
                    ckpt.save_phase(
                        experiment_id=exp_id,
                        phase="investigation",
                        candidate_idx=-1,
                        seed_idx=0,
                        model_state_dict={},
                        optimizer_state_dict={},
                        step=0,
                        metrics={"candidate_idx": prog_idx + 1},
                    )
                except Exception as e:
                    logger.debug("Investigation checkpoint save failed: %s", e)

            # Complete experiment
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=self._analyze_results(results, exp_id, nb, context=context),
                llm_analysis=llm_analysis,
            )

            # Auto-escalate to validation
            self._auto_escalate(results, config, nb, phase="investigation")

            # Clean up investigation checkpoints on success
            if not config.keep_checkpoints:
                try:
                    ckpt.cleanup(exp_id)
                except Exception:
                    pass

            with self._lock:
                self._progress.status = "completed"
                self._progress.elapsed_seconds = time.time() - t_start
                self._progress.aria_message = summary.split("\n")[-1] if summary else "Investigation complete."

            self._emit_event("investigation_completed", {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
            })

        except Exception as e:
            error = traceback.format_exc()
            logger.error("Investigation failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Investigation failure: {str(e)[:240]}",
                reproduction_steps=["python -m pytest tests/test_integration.py -k \"investigation\" -x --tb=short"],
                acceptance_tests=["python -m pytest tests/test_integration.py -k \"investigation\" -x --tb=short"],
                trigger_payload={"mode": "investigation", "error": str(e)},
            )
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
            self._emit_event("experiment_failed", {
                "experiment_id": exp_id,
                "error": str(e),
            })
        finally:
            self._live_training_context = None
            nb.close()
            self._run_pending_scale_up()

    # ── Validation Phase ──

    def _run_validation_thread(self, exp_id: str, result_ids: List[str],
                                config: RunConfig, hypothesis: str):
        """Execute validation phase in background."""
        self._live_training_context = {"exp_id": exp_id, "phase": "validation"}
        nb = self._make_notebook()
        t_start = time.time()
        ckpt = CheckpointManager(config.checkpoint_dir)

        # Load phase checkpoint to find where we left off
        resume_from_candidate = 0
        ckpt_state = ckpt.load_phase(exp_id, "validation", -1, 0)
        if ckpt_state:
            resume_from_candidate = ckpt_state.get("candidate_idx", 0)
            logger.info("Resuming validation from candidate %d", resume_from_candidate)

        try:
            results = {
                "total": len(result_ids), "stage0_passed": 0, "stage05_passed": 0,
                "stage1_passed": 0, "novel_count": 0,
                "best_loss_ratio": None, "best_novelty_score": None,
                "survivors": [], "validation_results": [],
            }

            dev = resolve_device(config.device)
            dev_str = str(dev)

            val_config = RunConfig.from_dict(config.to_dict())
            val_config.stage1_steps = config.validation_steps
            val_config.stage1_batch_size = config.validation_batch_size
            val_config.max_seq_len = config.validation_seq_len

            # Fetch all sources at once to avoid N+1 queries
            program_details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
            source_map = {d.get("result_id"): d for d in program_details if d.get("result_id")}

            for prog_idx, source_result_id in enumerate(result_ids):
                if prog_idx < resume_from_candidate:
                    continue
                if self._stop_event.is_set():
                    break

                with self._lock:
                    self._progress.current_program = prog_idx + 1
                    self._progress.status = "validating"
                    self._progress.aria_message = (
                        f"Validating {prog_idx + 1}/{len(result_ids)}: "
                        f"{source_result_id[:8]}... "
                        f"({config.validation_n_seeds} seeds, "
                        f"{config.validation_steps} steps)"
                    )
                    self._progress.elapsed_seconds = time.time() - t_start

                self._emit_event("validation_progress", {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "starting",
                })

                # Fetch source and leaderboard entry
                source = source_map.get(source_result_id)
                if source is None:
                    continue

                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source") or "graph_synthesis"

                # Get best training program from investigation
                leaderboard_entries = nb.get_leaderboard()
                best_tp_json = None
                for entry in leaderboard_entries:
                    if entry.get("result_id") == source_result_id:
                        best_tp_json = entry.get("investigation_best_training")
                        break

                # Multi-seed evaluation (threaded validation)
                seed_results = []
                for seed in range(config.validation_n_seeds):
                    if self._stop_event.is_set():
                        break

                    torch.manual_seed(seed * 42 + 7)

                    # Reconstruct model fresh
                    init_scheme = "default"
                    try:
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ...morphological_box import ArchSpec
                            from ...arch_builder import build_model, BuildConfig
                            spec_data = self._cached_json_load(arch_spec_json_str)
                            spec = ArchSpec(**spec_data)
                            build_cfg = BuildConfig(
                                dim=config.model_dim,
                                n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len,
                            )
                            model = build_model(spec, build_cfg)
                        elif graph_json_str:
                            graph = graph_from_json(graph_json_str)
                            layer_graphs = [graph] * config.n_layers
                            model = compile_model(
                                layer_graphs,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len,
                            )
                        else:
                            continue
                        # Multi-init: use Xavier uniform for the last seed
                        if seed == config.validation_n_seeds - 1:
                            init_scheme = "xavier_uniform"
                            for p in model.parameters():
                                if p.dim() >= 2:
                                    nn.init.xavier_uniform_(p)
                    except Exception as e:
                        logger.debug(f"Model reconstruction failed: {e}")
                        continue

                    self._emit_event("validation_progress", {
                        "experiment_id": exp_id,
                        "current": prog_idx + 1,
                        "total": len(result_ids),
                        "source_result_id": source_result_id,
                        "seed": seed + 1,
                        "total_seeds": config.validation_n_seeds,
                        "status": f"seed {seed + 1}/{config.validation_n_seeds}",
                    })

                    # Train (use best training program if available)
                    if best_tp_json:
                        try:
                            tp_data = self._cached_json_load(best_tp_json)
                            tp = synthesize_training_program(
                                n_steps=config.validation_steps,
                                max_seq_len=config.validation_seq_len,
                                seed=tp_data.get("seed", seed),
                            )
                            s1_result = self._train_with_program(
                                model,
                                tp,
                                val_config,
                                dev,
                                seed=self._stable_seed(exp_id, source_result_id, seed, "validation_inline_tp"),
                            )
                        except Exception:
                            s1_result = self._micro_train(
                                model,
                                val_config,
                                dev,
                                seed=self._stable_seed(exp_id, source_result_id, seed, "validation_inline_micro"),
                            )
                    else:
                        s1_result = self._micro_train(
                            model,
                            val_config,
                            dev,
                            seed=self._stable_seed(exp_id, source_result_id, seed, "validation_inline_micro"),
                        )

                    seed_results.append({
                        "seed": seed,
                        "init_scheme": init_scheme,
                        "passed": s1_result.get("passed", False),
                        "loss_ratio": s1_result.get("loss_ratio"),
                        "final_loss": s1_result.get("final_loss"),
                        "n_train_steps": s1_result.get("n_train_steps"),
                        "final_lr": s1_result.get("final_lr"),
                        "training_program_json": s1_result.get("training_program_json"),
                        "optimizer_class": s1_result.get("optimizer_class"),
                        "optimizer_lr": s1_result.get("optimizer_lr"),
                        "optimizer_weight_decay": s1_result.get("optimizer_weight_decay"),
                        "optimizer_momentum": s1_result.get("optimizer_momentum"),
                        "optimizer_beta1": s1_result.get("optimizer_beta1"),
                        "optimizer_beta2": s1_result.get("optimizer_beta2"),
                    })

                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()


                # Skip candidates where no seed could reconstruct the model
                if not seed_results:
                    logger.debug(
                        f"Threaded validation: skipping {source_result_id[:8]} — "
                        f"model failed to reconstruct for all {config.validation_n_seeds} seeds"
                    )
                    continue

                # Compute validation metrics
                passed_seeds = [r for r in seed_results if r.get("passed")]
                loss_ratios = [r["loss_ratio"] for r in seed_results
                               if r.get("loss_ratio") is not None]

                val_loss_ratio = (sum(loss_ratios) / len(loss_ratios)
                                  if loss_ratios else None)
                multi_seed_std = 0.0
                if len(loss_ratios) > 1:
                    mean_lr = sum(loss_ratios) / len(loss_ratios)
                    multi_seed_std = (
                        sum((lr - mean_lr) ** 2 for lr in loss_ratios)
                        / len(loss_ratios)
                    ) ** 0.5

                # Init sensitivity: std between default and xavier seeds
                init_sensitivity_std = None
                default_losses = [
                    r["loss_ratio"] for r in seed_results
                    if r.get("init_scheme") == "default" and r.get("loss_ratio") is not None
                ]
                xavier_losses = [
                    r["loss_ratio"] for r in seed_results
                    if r.get("init_scheme") == "xavier_uniform" and r.get("loss_ratio") is not None
                ]
                if default_losses and xavier_losses:
                    default_mean = sum(default_losses) / len(default_losses)
                    xavier_mean = sum(xavier_losses) / len(xavier_losses)
                    init_sensitivity_std = abs(default_mean - xavier_mean)

                # Baseline comparison at validation scale
                val_baseline_ratio = None
                if loss_ratios:
                    best_seed = min(
                        (r for r in seed_results if r.get("final_loss") is not None),
                        key=lambda r: r["final_loss"],
                        default=None,
                    )
                    if best_seed is not None:
                        try:
                            baseline = self._get_baseline()
                            baseline_steps = int(best_seed.get("n_train_steps") or config.validation_steps)
                            baseline_recipe = self._resolve_baseline_recipe(
                                best_seed, default_lr=config.stage1_lr)
                            bl_data_fn, bl_data_tag, bl_cache = self._make_baseline_data_fn(config)
                            val_baseline_ratio = baseline.compare(
                                best_seed["final_loss"],
                                d_model=config.model_dim,
                                seq_len=min(128, config.validation_seq_len),
                                n_steps=max(1, baseline_steps),
                                vocab_size=config.vocab_size,
                                batch_size=config.validation_batch_size,
                                lr=baseline_recipe["lr"],
                                device=dev_str,
                                n_layers=config.n_layers,
                                optimizer_name=baseline_recipe["optimizer_name"],
                                weight_decay=baseline_recipe["weight_decay"],
                                momentum=baseline_recipe["momentum"],
                                betas=baseline_recipe["betas"],
                                data_fn=bl_data_fn,
                                data_tag=bl_data_tag,
                                cache_data_fn=bl_cache,
                            )
                            # Optional: Validation baseline comparison (using val split)
                            v_loss = best_seed.get("validation_loss")
                            if v_loss is not None:
                                try:
                                    v_data_fn, v_data_tag, v_cache = self._make_baseline_data_fn(config, split="val")
                                    v_baseline_ratio = baseline.compare(
                                        v_loss,
                                        d_model=config.model_dim,
                                        seq_len=min(128, int(getattr(config, "validation_seq_len", 128))),
                                        n_steps=max(1, baseline_steps),
                                        vocab_size=config.vocab_size,
                                        batch_size=int(getattr(config, "validation_batch_size", 4)),
                                        lr=baseline_recipe["lr"],
                                        device=dev_str,
                                        n_layers=config.n_layers,
                                        optimizer_name=baseline_recipe["optimizer_name"],
                                        weight_decay=baseline_recipe["weight_decay"],
                                        momentum=baseline_recipe["momentum"],
                                        betas=baseline_recipe["betas"],
                                        data_fn=v_data_fn,
                                        data_tag=v_data_tag,
                                        cache_data_fn=v_cache,
                                    )
                                    program_metrics["validation_baseline_loss_ratio"] = v_baseline_ratio
                                except Exception:
                                    pass
                        except Exception:
                            pass

                # Parameter-normalized baseline comparison
                val_normalized_ratio = None
                val_param_efficiency = None
                source_params = (source.get("param_count")
                                 or source.get("graph_n_params_estimate")
                                 or 0) if source else 0
                if loss_ratios and best_seed is not None and source_params > 0:
                    try:
                        baseline = self._get_baseline()
                        baseline_steps = int(best_seed.get("n_train_steps") or config.validation_steps)
                        baseline_recipe = self._resolve_baseline_recipe(
                            best_seed, default_lr=config.stage1_lr)
                        bl_data_fn, bl_data_tag, bl_cache = self._make_baseline_data_fn(config)
                        norm_result = baseline.compare_normalized(
                            best_seed["final_loss"],
                            program_params=int(source_params),
                            d_model=config.model_dim,
                            seq_len=min(128, config.validation_seq_len),
                            n_steps=max(1, baseline_steps),
                            vocab_size=config.vocab_size,
                            batch_size=config.validation_batch_size,
                            lr=baseline_recipe["lr"],
                            device=dev_str,
                            n_layers=config.n_layers,
                            optimizer_name=baseline_recipe["optimizer_name"],
                            weight_decay=baseline_recipe["weight_decay"],
                            momentum=baseline_recipe["momentum"],
                            betas=baseline_recipe["betas"],
                            data_fn=bl_data_fn,
                            data_tag=bl_data_tag,
                            cache_data_fn=bl_cache,
                        )
                        val_normalized_ratio = norm_result.get("normalized_ratio")
                        val_param_efficiency = norm_result.get("param_efficiency")
                    except Exception:
                        pass

                if len(passed_seeds) > 0:
                    results["stage1_passed"] += 1
                results["stage0_passed"] += 1
                results["stage05_passed"] += 1

                # OOD robustness check (#54): test with reference recipes
                ood_result = None
                if len(passed_seeds) > 0:
                    _gjs_t = graph_json_str
                    _asjs_t = arch_spec_json_str
                    _ms_t = model_source
                    _cfg_t = config

                    def _make_model_t():
                        if _ms_t == "morphological_box" and _asjs_t:
                            from ...morphological_box import ArchSpec
                            from ...arch_builder import build_model, BuildConfig
                            spec = ArchSpec(**json.loads(_asjs_t))
                            bc = BuildConfig(
                                dim=_cfg_t.model_dim,
                                n_layers=_cfg_t.n_layers,
                                vocab_size=_cfg_t.vocab_size,
                                max_seq_len=_cfg_t.validation_seq_len)
                            return build_model(spec, bc)
                        else:
                            g = graph_from_json(_gjs_t)
                            return compile_model(
                                [g] * _cfg_t.n_layers,
                                vocab_size=_cfg_t.vocab_size,
                                max_seq_len=_cfg_t.validation_seq_len)

                    try:
                        ood_result = self._ood_robustness_check(
                            _make_model_t, config, dev,
                            n_steps=min(300, config.validation_steps // 3),
                            seed=self._stable_seed(
                                exp_id, source_result_id, 0, "ood"),
                        )
                    except Exception as e:
                        logger.debug("OOD robustness check failed: %s", e)

                # Hyperparameter sensitivity check (#57)
                sensitivity_result = None
                if len(passed_seeds) > 0 and val_loss_ratio is not None:
                    try:
                        sensitivity_result = self._sensitivity_check(
                            _make_model_t, config, dev,
                            base_loss_ratio=val_loss_ratio,
                            n_steps=min(300, config.validation_steps // 3),
                            seed=self._stable_seed(
                                exp_id, source_result_id, 0, "sensitivity"),
                        )
                    except Exception as e:
                        logger.debug("Sensitivity check failed: %s", e)

                # Determine if breakthrough — requires both raw AND normalized thresholds
                ood_ok = (ood_result is not None
                          and ood_result.get("ood_robustness", 0) >= 0.67)
                hp_ok = (sensitivity_result is not None
                         and sensitivity_result.get("hp_robustness", 0) >= 0.75)
                nov_conf = source.get("novelty_confidence", 0) if source else 0
                novelty_valid = False
                if source:
                    novelty_valid = bool(source.get("novelty_valid_for_promotion"))
                    if not novelty_valid and source.get("cka_source") == "artifact":
                        novelty_valid = True

                raw_threshold = config.breakthrough_raw_threshold
                norm_threshold = config.breakthrough_normalized_threshold
                raw_ok = (val_baseline_ratio is not None
                          and val_baseline_ratio < raw_threshold)
                norm_ok = (val_normalized_ratio is None
                           or val_normalized_ratio < norm_threshold)
                is_breakthrough = (
                    raw_ok
                    and norm_ok
                    and multi_seed_std <= 0.03
                    and len(passed_seeds) >= 5
                    and len(passed_seeds) == config.validation_n_seeds
                    and (ood_result is None or ood_ok)
                    and (sensitivity_result is None or hp_ok)
                    and nov_conf >= 0.5
                    and novelty_valid
                )

                # FLOP gate: reject breakthrough if >5x baseline FLOPs per token
                flop_gated = False
                if is_breakthrough and source_params > 0:
                    candidate_fpt = source_params * 2.0
                    baseline_fpt_gate = 2.0 * config.model_dim ** 2 * config.n_layers
                    if candidate_fpt > 5.0 * baseline_fpt_gate:
                        is_breakthrough = False
                        flop_gated = True
                        logger.info(
                            "FLOP gate downgraded %s: %.0f FPT > 5x baseline %.0f",
                            source_result_id[:8], candidate_fpt, baseline_fpt_gate,
                        )

                # Scaling law comparison gate
                scaling_result = None
                scaling_param_efficiency = None
                scaling_flop_efficiency = None
                scaling_gate_passed_val = None
                scaling_best_family = None
                scaling_confidence = None
                if is_breakthrough and config.enable_scaling_comparison:
                    try:
                        scaling_mgr = self._get_scaling_reference_manager()
                        bl_data_fn, bl_data_tag, _ = self._make_baseline_data_fn(config)
                        candidate_flops = (source.get("flops_forward", 0) or 0)
                        if candidate_flops <= 0:
                            candidate_flops = source_params * 2

                        scaling_result = scaling_mgr.compare_candidate(
                            candidate_loss=best_seed_loss,
                            candidate_params=source_params,
                            candidate_flops=candidate_flops,
                            d_model=config.model_dim,
                            n_steps=config.validation_steps,
                            seq_len=config.validation_seq_len,
                            vocab_size=config.vocab_size,
                            batch_size=config.validation_batch_size,
                            lr=config.stage1_lr,
                            device=dev_str,
                            data_fn=bl_data_fn, data_tag=bl_data_tag,
                            families=config.scaling_reference_families.split(","),
                            param_efficiency_target=config.scaling_param_efficiency_target,
                            flop_ceiling=config.scaling_flop_ceiling,
                        )
                        scaling_param_efficiency = scaling_result.best_param_efficiency
                        scaling_flop_efficiency = scaling_result.flop_efficiency
                        scaling_gate_passed_val = scaling_result.scaling_gate_passed
                        scaling_best_family = scaling_result.best_param_efficiency_family
                        scaling_confidence = scaling_result.confidence

                        if not scaling_result.scaling_gate_passed:
                            is_breakthrough = False
                            logger.info(
                                "Scaling gate downgraded %s: param_eff=%.2f (need %.1f), flop_eff=%.2f",
                                source_result_id[:8],
                                scaling_result.best_param_efficiency,
                                config.scaling_param_efficiency_target,
                                scaling_result.flop_efficiency,
                            )
                    except Exception as e:
                        logger.debug("Scaling comparison failed: %s", e)

                # Quantization eval: test INT8 retention for all validation candidates
                quant_int8_retention = None
                quant_quality_per_byte = None
                if best_seed is not None:
                    try:
                        from ...eval.quantization import evaluate_sparse_quant_quality
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ...morphological_box import ArchSpec
                            from ...arch_builder import build_model, BuildConfig
                            _spec = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            quant_model = build_model(_spec, _bc).to(dev)
                        else:
                            quant_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        quant_batches = [
                            torch.randint(0, config.vocab_size,
                                          (2, min(128, config.validation_seq_len)),
                                          device=dev)
                            for _ in range(4)
                        ]
                        quant_result = evaluate_sparse_quant_quality(
                            quant_model, quant_batches, dev,
                            target_sparsity=0.5, bits=8)
                        if quant_result is not None:
                            quant_int8_retention = quant_result.get("full_retention")
                            quant_quality_per_byte = quant_result.get("quality_per_byte")
                            if is_breakthrough and quant_int8_retention is not None and quant_int8_retention < 0.80:
                                is_breakthrough = False
                                logger.info(
                                    "Quant gate downgraded %s: INT8 retention=%.3f < 0.80",
                                    source_result_id[:8], quant_int8_retention,
                                )
                        del quant_model
                    except Exception as e:
                        logger.debug("Quantization eval skipped: %s", e)

                # Long-context sweep (informational, non-blocking)
                long_context_score = None
                long_context_details = None
                if best_seed is not None:
                    try:
                        from ...eval.long_context import run_long_context_sweep
                        base_loss_val = best_seed.get("final_loss", 0)
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _asjs_lc2 = arch_spec_json_str
                            _cfg_lc2 = config
                            def _make_model_lc2():
                                from ...morphological_box import ArchSpec
                                from ...arch_builder import build_model, BuildConfig
                                _sp2 = ArchSpec(**json.loads(_asjs_lc2))
                                _bc3 = BuildConfig(
                                    dim=_cfg_lc2.model_dim, n_layers=_cfg_lc2.n_layers,
                                    vocab_size=_cfg_lc2.vocab_size, max_seq_len=1024)
                                return build_model(_sp2, _bc3)
                        else:
                            _gjs_lc2 = graph_json_str
                            _cfg_lc2 = config
                            def _make_model_lc2():
                                return compile_model(
                                    [graph_from_json(_gjs_lc2)] * _cfg_lc2.n_layers,
                                    vocab_size=_cfg_lc2.vocab_size, max_seq_len=1024)
                        from ...eval.long_context import run_long_context_sweep
                        from ...eval.passkey import evaluate_long_context_retrieval

                        lc_result = run_long_context_sweep(
                            _make_model_lc2, config.vocab_size, dev,
                            base_loss=base_loss_val, seq_lens=(512, 1024),
                            n_steps=200, batch_size=2,
                        )
                        
                        # Retrieval test (needle-in-a-haystack)
                        retr_model = _make_model_lc2().to(dev)
                        retr_result = evaluate_long_context_retrieval(
                            retr_model, config.vocab_size, dev,
                            lengths=[256, 512, 1024]
                        )
                        del retr_model
                        
                        # Combine scaling score and retrieval aggregate (50/50)
                        scaling_score = lc_result.get("long_context_score", 0.0)
                        retrieval_score = retr_result.get(
                            "retrieval_aggregate_score",
                            retr_result.get("retrieval_score", 0.0),
                        )
                        assoc_retrieval_score = retr_result.get("assoc_retrieval_score", retr_result.get("retrieval_score", 0.0))
                        passkey_score = retr_result.get("passkey_score", 0.0)
                        multi_hop_score = retr_result.get("multi_hop_score", 0.0)
                        long_context_score = (scaling_score * 0.5) + (retrieval_score * 0.5)

                        long_context_details = {
                            "scaling": lc_result,
                            "retrieval": retr_result,
                            "scaling_score": scaling_score,
                            "assoc_retrieval_score": assoc_retrieval_score,
                            "multi_hop_score": multi_hop_score,
                            "passkey_score": passkey_score,
                            "retrieval_aggregate_score": retrieval_score,
                            "combined_score": long_context_score,
                            "benchmark_version": "v3_assoc_multihop_passkey",
                        }

                        logger.info(
                            "Long-context check (v2): scaling=%.2f, assoc=%.2f, multi_hop=%.2f, passkey=%.2f, retrieval=%.2f, combined=%.2f",
                            scaling_score,
                            assoc_retrieval_score,
                            multi_hop_score,
                            passkey_score,
                            retrieval_score,
                            long_context_score,
                        )
                    except Exception as e:
                        logger.debug("Long-context sweep skipped: %s", e)

                # Noise sensitivity (informational, non-blocking)
                noise_score = None
                if best_seed is not None:
                    try:
                        from ...eval.noise_sensitivity import evaluate_noise_sensitivity
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ...morphological_box import ArchSpec
                            from ...arch_builder import build_model, BuildConfig
                            _spec_ns2 = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_ns2 = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            ns_model = build_model(_spec_ns2, _bc_ns2).to(dev)
                        else:
                            ns_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        ns_batches = [
                            torch.randint(0, config.vocab_size,
                                          (2, min(128, config.validation_seq_len)),
                                          device=dev)
                            for _ in range(4)
                        ]
                        ns_result = evaluate_noise_sensitivity(
                            ns_model, ns_batches, dev)
                        noise_score = ns_result.get("noise_sensitivity_score")
                        del ns_model
                    except Exception as e:
                        logger.debug("Noise sensitivity skipped: %s", e)

                # Activation sparsity analysis (informational, non-blocking)
                activation_sparsity_score = None
                dead_neuron_ratio = None
                if best_seed is not None:
                    try:
                        from ...eval.sparsity import evaluate_activation_sparsity
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_as = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_as = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            as_model = build_model(_spec_as, _bc_as).to(dev)
                        else:
                            as_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        as_batches = [
                            torch.randint(0, config.vocab_size,
                                          (2, min(128, config.validation_seq_len)),
                                          device=dev)
                            for _ in range(4)
                        ]
                        as_result = evaluate_activation_sparsity(
                            as_model, as_batches, dev)
                        activation_sparsity_score = as_result.get("activation_sparsity_score")
                        dead_neuron_ratio = as_result.get("dead_neuron_ratio")
                        del as_model
                    except Exception as e:
                        logger.debug("Activation sparsity eval skipped: %s", e)

                # Routing heatmap / collapse detection (informational, non-blocking)
                routing_collapse_score = None
                if best_seed is not None:
                    try:
                        from ...eval.routing_heatmap import evaluate_routing_heatmap
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_rh = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_rh = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            rh_model = build_model(_spec_rh, _bc_rh).to(dev)
                        else:
                            rh_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        rh_batches = [
                            torch.randint(0, config.vocab_size,
                                          (2, min(128, config.validation_seq_len)),
                                          device=dev)
                            for _ in range(4)
                        ]
                        rh_result = evaluate_routing_heatmap(
                            rh_model, rh_batches, dev)
                        if rh_result.get("has_routing"):
                            routing_collapse_score = rh_result.get("routing_collapse_score")
                        del rh_model
                    except Exception as e:
                        logger.debug("Routing heatmap eval skipped: %s", e)

                # WikiText perplexity (informational, non-blocking)
                wikitext_perplexity = None
                wikitext_score = None
                if best_seed is not None:
                    try:
                        from ...eval.wikitext_eval import evaluate_wikitext_perplexity
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_wt = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_wt = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            wt_model = build_model(_spec_wt, _bc_wt).to(dev)
                        else:
                            wt_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        wt_result = evaluate_wikitext_perplexity(
                            wt_model, config.vocab_size, dev,
                            n_train_steps=200, seq_len=min(128, config.validation_seq_len))
                        wikitext_perplexity = wt_result.get("wikitext_perplexity")
                        wikitext_score = wt_result.get("wikitext_score")
                        if wikitext_perplexity is not None:
                            logger.info("WikiText ppl=%.1f score=%.3f",
                                        wikitext_perplexity, wikitext_score or 0)
                        del wt_model
                    except Exception as e:
                        logger.debug("WikiText eval skipped: %s", e)

                # TinyStories validation (informational, non-blocking)
                tinystories_perplexity = None
                tinystories_score = None
                if best_seed is not None:
                    try:
                        from ...eval.tinystories_eval import evaluate_tinystories
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_ts = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_ts = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            ts_model = build_model(_spec_ts, _bc_ts).to(dev)
                        else:
                            ts_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        ts_result = evaluate_tinystories(
                            ts_model, config.vocab_size, dev,
                            n_train_steps=200, seq_len=min(128, config.validation_seq_len))
                        tinystories_perplexity = ts_result.get("tinystories_perplexity")
                        tinystories_score = ts_result.get("tinystories_score")
                        del ts_model
                    except Exception as e:
                        logger.debug("TinyStories eval skipped: %s", e)

                # Cross-task robustness (informational, non-blocking)
                cross_task_score = None
                if best_seed is not None:
                    try:
                        from ...eval.cross_task_eval import evaluate_cross_task_robustness
                        _gjs_ct = graph_json_str
                        _asjs_ct = arch_spec_json_str
                        _ms_ct = model_source
                        _cfg_ct = config
                        def _make_ct_model():
                            if _ms_ct == "morphological_box" and _asjs_ct:
                                _sp = ArchSpec(**json.loads(_asjs_ct))
                                _bc = BuildConfig(
                                    dim=_cfg_ct.model_dim, n_layers=_cfg_ct.n_layers,
                                    vocab_size=_cfg_ct.vocab_size,
                                    max_seq_len=_cfg_ct.validation_seq_len)
                                return build_model(_sp, _bc)
                            return compile_model(
                                [graph_from_json(_gjs_ct)] * _cfg_ct.n_layers,
                                vocab_size=_cfg_ct.vocab_size,
                                max_seq_len=_cfg_ct.validation_seq_len)
                        ct_result = evaluate_cross_task_robustness(
                            _make_ct_model, config.vocab_size, dev,
                            n_train_steps=100, seq_len=min(128, config.validation_seq_len))
                        cross_task_score = ct_result.get("cross_task_score")
                    except Exception as e:
                        logger.debug("Cross-task eval skipped: %s", e)

                # Efficiency wall (informational, non-blocking)
                efficiency_wall_score = None
                max_viable_seq_len = None
                scaling_regime = None
                if best_seed is not None:
                    try:
                        from ...eval.efficiency_wall import evaluate_efficiency_wall
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_ew = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_ew = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size, max_seq_len=1024)
                            ew_model = build_model(_spec_ew, _bc_ew).to(dev)
                        else:
                            ew_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size, max_seq_len=1024).to(dev)
                        ew_result = evaluate_efficiency_wall(
                            ew_model, config.vocab_size, dev,
                            seq_lens=(64, 128, 256, 512), batch_size=2)
                        efficiency_wall_score = ew_result.get("efficiency_wall_score")
                        max_viable_seq_len = ew_result.get("max_viable_seq_len")
                        scaling_regime = ew_result.get("scaling_regime")
                        del ew_model
                    except Exception as e:
                        logger.debug("Efficiency wall eval skipped: %s", e)

                tier = "breakthrough" if is_breakthrough else "validation"

                validation_entry = {
                    "result_id": source_result_id,
                    "val_loss_ratio": val_loss_ratio,
                    "val_baseline_ratio": val_baseline_ratio,
                    "val_normalized_ratio": val_normalized_ratio,
                    "param_efficiency": val_param_efficiency,
                    "multi_seed_std": multi_seed_std,
                    "seeds_passed": len(passed_seeds),
                    "total_seeds": config.validation_n_seeds,
                    "is_breakthrough": is_breakthrough,
                    "flop_gated": flop_gated,
                    "quant_int8_retention": quant_int8_retention,
                    "quant_quality_per_byte": quant_quality_per_byte,
                    "long_context_score": long_context_score,
                    "noise_sensitivity_score": noise_score,
                    "init_sensitivity_std": init_sensitivity_std,
                    "novelty_confidence": nov_conf,
                    "ood_robustness": ood_result,
                    "sensitivity": sensitivity_result,
                    "activation_sparsity_score": activation_sparsity_score,
                    "dead_neuron_ratio": dead_neuron_ratio,
                    "routing_collapse_score": routing_collapse_score,
                    "wikitext_perplexity": wikitext_perplexity,
                    "wikitext_score": wikitext_score,
                    "tinystories_perplexity": tinystories_perplexity,
                    "tinystories_score": tinystories_score,
                    "cross_task_score": cross_task_score,
                    "efficiency_wall_score": efficiency_wall_score,
                    "max_viable_seq_len": max_viable_seq_len,
                    "scaling_regime": scaling_regime,
                }
                results["validation_results"].append(validation_entry)

                if val_loss_ratio and (results["best_loss_ratio"] is None
                                       or val_loss_ratio < results["best_loss_ratio"]):
                    results["best_loss_ratio"] = val_loss_ratio
                source_novelty = source.get("novelty_score")
                if source_novelty is not None and (
                    results["best_novelty_score"] is None
                    or source_novelty > results["best_novelty_score"]
                ):
                    results["best_novelty_score"] = source_novelty

                # Update leaderboard — find the actual entry for this result
                for entry in nb.get_leaderboard(limit=200):
                    if entry.get("result_id") == source_result_id:
                        nb.promote_to_tier(
                            entry_id=entry["entry_id"],
                            tier=tier,
                            validation_loss_ratio=val_loss_ratio,
                            validation_baseline_ratio=val_baseline_ratio,
                            validation_multi_seed_std=multi_seed_std,
                            validation_passed=len(passed_seeds) > 0,
                            normalized_baseline_ratio=val_normalized_ratio,
                            param_efficiency=val_param_efficiency,
                            quant_int8_retention=quant_int8_retention,
                            quant_quality_per_byte=quant_quality_per_byte,
                            robustness_long_ctx_score=long_context_score,
                            robustness_noise_score=noise_score,
                            init_sensitivity_std=init_sensitivity_std,
                            fp_jacobian_spectral_norm=source.get("fp_jacobian_spectral_norm"),
                            scaling_param_efficiency=scaling_param_efficiency,
                            scaling_flop_efficiency=scaling_flop_efficiency,
                            scaling_gate_passed=scaling_gate_passed_val,
                            scaling_best_family=scaling_best_family,
                            scaling_confidence=scaling_confidence,
                            activation_sparsity_score=activation_sparsity_score,
                            dead_neuron_ratio=dead_neuron_ratio,
                            routing_collapse_score=routing_collapse_score,
                            wikitext_perplexity=wikitext_perplexity,
                            wikitext_score=wikitext_score,
                            tinystories_perplexity=tinystories_perplexity,
                            tinystories_score=tinystories_score,
                            cross_task_score=cross_task_score,
                            efficiency_wall_score=efficiency_wall_score,
                            max_viable_seq_len=max_viable_seq_len,
                            scaling_regime=scaling_regime,
                        )
                        # Store detailed benchmark payload
                        external_benchmarks_payload = {}
                        if scaling_result is not None:
                            scaling_payload = scaling_result.to_dict()
                            if isinstance(scaling_payload, dict):
                                external_benchmarks_payload.update(scaling_payload)
                                external_benchmarks_payload["scaling_comparison"] = scaling_payload
                        if long_context_details is not None:
                            external_benchmarks_payload["long_context"] = long_context_details
                        if external_benchmarks_payload:
                            nb.set_external_benchmarks(source_result_id, external_benchmarks_payload)
                        break

                # Breakthrough detection
                if is_breakthrough:
                    ctx = build_validation_context(
                        [source], [validation_entry])
                    announcement = self.aria.announce_breakthrough(ctx)
                    nb.add_entry(ExperimentEntry(
                        entry_type="insight",
                        title="BREAKTHROUGH DETECTED",
                        content=announcement,
                        experiment_id=exp_id,
                        tags=["breakthrough"],
                    ))
                    self._emit_event("breakthrough_detected", {
                        "experiment_id": exp_id,
                        "result_id": source_result_id,
                        "val_loss_ratio": val_loss_ratio,
                        "val_baseline_ratio": val_baseline_ratio,
                        "multi_seed_std": multi_seed_std,
                        "announcement": announcement,
                    })

                # Record validation result
                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=source.get("graph_fingerprint",
                                                 source_result_id),
                    graph_json=graph_json_str or "{}",
                    stage0_passed=True,
                    stage05_passed=True,
                    stage1_passed=len(passed_seeds) > 0,
                    loss_ratio=val_loss_ratio,
                    baseline_loss_ratio=val_baseline_ratio,
                    novelty_score=source.get("novelty_score"),
                    novelty_confidence=source.get("novelty_confidence"),
                    novelty_raw_score=source.get("novelty_raw_score"),
                    novelty_z_score=source.get("novelty_z_score"),
                    novelty_reference_version=source.get("novelty_reference_version"),
                    novelty_valid_for_promotion=source.get("novelty_valid_for_promotion"),
                    novelty_validity_reason=source.get("novelty_validity_reason"),
                    novelty_requires_justification=source.get("novelty_requires_justification"),
                    model_source=model_source,
                    arch_spec_json=arch_spec_json_str,
                )

                # Save checkpoint after each candidate completes
                try:
                    ckpt.save_phase(
                        experiment_id=exp_id,
                        phase="validation",
                        candidate_idx=prog_idx + 1,
                        seed_idx=0,
                        model_state_dict={},
                        optimizer_state_dict={},
                        step=0,
                        metrics={"completed_candidate": prog_idx},
                    )
                    # Also save a progress marker at index -1 for resume
                    ckpt.save_phase(
                        experiment_id=exp_id,
                        phase="validation",
                        candidate_idx=-1,
                        seed_idx=0,
                        model_state_dict={},
                        optimizer_state_dict={},
                        step=0,
                        metrics={"candidate_idx": prog_idx + 1},
                    )
                except Exception as e:
                    logger.debug("Validation checkpoint save failed: %s", e)

            # Complete experiment
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=self._analyze_results(results, exp_id, nb, context=context),
                llm_analysis=llm_analysis,
            )

            # Clean up validation checkpoints on success
            if not config.keep_checkpoints:
                try:
                    ckpt.cleanup(exp_id)
                except Exception:
                    pass

            with self._lock:
                self._progress.status = "completed"
                self._progress.elapsed_seconds = time.time() - t_start
                self._progress.aria_message = summary.split("\n")[-1] if summary else "Validation complete."

            self._emit_event("validation_completed", {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
            })

        except Exception as e:
            error = traceback.format_exc()
            logger.error("Validation failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Validation failure: {str(e)[:240]}",
                reproduction_steps=["python -m pytest tests/test_integration.py -k \"validation\" -x --tb=short"],
                acceptance_tests=["python -m pytest tests/test_integration.py -k \"validation\" -x --tb=short"],
                trigger_payload={"mode": "validation", "error": str(e)},
            )
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
            self._emit_event("experiment_failed", {
                "experiment_id": exp_id,
                "error": str(e),
            })
        finally:
            self._live_training_context = None
            nb.close()

    # ── Auto-Escalation Pipeline ──

    def _run_scale_up_thread(self, exp_id: str, result_ids: List[str],
                              config: RunConfig, hypothesis: str):
        """Execute scale-up training in background."""
        self._live_training_context = {"exp_id": exp_id, "phase": "scale_up"}
        nb = self._make_notebook()
        t_start = time.time()
        try:
            # graph_from_json already imported at module level
            results = {
                "total": len(result_ids), "stage0_passed": 0, "stage05_passed": 0,
                "stage1_passed": 0, "novel_count": 0,
                "best_loss_ratio": None, "best_novelty_score": None,
                "survivors": [],
            }

            dev = resolve_device(config.device)
            dev_str = str(dev)

            # Create a modified config for scale-up training
            scale_config = RunConfig.from_dict(config.to_dict())
            scale_config.stage1_steps = config.scale_up_steps
            scale_config.stage1_batch_size = config.scale_up_batch_size
            scale_config.max_seq_len = config.scale_up_seq_len

            for prog_idx, source_result_id in enumerate(result_ids):
                if self._stop_event.is_set():
                    break

                with self._lock:
                    self._progress.current_program = prog_idx + 1
                    self._progress.status = "training"
                    self._progress.aria_message = (
                        f"Scale-up {prog_idx + 1}/{len(result_ids)}: "
                        f"training {source_result_id[:8]}... "
                        f"({config.scale_up_steps} steps, batch={config.scale_up_batch_size})"
                    )
                    self._progress.elapsed_seconds = time.time() - t_start

                self._emit_event("scale_up_progress", {
                    "experiment_id": exp_id,
                    "current_program": prog_idx + 1,
                    "total_programs": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "starting",
                })

                # Fetch source program
                source_program = nb.get_program_detail(source_result_id)
                if source_program is None:
                    self._emit_event("scale_up_progress", {
                        "experiment_id": exp_id,
                        "current_program": prog_idx + 1,
                        "total_programs": len(result_ids),
                        "source_result_id": source_result_id,
                        "status": "skipped",
                        "error": "Source program not found",
                    })
                    continue

                # Reconstruct graph from stored JSON
                graph_json_str = source_program.get("graph_json")
                if not graph_json_str:
                    continue

                try:
                    graph = graph_from_json(graph_json_str)
                except Exception as e:
                    self._emit_event("scale_up_progress", {
                        "experiment_id": exp_id,
                        "current_program": prog_idx + 1,
                        "total_programs": len(result_ids),
                        "source_result_id": source_result_id,
                        "status": "error",
                        "error": f"Graph deserialization failed: {e}",
                    })
                    continue

                # Compile model
                try:
                    layer_graphs = [graph] * config.n_layers
                    model = compile_model(
                        layer_graphs,
                        vocab_size=config.vocab_size,
                        max_seq_len=config.scale_up_seq_len,
                    )
                except Exception as e:
                    self._emit_event("scale_up_progress", {
                        "experiment_id": exp_id,
                        "current_program": prog_idx + 1,
                        "total_programs": len(result_ids),
                        "source_result_id": source_result_id,
                        "status": "error",
                        "error": f"Compilation failed: {e}",
                    })
                    continue

                results["stage0_passed"] += 1
                results["stage05_passed"] += 1

                # Run scale-up training
                s1_result = self._micro_train(
                    model,
                    scale_config,
                    dev,
                    seed=self._stable_seed(exp_id, source_result_id, "scale_up"),
                )

                program_metrics = self._extract_graph_metrics(graph)
                # Store scale-up provenance in model_source (a valid column)
                # rather than as separate columns that don't exist in schema
                program_metrics["model_source"] = "graph_synthesis"

                s1_passed = s1_result.get("passed", False)
                loss_ratio = s1_result.get("loss_ratio")
                final_loss = s1_result.get("final_loss")
                throughput = s1_result.get("throughput")
                training_curve = s1_result.get("training_curve")

                # Training metrics
                for key in ["initial_loss", "min_loss", "loss_improvement_rate",
                            "avg_step_time_ms", "total_train_time_ms",
                            "max_grad_norm", "mean_grad_norm", "grad_norm_std",
                            "n_train_steps", "final_lr",
                            "validation_loss", "validation_loss_ratio", "generalization_gap",
                            "discovery_loss", "discovery_loss_ratio"]:
                    program_metrics[key] = s1_result.get(key)
                self._merge_s1_telemetry(program_metrics, s1_result)

                if s1_passed:
                    results["stage1_passed"] += 1
                    # Baseline comparison at scale
                    if final_loss is not None:
                        try:
                            baseline = self._get_baseline()
                            baseline_steps = int(s1_result.get("n_train_steps") or config.scale_up_steps)
                            baseline_recipe = self._resolve_baseline_recipe(
                                s1_result, default_lr=config.stage1_lr)
                            bl_data_fn, bl_data_tag, bl_cache = self._make_baseline_data_fn(config)
                            baseline_ratio = baseline.compare(
                                final_loss,
                                d_model=config.model_dim,
                                seq_len=min(128, config.scale_up_seq_len),
                                n_steps=max(1, baseline_steps),
                                vocab_size=config.vocab_size,
                                batch_size=config.scale_up_batch_size,
                                lr=baseline_recipe["lr"],
                                device=dev_str,
                                n_layers=config.n_layers,
                                optimizer_name=baseline_recipe["optimizer_name"],
                                weight_decay=baseline_recipe["weight_decay"],
                                momentum=baseline_recipe["momentum"],
                                betas=baseline_recipe["betas"],
                                data_fn=bl_data_fn,
                                data_tag=bl_data_tag,
                                cache_data_fn=bl_cache,
                            )
                            program_metrics["baseline_loss_ratio"] = baseline_ratio
                            
                            # Optional: Validation baseline comparison (using val split)
                            val_loss = s1_result.get("validation_loss")
                            if val_loss is not None:
                                v_data_fn, v_data_tag, v_cache = self._make_baseline_data_fn(config, split="val")
                                v_baseline_ratio = baseline.compare(
                                    val_loss,
                                    d_model=config.model_dim,
                                    seq_len=min(128, config.scale_up_seq_len),
                                    n_steps=max(1, baseline_steps),
                                    vocab_size=config.vocab_size,
                                    batch_size=config.scale_up_batch_size,
                                    lr=baseline_recipe["lr"],
                                    device=dev_str,
                                    n_layers=config.n_layers,
                                    optimizer_name=baseline_recipe["optimizer_name"],
                                    weight_decay=baseline_recipe["weight_decay"],
                                    momentum=baseline_recipe["momentum"],
                                    betas=baseline_recipe["betas"],
                                    data_fn=v_data_fn,
                                    data_tag=v_data_tag,
                                    cache_data_fn=v_cache,
                                )
                                program_metrics["validation_baseline_loss_ratio"] = v_baseline_ratio
                        except Exception:
                            pass

                program_metrics["stage_at_death"] = "survived" if s1_passed else "stage1"

                # Diagnostic tasks for S1 survivors
                if s1_passed and model is not None:
                    try:
                        diag = run_diagnostic_suite(model, device=dev_str)
                        program_metrics["diagnostic_tasks_json"] = json.dumps(diag.to_dict())
                        program_metrics["diagnostic_score"] = diag.diagnostic_score
                    except Exception:
                        pass

                # Benchmark evals (non-blocking) for scale-up survivors
                if s1_passed and model is not None:
                    eval_seq_len = min(128, config.scale_up_seq_len)
                    try:
                        from ...eval.wikitext_eval import evaluate_wikitext_perplexity
                        wt_result = evaluate_wikitext_perplexity(
                            model, config.vocab_size, dev_str,
                            n_train_steps=200, seq_len=eval_seq_len)
                        program_metrics["wikitext_perplexity"] = wt_result.get("wikitext_perplexity")
                        program_metrics["wikitext_score"] = wt_result.get("wikitext_score")
                        if program_metrics.get("wikitext_perplexity") is not None:
                            logger.info("Scale-up WikiText ppl=%.1f score=%.3f",
                                        program_metrics["wikitext_perplexity"],
                                        program_metrics.get("wikitext_score") or 0)
                    except Exception as e:
                        logger.debug("Scale-up WikiText eval skipped: %s", e)
                    try:
                        from ...eval.tinystories_eval import evaluate_tinystories
                        ts_result = evaluate_tinystories(
                            model, config.vocab_size, dev_str,
                            n_train_steps=200, seq_len=eval_seq_len)
                        program_metrics["tinystories_perplexity"] = ts_result.get("tinystories_perplexity")
                        program_metrics["tinystories_score"] = ts_result.get("tinystories_score")
                        if program_metrics.get("tinystories_perplexity") is not None:
                            logger.info("Scale-up TinyStories ppl=%.1f score=%.3f",
                                        program_metrics["tinystories_perplexity"],
                                        program_metrics.get("tinystories_score") or 0)
                    except Exception as e:
                        logger.debug("Scale-up TinyStories eval skipped: %s", e)

                # Novelty — compute behavioral fingerprint for S1 survivors
                fp = None
                calibration_row = None
                if s1_passed and model is not None:
                    try:
                        fp = compute_fingerprint(
                            model,
                            seq_len=min(64, config.scale_up_seq_len),
                            model_dim=config.model_dim,
                            vocab_size=config.vocab_size,
                            device=dev_str,
                        )
                        program_metrics["cka_source"] = fp.cka_source
                        program_metrics["cka_artifact_version"] = fp.cka_artifact_version
                        program_metrics["cka_probe_protocol_hash"] = fp.cka_probe_protocol_hash
                        program_metrics["cka_reference_quality"] = fp.cka_reference_quality
                        calibration_row = self._ensure_novelty_calibration(nb, config, fp)
                    except Exception:
                        pass

                calibration = None
                if calibration_row:
                    calibration = {
                        "noise_floor_mean": calibration_row.get("noise_floor_mean"),
                        "noise_floor_std": calibration_row.get("noise_floor_std"),
                    }
                nov = novelty_score(graph, fingerprint=fp, calibration=calibration)
                n_score = nov.overall_novelty
                novelty_valid, novelty_valid_reason, novelty_requires_justification = (
                    self._resolve_novelty_promotion_validity(
                        config,
                        nov.novelty_valid_for_promotion,
                        nov.novelty_validity_reason,
                    )
                )
                program_metrics["novelty_raw_score"] = nov.raw_novelty
                program_metrics["novelty_z_score"] = nov.novelty_z_score
                program_metrics["novelty_reference_version"] = (
                    nov.novelty_reference_version
                    or (fp.novelty_reference_version if fp is not None else None)
                )
                program_metrics["novelty_valid_for_promotion"] = int(novelty_valid)
                program_metrics["novelty_validity_reason"] = novelty_valid_reason
                program_metrics["novelty_requires_justification"] = int(
                    novelty_requires_justification
                )
                if s1_passed and n_score > 0.5:
                    results["novel_count"] += 1
                    results["survivors"].append({
                        "fingerprint": graph.fingerprint(),
                        "novelty": n_score,
                        "loss_ratio": loss_ratio,
                        "novelty_valid_for_promotion": novelty_valid,
                    })

                if loss_ratio and (results["best_loss_ratio"] is None
                                   or loss_ratio < results["best_loss_ratio"]):
                    results["best_loss_ratio"] = loss_ratio
                if n_score and (results["best_novelty_score"] is None
                                or n_score > results["best_novelty_score"]):
                    results["best_novelty_score"] = n_score

                result_id = nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=graph.fingerprint(),
                    graph_json=graph_to_json(graph),
                    stage0_passed=True, stage05_passed=True,
                    stage1_passed=s1_passed,
                    loss_ratio=loss_ratio, final_loss=final_loss,
                    throughput=throughput, novelty_score=n_score,
                    structural_novelty=nov.structural_novelty,
                    behavioral_novelty=nov.behavioral_novelty,
                    most_similar_to=nov.most_similar_to,
                    novelty_confidence=nov.novelty_confidence,
                    **program_metrics,
                )

                if training_curve and result_id:
                    try:
                        nb.store_training_curve(result_id, training_curve)
                    except Exception:
                        pass

                self._emit_event("scale_up_progress", {
                    "experiment_id": exp_id,
                    "current_program": prog_idx + 1,
                    "total_programs": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "completed",
                    "passed": s1_passed,
                    "loss_ratio": round(loss_ratio, 4) if loss_ratio else None,
                    "final_loss": round(final_loss, 4) if final_loss else None,
                })

                # Cleanup
                del model
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            # Guard: if no programs were processed at all, fail with clear reason
            if results["stage0_passed"] == 0 and results["total"] > 0:
                reason = (f"All {results['total']} source programs were skipped "
                          f"(not found or failed to compile). "
                          f"Result IDs: {', '.join(r[:12] for r in result_ids)}")
                logger.warning("Scale-up produced no results: %s", reason)
                nb.fail_experiment(exp_id, reason)
                with self._lock:
                    self._progress.status = "failed"
                    self._progress.error = reason
                    self._progress.aria_message = self.aria.react_to_failure(reason)
                self._emit_event("experiment_failed", {
                    "experiment_id": exp_id, "error": reason,
                })
                return

            # Complete experiment
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)
            insights = self._analyze_results(results, exp_id, nb, context=context)

            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=insights,
                llm_analysis=llm_analysis,
            )

            self._auto_recommend(results, config, hypothesis, nb)

            with self._lock:
                self._progress.status = "completed"
                self._progress.elapsed_seconds = time.time() - t_start
                self._progress.aria_message = summary.split("\n")[-1] if summary else "Scale-up complete."

            self._emit_event("scale_up_completed", {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
            })

        except Exception as e:
            error = traceback.format_exc()
            logger.error("Scale-up failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Scale-up failure: {str(e)[:240]}",
                reproduction_steps=["python -m pytest tests/test_integration.py -k \"scale_up\" -x --tb=short"],
                acceptance_tests=["python -m pytest tests/test_integration.py -k \"scale_up\" -x --tb=short"],
                trigger_payload={"mode": "scale_up", "error": str(e)},
            )
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
            self._emit_event("experiment_failed", {
                "experiment_id": exp_id,
                "error": str(e),
            })
        finally:
            self._live_training_context = None
            nb.close()

    def _run_evolution_thread(self, exp_id: str, config: RunConfig,
                               hypothesis: str):
        """Execute evolutionary search in background."""
        nb = self._make_notebook()
        t_start = time.time()
        try:
            from ...search.evolution import EvolutionConfig, evolutionary_search

            grammar = self._build_grammar_config(config)

            evo_config = EvolutionConfig(
                population_size=config.population_size,
                n_generations=config.n_generations,
                tournament_size=config.tournament_size,
                mutation_rate=config.mutation_rate,
                crossover_rate=config.crossover_rate,
                elitism=config.elitism,
                fitness_weight=config.fitness_weight,
                novelty_weight=config.novelty_weight,
                grammar_config=grammar,
            )

            fitness_cache: dict = {}
            eval_counters = {"total": 0, "s0": 0, "s1": 0}

            def on_evaluate(graph, fitness, sandbox_result, s1_result):
                self._on_program_evaluated(graph, fitness, sandbox_result, s1_result, 
                                           eval_counters, nb, exp_id, model_source="evolution")

            fitness_fn = self._make_fitness_fn(
                config, on_evaluate=on_evaluate, fitness_cache=fitness_cache)

            def gen_callback(gen, population):
                if self._stop_event.is_set():
                    return
                fitnesses = [ind.fitness for ind in population]
                avg_fit = sum(fitnesses) / len(fitnesses) if fitnesses else 0
                best_fit = max(fitnesses) if fitnesses else 0
                with self._lock:
                    self._progress.current_generation = gen + 1
                    self._progress.status = "evaluating"
                    self._progress.best_fitness = best_fit
                    self._progress.avg_fitness = avg_fit
                    self._progress.elapsed_seconds = time.time() - t_start
                    self._progress.aria_message = (
                        f"Generation {gen + 1}/{config.n_generations}: "
                        f"best={best_fit:.3f}, avg={avg_fit:.3f}"
                    )
                self._emit_event("evolution_generation", {
                    "experiment_id": exp_id,
                    "generation": gen + 1,
                    "total_generations": config.n_generations,
                    "best_fitness": best_fit,
                    "avg_fitness": avg_fit,
                    "population_size": len(population),
                })
                try:
                    nb.add_entry(ExperimentEntry(
                        entry_type="live_feed",
                        title=f"Evolution generation {gen + 1}/{config.n_generations}",
                        content=(
                            f"Gen {gen + 1}/{config.n_generations}: "
                            f"best={best_fit:.3f}, avg={avg_fit:.3f}, "
                            f"pop={len(population)}"
                        ),
                        experiment_id=exp_id,
                        metadata={
                            "live_feed_type": "evo_gen",
                            "payload": {
                                "experiment_id": exp_id,
                                "generation": gen + 1,
                                "total_generations": config.n_generations,
                                "best_fitness": best_fit,
                                "avg_fitness": avg_fit,
                                "population_size": len(population),
                            },
                        },
                    ))
                except Exception as e:
                    logger.debug("Failed to persist evolution generation feed entry: %s", e)

            def novelty_fn(graph, all_graphs):
                """Structural novelty relative to current population."""
                nov = novelty_score(graph)
                # Penalize duplicates within population
                my_fp = graph.fingerprint()
                dup_count = sum(1 for g in all_graphs
                                if g.fingerprint() == my_fp) - 1
                penalty = max(0, 1 - dup_count * 0.3)
                return nov.structural_novelty * penalty

            population = evolutionary_search(
                fitness_fn=fitness_fn,
                novelty_fn=novelty_fn,
                config=evo_config,
                callback=gen_callback,
            )

            results = {
                "total": eval_counters["total"],
                "stage0_passed": eval_counters["s0"],
                "stage05_passed": eval_counters["s0"],
                "stage1_passed": eval_counters["s1"],
                "novel_count": sum(1 for ind in population if ind.novelty > 0.5),
                "best_loss_ratio": 1.0 - max((ind.fitness for ind in population), default=0),
                "best_novelty_score": max((ind.novelty for ind in population), default=0),
                "survivors": [],
            }

            for ind in population[:20]:
                if ind.fitness > 0.2:
                    results["survivors"].append({
                        "fingerprint": ind.fingerprint,
                        "novelty": ind.novelty,
                        "loss_ratio": 1.0 - ind.fitness,
                    })

            nb.update_op_success_rates(exp_id)
            nb.update_failure_signatures(exp_id)

            # Rich context for Aria
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            # Validate hypothesis
            try:
                validation = self.aria.validate_hypothesis(hypothesis, results, context)
                if validation:
                    nb.add_entry(ExperimentEntry(
                        entry_type="analysis",
                        title="Hypothesis Validation",
                        content=validation.get("explanation", ""),
                        experiment_id=exp_id,
                        metadata={"validated": validation.get("validated", False)},
                    ))
            except Exception as e:
                logger.warning("Hypothesis validation failed for %s: %s", exp_id, e)

            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=self._analyze_results(results, exp_id, nb, context=context),
                llm_analysis=llm_analysis,
            )

            # Auto-recommend next experiment
            self._auto_recommend(results, config, hypothesis, nb)

            # Auto-scale-up and auto-report
            self._maybe_auto_scale_up(results, config, nb)
            self._maybe_auto_report(config, nb, reason="evolution_complete")

            with self._lock:
                self._progress.status = "completed"
                self._progress.elapsed_seconds = time.time() - t_start
                self._progress.aria_message = summary.split("\n")[-1] if summary else "Evolution complete."

            self._emit_event("evolution_completed", {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
            })

        except Exception as e:
            error = traceback.format_exc()
            logger.error("Evolution failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Evolution failure: {str(e)[:240]}",
                reproduction_steps=["python -m pytest tests/test_integration.py -k \"evolution\" -x --tb=short"],
                acceptance_tests=["python -m pytest tests/test_integration.py -k \"evolution\" -x --tb=short"],
                trigger_payload={"mode": "evolution", "error": str(e)},
            )
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
            self._emit_event("experiment_failed", {
                "experiment_id": exp_id,
                "error": str(e),
            })
        finally:
            nb.close()
            self._run_pending_scale_up()

    def _run_novelty_thread(self, exp_id: str, config: RunConfig,
                             hypothesis: str):
        """Execute novelty search in background."""
        nb = self._make_notebook()
        t_start = time.time()
        try:
            from ...search.novelty_search import NoveltySearchConfig, novelty_search

            grammar = self._build_grammar_config(config)

            ns_config = NoveltySearchConfig(
                archive_size=config.archive_size,
                k_nearest=config.k_nearest,
                archive_threshold=config.archive_threshold,
                novelty_weight=config.novelty_weight,
                fitness_weight=config.fitness_weight,
                population_size=config.population_size,
                n_generations=config.n_generations,
                grammar_config=grammar,
            )

            dev = resolve_device(config.device)
            dev_str = str(dev)

            fitness_cache: dict = {}
            fingerprint_cache: dict = {}
            eval_counters = {"total": 0, "s0": 0, "s1": 0}

            def on_evaluate(graph, fitness, sandbox_result, s1_result):
                self._on_program_evaluated(graph, fitness, sandbox_result, s1_result, 
                                           eval_counters, nb, exp_id, model_source="novelty")

            def combined_fitness_fn(graph):
                """Compile once, run sandbox + micro-train + fingerprint in one pass."""
                gfp = graph.fingerprint()
                if gfp in fitness_cache:
                    return fitness_cache[gfp]

                sandbox_result = None
                s1_result = None
                try:
                    layer_graphs = [graph] * config.n_layers
                    model = compile_model(
                        layer_graphs,
                        vocab_size=config.vocab_size,
                        max_seq_len=config.max_seq_len,
                    )
                    sandbox_result = self._safe_eval_for_stage(
                        model,
                        stage_tag="evolution_combined_fitness",
                        batch_size=2,
                        seq_len=min(128, config.max_seq_len),
                        vocab_size=config.vocab_size,
                        device=dev_str,
                    )
                    if not sandbox_result.passed:
                        del model
                        fitness = 0.0
                        fitness_cache[gfp] = fitness
                        on_evaluate(graph, fitness, sandbox_result, s1_result)
                        return fitness

                    # Compute behavioral fingerprint while model is in memory
                    try:
                        bfp = compute_fingerprint(
                            model,
                            seq_len=min(64, config.max_seq_len),
                            model_dim=config.model_dim,
                            vocab_size=config.vocab_size,
                            device=dev_str,
                        )
                        fingerprint_cache[gfp] = bfp
                    except Exception as e:
                        logger.debug("Fingerprint computation failed: %s", e)

                    s1_result = self._micro_train(
                        model,
                        config,
                        dev,
                        seed=self._stable_seed("fitness", gfp),
                    )
                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                    if s1_result.get("passed"):
                        fitness, _components = self._compute_multi_objective_fitness(
                            s1_result, sandbox_result, graph, config)
                    else:
                        fitness = 0.1
                except Exception:
                    fitness = 0.0

                fitness_cache[gfp] = fitness
                on_evaluate(graph, fitness, sandbox_result, s1_result)
                return fitness

            def fingerprint_fn(graph):
                return fingerprint_cache.get(graph.fingerprint())

            def gen_callback(gen, population, archive):
                if self._stop_event.is_set():
                    return
                fitnesses = [ind.fitness for ind in population]
                novelties = [ind.novelty for ind in population]
                avg_fit = sum(fitnesses) / len(fitnesses) if fitnesses else 0
                best_fit = max(fitnesses) if fitnesses else 0
                with self._lock:
                    self._progress.current_generation = gen + 1
                    self._progress.status = "evaluating"
                    self._progress.best_fitness = best_fit
                    self._progress.avg_fitness = avg_fit
                    self._progress.archive_size = archive.size()
                    self._progress.elapsed_seconds = time.time() - t_start
                    self._progress.aria_message = (
                        f"Generation {gen + 1}/{config.n_generations}: "
                        f"archive={archive.size()}, best_fit={best_fit:.3f}"
                    )
                self._emit_event("novelty_generation", {
                    "experiment_id": exp_id,
                    "generation": gen + 1,
                    "total_generations": config.n_generations,
                    "best_fitness": best_fit,
                    "avg_fitness": avg_fit,
                    "archive_size": archive.size(),
                    "best_novelty": max(novelties) if novelties else 0,
                })
                try:
                    best_novelty = max(novelties) if novelties else 0
                    nb.add_entry(ExperimentEntry(
                        entry_type="live_feed",
                        title=f"Novelty generation {gen + 1}/{config.n_generations}",
                        content=(
                            f"Gen {gen + 1}/{config.n_generations}: "
                            f"best_fit={best_fit:.3f}, archive={archive.size()}, "
                            f"novelty={best_novelty:.3f}"
                        ),
                        experiment_id=exp_id,
                        metadata={
                            "live_feed_type": "nov_gen",
                            "payload": {
                                "experiment_id": exp_id,
                                "generation": gen + 1,
                                "total_generations": config.n_generations,
                                "best_fitness": best_fit,
                                "avg_fitness": avg_fit,
                                "archive_size": archive.size(),
                                "best_novelty": best_novelty,
                            },
                        },
                    ))
                except Exception as e:
                    logger.debug("Failed to persist novelty generation feed entry: %s", e)

            ns_result = novelty_search(
                fitness_fn=combined_fitness_fn,
                fingerprint_fn=fingerprint_fn,
                config=ns_config,
                callback=gen_callback,
                stop_check=self._stop_event.is_set,
            )

            results = {
                "total": eval_counters["total"],
                "stage0_passed": eval_counters["s0"],
                "stage05_passed": eval_counters["s0"],
                "stage1_passed": eval_counters["s1"],
                "novel_count": sum(1 for ind in ns_result.best_individuals if ind.novelty > 0.5),
                "best_loss_ratio": None,
                "best_novelty_score": None,
                "survivors": [],
                "archive_size": ns_result.archive_size,
            }

            for ind in ns_result.best_individuals[:20]:
                if ind.fitness > 0.2:
                    results["survivors"].append({
                        "fingerprint": ind.fingerprint,
                        "novelty": ind.novelty,
                        "loss_ratio": 1.0 - ind.fitness,
                    })

            if results["survivors"]:
                results["best_loss_ratio"] = min(s["loss_ratio"] for s in results["survivors"])
                results["best_novelty_score"] = max(s["novelty"] for s in results["survivors"])

            nb.update_op_success_rates(exp_id)
            nb.update_failure_signatures(exp_id)

            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            try:
                validation = self.aria.validate_hypothesis(hypothesis, results, context)
                if validation:
                    nb.add_entry(ExperimentEntry(
                        entry_type="analysis",
                        title="Hypothesis Validation",
                        content=validation.get("explanation", ""),
                        experiment_id=exp_id,
                        metadata={"validated": validation.get("validated", False)},
                    ))
            except Exception:
                pass

            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=self._analyze_results(results, exp_id, nb, context=context),
                llm_analysis=llm_analysis,
            )

            # Auto-recommend next experiment
            self._auto_recommend(results, config, hypothesis, nb)

            # Auto-scale-up and auto-report
            self._maybe_auto_scale_up(results, config, nb)
            self._maybe_auto_report(config, nb, reason="novelty_search_complete")

            with self._lock:
                self._progress.status = "completed"
                self._progress.elapsed_seconds = time.time() - t_start
                self._progress.aria_message = summary.split("\n")[-1] if summary else "Novelty search complete."

            self._emit_event("novelty_completed", {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
                "archive_size": ns_result.archive_size,
            })

        except Exception as e:
            error = traceback.format_exc()
            logger.error("Novelty search failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Novelty search failure: {str(e)[:240]}",
                reproduction_steps=["python -m pytest tests/test_integration.py -k \"novelty\" -x --tb=short"],
                acceptance_tests=["python -m pytest tests/test_integration.py -k \"novelty\" -x --tb=short"],
                trigger_payload={"mode": "novelty", "error": str(e)},
            )
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
            self._emit_event("experiment_failed", {
                "experiment_id": exp_id,
                "error": str(e),
            })
        finally:
            nb.close()
            self._run_pending_scale_up()

    # ── Scale-Up Mode ──

    def _micro_train(self, model: nn.Module, config: RunConfig,
                     dev: torch.device, seed: int = 42) -> Dict:
        """Run Stage 1 micro-training with comprehensive metric capture.

        Uses deterministic seeding per step so all candidates see the same
        training data in the same order, enabling fair comparison (#56).
        """
        from research.scientist.perf import PerfTracer, GPUStarvationDetector, OpKernelProfiler
        trace_enabled = bool(getattr(config, "enable_perf_tracing", False))
        tracer = PerfTracer() if trace_enabled else None
        starvation_detector = GPUStarvationDetector(threshold_ms=2.0)
        op_profiler = OpKernelProfiler(
            enabled=bool(getattr(config, "enable_kernel_profiling", False)),
            top_k=max(1, int(getattr(config, "kernel_profile_top_k", 20) or 20)),
        )
        
        result: Dict[str, Any] = {"passed": False}
        collect_curve = bool(getattr(config, "collect_training_curve", False))
        grad_clip_norm = float(getattr(config, "gradient_clip_norm", 1.0) or 0.0)
        if grad_clip_norm < 0.0:
            grad_clip_norm = 0.0

        trace_totals_ms: Dict[str, float] = {
            "model_setup": 0.0,
            "data_sampling": 0.0,
            "forward_pass": 0.0,
            "backward_pass": 0.0,
            "optimizer_step": 0.0,
        }

        def _trace_ctx(name: str, use_gpu: bool = True):
            return tracer.trace(name, use_gpu=use_gpu) if tracer is not None else nullcontext()

        try:
            setup_t0 = time.perf_counter()
            with _trace_ctx("model_setup"):
                model = model.to(dev)
                model.train()
                opt_kwargs: Dict[str, Any] = {"lr": config.stage1_lr, "weight_decay": 0.01}
                if dev.type == "cuda":
                    use_fused = bool(getattr(config, "optimizer_fused", True))
                    use_foreach = bool(getattr(config, "optimizer_foreach", True))
                    if use_fused:
                        opt_kwargs["fused"] = True
                    elif use_foreach:
                        opt_kwargs["foreach"] = True
                try:
                    optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
                except Exception:
                    opt_kwargs.pop("fused", None)
                    opt_kwargs.pop("foreach", None)
                    optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
            trace_totals_ms["model_setup"] += (time.perf_counter() - setup_t0) * 1000.0

            result["optimizer_class"] = optimizer.__class__.__name__.lower()
            if optimizer.param_groups:
                pg0 = optimizer.param_groups[0]
                result["optimizer_lr"] = float(pg0.get("lr", config.stage1_lr))
                result["optimizer_weight_decay"] = float(pg0.get("weight_decay", 0.01))
                result["optimizer_momentum"] = float(pg0.get("momentum", 0.0))
                betas = pg0.get("betas")
                if isinstance(betas, tuple) and len(betas) == 2:
                    result["optimizer_beta1"] = float(betas[0])
                    result["optimizer_beta2"] = float(betas[1])

            initial_loss = None
            final_loss = None
            min_loss = float("inf")
            total_tokens = 0
            t_start = time.perf_counter()

            step_time_sum_ms = 0.0
            step_count = 0
            grad_norm_sum = 0.0
            grad_norm_sq_sum = 0.0
            grad_norm_max = 0.0
            grad_norm_count = 0
            training_curve: List[Dict] = [] if collect_curve else []
            kernel_profiles: List[Dict[str, Any]] = []

            seq_len = min(128, config.max_seq_len)
            random_mode = str(config.data_mode or "random").strip().lower() == "random"
            _seed_int = int(seed)

            def _make_random_batch(step: int) -> torch.Tensor:
                """Generate a deterministic random batch for a given step."""
                torch.manual_seed(_seed_int * 100_000 + step)
                return torch.randint(
                    0, int(config.vocab_size),
                    (config.stage1_batch_size, seq_len),
                    device=dev,
                )

            # --- Part 1: Discovery Evaluation (Fast) ---
            # Evaluate on a few random batches to get "discovery_loss"
            discovery_steps = min(5, config.stage1_steps // 10)
            discovery_losses = []
            model.eval()
            with torch.no_grad():
                for ds in range(discovery_steps):
                    d_batch = _make_random_batch(ds + 9999) # different offset
                    with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=True):
                        d_logits = model(d_batch)
                        d_loss = F.cross_entropy(d_logits[:, :-1].reshape(-1, config.vocab_size), d_batch[:, 1:].reshape(-1))
                    discovery_losses.append(d_loss.item())
            
            if discovery_losses:
                result["discovery_loss"] = sum(discovery_losses) / len(discovery_losses)
                # Note: discovery_loss_ratio needs a baseline; we'll compute it in _execute_experiment
            
            model.train()
            # --- Part 2: Main Training (Validation Channel) ---
            
            # Implementation of train/val split for Stage 1
            train_steps = int(config.stage1_steps * 0.8)
            config.stage1_steps - train_steps

            starvation_interval = max(1, int(getattr(config, "starvation_check_interval", 8) or 8))

            use_cuda_graph = bool(
                dev.type == "cuda"
                and bool(getattr(config, "enable_cuda_graphs", True))
                and random_mode
                and not op_profiler.enabled
                and not trace_enabled
                and not collect_curve
                and int(config.stage1_steps) >= 8
            )

            ran_cuda_graph = False
            if use_cuda_graph:
                try:
                    static_input_ids = torch.empty(
                        (config.stage1_batch_size, seq_len), dtype=torch.long, device=dev
                    )
                    captured_loss = torch.zeros((), device=dev)
                    captured_grad_norm = torch.zeros((), device=dev)
                    warmup_steps = max(1, int(getattr(config, "cuda_graph_warmup_steps", 3) or 3))

                    def _graph_step() -> Tuple[torch.Tensor, torch.Tensor]:
                        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=True):
                            logits = model(static_input_ids)
                            loss_t = F.cross_entropy(
                                logits[:, :-1].reshape(-1, logits.shape[-1]),
                                static_input_ids[:, 1:].reshape(-1),
                            )
                        optimizer.zero_grad(set_to_none=True)
                        loss_t.backward()
                        if grad_clip_norm > 0.0:
                            grad_norm_t = nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip_norm, foreach=True
                            )
                        else:
                            grad_norm_t = torch.zeros((), device=dev)
                        optimizer.step()
                        return loss_t, grad_norm_t

                    for wi in range(min(warmup_steps, int(config.stage1_steps))):
                        static_input_ids.copy_(_make_random_batch(wi), non_blocking=True)
                        loss_t, grad_norm_t = _graph_step()
                        captured_loss.copy_(loss_t.detach())
                        captured_grad_norm.copy_(torch.as_tensor(grad_norm_t, device=dev).detach())

                    torch.cuda.synchronize(dev)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        loss_t, grad_norm_t = _graph_step()
                        captured_loss.copy_(loss_t.detach())
                        captured_grad_norm.copy_(torch.as_tensor(grad_norm_t, device=dev).detach())

                    check_interval = max(1, int(getattr(config, "loss_check_interval", 8) or 8))
                    for step in range(config.stage1_steps):
                        if self._stop_event.is_set():
                            break
                        t_step = time.perf_counter()
                        static_input_ids.copy_(_make_random_batch(step), non_blocking=True)
                        graph.replay()
                        t_step_end = time.perf_counter()
                        step_time_ms = (t_step_end - t_step) * 1000.0
                        step_count += 1
                        step_time_sum_ms += step_time_ms
                        total_tokens += static_input_ids.numel()

                        should_check = (step == 0) or (step == config.stage1_steps - 1) or (step % check_interval == 0)
                        if not should_check:
                            continue

                        loss_val = float(captured_loss.item())
                        grad_norm = float(captured_grad_norm.item())
                        if not math.isfinite(loss_val):
                            result["error"] = f"NaN/Inf loss at step {step}"
                            result["n_train_steps"] = step
                            return result
                        if step == 0 and (not math.isfinite(grad_norm) or grad_norm <= 1e-10):
                            result["error"] = "zero_grad_precheck_failed"
                            result["n_train_steps"] = 0
                            result["max_grad_norm"] = grad_norm
                            result["mean_grad_norm"] = grad_norm
                            result["grad_norm_std"] = 0.0
                            return result
                        if step == 0:
                            initial_loss = loss_val
                        final_loss = loss_val
                        min_loss = min(min_loss, loss_val)
                        grad_norm_sum += grad_norm
                        grad_norm_sq_sum += grad_norm * grad_norm
                        grad_norm_max = max(grad_norm_max, grad_norm)
                        grad_norm_count += 1
                    ran_cuda_graph = True
                except Exception as e:
                    result["cuda_graph_fallback_reason"] = str(e)

            if not ran_cuda_graph:
                for step in range(config.stage1_steps):
                    if self._stop_event.is_set():
                        break

                    starvation_sample = (not random_mode) and ((step % starvation_interval) == 0)
                    if starvation_sample:
                        starvation_detector.start_wait()
                    data_t0 = time.perf_counter()
                    with _trace_ctx("data_sampling"):
                        if random_mode:
                            input_ids = _make_random_batch(step)
                        else:
                            input_ids = self._sample_training_input_ids(
                                config=config,
                                dev=dev,
                                batch_size=config.stage1_batch_size,
                                seq_len=seq_len,
                                seed=seed + step,
                            )
                    if starvation_sample:
                        starvation_detector.end_wait()
                    trace_totals_ms["data_sampling"] += (time.perf_counter() - data_t0) * 1000.0

                    t_step = time.perf_counter()

                    step_state: Dict[str, Any] = {}

                    def _run_step() -> None:
                        fwd_t0 = time.perf_counter()
                        with _trace_ctx("forward_pass"):
                            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                                    enabled=(dev.type == "cuda")):
                                logits = model(input_ids)
                                loss = F.cross_entropy(
                                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                                    input_ids[:, 1:].reshape(-1),
                                )
                        trace_totals_ms["forward_pass"] += (time.perf_counter() - fwd_t0) * 1000.0
                        step_state["loss"] = loss

                        bwd_t0 = time.perf_counter()
                        with _trace_ctx("backward_pass"):
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            if grad_clip_norm > 0.0:
                                step_state["grad_norm"] = nn.utils.clip_grad_norm_(
                                    model.parameters(), grad_clip_norm, foreach=(dev.type == "cuda")
                                ).item()
                            else:
                                step_state["grad_norm"] = 0.0
                        trace_totals_ms["backward_pass"] += (time.perf_counter() - bwd_t0) * 1000.0

                        opt_t0 = time.perf_counter()
                        with _trace_ctx("optimizer_step"):
                            optimizer.step()
                        trace_totals_ms["optimizer_step"] += (time.perf_counter() - opt_t0) * 1000.0

                    if step == 0 and op_profiler.enabled:
                        kernel_summary = op_profiler.profile_callable(_run_step)
                        if kernel_summary:
                            kernel_profiles.append({"step": step, **kernel_summary})
                        else:
                            _run_step()
                    else:
                        _run_step()

                    loss = step_state.get("loss")
                    grad_norm = float(step_state.get("grad_norm", 0.0))

                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        result["error"] = f"NaN/Inf loss at step {step}"
                        result["n_train_steps"] = step
                        return result

                    if step == 0 and (not math.isfinite(grad_norm) or grad_norm <= 1e-10):
                        result["error"] = "zero_grad_precheck_failed"
                        result["n_train_steps"] = 0
                        result["max_grad_norm"] = grad_norm
                        result["mean_grad_norm"] = grad_norm
                        result["grad_norm_std"] = 0.0
                        return result

                    if dev.type == "cuda" and (trace_enabled or op_profiler.enabled):
                        torch.cuda.synchronize(dev)

                    t_step_end = time.perf_counter()
                    step_time_ms = (t_step_end - t_step) * 1000

                    loss_val = loss.item()
                    if step == 0:
                        initial_loss = loss_val
                    final_loss = loss_val
                    min_loss = min(min_loss, loss_val)
                    total_tokens += input_ids.numel()

                    step_count += 1
                    step_time_sum_ms += step_time_ms
                    grad_norm_sum += grad_norm
                    grad_norm_sq_sum += grad_norm * grad_norm
                    grad_norm_max = max(grad_norm_max, grad_norm)
                    grad_norm_count += 1

                    # Record per-step data
                    if collect_curve:
                        training_curve.append({
                            "step": step,
                            "loss": loss_val,
                            "grad_norm": grad_norm,
                            "step_time_ms": step_time_ms,
                        })

                    # Emit live training step events for dashboard
                    ctx = getattr(self, "_live_training_context", None)
                    if ctx and step % 10 == 0:
                        self._emit_event("training_step", {
                            "experiment_id": ctx.get("exp_id", ""),
                            "step": step,
                            "loss": round(loss_val, 6),
                            "total_steps": config.stage1_steps,
                            "phase": ctx.get("phase", ""),
                        })

                    # Log training progress at start, midpoint, and end
                    total_steps = config.stage1_steps
                    if step == 0 or step == total_steps // 2 or step == total_steps - 1:
                        logger.debug(
                            "    train step %d/%d: loss=%.4f, grad_norm=%.3f, "
                            "step_time=%.1fms",
                            step + 1, total_steps, loss_val, grad_norm, step_time_ms,
                        )

            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000

            # Optional validation loss on heldout corpus split
            validation_loss = None
            validation_loss_ratio = None
            generalization_gap = None
            val_batches = max(1, int(getattr(config, "stage1_val_batches", 0) or 0))
            compute_val = bool(getattr(config, "stage1_compute_val_loss", True))
            val_batch_size = int(getattr(config, "stage1_val_batch_size", 0) or config.stage1_batch_size)
            val_frac = float(getattr(config, "corpus_val_fraction", 0.0) or 0.0)
            if compute_val and val_batches > 0 and val_frac > 0.0:
                if str(config.data_mode or "random").strip().lower() == "corpus":
                    try:
                        model.eval()
                        losses = []
                        with torch.no_grad():
                            for i in range(val_batches):
                                input_ids = self._sample_training_input_ids(
                                    config=config,
                                    dev=dev,
                                    batch_size=val_batch_size,
                                    seq_len=seq_len,
                                    seed=seed + 10_000 + i,
                                    split="val",
                                )
                                if input_ids is None:
                                    continue
                                with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                                        enabled=(dev.type == "cuda")):
                                    logits = model(input_ids)
                                    loss = F.cross_entropy(
                                        logits[:, :-1].reshape(-1, logits.shape[-1]),
                                        input_ids[:, 1:].reshape(-1),
                                    )
                                if loss is not None and torch.isfinite(loss):
                                    losses.append(float(loss.item()))
                        if losses:
                            validation_loss = sum(losses) / len(losses)
                    except Exception as e:
                        result["validation_loss_error"] = str(e)
                    finally:
                        model.train()

            # Optional discovery loss on random tokens (fast triage signal)
            discovery_loss = None
            discovery_loss_ratio = None
            discovery_batches = max(1, int(getattr(config, "stage1_discovery_batches", 0) or 0))
            compute_discovery = bool(getattr(config, "stage1_compute_discovery_loss", True))
            discovery_batch_size = int(getattr(config, "stage1_discovery_batch_size", 0) or config.stage1_batch_size)
            if compute_discovery and discovery_batches > 0:
                try:
                    model.eval()
                    losses = []
                    with torch.no_grad():
                        for i in range(discovery_batches):
                            torch.manual_seed(int(seed) * 10_000 + 3_000 + i)
                            input_ids = torch.randint(
                                0, int(config.vocab_size),
                                (discovery_batch_size, seq_len),
                                device=dev,
                            )
                            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                                    enabled=(dev.type == "cuda")):
                                logits = model(input_ids)
                                loss = F.cross_entropy(
                                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                                    input_ids[:, 1:].reshape(-1),
                                )
                            if loss is not None and torch.isfinite(loss):
                                losses.append(float(loss.item()))
                    if losses:
                        discovery_loss = sum(losses) / len(losses)
                except Exception as e:
                    result["discovery_loss_error"] = str(e)
                finally:
                    model.train()

            if validation_loss is not None and initial_loss:
                validation_loss_ratio = validation_loss / max(initial_loss, 1e-6)
            if validation_loss is not None and final_loss is not None:
                generalization_gap = validation_loss - final_loss
            if discovery_loss is not None and initial_loss:
                discovery_loss_ratio = discovery_loss / max(initial_loss, 1e-6)

            # Collect perf results
            if tracer is not None:
                result["perf_traces"] = tracer.get_report()
            else:
                result["perf_traces"] = {
                    "summary_ms": {k: round(v, 4) for k, v in trace_totals_ms.items()},
                    "traces": [],
                }
            result["gpu_starvation"] = starvation_detector.get_summary()
            if kernel_profiles:
                result["kernel_timing"] = {
                    "sample_count": len(kernel_profiles),
                    "samples": kernel_profiles,
                    "top_ops": kernel_profiles[0].get("top_ops", []),
                }

            if initial_loss and final_loss:
                result["loss_ratio"] = final_loss / max(initial_loss, 1e-6)
                result["final_loss"] = final_loss
                result["initial_loss"] = initial_loss
                result["min_loss"] = min_loss
                if validation_loss is not None:
                    result["validation_loss"] = validation_loss
                if validation_loss_ratio is not None:
                    result["validation_loss_ratio"] = validation_loss_ratio
                if generalization_gap is not None:
                    result["generalization_gap"] = generalization_gap
                if discovery_loss is not None:
                    result["discovery_loss"] = discovery_loss
                if discovery_loss_ratio is not None:
                    result["discovery_loss_ratio"] = discovery_loss_ratio
                result["throughput"] = total_tokens / (total_time_ms / 1000)
                result["passed"] = result["loss_ratio"] < config.stage1_loss_ratio_threshold
                if not result["passed"] and result.get("error_type") is None:
                    result["error_type"] = "failed_convergence"
                    result["error"] = f"Insufficient loss reduction: {result['loss_ratio']:.4f} >= {config.stage1_loss_ratio_threshold}"
                if initial_loss > 0:
                    result["loss_improvement_rate"] = (initial_loss - final_loss) / initial_loss

                # Timing stats
                result["avg_step_time_ms"] = (step_time_sum_ms / step_count) if step_count > 0 else 0.0
                result["total_train_time_ms"] = total_time_ms

                # Gradient norm stats
                if grad_norm_count > 0:
                    result["max_grad_norm"] = grad_norm_max
                    result["mean_grad_norm"] = grad_norm_sum / grad_norm_count
                    mean_gn = result["mean_grad_norm"]
                    var = max((grad_norm_sq_sum / grad_norm_count) - (mean_gn * mean_gn), 0.0)
                    result["grad_norm_std"] = var ** 0.5

                result["n_train_steps"] = step_count
                result["final_lr"] = config.stage1_lr  # constant for now
                if collect_curve:
                    result["training_curve"] = training_curve

                # Extract architecture-specific telemetry (MoE, MoD, MoR, etc.)
                arch_telemetry = self._extract_architecture_telemetry(model)
                result.update(arch_telemetry)

                # Behavioral fingerprint for S1 survivors (novelty scoring)
                if result.get("passed") and model is not None:
                    try:
                        _fp = compute_fingerprint(
                            model,
                            seq_len=min(64, config.max_seq_len),
                            model_dim=config.model_dim,
                            vocab_size=config.vocab_size,
                            device=str(dev),
                        )
                        result["_behavioral_fingerprint"] = _fp.to_dict()
                    except Exception as e_fp:
                        logger.debug("Fingerprint failed in S1 worker: %s", e_fp)

        except Exception as e:
            result["error"] = str(e)

        if result.get("final_loss") is not None and bool(getattr(config, "one_shot_pruning_baseline", False)):
            try:
                seq_len = min(128, int(config.max_seq_len))
                eval_batches = max(1, int(getattr(config, "one_shot_pruning_eval_batches", 4)))
                eval_batch_size = max(1, int(getattr(config, "one_shot_pruning_batch_size", 2)))

                eval_inputs = [
                    self._sample_training_input_ids(
                        config=config,
                        dev=dev,
                        batch_size=eval_batch_size,
                        seq_len=seq_len,
                        seed=seed + 100_000 + i,
                    )
                    for i in range(eval_batches)
                ]

                dense_eval_loss = estimate_lm_ce_loss(model, eval_inputs, dev)

                pruned_model = copy.deepcopy(model).to(dev)
                prune_info = apply_one_shot_pruning(
                    pruned_model,
                    target_sparsity=float(getattr(config, "one_shot_pruning_sparsity", 0.5)),
                    method=str(getattr(config, "one_shot_pruning_method", "wanda")),
                )
                pruned_eval_loss = estimate_lm_ce_loss(pruned_model, eval_inputs, dev)

                quality_retention = None
                if dense_eval_loss is not None and pruned_eval_loss is not None and pruned_eval_loss > 0:
                    quality_retention = max(0.0, min(1.5, dense_eval_loss / pruned_eval_loss))

                result["pruning_method"] = prune_info.method
                result["pruning_target_sparsity"] = prune_info.target_sparsity
                result["pruning_actual_sparsity"] = prune_info.actual_sparsity
                result["pruning_n_params_total"] = prune_info.n_params_total
                result["pruning_n_params_pruned"] = prune_info.n_params_pruned
                result["pruning_dense_eval_loss"] = dense_eval_loss
                result["pruning_pruned_eval_loss"] = pruned_eval_loss
                result["pruning_quality_retention"] = quality_retention
                if prune_info.n_params_total > 0:
                    result["pruning_active_params_estimate"] = (
                        prune_info.n_params_total - prune_info.n_params_pruned
                    )

                del pruned_model
            except Exception as e:
                result["pruning_error"] = str(e)

        # Finalize performance reports
        try:
            if tracer is not None:
                fallback_perf = tracer.get_report()
            else:
                fallback_perf = {
                    "summary_ms": {k: round(v, 4) for k, v in trace_totals_ms.items()},
                    "traces": [],
                }
            result["perf_report"] = result.get("perf_traces", fallback_perf)
            # Ensure throughput is included in perf_report for experiment-level aggregation
            if isinstance(result.get("throughput"), (int, float)):
                result["perf_report"]["avg_throughput_tok_s"] = float(result["throughput"])
            
            result["starvation_report"] = result.get("gpu_starvation", starvation_detector.get_summary())
            if "kernel_timing" in result:
                result["kernel_timings_ms"] = result["kernel_timing"]
        except Exception as e:
            result["perf_error"] = str(e)

        try:
            result.update(self._extract_architecture_telemetry(model))
        except Exception as e:
            logger.debug("Architecture telemetry extract failed: %s", e)

        return result

    def _micro_train_async(self, model: nn.Module, config: RunConfig, seed: int, dev: torch.device) -> Dict:
        """Async worker entry point for training a pre-compiled model."""
        try:
            return self._micro_train(model, config, dev, seed=seed)
        except Exception as e:
            return {"error": str(e), "passed": False}

    def _train_with_program(self, model: nn.Module, program,
                            config: RunConfig,
                            dev: torch.device,
                            seed: int = 42) -> Dict:
        """Train a model using a synthesized TrainingProgram.

        Returns same metrics dict as _micro_train() plus training_program_json.
        """
        from research.scientist.perf import PerfTracer, GPUStarvationDetector, KernelTimer
        tracer = PerfTracer()
        starvation_detector = GPUStarvationDetector(threshold_ms=2.0)
        kernel_timer = KernelTimer(model, enabled=bool(getattr(config, "enable_kernel_profiling", False)))
        
        result: Dict[str, Any] = {"passed": False}

        try:
            with tracer.trace("model_setup"):
                model = model.to(dev)
                model.train()

            # Apply init scheme
            if program.init_scheme == "small":
                for p in model.parameters():
                    if p.dim() >= 2:
                        nn.init.normal_(p, std=program.init_scale)
            elif program.init_scheme == "orthogonal":
                for m in model.modules():
                    if isinstance(m, (nn.Linear, nn.Conv1d)):
                        nn.init.orthogonal_(m.weight, gain=program.init_scale)
            elif program.init_scheme == "spectral":
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)

            # Create optimizer from program
            try:
                optimizer = program.optimizer.create(model.parameters())
            except Exception:
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=3e-4, weight_decay=0.01)

            result["optimizer_class"] = optimizer.__class__.__name__.lower()
            if optimizer.param_groups:
                pg0 = optimizer.param_groups[0]
                result["optimizer_lr"] = float(pg0.get("lr", 3e-4))
                result["optimizer_weight_decay"] = float(pg0.get("weight_decay", 0.01))
                result["optimizer_momentum"] = float(pg0.get("momentum", 0.0))
                betas = pg0.get("betas")
                if isinstance(betas, tuple) and len(betas) == 2:
                    result["optimizer_beta1"] = float(betas[0])
                    result["optimizer_beta2"] = float(betas[1])

            n_steps = program.n_steps
            batch_size = program.batch_size
            max_grad_norm_val = program.max_grad_norm

            initial_loss = None
            final_loss = None
            min_loss = float("inf")
            total_tokens = 0
            t_start = time.perf_counter()

            step_times: List[float] = []
            grad_norms: List[float] = []
            training_curve: List[Dict] = []

            seq_len = min(128, config.max_seq_len)
            # Apply curriculum seq_len schedule
            try:
                base_seq = program.curriculum.get_seq_len(0, n_steps)
                if base_seq and base_seq > 0:
                    seq_len = min(base_seq, config.max_seq_len)
            except Exception:
                pass

            for step in range(n_steps):
                if self._stop_event.is_set():
                    break

                # Update seq_len from curriculum
                try:
                    curr_seq = program.curriculum.get_seq_len(step, n_steps)
                    if curr_seq and curr_seq > 0:
                        seq_len = min(curr_seq, config.max_seq_len)
                except Exception:
                    pass

                starvation_detector.start_wait()
                with tracer.trace("data_sampling"):
                    input_ids = self._sample_training_input_ids(
                        config=config,
                        dev=dev,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        seed=seed + step,
                    )
                starvation_detector.end_wait()

                t_step = time.perf_counter()

                with tracer.trace("forward_pass"):
                    with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                            enabled=(dev.type == "cuda")):
                        logits = model(input_ids)
                        # Use synthesized loss if possible
                        try:
                            loss = program.loss.compute(
                                logits[:, :-1].reshape(-1, logits.shape[-1]),
                                input_ids[:, 1:].reshape(-1),
                            )
                        except Exception:
                            loss = F.cross_entropy(
                                logits[:, :-1].reshape(-1, logits.shape[-1]),
                                input_ids[:, 1:].reshape(-1),
                            )

                if torch.isnan(loss) or torch.isinf(loss):
                    result["error"] = f"NaN/Inf loss at step {step}"
                    result["n_train_steps"] = step
                    return result

                with tracer.trace("backward_pass"):
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm_val).item()
                    optimizer.step()
            

                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)

                t_step_end = time.perf_counter()
                step_time_ms = (t_step_end - t_step) * 1000

                loss_val = loss.item()
                if step == 0:
                    initial_loss = loss_val
                final_loss = loss_val
                min_loss = min(min_loss, loss_val)
                total_tokens += input_ids.numel()

                step_times.append(step_time_ms)
                grad_norms.append(grad_norm)

                training_curve.append({
                    "step": step,
                    "loss": loss_val,
                    "grad_norm": grad_norm,
                    "step_time_ms": step_time_ms,
                })

                # Emit live training step events for dashboard
                ctx = getattr(self, "_live_training_context", None)
                if ctx and step % 10 == 0:
                    self._emit_event("training_step", {
                        "experiment_id": ctx.get("exp_id", ""),
                        "step": step,
                        "loss": round(loss_val, 6),
                        "total_steps": n_steps,
                        "phase": ctx.get("phase", ""),
                    })

            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000

            if initial_loss and final_loss:
                result["loss_ratio"] = final_loss / max(initial_loss, 1e-6)
                result["final_loss"] = final_loss
                result["initial_loss"] = initial_loss
                result["min_loss"] = min_loss
                result["throughput"] = total_tokens / (total_time_ms / 1000)
                result["passed"] = result["loss_ratio"] < config.stage1_loss_ratio_threshold
                if not result["passed"] and result.get("error_type") is None:
                    result["error_type"] = "failed_convergence"
                    result["error"] = f"Insufficient loss reduction during investigation: {result['loss_ratio']:.4f}"
                    result["loss_improvement_rate"] = (initial_loss - final_loss) / initial_loss

                result["avg_step_time_ms"] = sum(step_times) / len(step_times) if step_times else 0
                result["total_train_time_ms"] = total_time_ms

                if grad_norms:
                    result["max_grad_norm"] = max(grad_norms)
                    result["mean_grad_norm"] = sum(grad_norms) / len(grad_norms)
                    mean_gn = result["mean_grad_norm"]
                    result["grad_norm_std"] = (
                        sum((g - mean_gn) ** 2 for g in grad_norms) / len(grad_norms)
                    ) ** 0.5

                result["n_train_steps"] = len(step_times)
                result["final_lr"] = getattr(optimizer, 'defaults', {}).get('lr', 3e-4)
                result["training_curve"] = training_curve
                result["training_program_json"] = json.dumps(program.to_dict())

                # Extract architecture-specific telemetry (MoE, MoD, MoR, etc.)
                arch_telemetry = self._extract_architecture_telemetry(model)
                result.update(arch_telemetry)

        except Exception as e:
            result["error"] = str(e)

        # Finalize performance reports
        try:
            result["perf_report"] = tracer.get_report()
            # Ensure throughput is included in perf_report for experiment-level aggregation
            if isinstance(result.get("throughput"), (int, float)):
                result["perf_report"]["avg_throughput_tok_s"] = float(result["throughput"])
                
            result["starvation_report"] = starvation_detector.get_summary()
            if kernel_timer.enabled:
                result["kernel_timings_ms"] = kernel_timer.synchronize_and_get_timings()
        except Exception as e:
            result["perf_error"] = str(e)

        return result

    # ── OOD Robustness Testing (#54) ──

    # Hand-designed reference training recipes for out-of-distribution testing.
    # Each recipe exercises a different optimizer/LR/schedule to test whether
    # a candidate's learnability is robust or just an artifact of one recipe.
    def _generate_candidates(self, config: RunConfig, n: int,
                             source: str = "graph_synthesis") -> List[ModelCandidate]:
        """Generate candidate models from the specified source.

        source: "graph_synthesis", "morphological_box", or "mixed"
        Returns candidates that pass Stage 0 smoke test.
        """
        candidates: List[ModelCandidate] = []
        dev_str = str(resolve_device(config.device))

        if source == "mixed":
            n_morph = int(n * config.morph_ratio)
            n_graph = n - n_morph
            candidates.extend(
                self._generate_candidates(config, n_graph, "graph_synthesis"))
            candidates.extend(
                self._generate_candidates(config, n_morph, "morphological_box"))
            return candidates

        if source == "morphological_box":
            try:
                from ...morphological_box import roll, describe_spec
                from ...arch_builder import build_model, BuildConfig

                sparse_weight_options = (
                    "structured_sparse",
                    "semi_structured_2_4",
                    "block_sparse",
                )

                build_cfg = BuildConfig(
                    dim=config.model_dim,
                    n_layers=config.n_layers,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )

                for i in range(n):
                    if self._stop_event.is_set():
                        break
                    try:
                        fixed_choices: Dict[str, str] = {}
                        if bool(getattr(config, "morph_focus_sparse", False)):
                            explicit_sparse = str(getattr(config, "morph_sparse_weight_storage", "") or "").strip()
                            if explicit_sparse in sparse_weight_options:
                                fixed_choices["weight_storage"] = explicit_sparse
                            else:
                                fixed_choices["weight_storage"] = sparse_weight_options[i % len(sparse_weight_options)]
                        fixed_routing = str(getattr(config, "morph_compute_routing", "") or "").strip()
                        if fixed_routing:
                            fixed_choices["compute_routing"] = fixed_routing
                        fixed_channel = str(getattr(config, "morph_channel_mixing", "") or "").strip()
                        if fixed_channel:
                            fixed_choices["channel_mixing"] = fixed_channel

                        spec = roll(seed=i + int(time.time() * 1000) % 100000,
                                    generation=0,
                                    fixed=fixed_choices or None)
                        model = build_model(spec, build_cfg)
                        desc = describe_spec(spec)

                        # Quick smoke test
                        sandbox_result = self._safe_eval_for_stage(
                            model,
                            stage_tag="morph_candidate_gen",
                            batch_size=2,
                            seq_len=min(128, config.max_seq_len),
                            vocab_size=config.vocab_size,
                            device=dev_str,
                        )
                        if sandbox_result.passed:
                            import json as _json
                            candidates.append(ModelCandidate(
                                source="morphological_box",
                                model=model,
                                description=desc,
                                arch_spec=spec,
                                arch_spec_json=_json.dumps(spec.to_dict()),
                                fingerprint=spec.id,
                            ))
                        else:
                            del model
                    except Exception as e:
                        logger.debug(f"Morphological candidate {i} failed: {e}")
                        continue
            except ImportError:
                logger.warning("morphological_box or arch_builder not available")
            return candidates

        # Default: graph_synthesis
        grammar = self._build_grammar_config(config)

        graphs = batch_generate(n, grammar)
        for graph in graphs:
            if self._stop_event.is_set():
                break
            validation = validate_graph(
                graph,
                max_ops=max(1, int(config.max_ops)),
                max_depth=max(1, int(config.max_depth)),
                min_splits=config.min_splits,
            )
            if not validation.valid:
                continue
            try:
                layer_graphs = [graph] * config.n_layers
                model = compile_model(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
                sandbox_result = self._safe_eval_for_stage(
                    model,
                    stage_tag="graph_candidate_gen",
                    batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=config.vocab_size,
                    device=dev_str,
                )
                if sandbox_result.passed:
                    candidates.append(ModelCandidate(
                        source="graph_synthesis",
                        model=model,
                        description=graph_summary(graph),
                        graph=graph,
                        graph_json=graph_to_json(graph),
                        fingerprint=graph.fingerprint(),
                    ))
                else:
                    del model
            except Exception:
                continue

        return candidates

    # ── Training with synthesized programs ──

    def _build_grammar_config(self, config: RunConfig,
                              excluded_ops: Optional[Set[str]] = None,
                              op_weights: Optional[Dict[str, float]] = None) -> GrammarConfig:
        """Create a GrammarConfig from a RunConfig with standardized defaults."""
        from ...synthesis.grammar import GrammarConfig

        # Exotic mode: use the exotic preset as base, then layer on
        # excluded_ops and learned op_weights
        if getattr(config, "_exotic_mode", False):
            grammar = GrammarConfig.exotic(model_dim=config.model_dim)
            # Merge with defaults (non-causal ops) rather than replacing
            grammar.excluded_ops = grammar.excluded_ops | (excluded_ops or set())
            # Merge learned op_weights (exotic preset weights take precedence
            # only when learned weight is default 1.0)
            if op_weights:
                for op_name, w in op_weights.items():
                    existing = grammar.op_weights.get(op_name, 1.0)
                    # If the learned system penalizes an op, respect it even in exotic mode
                    if w < 1.0:
                        grammar.op_weights[op_name] = existing * w
                    else:
                        grammar.op_weights.setdefault(op_name, w)
            return grammar

        # Pick up structured_sparsity_bias from mode recommendation or config
        sparsity_bias = getattr(self, "_structured_sparsity_bias_override",
                                getattr(config, "structured_sparsity_bias", 0.0))

        # Merge API-provided op_weights with learned op_weights
        merged_op_weights = dict(op_weights or {})
        if config.op_weights:
            # API overrides take precedence over learned weights
            merged_op_weights.update(config.op_weights)

        grammar = GrammarConfig(
            model_dim=config.model_dim,
            min_depth=config.min_depth,
            max_depth=min(config.max_depth, 12),
            max_ops=min(config.max_ops, 20),
            residual_prob=config.residual_prob,
            split_prob=config.grammar_split_prob,
            merge_prob=config.grammar_merge_prob,
            risky_op_prob=config.grammar_risky_op_prob,
            freq_domain_prob=config.grammar_freq_domain_prob,
            structured_sparsity_bias=sparsity_bias,
            op_weights=merged_op_weights,
            min_splits=config.min_splits,
            three_way_split_prob=config.three_way_split_prob,
            branch_depth=config.branch_depth,
            max_recursion_depth=config.max_recursion_depth,
        )
        # Merge learned excluded_ops with defaults (non-causal ops)
        if excluded_ops:
            grammar.excluded_ops = grammar.excluded_ops | excluded_ops
        # Apply specialized weights
        grammar.category_weights["math_space"] = config.math_space_weight
        # Apply custom category weights from API (overrides defaults)
        if config.category_weights:
            grammar.category_weights.update(config.category_weights)

        # Apply Bayesian op priors from compressed learning (optional)
        try:
            from pathlib import Path
            import json as _json
            priors_path = Path("research/runtime/learning/op_priors.json")
            if priors_path.exists():
                payload = _json.loads(priors_path.read_text())
                op_penalties = payload.get("op_penalties", {}) if isinstance(payload, dict) else {}
                if isinstance(op_penalties, dict):
                    for op_name, penalty in op_penalties.items():
                        try:
                            p = float(penalty)
                        except Exception:
                            continue
                        # Convert penalty (0..1) into weight multiplier (1..0.5)
                        mult = max(0.5, 1.0 - 0.5 * max(0.0, min(1.0, p)))
                        grammar.op_weights[op_name] = grammar.op_weights.get(op_name, 1.0) * mult
        except Exception:
            pass

        # Apply cluster-based suggestions (optional)
        try:
            from pathlib import Path
            import json as _json
            sugg_path = Path("research/runtime/learning/cluster_suggestions.json")
            if sugg_path.exists():
                payload = _json.loads(sugg_path.read_text())
                if isinstance(payload, dict):
                    op_weight_suggestions = payload.get("op_weight_suggestions") or payload.get("op_weights") or {}
                    op_penalties = payload.get("op_penalties") or {}
                    op_promotions = payload.get("op_promotions") or {}
                    avoid_patterns = payload.get("avoid_patterns") or []
                    promote_patterns = payload.get("promote_patterns") or []

                    def _apply_mult(op_name: str, mult: float):
                        if not op_name:
                            return
                        m = max(0.2, min(3.0, float(mult)))
                        grammar.op_weights[op_name] = grammar.op_weights.get(op_name, 1.0) * m

                    for op_name, mult in op_weight_suggestions.items():
                        try:
                            _apply_mult(op_name, float(mult))
                        except Exception:
                            continue

                    for op_name, p in op_penalties.items():
                        try:
                            penalty = max(0.0, min(1.0, float(p)))
                        except Exception:
                            continue
                        _apply_mult(op_name, 1.0 - 0.4 * penalty)

                    for op_name, p in op_promotions.items():
                        try:
                            promo = max(0.0, min(1.0, float(p)))
                        except Exception:
                            continue
                        _apply_mult(op_name, 1.0 + 0.4 * promo)

                    def _ops_from_pattern(pat: str):
                        if "->" in pat:
                            parts = [p.strip() for p in pat.split("->", 1)]
                        elif "," in pat:
                            parts = [p.strip() for p in pat.split(",", 1)]
                        else:
                            parts = [pat.strip()]
                        return [p for p in parts if p]

                    for pat in avoid_patterns:
                        for op_name in _ops_from_pattern(str(pat)):
                            _apply_mult(op_name, 0.85)
                    for pat in promote_patterns:
                        for op_name in _ops_from_pattern(str(pat)):
                            _apply_mult(op_name, 1.1)
        except Exception:
            pass

        return grammar

    def _sample_training_input_ids(
        self,
        config: RunConfig,
        dev: torch.device,
        batch_size: int,
        seq_len: int,
        seed: int,
        split: str = "train",
    ) -> torch.Tensor:
        """Sample input IDs from configured data source with deterministic seed."""
        mode = str(config.data_mode or "random").strip().lower()
        generator = torch.Generator(device=dev)
        generator.manual_seed(int(seed))

        if mode == "huggingface":
            batcher = self._get_hf_batcher(config)
            if batcher is not None:
                batch = batcher.sample_batch(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    generator=generator,
                    device=dev,
                    split=split,
                )
                if batch is not None:
                    return batch
            # Fall through to random on failure

        if mode == "hydra":
            batch = self._get_hydra_batch(config, batch_size, seq_len, dev)
            if batch is not None:
                return batch
            # Fall through to random on failure

        if mode == "corpus":
            batcher = self._get_corpus_batcher(config)
            if batcher is not None:
                batch = batcher.sample_batch(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    generator=generator,
                    device=dev,
                    split=split,
                )
                if batch is not None:
                    return batch

        return torch.randint(
            0,
            int(config.vocab_size),
            (batch_size, seq_len),
            device=dev,
            generator=generator,
        )

    def _make_baseline_data_fn(self, config: RunConfig, split: str = "train"):
        """Build a data_fn for baseline training when using real data.

        Returns (data_fn, data_tag, cache_data_fn) tuple. data_fn is None for
        random mode (baseline uses its own random tokens). data_tag is a cache
        key suffix. cache_data_fn indicates safe caching for data_fn.
        """
        mode = str(config.data_mode or "random").strip().lower()
        if mode == "huggingface":
            ds_name = str(config.hf_dataset or "").strip()
            subset = str(config.hf_subset or "").strip()
            data_tag = f"hf:{ds_name}:{subset}:{config.hf_split}:{split}"
            step_state = {"step": 0}

            def data_fn(batch_size, seq_len, dev):
                step = step_state["step"]
                step_state["step"] = step + 1
                generator = torch.Generator(device=dev)
                generator.manual_seed(1337 + step)
                batcher = self._get_hf_batcher(config)
                if batcher is not None:
                    batch = batcher.sample_batch(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        generator=generator,
                        device=dev,
                        split=str(split or "train").lower(),
                    )
                    if batch is not None:
                        return batch
                return torch.randint(0, config.vocab_size, (batch_size, seq_len), device=dev, generator=generator)

            return data_fn, data_tag, True
        if mode == "hydra":
            def data_fn(batch_size, seq_len, dev):
                batch = self._get_hydra_batch(config, batch_size, seq_len, dev)
                if batch is not None:
                    return batch
                return torch.randint(0, config.vocab_size, (batch_size, seq_len), device=dev)
            return data_fn, "hydra", False
        if mode == "corpus":
            path = str(config.corpus_path or "").strip()
            version = self._corpus_version_tag(path)
            train_frac = float(getattr(config, "corpus_train_fraction", 0.9) or 0.9)
            val_frac = float(getattr(config, "corpus_val_fraction", 0.1) or 0.1)
            fmt = str(config.corpus_format or "auto")
            text_key = str(config.corpus_text_key or "text")
            tok = str(config.tokenizer_mode or "byte")
            max_chars = int(config.corpus_max_chars)
            split_tag = str(split or "train").lower()
            data_tag = (
                f"corpus:{version}:{fmt}:{text_key}:{tok}:{max_chars}:"
                f"train{train_frac:.3f}:val{val_frac:.3f}:split{split_tag}"
            )
            step_state = {"step": 0}

            def data_fn(batch_size, seq_len, dev):
                step = step_state["step"]
                step_state["step"] = step + 1
                generator = torch.Generator(device=dev)
                generator.manual_seed(1337 + step)
                batcher = self._get_corpus_batcher(config)
                if batcher is not None:
                    batch = batcher.sample_batch(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        generator=generator,
                        device=dev,
                        split=split_tag,
                    )
                    if batch is not None:
                        return batch
                return torch.randint(0, config.vocab_size, (batch_size, seq_len), device=dev, generator=generator)

            return data_fn, data_tag, True
        return None, "random", False

    def _run_pending_investigation(self):
        """Launch pending auto-investigation if queued."""
        pending = getattr(self, "_pending_investigation", None)
        if pending is None:
            return
        self._pending_investigation = None

        if self.is_running:
            return

        try:
            self.start_investigation(
                result_ids=pending["result_ids"],
                config=pending["config"],
                hypothesis=pending["hypothesis"],
            )
        except Exception as e:
            logger.warning(f"Failed to launch auto-investigation: {e}")

    def _run_pending_validation(self):
        """Launch pending auto-validation if queued."""
        pending = getattr(self, "_pending_validation", None)
        if pending is None:
            return
        self._pending_validation = None

        if self.is_running:
            return

        try:
            self.start_validation(
                result_ids=pending["result_ids"],
                config=pending["config"],
                hypothesis=pending["hypothesis"],
                trigger="auto_escalate",
            )
        except Exception as e:
            logger.warning(f"Failed to launch auto-validation: {e}")

    # ── Evolution & Novelty Search ──
