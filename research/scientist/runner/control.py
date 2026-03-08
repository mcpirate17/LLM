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

import logging
logger = logging.getLogger(__name__)

from ._types import RunConfig, LiveProgress, _LIVE_LOSS_CURVE_MAX_POINTS, _TRAINING_STEP_SSE_EVERY


class _ControlMixin:
    """Start/stop experiment methods, events, chat actions."""

    def start_experiment(self, config: RunConfig,
                         hypothesis: Optional[str] = None,
                         preregistration: Optional[Dict[str, Any]] = None,
                         exploratory: bool = False) -> str:
        """Start an experiment in a background thread. Returns experiment ID."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        config, prescreen = self.prescreen_run_config(
            config,
            mode="single",
            auto_harden=True,
        )

        self._ensure_math_spaces()
        self._stop_event.clear()
        self._set_aria_cycle_phase(
            "idle",
            continuous_active=False,
            cycle_index=0,
            selected_mode=None,
            note="Single-run experiment started.",
            emit_event=False,
        )

        # Pre-generate experiment ID
        nb = self._make_notebook()

        # Populate refuted hypotheses cache for similarity gating
        self._populate_refuted_cache(nb)

        hypothesis_metadata = {
            "source": "user_input" if hypothesis is not None else "unknown",
            "llm_used": False,
            "fallback_used": False,
            "used_context": False,
            "review_status": "not_reviewed",
            "confidence": None,
            "critique": None,
        }
        if hypothesis is None:
            context = self._build_start_experiment_hypothesis_context(nb, config)
            llm_available = self.aria._get_llm() is not None
            if llm_available and not (context or "").strip():
                context = build_manual_start_fallback_context(config.to_dict())
            result = None
            if context:
                result = self.aria.formulate_hypothesis(
                    context=context,
                    return_metadata=True,
                )
                hypothesis_metadata["used_context"] = True
            else:
                result = self.aria.formulate_hypothesis(return_metadata=True)

            if isinstance(result, tuple):
                hypothesis, meta = result
                hypothesis_metadata.update(meta or {})
            else:
                hypothesis = result
                hypothesis_metadata["source"] = (
                    "rule_based_fallback" if context else "rule_based"
                )

            if context:
                hypothesis_metadata["context_char_count"] = len(context)

        # Preflight hypothesis critique
        critique = None
        if hypothesis:
            try:
                critique_context = self._build_start_experiment_hypothesis_context(
                    nb, config,
                ) if hypothesis_metadata.get("source") == "user_input" else ""
                critique = self.aria.critique_hypothesis(
                    hypothesis, context=critique_context,
                )
                hypothesis_metadata["preflight_critique"] = critique
                hypothesis_metadata["critique"] = critique
                hypothesis_metadata["critique_confidence"] = critique.get("confidence")
                hypothesis_metadata["review_status"] = f"preflight_{critique.get('gate', 'warn')}"
            except Exception as e:
                logger.warning(f"Hypothesis critique failed: {e}")

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="synthesis",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_experiment",
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=config.n_programs,
                aria_message=self.aria.greet(),
                hypothesis_critique=critique,
            )

        self._emit_event("experiment_started", {
            "experiment_id": exp_id,
            "hypothesis": hypothesis,
            "config": config.to_dict(),
            "prescreen": prescreen,
            "aria_greeting": self.aria.greet(),
            "hypothesis_critique": critique,
        })

        self._thread = threading.Thread(
            target=self._run_experiment_thread,
            args=(exp_id, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def _build_start_experiment_hypothesis_context(
        self, nb: LabNotebook, config: RunConfig,
    ) -> str:
        """Build context for hypothesis generation in manual start_experiment.

        Ensures manual starts use the same context-aware hypothesis pathway as
        continuous mode whenever history/analytics are available.
        """
        try:
            recent = nb.get_recent_experiments(10)
            leaderboard = nb.get_leaderboard(limit=20)
            analytics_data = self._gather_analytics_data(nb)
            context = build_mode_selection_context(
                recent_experiments=recent,
                leaderboard=leaderboard,
                analytics_data=analytics_data,
                current_mode="synthesis",
                n_experiments_in_session=len(recent),
                cost_spent=self.aria.total_cost,
                budget=config.max_cost_dollars,
            )
            if config.max_cost_dollars > 0:
                context += (f"\n\nBudget: ${self.aria.total_cost:.2f} spent "
                            f"of ${config.max_cost_dollars:.2f}")
            return context
        except Exception as e:
            logger.debug("Failed to build manual hypothesis context: %s", e)
            return build_manual_start_fallback_context(config.to_dict())

    def start_continuous(self, config: RunConfig) -> str:
        """Start continuous experiment mode in background."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        config, _ = self.prescreen_run_config(
            config,
            mode="continuous",
            auto_harden=True,
        )

        self._ensure_math_spaces()
        self._stop_event.clear()
        with self._lock:
            self._aria_cycle_paused = False

        config.continuous = True
        self._set_aria_cycle_phase(
            "planning",
            continuous_active=True,
            cycle_index=0,
            selected_mode=None,
            note="Continuous session initialized.",
        )

        limits = []
        if config.max_experiments > 0:
            limits.append(f"max_experiments={config.max_experiments}")
        if config.max_time_minutes > 0:
            limits.append(f"max_time={config.max_time_minutes}min")
        if config.max_cost_dollars > 0:
            limits.append(f"max_cost=${config.max_cost_dollars:.2f}")
        logger.info(
            "Starting continuous session: %d programs/cycle, dim=%d, "
            "depth=%d, ops=%d, device=%s [%s]",
            config.n_programs, config.model_dim, config.max_depth,
            config.max_ops, config.device,
            ", ".join(limits) if limits else "no limits",
        )

        with self._lock:
            self._progress = LiveProgress(
                status="generating",
                aria_message=f"{self.aria.NAME} entering continuous research mode...",
            )

        self._thread = threading.Thread(
            target=self._run_continuous_thread,
            args=(config,),
            daemon=True,
        )
        self._thread.start()
        return "continuous"

    def start_fingerprint_refinement(
        self,
        result_ids: List[str],
        config: RunConfig,
        hypothesis: Optional[str] = None,
    ) -> str:
        """Start local mutation refinement around selected fingerprint sources."""
        ids = [rid.strip() for rid in result_ids if str(rid).strip()]
        if not ids:
            raise ValueError("result_ids required for fingerprint refinement")

        refine_config = RunConfig.from_dict(config.to_dict())
        refine_config.model_source = "fingerprint_refine"
        refine_config.refine_source_result_ids = ",".join(ids)
        if refine_config.refine_mutations_per_source <= 0:
            refine_config.refine_mutations_per_source = 1

        source_stage1_passed = 0
        recent_synthesis_s1_rate = 0.0
        source_rows: List[Dict[str, Any]] = []
        recommendation: Optional[Dict[str, Any]] = None
        try:
            nb = self._make_notebook()
            recent = self._recent_synthesis_health(nb, window=5)
            recent_synthesis_s1_rate = float(recent.get("s1_rate") or 0.0)
            for rid in ids:
                row = nb.get_program_detail(rid)
                if row and row.get("stage1_passed"):
                    source_stage1_passed += 1
                if isinstance(row, dict):
                    source_rows.append(row)

            requested_intent = str(refine_config.refine_intent or "balanced").strip().lower()
            if requested_intent in {"recommended", "auto"}:
                # Auto-run RefinementAnalyzer if no pre-computed analysis
                if not refine_config.refine_analysis_json and source_rows:
                    try:
                        from ..analytics import ExperimentAnalytics, RefinementAnalyzer
                        analytics = ExperimentAnalytics(nb)
                        analyzer = RefinementAnalyzer(analytics)
                        primary_row = source_rows[0]
                        primary_id = primary_row.get("result_id", ids[0])
                        analysis = analyzer.analyze_program_for_refinement(primary_id, primary_row)
                        recipe = analysis.get("recipe", {})
                        resolved_intent = recipe.get("recommended_intent", "balanced")
                        recommendation = {
                            "intent": resolved_intent,
                            "rationale": recipe.get("primary_target", ""),
                            "evidence": recipe.get("grammar_hints", {}),
                        }
                        refine_config.refine_analysis_json = json.dumps(analysis)
                    except Exception as e:
                        logger.warning("RefinementAnalyzer failed, falling back: %s", e)
                        resolved_intent, recommendation = self._recommend_refinement_intent(
                            nb, source_rows,
                        )
                else:
                    resolved_intent, recommendation = self._recommend_refinement_intent(
                        nb,
                        source_rows,
                    )
                refine_config.refine_intent = resolved_intent
            nb.close()
        except Exception:
            recent_synthesis_s1_rate = 0.0

        if hypothesis is None:
            intent_spec = self._refinement_intent_spec(refine_config.refine_intent)
            source_rule = (
                f"source_selection_rule=result_ids({len(ids)}) with "
                f"stage1_survivor_sources={source_stage1_passed}/{len(ids)}"
            )
            mutation_plan = (
                "mutation_mechanism=evolution_local_neighborhood("
                f"operators=op_replace|config_tweak|edge_rewire, mutation_rate={refine_config.mutation_rate:.2f}, "
                f"mutations_per_source={refine_config.refine_mutations_per_source}, "
                f"pool_multiplier={max(1, int(refine_config.refine_pool_multiplier or 1))})"
            )
            baseline_s1 = f"recent_synthesis_s1_rate={recent_synthesis_s1_rate:.3f}"
            success_criteria = (
                "success_criteria=(stage0_pass_rate>=0.95 AND stage05_pass_rate>=0.70) "
                "AND (delta_s1_rate>=+0.03_vs_recent OR best_loss_ratio<=0.98*parent_loss_ratio)"
            )
            fallback_plan = (
                "fallback_plan=if(no_stage1_improvement OR no_stage1_sources) "
                "queue_ablation_suite_and_novelty_mode"
            )
            recommendation_clause = ""
            if recommendation:
                recommendation_clause = (
                    " recommended_intent="
                    f"{recommendation.get('intent')}"
                    f" rationale={recommendation.get('rationale')}"
                    f" evidence={recommendation.get('evidence')}"
                    ";"
                )
            hypothesis = (
                "Fingerprint refinement hypothesis: "
                f"{source_rule}; "
                f"{mutation_plan}; "
                f"intent={intent_spec['name']} weights={intent_spec['weights']} "
                f"score={intent_spec['formula']}; "
                f"{recommendation_clause} "
                f"{baseline_s1}; "
                f"{success_criteria}; "
                f"{fallback_plan}."
            )

        return self.start_experiment(refine_config, hypothesis=hypothesis)

    def _recommend_refinement_intent(
        self,
        nb: LabNotebook,
        source_rows: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        """Recommend refinement intent from historical quality/novelty/compression evidence."""
        if not source_rows:
            return "balanced", {
                "intent": "balanced",
                "rationale": "no_source_rows",
                "evidence": {"source_count": 0},
            }

        op_success = self._op_success_lookup(nb)
        sparse_hint_ops = ("sparse", "gate", "topk", "mask", "threshold", "skip", "mixture")

        loss_values: List[float] = []
        novelty_values: List[float] = []
        param_values: List[float] = []
        op_success_values: List[float] = []
        sparse_ratios: List[float] = []

        for row in source_rows:
            loss = row.get("loss_ratio")
            novelty = row.get("novelty_score")
            params = row.get("param_count") or row.get("graph_n_params_estimate")

            if isinstance(loss, (int, float)):
                loss_values.append(float(loss))
            if isinstance(novelty, (int, float)):
                novelty_values.append(float(novelty))
            if isinstance(params, (int, float)) and float(params) > 0:
                param_values.append(float(params))

            ops: List[str] = []
            graph_json = row.get("graph_json")
            if isinstance(graph_json, str) and graph_json.strip():
                try:
                    graph_data = json.loads(graph_json)
                    nodes = graph_data.get("nodes", {}) if isinstance(graph_data, dict) else {}
                    for nd in nodes.values():
                        if not isinstance(nd, dict):
                            continue
                        op_name = str(nd.get("op_name") or "").strip().lower()
                        if not op_name or op_name == "input":
                            continue
                        ops.append(op_name)
                except Exception:
                    ops = []

            if ops:
                scores = [float(op_success.get(op, 0.5)) for op in ops]
                op_success_values.append(sum(scores) / len(scores))
                sparse_ratio = sum(
                    1.0 for op in ops if any(token in op for token in sparse_hint_ops)
                ) / len(ops)
                sparse_ratios.append(float(sparse_ratio))

        mean_loss = (sum(loss_values) / len(loss_values)) if loss_values else None
        mean_novelty = (sum(novelty_values) / len(novelty_values)) if novelty_values else None
        mean_params = (sum(param_values) / len(param_values)) if param_values else None
        mean_op_success = (
            sum(op_success_values) / len(op_success_values)
        ) if op_success_values else None
        mean_sparse_ratio = (
            sum(sparse_ratios) / len(sparse_ratios)
        ) if sparse_ratios else None

        intent = "balanced"
        rationale = "mixed_signals"
        if ((mean_loss is not None and mean_loss >= 0.75)
                or (mean_op_success is not None and mean_op_success < 0.35)):
            intent = "quality"
            rationale = "weak_quality_signal"
        elif (mean_params is not None and mean_params >= 500_000
              and (mean_loss is None or mean_loss <= 0.80)):
            intent = "compression"
            rationale = "high_parameter_budget"
        elif mean_novelty is not None and mean_novelty < 0.45:
            intent = "novelty"
            rationale = "low_novelty_signal"
        elif (mean_sparse_ratio is not None and mean_sparse_ratio < 0.10
              and mean_params is not None and mean_params >= 1_000_000):
            intent = "sparsity"
            rationale = "sparse_operator_gap"
        elif (mean_params is not None and mean_params > 0
              and mean_loss is not None and mean_loss < 0.60):
            # Good quality but check FLOP efficiency
            baseline_params = 6 * 256 ** 2  # ~393K for a minimal 2-layer transformer
            if mean_params > 3 * baseline_params:
                intent = "compression"
                rationale = "low_flop_efficiency"

        recommendation = {
            "intent": intent,
            "rationale": rationale,
            "evidence": {
                "source_count": len(source_rows),
                "mean_loss_ratio": mean_loss,
                "mean_novelty": mean_novelty,
                "mean_params": mean_params,
                "mean_op_success": mean_op_success,
                "mean_sparse_op_ratio": mean_sparse_ratio,
            },
        }
        return intent, recommendation

    def start_investigation(self, result_ids: List[str], config: RunConfig,
                            hypothesis: Optional[str] = None,
                            preregistration: Optional[Dict[str, Any]] = None,
                            exploratory: bool = False,
                            force: bool = False) -> str:
        """Start investigation phase for selected candidates.

        Args:
            force: Skip tier and already-investigated guards.  Allows
                   re-investigating candidates with different config
                   (e.g. longer steps, different data mode).
        """
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()

        if not force:
            # Tier guard: reject result IDs already at investigation tier or beyond
            tiers = nb.get_tiers_for_result_ids(result_ids)
            already_done = {
                rid: tier for rid, tier in tiers.items()
                if tier in ("investigation", "validation", "breakthrough")
            }
            if already_done:
                nb.close()
                labels = ", ".join(f"{rid} ({tier})" for rid, tier in already_done.items())
                raise ValueError(
                    f"Cannot investigate: {len(already_done)} candidate(s) already "
                    f"at or beyond investigation tier: {labels}"
                )
        else:
            logger.info("Force re-investigation: skipping tier/fingerprint guards for %s",
                        ", ".join(r[:8] for r in result_ids))
            # Reset tier to screening so the investigation can re-promote
            for rid in result_ids:
                try:
                    nb.conn.execute(
                        "UPDATE leaderboard SET tier = 'screening', "
                        "investigation_passed = NULL, investigation_loss_ratio = NULL, "
                        "investigation_robustness = NULL, investigation_best_training = NULL "
                        "WHERE result_id = ?", (rid,))
                except Exception:
                    pass
            try:
                nb.conn.commit()
            except Exception:
                pass

        source = "user_input" if hypothesis is not None else "runner_template"
        if hypothesis is None:
            hypothesis = (
                f"Investigation: deep study of {len(result_ids)} screening survivors "
                f"with multiple training programs to test robustness."
            )

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="investigation",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source=source,
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_investigation",
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=len(result_ids),
                aria_message=f"{self.aria.NAME}: Starting investigation of {len(result_ids)} candidate(s)...",
            )

        self._emit_event("investigation_started", {
            "experiment_id": exp_id,
            "hypothesis": hypothesis,
            "result_ids": result_ids,
            "n_training_programs": config.n_training_programs,
        })

        self._thread = threading.Thread(
            target=self._run_investigation_thread,
            args=(exp_id, result_ids, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def start_validation(self, result_ids: List[str], config: RunConfig,
                         hypothesis: Optional[str] = None,
                         preregistration: Optional[Dict[str, Any]] = None,
                         exploratory: bool = False,
                         trigger: str = "manual",
                         force: bool = False) -> str:
        """Start validation phase for investigation survivors."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()

        # Tier guards can be bypassed explicitly for manual override workflows.
        tiers = nb.get_tiers_for_result_ids(result_ids)
        if not force:
            already_validated = {
                rid: tier for rid, tier in tiers.items()
                if tier in ("validation", "breakthrough")
            }
            if already_validated:
                nb.close()
                labels = ", ".join(f"{rid} ({tier})" for rid, tier in already_validated.items())
                raise ValueError(
                    f"Cannot validate: {len(already_validated)} candidate(s) already "
                    f"at or beyond validation tier: {labels}"
                )
            # Warn if known-screening candidates haven't been investigated
            # (result_ids without leaderboard entries are allowed — they may
            # come from auto-escalation paths that create entries mid-flight)
            not_investigated = {
                rid for rid in result_ids
                if tiers.get(rid) == "screening"
            }
            if not_investigated:
                nb.close()
                raise ValueError(
                    f"Cannot validate: {len(not_investigated)} candidate(s) are still "
                    f"at screening tier (not investigated): {', '.join(not_investigated)}"
                )

        source = "user_input" if hypothesis is not None else "runner_template"
        if hypothesis is None:
            hypothesis = (
                f"Validation: publication-grade testing of {len(result_ids)} "
                f"investigation survivors with multi-seed evaluation."
            )

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="validation",
            config=self._validation_config_with_result_ids(config, result_ids, trigger),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source=source,
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_validation",
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=len(result_ids),
                aria_message=f"{self.aria.NAME}: Starting validation of {len(result_ids)} candidate(s)...",
            )

        self._emit_event("validation_started", {
            "experiment_id": exp_id,
            "hypothesis": hypothesis,
            "result_ids": result_ids,
        })

        self._thread = threading.Thread(
            target=self._run_validation_thread,
            args=(exp_id, result_ids, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def start_scale_up(self, result_ids: List[str], config: RunConfig,
                       hypothesis: Optional[str] = None,
                       preregistration: Optional[Dict[str, Any]] = None,
                       exploratory: bool = False) -> str:
        """Start scale-up validation of specific programs in a background thread."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()
        source = "user_input" if hypothesis is not None else "runner_template"
        if hypothesis is None:
            hypothesis = (
                f"Scale-up validation: testing whether {len(result_ids)} "
                f"top performer(s) maintain their advantage at 10x training scale."
            )

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="scale_up",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source=source,
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_scale_up",
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=len(result_ids),
                aria_message=f"{self.aria.NAME}: Starting scale-up validation of {len(result_ids)} program(s)...",
            )

        self._emit_event("scale_up_started", {
            "experiment_id": exp_id,
            "hypothesis": hypothesis,
            "result_ids": result_ids,
            "config": {
                "steps": config.scale_up_steps,
                "batch_size": config.scale_up_batch_size,
                "seq_len": config.scale_up_seq_len,
            },
        })

        self._thread = threading.Thread(
            target=self._run_scale_up_thread,
            args=(exp_id, result_ids, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def start_evolution(self, config: RunConfig,
                        hypothesis: Optional[str] = None,
                        preregistration: Optional[Dict[str, Any]] = None,
                        exploratory: bool = False) -> str:
        """Start evolutionary search in a background thread."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()
        hypothesis_metadata = self._build_hypothesis_metadata(
            source="user_input" if hypothesis is not None else "unknown",
            llm_used=False,
            fallback_used=False,
            used_context=False,
        )
        if hypothesis is None:
            result = self.aria.formulate_hypothesis(return_metadata=True)
            if isinstance(result, tuple):
                hypothesis, meta = result
                hypothesis_metadata.update(meta or {})
            else:
                hypothesis = result
                hypothesis_metadata["source"] = "rule_based"

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="evolution",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_evolution",
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_generations=config.n_generations,
                aria_message=f"{self.aria.NAME}: Starting evolutionary search...",
            )

        self._emit_event("evolution_started", {
            "experiment_id": exp_id,
            "hypothesis": hypothesis,
            "config": config.to_dict(),
        })

        self._thread = threading.Thread(
            target=self._run_evolution_thread,
            args=(exp_id, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def start_novelty_search(self, config: RunConfig,
                             hypothesis: Optional[str] = None,
                             preregistration: Optional[Dict[str, Any]] = None,
                             exploratory: bool = False) -> str:
        """Start novelty search in a background thread."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()
        hypothesis_metadata = self._build_hypothesis_metadata(
            source="user_input" if hypothesis is not None else "unknown",
            llm_used=False,
            fallback_used=False,
            used_context=False,
        )
        if hypothesis is None:
            result = self.aria.formulate_hypothesis(return_metadata=True)
            if isinstance(result, tuple):
                hypothesis, meta = result
                hypothesis_metadata.update(meta or {})
            else:
                hypothesis = result
                hypothesis_metadata["source"] = "rule_based"

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="novelty",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_novelty_search",
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_generations=config.n_generations,
                aria_message=f"{self.aria.NAME}: Starting novelty search...",
            )

        self._emit_event("novelty_started", {
            "experiment_id": exp_id,
            "hypothesis": hypothesis,
            "config": config.to_dict(),
        })

        self._thread = threading.Thread(
            target=self._run_novelty_thread,
            args=(exp_id, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def start_resume(self, experiment_id: str, config: Optional[RunConfig] = None) -> str:
        """Resume an interrupted experiment from its last checkpoint.

        Looks up the experiment in the notebook, reconstructs config if needed,
        and dispatches to the appropriate thread based on experiment type.
        """
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()
        exp_data = nb.get_resumable_experiment(experiment_id)
        if exp_data is None:
            nb.close()
            raise ValueError(
                f"Experiment {experiment_id} not found or not resumable "
                "(must be 'running' or 'failed')")

        exp_type = exp_data["experiment_type"]
        exp_data.get("hypothesis", "")

        # Reconstruct config from stored config_json
        if config is None:
            try:
                config_dict = json.loads(exp_data["config_json"])
                config = RunConfig.from_dict(config_dict)
            except Exception:
                nb.close()
                raise ValueError(
                    f"Cannot reconstruct config for experiment {experiment_id}")

        config.resume_experiment_id = experiment_id

        # Mark experiment as running again if it was failed
        if exp_data["status"] == "failed":
            nb.conn.execute(
                "UPDATE experiments SET status = 'running' WHERE experiment_id = ?",
                (experiment_id,),
            )
            nb.conn.commit()
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=experiment_id,
                status="resuming",
                aria_message=f"Resuming {exp_type} experiment {experiment_id}...",
            )

        self._emit_event("experiment_resuming", {
            "experiment_id": experiment_id,
            "experiment_type": exp_type,
        })

        if exp_type == "continuous" or config.continuous:
            self._thread = threading.Thread(
                target=self._run_continuous_thread,
                args=(config,),
                daemon=True,
            )
        else:
            logger.warning("Resume for experiment type '%s' not yet supported, "
                           "falling back to continuous", exp_type)
            config.continuous = True
            self._thread = threading.Thread(
                target=self._run_continuous_thread,
                args=(config,),
                daemon=True,
            )

        self._thread.start()
        return experiment_id

    def stop(self):
        """Stop the current experiment gracefully."""
        self._stop_event.set()
        self.aria.state.mood = "contemplative"
        with self._lock:
            self._aria_cycle_paused = False
        self._set_aria_cycle_phase(
            "stopping",
            continuous_active=self.is_running,
            note="Stop requested; wrapping up current work.",
        )
        with self._lock:
            self._progress.status = "stopped"
            self._progress.aria_message = "Stopping... wrapping up current evaluation."
            
            # Z17: Clear global native-runner counters immediately on stop
            reset_native_runner_telemetry()
            
        self._emit_event("experiment_stopping", {})

    # ── Routing Benchmark Harness (Track C) ──

    def get_events(self, timeout: float = 30.0):
        """Generator yielding events for SSE streaming."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                event = self._event_queue.get(timeout=1.0)
                yield event
            except queue.Empty:
                # Send keepalive
                yield {"type": "keepalive", "data": {}, "timestamp": time.time()}

    # ── Start / Stop ──

    def _emit_event(self, event_type: str, data: Dict):
        """Push an event for SSE consumers."""
        # training_step can emit at very high frequency; throttle SSE pressure so
        # structural live-feed events (program_evaluated/validation_progress/etc.)
        # are not dropped when the event queue is saturated.
        should_enqueue = True
        if event_type == "training_step":
            step = int(data.get("step") or 0)
            total_steps = int(data.get("total_steps") or 0)
            should_enqueue = (
                step <= 1
                or (step % _TRAINING_STEP_SSE_EVERY == 0)
                or (total_steps > 0 and step >= total_steps)
            )

        payload = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        }

        try:
            if should_enqueue:
                self._event_queue.put_nowait(payload)
        except queue.Full:
            if event_type != "training_step":
                # Prefer dropping one stale queued event so critical lifecycle
                # events can still be delivered to the dashboard feed.
                try:
                    self._event_queue.get_nowait()
                    self._event_queue.put_nowait(payload)
                except Exception:
                    pass
        # Buffer training_step events for REST retrieval (dashboard chart restore).
        # Keep a deep enough history so the dashboard can reconstruct near-full
        # curves for long validation/investigation runs.
        if event_type == "training_step":
            curve = self._live_loss_curve
            exp_id = data.get("experiment_id", "")
            if curve and curve[0].get("experiment_id") != exp_id:
                curve.clear()
            curve.append(data)
            if len(curve) > _LIVE_LOSS_CURVE_MAX_POINTS:
                del curve[:len(curve) - _LIVE_LOSS_CURVE_MAX_POINTS]

    @staticmethod
    def _aria_phase_label(phase: str) -> str:
        labels = {
            "idle": "Idle",
            "planning": "Planning",
            "running": "Running",
            "analyzing": "Analyzing",
            "paused": "Paused",
            "stopping": "Stopping",
            "completed": "Completed",
            "failed": "Failed",
        }
        return labels.get(phase, phase.replace("_", " ").title())

    def _set_aria_cycle_phase(
        self,
        phase: str,
        *,
        cycle_index: Optional[int] = None,
        selected_mode: Optional[str] = None,
        note: Optional[str] = None,
        continuous_active: Optional[bool] = None,
        emit_event: bool = True,
    ) -> None:
        """Track Aria's continuous cycle phase for observability APIs/UI."""
        with self._lock:
            payload: Dict[str, Any] = {
                "phase": str(phase or "idle"),
                "phase_label": self._aria_phase_label(str(phase or "idle")),
                "last_transition_ts": time.time(),
            }
            if cycle_index is not None:
                payload["cycle_index"] = int(cycle_index)
            if selected_mode is not None:
                payload["selected_mode"] = str(selected_mode)
            if note is not None:
                payload["last_note"] = str(note)
            if continuous_active is not None:
                payload["continuous_active"] = bool(continuous_active)
            if phase == "running" and selected_mode is not None:
                payload["last_completed_mode"] = None
            if phase in {"analyzing", "completed", "failed"} and selected_mode is not None:
                payload["last_completed_mode"] = str(selected_mode)

            self._aria_cycle_status.update(payload)
            snapshot = dict(self._aria_cycle_status)

        if emit_event:
            self._emit_event("aria_cycle_phase", snapshot)

    def get_aria_cycle_status(self) -> Dict[str, Any]:
        """Return latest Aria cycle status for dashboard/API polling."""
        with self._lock:
            cycle = dict(self._aria_cycle_status)
            progress = self._progress.to_dict()
            last_cycle = dict(self._last_cycle_summary) if self._last_cycle_summary else None
            cycle_history = [dict(item) for item in self._aria_cycle_history[-10:]]
            cycle_paused = bool(self._aria_cycle_paused)
        cycle["is_running"] = self.is_running
        cycle["progress_status"] = progress.get("status")
        cycle["aria_message"] = progress.get("aria_message")
        cycle["experiment_id"] = progress.get("experiment_id")
        cycle["last_cycle_summary"] = last_cycle
        cycle["cycle_history"] = cycle_history
        cycle["cycle_paused"] = cycle_paused
        return cycle

    def pause_aria_cycle(self) -> Dict[str, Any]:
        """Pause continuous cycle progression between experiment iterations."""
        with self._lock:
            self._aria_cycle_paused = True
            running = self.is_running
        note = (
            "Pause requested; pausing before the next cycle."
            if running
            else "Cycle is paused. Start continuous mode to resume execution."
        )
        self._set_aria_cycle_phase(
            "paused",
            continuous_active=running,
            note=note,
        )
        self._emit_event("aria_cycle_paused", {"note": note})
        return self.get_aria_cycle_status()

    def resume_aria_cycle(self) -> Dict[str, Any]:
        """Resume continuous cycle progression."""
        with self._lock:
            self._aria_cycle_paused = False
            running = self.is_running
            cycle_index = int(self._aria_cycle_status.get("cycle_index") or 0)
        self._set_aria_cycle_phase(
            "planning" if running else "idle",
            continuous_active=running,
            cycle_index=cycle_index,
            note="Cycle resumed." if running else "Cycle resumed and awaiting start.",
        )
        self._emit_event("aria_cycle_resumed", {"running": running})
        return self.get_aria_cycle_status()

    def _wait_for_cycle_resume(self, cycle_index: int) -> None:
        """Block between cycles while paused, unless stop is requested."""
        with self._lock:
            paused = bool(self._aria_cycle_paused)
        if not paused:
            return
        self._set_aria_cycle_phase(
            "paused",
            continuous_active=True,
            cycle_index=cycle_index,
            note="Cycle paused; waiting for resume.",
        )
        while not self._stop_event.is_set():
            with self._lock:
                paused = bool(self._aria_cycle_paused)
            if not paused:
                break
            time.sleep(0.5)

    def _build_aria_cycle_summary(
        self,
        *,
        cycle_index: int,
        selected_mode: str,
        mode_reasoning: str,
        mode_confidence: Optional[float],
        before_progress: Dict[str, Any],
        after_progress: Dict[str, Any],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a compact cycle summary payload for SSE/UI/chat consumers."""
        before_total = int(before_progress.get("total_programs") or 0)
        after_total = int(after_progress.get("total_programs") or 0)
        before_s1 = int(before_progress.get("stage1_passed") or 0)
        after_s1 = int(after_progress.get("stage1_passed") or 0)

        summary = {
            "cycle_index": int(cycle_index),
            "mode": str(selected_mode or "synthesis"),
            "reasoning": str(mode_reasoning or ""),
            "confidence": float(mode_confidence or 0.0),
            "status": "failed" if error else "completed",
            "programs_total": after_total,
            "stage1_survivors": after_s1,
            "delta_programs": max(0, after_total - before_total),
            "delta_stage1_survivors": max(0, after_s1 - before_s1),
            "aria_message": str(after_progress.get("aria_message") or ""),
            "timestamp": time.time(),
            "before": dict(before_progress or {}),
            "after": dict(after_progress or {}),
        }
        if error:
            summary["error"] = str(error)
        return summary

    def execute_chat_action(self, action: Dict[str, Any], nb) -> Dict[str, Any]:
        """Execute an action dispatched from Aria's chat response.

        Supported types: adjust_config, adjust_grammar, start_experiment, edit_file.
        """
        action_type = str(action.get("type") or "").strip()

        if action_type == "adjust_config":
            changes = action.get("changes") or {}
            if not isinstance(changes, dict) or not changes:
                return {"status": "error", "error": "No changes provided"}
            # Apply via _config_with_overrides on a fresh default config
            base = RunConfig()
            effective, report = self._config_with_overrides(base, changes)
            # Store as the new defaults for future experiments
            self._last_chat_config_overrides = changes
            nb.log_learning_event(
                "chat_config_adjusted",
                f"Aria adjusted config: {report.get('applied', {})}",
                changes=report.get("applied", {}),
                ignored=report.get("ignored", {}),
            )
            return {"status": "applied", "changes": report.get("applied", {}),
                    "ignored": report.get("ignored", {})}

        elif action_type == "adjust_grammar":
            weights = action.get("weights") or {}
            if not isinstance(weights, dict) or not weights:
                return {"status": "error", "error": "No weights provided"}
            # Validate values are numeric
            clean_weights = {}
            for k, v in weights.items():
                try:
                    clean_weights[str(k)] = float(v)
                except (ValueError, TypeError):
                    pass
            if not clean_weights:
                return {"status": "error", "error": "No valid numeric weights"}
            self._grammar_weight_overrides.update(clean_weights)
            nb.log_learning_event(
                "chat_grammar_adjusted",
                f"Aria adjusted grammar weights: {clean_weights}",
                weights=clean_weights,
                all_overrides=dict(self._grammar_weight_overrides),
            )
            return {"status": "applied", "weights": clean_weights}

        elif action_type == "start_experiment":
            if self.is_running:
                return {"status": "busy", "error": "An experiment is already running"}
            mode = str(action.get("mode") or "synthesis").strip().lower()
            config_overrides = action.get("config") or {}
            config = RunConfig()
            if isinstance(config_overrides, dict):
                for k, v in config_overrides.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
            try:
                if mode in {"sparse_morph", "sparse_morphology", "sparse_morphological"}:
                    config.model_source = "morphological_box"
                    config.morph_focus_sparse = True
                    config.n_programs = max(120, int(config.n_programs))
                    config.n_layers = max(1, min(int(config.n_layers), 4))
                    config.max_depth = max(2, min(int(config.max_depth), 6))
                    config.max_ops = max(4, min(int(config.max_ops), 10))
                    exp_id = self.start_experiment(config)
                if mode == "evolution":
                    exp_id = self.start_evolution(config)
                elif mode == "novelty":
                    exp_id = self.start_novelty_search(config)
                elif mode in {"sparse_morph", "sparse_morphology", "sparse_morphological"}:
                    pass
                else:
                    exp_id = self.start_experiment(config)
                return {"status": "started", "experiment_id": exp_id, "mode": mode}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        elif action_type == "edit_file":
            return self._execute_edit_file_action(action, nb)

        elif action_type == "maintain_database":
            return self._execute_maintain_database_action(action, nb)

        else:
            return {"status": "error", "error": f"Unknown action type: {action_type}"}

    def _execute_edit_file_action(self, action: Dict[str, Any], nb) -> Dict[str, Any]:
        """Execute an edit_file action with safety rails."""
        import py_compile
        import shutil

        path = str(action.get("path") or "").strip()
        search = str(action.get("search") or "")
        replace = str(action.get("replace") or "")
        description = str(action.get("description") or "Chat-initiated edit")

        # Safety: reject path traversal
        if ".." in path:
            return {"status": "error", "error": "Path traversal (..) not allowed"}

        # Safety: allow edits only within known project subpaths
        allowed_prefixes = (
            "research/",
            "scientist/", "synthesis/", "eval/", "search/", "training/",
            "dashboard/", "tests/", "tools/", "mathspaces/",
        )
        if not any(path.startswith(prefix) for prefix in allowed_prefixes):
            return {"status": "error", "error": "Path must be under research/ or a known project folder"}

        # Safety: only .py and .js files
        if not (path.endswith(".py") or path.endswith(".js")):
            return {"status": "error", "error": "Only .py and .js files can be edited"}

        # Resolve to absolute path.
        # project_root is typically <repo>/research when running from the package layout.
        # If the incoming path already starts with research/, resolve from repo root;
        # otherwise resolve from project_root directly.
        # __file__ is runner/control.py; go up 3 levels to reach research/
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        repo_root = os.path.dirname(project_root)
        if path.startswith("research/"):
            abs_path = os.path.normpath(os.path.join(repo_root, path))
        else:
            abs_path = os.path.normpath(os.path.join(project_root, path))

        # Double-check resolved path is under project
        if not abs_path.startswith(project_root):
            return {"status": "error", "error": "Resolved path escapes project directory"}

        if not os.path.isfile(abs_path):
            return {"status": "error", "error": f"File not found: {path}"}

        # Read current content
        with open(abs_path, "r") as f:
            content = f.read()

        if search not in content:
            return {"status": "error", "error": "Search string not found in file"}

        # Create backup
        timestamp = int(time.time())
        backup_path = f"{abs_path}.bak.{timestamp}"
        shutil.copy2(abs_path, backup_path)

        # Apply edit
        new_content = content.replace(search, replace, 1)
        with open(abs_path, "w") as f:
            f.write(new_content)

        # Syntax check for .py files
        if path.endswith(".py"):
            try:
                py_compile.compile(abs_path, doraise=True)
            except py_compile.PyCompileError as e:
                # Restore backup
                shutil.copy2(backup_path, abs_path)
                os.remove(backup_path)
                return {"status": "error", "error": f"Syntax error after edit, reverted: {e}"}

        # Log to notebook
        nb.log_learning_event(
            "chat_file_edited",
            f"Aria edited {path}: {description}",
            path=path,
            backup=backup_path,
            description=description,
        )

        return {"status": "applied", "path": path, "backup": backup_path,
                "description": description}

    # ── Database Maintenance Actions ──────────────────────────────────────

    def _execute_maintain_database_action(
        self, action: Dict[str, Any], nb: LabNotebook,
    ) -> Dict[str, Any]:
        """Execute a database maintenance operation.

        Allowed operations:
          purge_empty_experiments  — delete failed experiments with no results
          purge_junk_programs      — delete S0 failures with no error classification
          reset_op_stats           — reset op_success_rates for specific ops
          clear_toxic_signatures   — remove failure_signatures for specific ops
          vacuum                   — reclaim disk space
          backfill_failure_signatures — one-time backfill from existing results
        """
        operation = str(action.get("operation") or "").strip()
        if operation not in self._MAINTENANCE_OPS:
            return {
                "status": "error",
                "error": f"Unknown maintenance operation: {operation}. "
                         f"Allowed: {', '.join(sorted(self._MAINTENANCE_OPS))}",
            }

        try:
            if operation == "purge_empty_experiments":
                n = nb.purge_empty_experiments()
                nb.log_learning_event(
                    "maintenance_purge_experiments",
                    f"Aria purged {n} empty failed experiments",
                )
                return {"status": "applied", "deleted_experiments": n}

            elif operation == "purge_junk_programs":
                # Delete S0 failures with no error_type (no learning signal)
                cur = nb.conn.execute(
                    "DELETE FROM program_results "
                    "WHERE (stage0_passed = 0 OR stage0_passed IS NULL) "
                    "AND (error_type IS NULL OR error_type = '')"
                )
                n = cur.rowcount
                nb._maybe_commit()
                nb.log_learning_event(
                    "maintenance_purge_junk",
                    f"Aria purged {n} junk S0 failure records",
                )
                return {"status": "applied", "deleted_programs": n}

            elif operation == "reset_op_stats":
                ops = action.get("ops") or []
                if not isinstance(ops, list) or not ops:
                    return {"status": "error", "error": "Provide 'ops' list of op names to reset"}
                op_names = [str(o).strip() for o in ops if str(o).strip()]
                if not op_names:
                    return {"status": "error", "error": "No valid op names provided"}
                placeholders = ",".join("?" * len(op_names))
                cur = nb.conn.execute(
                    f"DELETE FROM op_success_rates WHERE op_name IN ({placeholders})",
                    op_names,
                )
                n = cur.rowcount
                nb._maybe_commit()
                nb.log_learning_event(
                    "maintenance_reset_op_stats",
                    f"Aria reset op stats for {op_names} ({n} rows)",
                    ops=op_names,
                )
                return {"status": "applied", "ops_reset": op_names, "rows_deleted": n}

            elif operation == "clear_toxic_signatures":
                ops = action.get("ops") or []
                if not isinstance(ops, list) or not ops:
                    return {"status": "error", "error": "Provide 'ops' list of op names to clear signatures for"}
                total = 0
                for op in ops:
                    op = str(op).strip()
                    if not op:
                        continue
                    cur = nb.conn.execute(
                        "DELETE FROM failure_signatures WHERE signature LIKE ?",
                        (f"%{op}%",),
                    )
                    total += cur.rowcount
                nb._maybe_commit()
                nb.log_learning_event(
                    "maintenance_clear_toxic",
                    f"Aria cleared {total} toxic signatures for {ops}",
                    ops=[str(o).strip() for o in ops],
                )
                return {"status": "applied", "signatures_deleted": total, "ops": [str(o).strip() for o in ops]}

            elif operation == "vacuum":
                nb.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                # VACUUM requires isolation_level=None; run on a fresh connection
                import sqlite3
                vac_conn = sqlite3.connect(nb.db_path, isolation_level=None)
                vac_conn.execute("VACUUM")
                vac_conn.close()
                nb.log_learning_event(
                    "maintenance_vacuum",
                    "Aria ran VACUUM to reclaim disk space",
                )
                return {"status": "applied", "operation": "vacuum"}

            elif operation == "backfill_failure_signatures":
                n = nb.backfill_failure_signatures()
                return {"status": "applied", "signatures_created": n}

        except Exception as e:
            logger.warning("Maintenance action %s failed: %s", operation, e)
            return {"status": "error", "error": str(e)[:200]}

        return {"status": "error", "error": "Unreachable"}

    @property
    def last_recommendation(self) -> Optional[Dict]:
        """Last auto-generated recommendation after experiment completion."""
        with self._lock:
            rec = self._last_recommendation
            # Clear after reading so dashboard only shows it once
            self._last_recommendation = None
            return rec

