"""Execution mixin: screening thread + core experiment logic.

# INVESTIGATION NOTE — S0.5 gate (2026-03-20)
# ────────────────────────────────────────────
# Diagnosis found 0 S0.5 failures across 500 programs. Investigation:
#
# a) S0.5 is computed on EVERY candidate that passes S0 (line ~1332).
#    It is not gated by any other code path — all S0 survivors reach it.
#
# b) stability_score CAN be < 0.5. It is `checks_passed / total_checks`
#    where total_checks = 6 (random, extreme, sequential, high_id,
#    causality, training_dynamics). Passing 2/6 = 0.33 < 0.5.
#    However, for a model that compiled and ran a forward pass:
#    - Tests 1-4 (forward-pass probes) almost always pass if S0 passed,
#      because S0 already verified a forward pass with no NaN/Inf.
#    - Test 5 (causality) passes when diff < 0.05, which is true for
#      most architectures unless they use non-causal ops (attention
#      without masking, bidirectional ops).
#    - Test 6 (training dynamics) needs 20 steps without NaN and CV < 0.25.
#    In practice, models that survive S0 typically pass 5/6 or 6/6 checks,
#    giving stability_score >= 0.83. The 0.5 threshold is too low to filter
#    anything that S0 didn't already catch.
#
# c) causality_passed CAN be False — when diff >= 0.05 or the check throws.
#    This happens with non-causal ops. However, such architectures typically
#    also fail S0 (safe_eval catches NaN from unbounded attention) or get
#    killed by rapid screening. The S0.5 causality gate is defense-in-depth
#    but rarely the first filter to fire.
#
# d) CONCLUSION: S0.5 is not vacuous — it can theoretically reject models.
#    But it is effectively redundant given the current pipeline ordering:
#    S0 (safe_eval) already rejects models that produce NaN/Inf, which is
#    the same failure mode that would push stability_score below 0.5.
#    The 0.5 threshold (config.stage05_stability_threshold) should be raised
#    to ~0.67 (4/6 checks) to catch models with marginal stability that
#    currently waste rapid-screen budget. This is NOT a bug fix — it's a
#    threshold calibration issue for a future tuning pass.
"""

from __future__ import annotations

import json
import math
import random
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from ..json_utils import fast_dumps, json_safe
from ..runtime_events import publish_lifecycle_event

import torch

from ...synthesis.grammar import GrammarConfig, batch_generate
from ..native_runner import compile_model_native_first as _compile_model_native
from ...synthesis.compiler import compile_model as _compile_model_legacy
from ...synthesis.validator import validate_graph
from ..refinement_scoring import rank_synthesis_candidates_by_stability
from ...eval.flops import estimate_flops
from ...eval.perf_budget import evaluate_perf_budget_gate
from .execution_screening_graphs import (
    analyze_graph_for_screening,
    structural_gate_failure,
    toxic_failure_ratio,
)
from .screening_candidate_rank import judgment_rerank
from .screening_signal_weights import (
    apply_insight_adjustments,
    build_signal_weight_maps,
)
from .failure_provenance import infer_graph_failure_provenance
from ..notebook import LabNotebook, ExperimentEntry

import logging

logger = logging.getLogger(__name__)


def _log_learning_event_compat(nb: LabNotebook, *args, **kwargs) -> None:
    getattr(nb, "log_learning_event")(*args, **kwargs)


# Gate 5 constant: routing/MoE/sparse/compression ops required for efficiency scoring.
# Module-level to avoid re-instantiation per graph in the screening loop.
_EFFICIENCY_OPS = frozenset(
    {
        "arch_router",
        "compute_budget_router",
        "difficulty_blend_3way",
        "depth_weighted_proj",
        "block_sparse_linear",
        "learned_token_gate",
        "dual_compression_blend",
        "confidence_token_gate",
        "gated_delta",
        "gated_linear",
        "gather_topk",
        "hetero_moe",
        "latent_attention_compressor",
        "score_depth_blend",
        "depth_token_mask",
        "moe_2expert",
        "moe_topk",
        "sparse_bottleneck_moe",
        "nm_sparse_linear",
        "padic_gate",
        "adaptive_rank_gate",
        "relu_gated_moe",
        "route_lanes",
        "route_recursion",
        "route_topk",
        "signal_conditioned_compression",
        "sparse_threshold",
        "ternary_projection",
        "adjacent_token_merge",
        "topk_gate",
        "tropical_gate",
        "tropical_moe",
        "tropical_router",
        "hybrid_token_gate",
        "sparse_span_builder",
        "hybrid_sparse_router",
        "lane_conditioned_block",
        "default_path",
    }
)


def _record_screening_failure(
    *,
    nb,
    exp_id: str,
    graph,
    stage0_passed: bool,
    stage05_passed: bool,
    error_type: str | None = None,
    error_message: str | None = None,
    stage_at_death: str | None = None,
    stability_score: float | None = None,
    extra_metrics: Dict[str, Any] | None = None,
) -> None:
    """Persist an early-screening failure for coverage-oriented runs."""
    try:
        persisted_extra_metrics = dict(extra_metrics or {})
        if extra_metrics:
            persisted_extra_metrics.update(screening_wikitext_fields(extra_metrics))
            persisted_extra_metrics.update(screening_probe_fields(extra_metrics))
        failure_provenance = infer_graph_failure_provenance(
            graph,
            error_type=error_type,
            error_message=error_message,
        )
        nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=graph.fingerprint(),
            graph_json=json.dumps(graph.to_dict(), separators=(",", ":")),
            bypass_quality_gate=True,
            stage0_passed=stage0_passed,
            stage05_passed=stage05_passed,
            stage1_passed=False,
            graph_n_ops=graph.n_ops(),
            graph_n_unique_ops=len(
                {
                    node.op_name
                    for node in graph.nodes.values()
                    if not node.is_input and not getattr(node, "is_output", False)
                }
            ),
            graph_has_gradient_path=graph.has_gradient_path(),
            stability_score=stability_score,
            error_type=error_type,
            error_message=error_message,
            stage_at_death=stage_at_death,
            failure_op=failure_provenance["failure_op"],
            failure_details_json=failure_provenance["failure_details_json"],
            **persisted_extra_metrics,
        )
    except Exception as exc:
        logger.debug("Failed to persist screening failure for %s: %s", exp_id, exc)


from ._types import RunConfig, LiveProgress
from ._helpers import (
    _native_proactive_gating,
    clear_gpu_memory,
    graph_observed_routing_ops,
    graph_routing_ops,
    routing_fast_lane_fields,
    screening_probe_fields,
    screening_wikitext_fields,
)

# S0.75 initial-loss threshold: architectures with initial CE loss above this
# are killed before rapid screening. Calibrated from diagnosis (2026-03-20):
# architectures with init_loss > 50 have deep unscaled projection chains and
# cannot reach the random-baseline floor (~10.94) within 500 S1 steps.
# Normal architectures start at init_loss ~11–16 (near ln(vocab_size)=11.52).
INITIAL_LOSS_THRESHOLD: float = 50.0

# Number of gradient steps for S0.75 mini-train probe
_S075_PROBE_STEPS: int = 5



def _make_experiment_results() -> Dict[str, Any]:
    """Create a fresh experiment results dict with all counters zeroed."""
    return {
        "total": 0,
        "stage0_passed": 0,
        "stage05_passed": 0,
        "rapid_screening_killed": 0,
        "rapid_screening_kill_reasons": {},
        "stage09_passed": 0,
        "stage1_passed": 0,
        "novel_count": 0,
        "best_loss_ratio": None,
        "best_novelty_score": None,
        "survivors": [],
        "skipped_proactive_gating": 0,
        "proactive_gating_failures": [],
        "funnel_counts": {
            "raw_generated": 0,
            "post_batch_dedup": 0,
            "judgment_filtered": 0,
            "post_judgment": 0,
            "screening_considered": 0,
            "dropped_runtime_dedup": 0,
            "dropped_toxic": 0,
            "dropped_proactive_gating": 0,
            "dropped_invalid_graph": 0,
            "dropped_runtime_error": 0,
            "stage0_attempted": 0,
            "stage0_passed": 0,
            "dropped_stage0": 0,
            "stage05_passed": 0,
            "dropped_stage05": 0,
            "dropped_s075_high_init": 0,
            "rapid_screen_attempted": 0,
            "dropped_rapid_screening": 0,
            "stage1_queued": 0,
            "stage09_completed": 0,
            "stage09_survived": 0,
            "stage1_completed": 0,
            "stage1_survived": 0,
            "persisted_rows": 0,
            "dropped_persistence_quality_gate": 0,
        },
    }


def _make_stage1_screening_config(config: RunConfig) -> RunConfig:
    """Strip expensive post-train eval from candidate-screening Stage 1.

    HellaSwag (~3.9s) and BLiMP (~2s) are fast enough to run on every S1
    passer.  They feed composite_score which drives auto-escalation —
    without them, screening entries stay below the 62.7 threshold and the
    investigation queue starves. Keep a cheap real-token LM probe and the
    lightweight binding/induction probes enabled so Stage 1 can reject
    retrieval-dead or text-dead architectures early.
    """
    stage1_config = config.copy()
    # Keep post-eval ENABLED so fast probes run on S1 passers.
    stage1_config.profile_disable_post_eval = False
    stage1_config.stage1_compute_val_loss = False
    stage1_config.stage1_compute_discovery_loss = False
    stage1_config.skip_screening_wikitext = False
    stage1_config.skip_screening_hellaswag = False  # ~3.9s, feeds composite
    stage1_config.skip_screening_blimp = False  # ~2s, feeds composite
    stage1_config.skip_binding_probes = False
    stage1_config.binding_probe_train_batch_size = max(
        1, int(getattr(config, "binding_probe_train_batch_size", 0) or 2)
    )
    stage1_config.binding_probe_eval_batch_size = max(
        1, int(getattr(config, "binding_probe_eval_batch_size", 0) or 4)
    )
    stage1_config.skip_post_s1_fingerprint = True
    stage1_config.skip_post_s1_triage = True
    stage1_config.collect_training_curve = False
    return stage1_config


class _ExecutionScreeningMixin:
    """Screening experiment thread and core experiment execution."""

    __slots__ = ()

    def _log_learning_event_compat(self, nb: LabNotebook, *args, **kwargs) -> None:
        getattr(nb, "log_learning_event")(*args, **kwargs)

    def _publish_screening_terminal_event(
        self,
        *,
        event_type: str,
        exp_id: str,
        payload: dict,
    ) -> None:
        publish_lifecycle_event(
            notebook_path=self.notebook_path,
            event_type=event_type,
            producer="runner.execution_screening",
            run_id=exp_id,
            payload=payload,
        )

    def _complete_experiment_compat(
        self,
        *,
        nb,
        experiment_id: str,
        results: dict,
        aria_summary: str,
        insights,
        llm_analysis: str | None,
    ) -> None:
        getattr(nb, "complete_experiment")(
            experiment_id=experiment_id,
            results=results,
            aria_summary=aria_summary,
            aria_mood=self.aria.state.mood,
            insights=insights,
            llm_analysis=llm_analysis,
        )

    def _fail_experiment_compat(
        self,
        *,
        nb,
        experiment_id: str,
        error: str,
    ) -> None:
        getattr(nb, "fail_experiment")(experiment_id, error)

    def _run_experiment_thread(self, exp_id: str, config: RunConfig, hypothesis: str):
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
                results, config, hypothesis, nb
            )

            summary = self.aria.experiment_summary(results, context=context)
            insights = self._analyze_results(results, exp_id, nb, context=context)

            # Store LLM analysis if available
            llm_analysis = self.aria.analyze_results(results, context=context)

            # Validate hypothesis
            try:
                validation = self.aria.validate_hypothesis(hypothesis, results, context)
                if validation:
                    nb.add_entry(
                        ExperimentEntry(
                            entry_type="analysis",
                            title="Hypothesis Validation",
                            content=validation.get("explanation", ""),
                            experiment_id=exp_id,
                            metadata={"validated": validation.get("validated", False)},
                        )
                    )
            except Exception as e:
                logger.warning("Hypothesis validation logging failed: %s", e)

            self._publish_screening_terminal_event(
                event_type="experiment_completed",
                exp_id=exp_id,
                payload={
                    "completed_at": time.time(),
                    "results": results,
                    "aria_summary": summary,
                    "aria_mood": self.aria.state.mood,
                    "insights": insights,
                    "llm_analysis": llm_analysis,
                    "mode": "screening",
                },
            )
            self._complete_experiment_compat(
                nb=nb,
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                insights=insights,
                llm_analysis=llm_analysis,
            )

            # Update op success rates and failure signatures after experiment.
            # _s0_op_counts tracks ALL compiled programs (pass + fail) so it
            # is the single source of truth.  Only fall back to
            # update_op_success_rates (program_results scan) when no in-memory
            # counts exist (e.g. investigation/validation modes).
            s0_op_counts = results.pop("_s0_op_counts", None)
            with nb.batch():
                if s0_op_counts:
                    nb.merge_op_failure_counts(s0_op_counts)
                else:
                    nb.update_op_success_rates(exp_id)
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
                            len(rehab_results),
                            ", ".join(rehab_results),
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

            # Flush async writes so auto-escalate can read back S1 survivors
            nb.flush_writes()
            # Auto-escalation pipeline (investigation/validation)
            results["experiment_id"] = exp_id
            self._auto_escalate(results, config, nb, phase="screening")

            # Auto-scale-up if criteria met (legacy, kept for backward compat)
            self._maybe_auto_scale_up(results, config, nb)

            # Auto-report for single experiments
            self._maybe_auto_report(config, nb, reason="experiment_complete")

            self._update_progress(
                status="completed",
                aria_message=summary.split("\n")[-1]
                if summary
                else "Experiment complete.",
            )

            self._emit_event(
                "experiment_completed",
                {
                    "experiment_id": exp_id,
                    "results": results,
                    "summary": summary,
                },
            )

        except Exception as e:
            error = traceback.format_exc()
            logger.error("Experiment failed (%s): %s\n%s", exp_id, e, error)
            try:
                self._invoke_code_healer(
                    nb=nb,
                    trigger_type="repeated_exception",
                    experiment_id=exp_id,
                    scope=f"Synthesis/experiment failure: {str(e)[:240]}",
                    reproduction_steps=[
                        'python -m pytest tests/test_integration.py -k "start_experiment" -x --tb=short'
                    ],
                    acceptance_tests=[
                        'python -m pytest tests/test_integration.py -k "start_experiment" -x --tb=short'
                    ],
                    trigger_payload={"mode": "synthesis", "error": str(e)},
                )
            except Exception:
                logger.warning(
                    "code_healer failed during experiment error handling", exc_info=True
                )
            self._publish_screening_terminal_event(
                event_type="experiment_failed",
                exp_id=exp_id,
                payload={
                    "completed_at": time.time(),
                    "error": str(e),
                    "results": None,
                    "mode": "screening",
                },
            )
            self._fail_experiment_compat(
                nb=nb,
                experiment_id=exp_id,
                error=str(e),
            )
            self._update_progress(
                status="failed",
                error=str(e),
                aria_message=self.aria.react_to_failure(str(e)),
            )
            self._emit_event(
                "experiment_failed",
                {
                    "experiment_id": exp_id,
                    "error": str(e),
                },
            )
        except BaseException as e:
            logger.critical(
                "Experiment thread KILLED (%s): %s\n%s",
                exp_id,
                e,
                traceback.format_exc(),
            )
            try:
                self._publish_screening_terminal_event(
                    event_type="experiment_failed",
                    exp_id=exp_id,
                    payload={
                        "completed_at": time.time(),
                        "error": f"FATAL: {e}",
                        "results": None,
                        "mode": "screening",
                        "fatal": True,
                    },
                )
                self._fail_experiment_compat(
                    nb=nb,
                    experiment_id=exp_id,
                    error=f"FATAL: {e}",
                )
                self._update_progress(status="failed", error=f"FATAL: {e}")
                self._emit_event(
                    "experiment_failed",
                    {"experiment_id": exp_id, "error": f"FATAL: {e}"},
                )
            except Exception:
                logger.error(
                    "Failed to emit failure event after fatal error", exc_info=True
                )
            raise
        finally:
            nb.close()
            # Launch queued auto-scale-up after notebook is closed
            self._run_pending_scale_up()

    def _prepare_grammar_config(
        self,
        exp_id: str,
        config: RunConfig,
        nb: LabNotebook,
        results: Dict,
        use_learned_grammar: bool = True,
    ) -> Tuple[GrammarConfig, Dict[str, float], Any]:
        """Build grammar config with learned weights, champion bias, and efficiency tuning.

        Returns:
            (grammar, failure_blocklist, analytics)
        """
        grammar_weights = None
        op_weights: Dict[str, float] = {}
        failure_blocklist: Dict[str, float] = {}
        champion_bias: Dict[str, float] = {}
        template_weights: Dict[str, float] = {}
        motif_weights: Dict[str, float] = {}
        analytics = None
        grammar_gate: Optional[Dict[str, Any]] = None
        from ..ml_influence_policy import component_is_allowed

        learned_grammar_allowed = component_is_allowed(
            "learned_grammar_weights", config
        )
        screening_signal_allowed = component_is_allowed(
            "screening_signal_weights", config
        )
        if use_learned_grammar and learned_grammar_allowed:
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
                        self._log_learning_event_compat(
                            nb,
                            "grammar_weights_blocked",
                            f"Blocked grammar weight update for {exp_id}: weak attribution evidence",
                            evidence=fast_dumps(json_safe(grammar_gate), safe=True),
                        )
                        grammar_weights = None
            except Exception as e:
                logger.warning(
                    "Failed computing learned grammar weights for %s: %s", exp_id, e
                )
        elif use_learned_grammar and not learned_grammar_allowed:
            logger.info(
                "Learned grammar weights requested but blocked by ML trust policy"
            )

            # Soft-penalize poorly-performing ops (no hard exclusion — causality
            # sandbox gate catches truly broken ops at eval time)
            op_weights: Dict[str, float] = {}
            try:
                rehab_cache = nb.get_op_rehabilitation_cache()
                if analytics is not None:
                    neg = analytics.negative_results_synthesis()
                    for op_info in neg.get("failed_ops", []):
                        if (
                            op_info.get("s1_rate", 1) == 0
                            and op_info.get("n_used", 0) >= 5
                            and op_info.get("confidence", 0) >= 0.7
                        ):
                            op_name = op_info["op_name"]
                            rehab = rehab_cache.get(op_name)
                            if (
                                rehab
                                and rehab.get("compile_passed")
                                and rehab.get("forward_passed")
                            ):
                                op_weights[op_name] = 0.5
                            elif op_info.get("failure_stage") == "compilation":
                                op_weights[op_name] = 0.15
                            else:
                                op_weights[op_name] = 0.1
                    for op_info in neg.get("weak_ops", []):
                        op_name = op_info.get("op_name", "")
                        penalty = op_info.get("penalty_weight", 1.0)
                        if op_name:
                            op_weights[op_name] = penalty
                    if op_weights:
                        self._log_learning_event_compat(
                            nb,
                            "weak_ops_penalized",
                            f"Soft-penalized {len(op_weights)} weak ops: "
                            f"{', '.join(f'{k}={v:.2f}' for k, v in sorted(op_weights.items()))}",
                            op_weights=op_weights,
                        )
            except Exception as e:
                logger.warning("Failed computing op penalties for %s: %s", exp_id, e)

            # Load failure-signature blocklist (op-pair bigrams with high fail rate)
            failure_blocklist: Dict[str, float] = {}
            try:
                failure_blocklist = nb.get_failure_signature_blocklist()
                if failure_blocklist:
                    self._log_learning_event_compat(
                        nb,
                        "failure_signatures_loaded",
                        f"Loaded {len(failure_blocklist)} toxic op-pair patterns",
                        signatures=sorted(failure_blocklist.keys())[:10],
                    )
            except Exception as e:
                logger.warning(
                    "Failed loading failure signatures for %s: %s", exp_id, e
                )

            # Champion bias pass: nudge category weights toward proven winners.
            # This biases the search toward high-performing projection/sparse patterns
            # and known-good structural/sequence motifs without hard-coding op-level picks.
            try:
                if analytics is not None:
                    # Use 7d windowed rates to avoid death spiral from
                    # stale lifetime data poisoning recently-fixed ops
                    _window_cutoff = time.time() - 604800  # 7 days
                    op_rates = analytics.op_success_rates(since_ts=_window_cutoff) or {}
                    if op_rates:
                        winning_ops = {"exp", "selective_scan", "tropical_center"}
                        projection_ops = {
                            "low_rank_proj",
                            "shared_basis_proj",
                            "tied_proj",
                        }
                        sparse_ops = {
                            "nm_sparse_linear",
                            "block_sparse_linear",
                            "semi_structured_2_4_linear",
                        }

                        def _is_reliable(
                            op_name: str, min_used: int = 10, min_s1: float = 0.25
                        ) -> bool:
                            info = op_rates.get(op_name) or {}
                            n_used = int(info.get("n_used") or 0)
                            s1_rate = float(info.get("s1_rate") or 0.0)
                            return n_used >= min_used and s1_rate >= min_s1

                        has_winners = any(_is_reliable(op) for op in winning_ops)
                        has_projection = any(_is_reliable(op) for op in projection_ops)
                        has_sparse = any(_is_reliable(op) for op in sparse_ops)

                        if has_winners:
                            champion_bias["structural"] = max(
                                champion_bias.get("structural", 1.0), 1.2
                            )
                            champion_bias["sequence"] = max(
                                champion_bias.get("sequence", 1.0), 1.2
                            )
                        if has_projection:
                            champion_bias["parameterized"] = max(
                                champion_bias.get("parameterized", 1.0), 1.4
                            )
                        if has_sparse:
                            champion_bias["parameterized"] = max(
                                champion_bias.get("parameterized", 1.0), 1.5
                            )
                            # Z7: If sparse ops are reliable, nudge the grammar hard toward them
                            champion_bias["_structured_sparsity_bias"] = 0.8

            except Exception as e:
                logger.warning("Failed computing champion bias for %s: %s", exp_id, e)

        if screening_signal_allowed:
            try:
                template_weights, motif_weights = build_signal_weight_maps(nb)
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug("Failed building signal weight maps: %s", e)
                template_weights, motif_weights = {}, {}
        else:
            template_weights, motif_weights = {}, {}
            logger.info("Screening signal weight maps disabled or blocked for this run")

        # Data-driven op/template/motif weights from accumulated S1 pass rates.
        # Template and motif weights share the same DB query internally
        # (_compute_metadata_weights), so computing them back-to-back is
        # efficient — the DB page cache serves the second query from memory.
        if analytics is not None and screening_signal_allowed:
            _window_cutoff = time.time() - 604800  # 7 days
            try:
                learned_op_weights = analytics.compute_op_weights(
                    since_ts=_window_cutoff
                )
                op_weights.update(learned_op_weights)
            except (TypeError, ValueError, KeyError) as e:
                logger.debug("Failed computing learned op weights: %s", e)
            try:
                learned_tpl_weights, learned_motif_weights = (
                    analytics.compute_template_and_motif_weights(
                        since_ts=_window_cutoff
                    )
                )
                if learned_tpl_weights:
                    template_weights.update(learned_tpl_weights)
                if learned_motif_weights:
                    motif_weights.update(learned_motif_weights)
            except (TypeError, ValueError, KeyError) as e:
                logger.debug("Failed computing template/motif weights: %s", e)
            # Synergy-driven boosts: ops that co-occur in S1 survivors
            # get their motifs/templates boosted to encourage recombination.
            try:
                syn_motif_boosts, syn_tpl_boosts = analytics.compute_synergy_boosts()
                for name, boost in syn_motif_boosts.items():
                    motif_weights[name] = motif_weights.get(name, 1.0) * boost
                for name, boost in syn_tpl_boosts.items():
                    template_weights[name] = template_weights.get(name, 1.0) * boost
            except (TypeError, ValueError, KeyError) as e:
                logger.debug("Failed computing synergy boosts: %s", e)

        op_weights = {**op_weights, **self._op_weights_overrides}
        grammar = self._build_grammar_config(config, op_weights=op_weights)
        explicit_template_weights = bool(getattr(config, "template_weights", None))
        # Merge learned template/motif weights, but preserve explicit template
        # weights supplied by callers such as targeted backfill.
        if explicit_template_weights:
            pass
        elif grammar.routing_mandatory:
            # Routing-first: only merge in weights that don't conflict
            for k, v in template_weights.items():
                grammar.template_weights.setdefault(k, v)
        else:
            grammar.template_weights = template_weights
        grammar.motif_weights = motif_weights
        # Apply Bayesian insight adjustments to grammar config
        if screening_signal_allowed:
            try:
                apply_insight_adjustments(
                    nb, grammar, grammar.template_weights, grammar.motif_weights
                )
            except Exception as e:
                logger.debug("Insight grammar adjustment failed: %s", e)

        if grammar_weights:
            old_weights = dict(grammar.category_weights)
            grammar.category_weights.update(grammar_weights)
            n_changed = sum(
                1
                for key, value in grammar_weights.items()
                if old_weights.get(key) != value
            )
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
            self._emit_event(
                "learning_event",
                {
                    "event_type": "grammar_weights_applied",
                    "experiment_id": exp_id,
                    "n_changed": n_changed,
                    "max_depth": int(config.max_depth),
                    "max_ops": int(config.max_ops),
                    "description": (
                        f"Applied learned grammar weights ({n_changed} categories changed; "
                        f"depth<= {int(config.max_depth)}, ops<= {int(config.max_ops)})"
                    ),
                },
            )

        if champion_bias:
            before_bias = dict(grammar.category_weights)
            for category, multiplier in champion_bias.items():
                if category == "_structured_sparsity_bias":
                    grammar.structured_sparsity_bias = float(multiplier)
                    continue
                base = float(grammar.category_weights.get(category, 1.0))
                grammar.category_weights[category] = round(
                    max(0.5, min(8.0, base * multiplier)), 2
                )
            self._log_learning_event_compat(
                nb,
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
            self._log_learning_event_compat(
                nb,
                "chat_grammar_overrides_applied",
                f"Applied chat-driven grammar overrides for {exp_id}",
                overrides=dict(self._grammar_weight_overrides),
                final_weights=dict(grammar.category_weights),
            )
            results["applied_grammar_weights"] = dict(grammar.category_weights)
        else:
            grammar.category_weights["math_space"] = config.math_space_weight

        # Efficiency bias: boost categories that produce compact/efficient architectures
        # Targets sparse, low-rank, MoE, and state-space ops per frontier micronization memo
        _eff_weight = getattr(config, "selection_efficiency_weight", 0.25)
        if _eff_weight >= 0.3:  # only apply when efficiency is prioritized
            _eff_boost = min(1.0 + _eff_weight, 2.0)  # 1.3-2.0x
            for _cat in ("structural", "parameterized"):
                _base = float(grammar.category_weights.get(_cat, 1.0))
                grammar.category_weights[_cat] = round(min(8.0, _base * _eff_boost), 2)
            # Boost specific efficiency-related ops
            for _op in (
                "moe_2expert",
                "moe_topk",
                "block_sparse_linear",
                "bottleneck_proj",
                "linear_proj_down",
                "selective_scan",
            ):
                grammar.op_weights[_op] = grammar.op_weights.get(_op, 1.0) * _eff_boost

        # Hyperbolic promotion: query recent hierarchy fitness from fingerprints
        if analytics is not None:
            try:
                hf = analytics.recent_hierarchy_fitness()
                if hf is not None:
                    grammar._hierarchy_fitness = hf
                    if hf > grammar.hyperbolic_promotion_threshold:
                        logger.info(
                            "Hierarchy detected (fitness=%.3f > %.2f): boosting hyperbolic ops",
                            hf,
                            grammar.hyperbolic_promotion_threshold,
                        )
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug("Hierarchy fitness lookup failed: %s", e)

        # Synthesized loss/optimizer exploration (20% of screening experiments)
        if random.random() < 0.2:
            config.loss_type = "synthesized"
        if random.random() < 0.2:
            config.optimizer_type = "synthesized"

        return grammar, failure_blocklist, analytics

    def _generate_and_filter_candidates(
        self,
        exp_id: str,
        config: RunConfig,
        nb: LabNotebook,
        grammar: GrammarConfig,
        analytics: Any,
        results: Dict,
        use_learned_grammar: bool = True,
    ) -> Tuple[List, Dict[int, float], Any, Optional[str], Any, int, float]:
        """Generate candidate graphs, dedup, rerank, and apply GBM prescreener.

        Returns:
            (graphs, judgment_scores, dev, dev_str, orchestrator, candidate_batch_size, t_start)
            If config.model_source == "morphological_box", returns
            ([], {}, None, None, None, 0, 0.0) after running morphological screening.
        """
        t_start = time.time()

        # Generate graphs
        if config.model_source == "morphological_box":
            self._run_morphological_screening(exp_id, config, nb, results, t_start)
            return [], {}, None, None, None, 0, 0.0

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
                        self._log_learning_event_compat(
                            nb,
                            "adaptive_synthesis_enabled",
                            f"Enabling budget-aware adaptive synthesis for {exp_id}",
                            frontier_size=len(frontier),
                        )
                except Exception as e:
                    logger.warning("Failed to initialize efficiency prior: %s", e)

            batch_seed = self._stable_seed(
                exp_id,
                config.mode,
                config.n_programs,
                config.model_source,
                "batch_generate",
            )
            logger.info(
                "Experiment %s: batch_generate base_seed=%d",
                exp_id[:8],
                batch_seed,
            )
            _bg_result = batch_generate(
                config.n_programs,
                grammar,
                base_seed=batch_seed,
                _use_adaptive_synthesis=use_adaptive,
                prior=prior,
            )
            graphs = _bg_result.graphs
            results["batch_generate_stats"] = {
                "base_seed": batch_seed,
                "n_attempted": _bg_result.n_attempted,
                "n_rejected_grammar": _bg_result.n_rejected_grammar,
                "n_rejected_dedup": _bg_result.n_rejected_dedup,
            }
        results["funnel_counts"]["raw_generated"] = len(graphs)
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
                self._log_learning_event_compat(
                    nb,
                    "architecture_distribution_shift",
                    f"Generated-op distribution shift recorded for synthesis experiment {exp_id}",
                    evidence=fast_dumps(json_safe(shift), safe=True),
                )
            else:
                self._log_learning_event_compat(
                    nb,
                    "architecture_distribution_snapshot",
                    f"Captured generated-op distribution for synthesis experiment {exp_id}",
                    evidence=fast_dumps(
                        json_safe({"op_distribution": op_distribution}), safe=True
                    ),
                )

        self._log_generated_graph_observation(nb, exp_id, graphs, grammar, config)
        dev, dev_str, orchestrator, candidate_batch_size = (
            self._prepare_screening_orchestrator(config, results)
        )
        graphs, _existing_fps = self._dedup_graph_candidates(
            nb=nb,
            graphs=graphs,
            grammar=grammar,
            config=config,
            exp_id=exp_id,
            results=results,
        )
        results["funnel_counts"]["post_batch_dedup"] = len(graphs)
        # judgment_scores maps graph id(graph) → score for persistence
        _judgment_scores: Dict[int, float] = {}
        if graphs:
            before_judgment = len(graphs)
            graphs = rank_synthesis_candidates_by_stability(graphs)
            results["stability_reranked"] = True
            ranked = judgment_rerank(
                graphs,
                nb,
                logger,
                log_event=_log_learning_event_compat,
            )
            if len(ranked) != before_judgment:
                results["judgment_filtered"] = before_judgment - len(ranked)
            _judgment_scores = {id(g): s for g, s in ranked}
            graphs = [g for g, _ in ranked]
            results["funnel_counts"]["judgment_filtered"] = max(
                0, before_judgment - len(graphs)
            )
        results["funnel_counts"]["post_judgment"] = len(graphs)
        graphs = self._run_gbm_prescreener(
            nb=nb,
            graphs=graphs,
            config=config,
            exp_id=exp_id,
            results=results,
        )

        self._update_progress(total_programs=len(graphs))
        return (
            graphs,
            _judgment_scores,
            dev,
            dev_str,
            orchestrator,
            candidate_batch_size,
            t_start,
        )

    def _screen_candidate_quality_gates(
        self,
        graph,
        fp: str,
        config: RunConfig,
        results: Dict,
        graph_analysis,
        _judgment_scores: Dict,
        i: int,
    ) -> Optional[str]:
        """Collect metrics, run proactive gating and graph validation.

        On pass: populates self._last_gate_program_metrics and
        self._last_gate_graph_analysis for the caller.

        Returns a skip-reason string if the candidate should be skipped,
        None if it passed.
        """
        # Collect all metrics for this program
        program_metrics: Dict[str, Any] = {}
        program_metrics.update(self._extract_graph_metrics(graph))
        j_score = _judgment_scores.get(id(graph))
        if j_score is not None:
            program_metrics["judgment_score"] = j_score

        # Estimate FLOPs
        try:
            flop_est = estimate_flops(
                graph,
                seq_len=min(128, config.max_seq_len),
                d_model=config.model_dim,
            )
            program_metrics["flops_forward"] = flop_est.flops_forward
            program_metrics["flops_per_param"] = flop_est.flops_per_param
            program_metrics["flops_per_token"] = flop_est.flops_per_token
        except Exception as e:
            logger.debug("FLOP estimate failed for %s: %s", graph.fingerprint()[:10], e)

        # Native Proactive Gating (Project Hephaestus)
        try:
            native_gating = _native_proactive_gating(graph)
            if not native_gating.get("passed", True):
                results.setdefault("skipped_proactive_gating", 0)
                results["skipped_proactive_gating"] += 1
                results["funnel_counts"]["dropped_proactive_gating"] += 1
                program_metrics["proactive_gating_reason"] = native_gating.get("reason")
                program_metrics["max_depth"] = native_gating.get("max_depth")
                program_metrics["n_toxic_motifs"] = native_gating.get("n_toxic_motifs")
                self._emit_event(
                    "program_evaluated",
                    {
                        "index": i,
                        "fingerprint": fp[:10],
                        "result": "skipped_proactive",
                        "reason": native_gating.get("reason"),
                        "max_depth": native_gating.get("max_depth"),
                    },
                )
                return "proactive_gating"
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
            results.setdefault("s0_validation_failures", 0)
            results["s0_validation_failures"] += 1
            results["funnel_counts"]["dropped_invalid_graph"] += 1
            self._emit_event(
                "program_evaluated",
                {
                    "index": i,
                    "fingerprint": fp[:10],
                    "result": "invalid",
                    "error": validation.errors[0] if validation.errors else "",
                },
            )
            return "invalid_graph"

        # Stash results for the caller
        self._last_gate_program_metrics = program_metrics
        self._last_gate_graph_analysis = graph_analysis
        return None

    def _screen_candidate_gates(
        self,
        graph,
        fp: str,
        failure_blocklist,
        config: RunConfig,
        results: Dict,
        nb: LabNotebook,
        _judgment_scores: Dict,
        _get_primitive,
        i: int,
    ) -> Optional[str]:
        """Run pre-compilation gates on a single candidate.

        Checks: runtime dedup, structural gate, toxic ratio, FLOPs estimate,
        proactive gating, graph validation.

        On pass: populates self._last_gate_program_metrics and
        self._last_gate_graph_analysis for the caller to consume.

        Returns a skip-reason string if the candidate should be skipped,
        None if it passed all gates.
        """
        # Real-time dedup: skip if evaluated by another process since experiment start
        if not getattr(config, "disable_runtime_dedup", False) and nb.has_fingerprint(
            fp
        ):
            results.setdefault("skipped_dedup_runtime", 0)
            results["skipped_dedup_runtime"] += 1
            results["funnel_counts"]["dropped_runtime_dedup"] += 1
            self._emit_event(
                "program_evaluated",
                {
                    "index": i,
                    "fingerprint": fp[:10],
                    "result": "skipped_dedup",
                },
            )
            return "dedup"

        # ── Structural gates + toxic pre-screen (single-pass analysis) ──
        graph_analysis = analyze_graph_for_screening(graph, _get_primitive)
        # gate8_retrieval_dead is opt-in via capability_first mode. The flag
        # lives on RunConfig as ``_capability_first_mode`` (the UI toggle);
        # the corresponding ``binding_capable_required`` field on
        # GrammarConfig is not visible at this call site. Use the RunConfig
        # flag as the source of truth so gate8 actually fires when the user
        # enables capability-first.
        gate_fail = structural_gate_failure(
            graph,
            routing_mandatory=bool(config.routing_mandatory),
            efficiency_ops=_EFFICIENCY_OPS,
            analysis=graph_analysis,
            binding_capable_required=bool(
                getattr(config, "_capability_first_mode", False)
            ),
        )
        if gate_fail is not None:
            results["funnel_counts"].setdefault("dropped_structural_gate", 0)
            results["funnel_counts"]["dropped_structural_gate"] += 1
            results["funnel_counts"].setdefault(f"dropped_{gate_fail}", 0)
            results["funnel_counts"][f"dropped_{gate_fail}"] += 1
            return gate_fail

        if failure_blocklist:
            _toxic_ratio = toxic_failure_ratio(failure_blocklist, graph_analysis)
            if _toxic_ratio >= 0.5:
                results.setdefault("skipped_toxic", 0)
                results["skipped_toxic"] += 1
                results["funnel_counts"]["dropped_toxic"] += 1
                self._emit_event(
                    "program_evaluated",
                    {
                        "index": i,
                        "fingerprint": fp[:10],
                        "result": "skipped_toxic",
                        "toxic_ratio": f"{_toxic_ratio:.2f}",
                    },
                )
                return "toxic"

        # Collect metrics, run proactive gating, validate
        skip = self._screen_candidate_quality_gates(
            graph=graph,
            fp=fp,
            config=config,
            results=results,
            graph_analysis=graph_analysis,
            _judgment_scores=_judgment_scores,
            i=i,
        )
        if skip is not None:
            return skip

        return None

    def _handle_s0_s05_failure(
        self,
        s0_passed: bool,
        s05_passed: bool,
        sandbox_result,
        results: Dict,
        config: RunConfig,
        nb: LabNotebook,
        exp_id: str,
        graph,
        fp: str,
        i: int,
    ) -> None:
        """Record S0/S0.5 failure: update funnel counts, emit event, persist."""
        if not s0_passed:
            results["funnel_counts"]["dropped_stage0"] += 1
        else:
            results["funnel_counts"]["dropped_stage05"] += 1
        error_type = sandbox_result.error_type or "unknown"
        results.setdefault("failure_error_types", {})
        results["failure_error_types"][error_type] = (
            results["failure_error_types"].get(error_type, 0) + 1
        )
        self._emit_event(
            "program_evaluated",
            {
                "index": i,
                "fingerprint": fp[:10],
                "result": "fail_s0" if not s0_passed else "fail_s05",
                "error": (sandbox_result.error or "")[:120] if not s0_passed else None,
                "error_type": error_type,
                "stability": f"{sandbox_result.stability_score:.2f}"
                if s0_passed and not s05_passed
                else None,
                "params": sandbox_result.param_count
                if sandbox_result.param_count
                else None,
                "memory_mb": f"{sandbox_result.peak_memory_mb:.1f}"
                if sandbox_result.peak_memory_mb
                else None,
                "has_nan": sandbox_result.has_nan_output
                or sandbox_result.has_nan_grad
                or None,
                "has_inf": sandbox_result.has_inf_output or None,
            },
        )
        if config.persist_screening_failures:
            _record_screening_failure(
                nb=nb,
                exp_id=exp_id,
                graph=graph,
                stage0_passed=bool(s0_passed),
                stage05_passed=bool(s05_passed),
                error_type=error_type,
                error_message=(sandbox_result.error or "")[:240] or None,
                stage_at_death="stage0" if not s0_passed else "stage05",
                stability_score=sandbox_result.stability_score,
            )

    def _screen_candidate_compile_eval(
        self,
        graph,
        graph_analysis,
        config: RunConfig,
        dev_str: str,
        results: Dict,
        nb: LabNotebook,
        exp_id: str,
        fp: str,
        program_metrics: Dict[str, Any],
        _s0_op_counts: Dict[str, Dict[str, int]],
        i: int,
    ) -> Optional[Dict[str, Any]]:
        """Compile a candidate graph and run S0/S0.5 sandbox eval.

        Returns a dict with keys (model, sandbox_result, layer_graphs,
        use_progressive, phase1_vocab) on success, or None if the candidate
        failed S0 or S0.5.
        """
        # Progressive screening: Phase 1 uses cheap qualifying vocab (32K)
        # for S0/S0.5/rapid.  Only Phase 1 survivors get recompiled at
        # the real vocab for S1 training.
        _use_progressive = (
            config.progressive_screening
            and config.vocab_size > config.qualifying_vocab_size
        )
        _phase1_vocab = (
            config.qualifying_vocab_size if _use_progressive else config.vocab_size
        )

        results["funnel_counts"]["stage0_attempted"] += 1
        # Z13: Defensive pause + GC to stabilize Torch Dynamo context if needed
        if i > 0 and i % 10 == 0:
            clear_gpu_memory()

            # More aggressive reset every 50 to clear Torch Dynamo cache
            if i % 50 == 0:
                try:
                    torch.compiler.reset()
                except (AttributeError, RuntimeError) as e:
                    logger.debug("torch.compiler.reset() unavailable: %s", e)

            time.sleep(0.1)

        layer_graphs = [graph] * config.n_layers
        _compile_t0 = time.perf_counter()
        # Capability-first templates produce novel graph topologies that
        # can trigger segfaults in the native C/Rust/Cython dispatch path
        # (the native kernels were written before these templates existed).
        # Fall back to pure PyTorch compilation which is slower but never
        # segfaults. Once the native kernels are hardened for the role-slot
        # op combinations, this guard can be removed.
        _compiler = (
            _compile_model_legacy
            if getattr(config, "_capability_first_mode", False)
            else _compile_model_native
        )
        model = _compiler(
            layer_graphs,
            vocab_size=_phase1_vocab,
            max_seq_len=config.max_seq_len,
        )
        _compile_ms = (time.perf_counter() - _compile_t0) * 1000.0
        program_metrics["compile_time_ms"] = (
            float(program_metrics.get("compile_time_ms", 0.0) or 0.0) + _compile_ms
        )
        results.setdefault("_compile_times_ms", []).append(_compile_ms)
        _eval_timeout = 60 if getattr(config, "_exotic_mode", False) else 30
        sandbox_result = self._safe_eval_for_stage(
            model,
            stage_tag="candidate_screening",
            batch_size=2,
            seq_len=min(128, config.max_seq_len),
            vocab_size=_phase1_vocab,
            device=dev_str,
            timeout_seconds=_eval_timeout,
        )
        program_metrics.update(self._extract_sandbox_metrics(sandbox_result))
        program_metrics["param_count"] = sandbox_result.param_count

        s0_passed = sandbox_result.passed
        s05_passed = (
            sandbox_result.stability_score >= config.stage05_stability_threshold
            and sandbox_result.causality_passed
        )

        if s0_passed:
            results["stage0_passed"] += 1
            results["funnel_counts"]["stage0_passed"] += 1
            with self._lock:
                self._progress.stage0_passed += 1
        if s05_passed:
            results["stage05_passed"] += 1
            results["funnel_counts"]["stage05_passed"] += 1
            with self._lock:
                self._progress.stage05_passed += 1

        # Track ALL compiled programs in _s0_op_counts so
        # merge_op_failure_counts sees both passes and failures.
        for op_name in graph_analysis.counted_ops:
            c = _s0_op_counts.setdefault(op_name, {"n_used": 0, "n_s0": 0, "n_s05": 0})
            c["n_used"] += 1
            if s0_passed:
                c["n_s0"] += 1
            if s05_passed:
                c["n_s05"] += 1

        if not s0_passed or not s05_passed:
            self._handle_s0_s05_failure(
                s0_passed=s0_passed,
                s05_passed=s05_passed,
                sandbox_result=sandbox_result,
                results=results,
                config=config,
                nb=nb,
                exp_id=exp_id,
                graph=graph,
                fp=fp,
                i=i,
            )
            return None

        return {
            "model": model,
            "sandbox_result": sandbox_result,
            "layer_graphs": layer_graphs,
            "use_progressive": _use_progressive,
            "phase1_vocab": _phase1_vocab,
        }

    def _screen_candidate_pipeline(
        self,
        i: int,
        graph,
        model,
        config: RunConfig,
        nb: LabNotebook,
        exp_id: str,
        fp: str,
        dev_str: str,
        results: Dict,
        program_metrics: Dict[str, Any],
        sandbox_result,
        layer_graphs: List,
        _use_progressive: bool,
        _phase1_vocab: int,
        stage1_config: RunConfig,
        stage09_enabled: bool,
        orchestrator,
        candidate_batch_size: int,
    ) -> bool:
        """Run S0.75 through S1 queue submission for a single candidate.

        Returns True if the candidate should be skipped (continue in the outer loop),
        False if it was successfully queued for S1.
        """
        # S0.75: Initial-loss pre-screen (5 gradient steps)
        # Architectures with init_loss > 50 have deep unscaled
        # projection chains and waste rapid-screen + S1 budget.
        try:
            _s075_dev = torch.device(dev_str)
            model.train()
            _s075_opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            _s075_ids = torch.randint(0, _phase1_vocab, (4, 64), device=_s075_dev)
            with torch.amp.autocast(
                device_type=_s075_dev.type,
                dtype=torch.bfloat16,
                enabled=(_s075_dev.type == "cuda"),
            ):
                _s075_logits = model(_s075_ids)
                _s075_loss = torch.nn.functional.cross_entropy(
                    _s075_logits[:, :-1].reshape(-1, _s075_logits.size(-1)),
                    _s075_ids[:, 1:].reshape(-1),
                )
            _s075_init_loss = _s075_loss.item()
            program_metrics["s075_initial_loss"] = _s075_init_loss

            if (
                not math.isnan(_s075_init_loss)
                and not math.isinf(_s075_init_loss)
                and _s075_init_loss > INITIAL_LOSS_THRESHOLD
            ):
                results["funnel_counts"]["dropped_s075_high_init"] += 1
                self._emit_event(
                    "program_evaluated",
                    {
                        "index": i,
                        "fingerprint": fp[:10],
                        "result": "fail_s075",
                        "initial_loss": round(_s075_init_loss, 1),
                        "threshold": INITIAL_LOSS_THRESHOLD,
                    },
                )
                if config.persist_screening_failures:
                    _record_screening_failure(
                        nb=nb,
                        exp_id=exp_id,
                        graph=graph,
                        stage0_passed=True,
                        stage05_passed=True,
                        error_type="high_initial_loss",
                        error_message=(
                            f"initial_loss={_s075_init_loss:.4f} > "
                            f"{INITIAL_LOSS_THRESHOLD:.4f}"
                        ),
                        stage_at_death="stage075",
                        stability_score=sandbox_result.stability_score,
                    )
                del _s075_opt
                return True

            # Clean up probe state before rapid screening
            _s075_opt.zero_grad(set_to_none=True)
            del _s075_opt
        except Exception as s075_err:
            logger.warning(
                "S0.75 probe failed for graph %d, skipping check: %s",
                i,
                s075_err,
            )

        # Rapid Screening: 150-step gradient health check
        # Catches exploding grads, NaN, stalled loss, routing collapse
        # BEFORE committing to full Stage 1 training budget.
        from ...eval.screening_rapid import RapidScreeningCheck

        rapid = RapidScreeningCheck()
        results["funnel_counts"]["rapid_screen_attempted"] += 1
        rapid_result = rapid.run(
            model,
            vocab_size=_phase1_vocab,
            seq_len=min(128, config.max_seq_len),
            batch_size=2,
            device=dev_str,
        )
        program_metrics["rapid_screening_passed"] = rapid_result.passed
        program_metrics["rapid_screening_elapsed_ms"] = rapid_result.elapsed_ms
        program_metrics["rapid_screening_steps_completed"] = rapid_result.metrics.get(
            "steps_completed"
        )
        program_metrics["rapid_screening_max_steps"] = rapid.max_steps
        program_metrics["rapid_screening_gpu_minutes_saved"] = (
            rapid_result.gpu_minutes_saved
        )
        program_metrics["rapid_screening_metrics"] = rapid_result.metrics
        # Extract screening loss checkpoints for failure diagnostics
        _rm = rapid_result.metrics
        for _step, _col in (
            (10, "screening_loss_10"),
            (25, "screening_loss_25"),
            (50, "screening_loss_50"),
        ):
            _key = f"loss_at_{_step}"
            if _key not in _rm and len(_rm.get("losses", [])) >= _step:
                _rm[_key] = _rm["losses"][_step - 1]
            if _rm.get(_key) is not None:
                program_metrics[_col] = _rm[_key]
        # Extract grad norms even for failures
        if _rm.get("max_grad_norm") is not None:
            program_metrics.setdefault("max_grad_norm", _rm["max_grad_norm"])
            program_metrics.setdefault("mean_grad_norm", _rm["mean_grad_norm"])
        if rapid_result.degraded:
            program_metrics["rapid_screening_degraded"] = True
            program_metrics["rapid_screening_degraded_reasons"] = (
                rapid_result.degraded_reasons
            )
        if not rapid_result.passed:
            results["rapid_screening_killed"] += 1
            results["funnel_counts"]["dropped_rapid_screening"] += 1
            kr = rapid_result.kill_reason or "unknown"
            results["rapid_screening_kill_reasons"][kr] = (
                results["rapid_screening_kill_reasons"].get(kr, 0) + 1
            )
            program_metrics["rapid_screening_kill_reason"] = rapid_result.kill_reason
            program_metrics["rapid_screening_kill_step"] = rapid_result.kill_step
            program_metrics["rapid_screening_kill_metric"] = rapid_result.kill_metric
            self._emit_event(
                "program_evaluated",
                {
                    "index": i,
                    "fingerprint": fp[:10],
                    "result": "fail_rapid_screening",
                    "kill_reason": rapid_result.kill_reason,
                    "kill_step": rapid_result.kill_step,
                    "gpu_minutes_saved": rapid_result.gpu_minutes_saved,
                },
            )
            if config.persist_screening_failures:
                _record_screening_failure(
                    nb=nb,
                    exp_id=exp_id,
                    graph=graph,
                    stage0_passed=True,
                    stage05_passed=True,
                    error_type="rapid_screening_error",
                    error_message=(
                        rapid_result.kill_reason or "rapid_screening_failed"
                    )[:240],
                    stage_at_death="rapid_screening",
                    stability_score=sandbox_result.stability_score,
                    extra_metrics=program_metrics,
                )
            return True

        # Phase 2: recompile at real vocab for S1 training
        if _use_progressive:
            del model
            clear_gpu_memory()
            _compile_t0 = time.perf_counter()
            _prog_compiler = (
                _compile_model_legacy
                if getattr(config, "_capability_first_mode", False)
                else _compile_model_native
            )
            model = _prog_compiler(
                layer_graphs,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            )
            _compile_ms = (time.perf_counter() - _compile_t0) * 1000.0
            program_metrics["compile_time_ms"] = (
                float(program_metrics.get("compile_time_ms", 0.0) or 0.0) + _compile_ms
            )
            results.setdefault("_compile_times_ms", []).append(_compile_ms)
            program_metrics["progressive_phase2_compiled"] = True

        routing_ops = graph_routing_ops(graph)
        observed_routing_ops = graph_observed_routing_ops(graph)
        if routing_ops:
            program_metrics["routing_fast_lane_applied"] = 1
            try:
                from ...eval.wikitext_eval import screening_wikitext_eval

                _compile_t0 = time.perf_counter()
                _fl_compiler = (
                    _compile_model_legacy
                    if getattr(config, "_capability_first_mode", False)
                    else _compile_model_native
                )
                fast_lane_model = _fl_compiler(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
                _compile_ms = (time.perf_counter() - _compile_t0) * 1000.0
                program_metrics["compile_time_ms"] = (
                    float(program_metrics.get("compile_time_ms", 0.0) or 0.0)
                    + _compile_ms
                )
                results.setdefault("_compile_times_ms", []).append(_compile_ms)
                fast_lane = screening_wikitext_eval(
                    fast_lane_model,
                    config.vocab_size,
                    "cpu",
                    seq_len=min(96, config.max_seq_len),
                    n_train_steps=24,
                    n_train_batches=8,
                    n_eval_batches=3,
                    batch_size=3,
                )
                fast_lane["routing_fast_lane_applied"] = 1
                fast_lane["routing_fast_lane_status"] = fast_lane.get(
                    "screening_wikitext_status"
                )
                fast_lane["routing_fast_lane_metric_version"] = "routing_fast_lane_v1"
                fast_lane["routing_fast_lane_perplexity"] = fast_lane.get(
                    "wikitext_perplexity"
                )
                fast_lane["routing_fast_lane_score"] = fast_lane.get("wikitext_score")
                fast_lane["routing_fast_lane_pre_perplexity"] = fast_lane.get(
                    "wikitext_pre_perplexity"
                )
                fast_lane["routing_fast_lane_ppl_improvement"] = fast_lane.get(
                    "wikitext_ppl_improvement"
                )
                fast_lane["routing_fast_lane_elapsed_ms"] = fast_lane.get("elapsed_ms")
                fast_lane["routing_fast_lane_budget"] = dict(
                    fast_lane.get("screening_wikitext_budget") or {}
                )
                fast_lane["routing_fast_lane_slope"] = fast_lane.get("screening_slope")
                fast_lane["routing_fast_lane_slope_consistent"] = fast_lane.get(
                    "screening_slope_consistent"
                )
                fast_lane["routing_fast_lane_routing_ops"] = routing_ops
                fast_lane["routing_observed_ops"] = observed_routing_ops
                program_metrics.update(routing_fast_lane_fields(fast_lane))
                logger.info(
                    "    Routing fast lane trigger_ops=%s observed_ops=%s score=%.3f status=%s (%.0fms)%s",
                    ",".join(routing_ops),
                    ",".join(observed_routing_ops) if observed_routing_ops else "-",
                    float(fast_lane.get("routing_fast_lane_score") or 0.0),
                    fast_lane.get("routing_fast_lane_status") or "unknown",
                    float(fast_lane.get("routing_fast_lane_elapsed_ms") or 0.0),
                    (
                        f" error={fast_lane.get('error')}"
                        if fast_lane.get("routing_fast_lane_status") != "ok"
                        and fast_lane.get("error")
                        else ""
                    ),
                )
                # HellaSwag fast lane probe (25 examples)
                try:
                    from ...eval.hellaswag_eval import screening_hellaswag_eval

                    hs_fl = screening_hellaswag_eval(
                        fast_lane_model,
                        config.vocab_size,
                        "cpu",
                        n_examples=25,
                    )
                    program_metrics["routing_fast_lane_hellaswag_acc"] = hs_fl.get(
                        "hellaswag_acc"
                    )
                except Exception as e_hs_fl:
                    logger.debug("Fast lane HellaSwag skipped: %s", e_hs_fl)

                del fast_lane_model
            except Exception as e_fast_lane:
                logger.warning("Routing fast lane failed: %s", e_fast_lane)
                program_metrics.setdefault("routing_fast_lane_applied", 1)
                program_metrics["routing_fast_lane_status"] = "eval_failed"
                program_metrics["routing_fast_lane_metric_version"] = (
                    "routing_fast_lane_v1"
                )
                program_metrics["routing_fast_lane_routing_ops_json"] = fast_dumps(
                    routing_ops
                )

        # Stage 1: Asynchronous Execution (Z6)
        self._update_progress(current_stage="queuing_s1")

        screening_seed = self._stable_seed(exp_id, i, "screening")
        orchestrator.submit(
            index=i,
            graph=graph,
            config=stage1_config,
            seed=screening_seed,
            payload={
                "metrics": program_metrics,
                "graph": graph,
                "batch_id": i // candidate_batch_size,
                "queue_kind": "candidate_screening",
                "screening_stage": "stage09" if stage09_enabled else "stage1",
                "screening_seed": screening_seed,
            },
            model=model,  # Reuse compiled model (at real vocab)
        )
        results["funnel_counts"]["stage1_queued"] += 1
        return False

    def _finalize_experiment_results(
        self,
        exp_id: str,
        orchestrator,
        nb: LabNotebook,
        results: Dict,
        config: RunConfig,
        t_start: float,
        _s0_op_counts: Dict[str, Dict[str, int]],
    ) -> None:
        """Drain orchestrator queue and log final experiment metrics."""
        # Wait for remaining asynchronous Stage 1 evaluations
        self._update_progress(status="finalizing_evaluations")

        while (
            orchestrator.job_queue.unfinished_tasks > 0
            or not orchestrator.result_queue.empty()
        ):
            if self._stop_event.is_set():
                break
            self._process_orchestrator_results(
                orchestrator, nb, exp_id, results, config
            )
            time.sleep(0.5)

        queue_telemetry = orchestrator.get_telemetry()
        orchestrator.shutdown()
        results["queue_telemetry"] = queue_telemetry
        results["elapsed_seconds"] = time.time() - t_start
        results["perf_report"] = self._build_experiment_perf_report(
            results, queue_telemetry=queue_telemetry
        )
        results["perf_budget_gate"] = evaluate_perf_budget_gate(results["perf_report"])
        results.pop("_perf_traces", None)
        results.pop("_gpu_starvation", None)
        results.pop("_kernel_timing", None)
        if _s0_op_counts:
            results["_s0_op_counts"] = _s0_op_counts

        elapsed = results.get("elapsed_seconds", time.time() - t_start)
        self._update_progress(
            elapsed_seconds=elapsed,
            status="analyzing",
            aria_message=self.aria.begin_analysis(),
        )

        best = results.get("best_loss_ratio")
        best_str = f", best loss={best:.4f}" if best else ""
        dedup_str = ""
        if results.get("skipped_dedup", 0) > 0:
            dedup_str = f", dedup={results['skipped_dedup']} ({results.get('dedup_rate', 0) * 100:.0f}%)"
        rapid_killed = results.get("rapid_screening_killed", 0)
        rapid_str = f", rapid_killed={rapid_killed}" if rapid_killed else ""
        stage09 = results.get("stage09_passed", 0)
        stage09_str = f", S0.9={stage09}" if stage09 else ""
        logger.info(
            "Experiment %s complete: %d programs → S0=%d → S0.5=%d → S1=%d "
            "(%.1fs)%s%s%s%s%s",
            exp_id[:8],
            results["total"],
            results["stage0_passed"],
            results["stage05_passed"],
            results["stage1_passed"],
            elapsed,
            best_str,
            dedup_str,
            rapid_str,
            stage09_str,
            f", native_gating={results.get('skipped_proactive_gating', 0)}"
            if results.get("skipped_proactive_gating")
            else "",
        )

        self._live_training_context = None

    def _execute_experiment(
        self,
        exp_id: str,
        config: RunConfig,
        nb: LabNotebook,
        use_learned_grammar: bool = True,
    ) -> Dict:
        """Core experiment logic shared by single and continuous modes."""
        self._live_training_context = {"exp_id": exp_id, "phase": "synthesis"}
        stage09_enabled = bool(
            getattr(config, "enable_stage09_cheap_train_gate", False)
        )
        stage1_config = (
            _make_stage1_screening_config(config) if stage09_enabled else config
        )
        with self._lock:
            # Z17: Explicitly reset progress object at start of execution
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=config.n_programs,
                aria_message=f"{self.aria.NAME}: Initializing experiment {exp_id[:8]}...",
            )

        results = _make_experiment_results()

        # Phase 1: Grammar preparation
        grammar, failure_blocklist, analytics = self._prepare_grammar_config(
            exp_id, config, nb, results, use_learned_grammar=use_learned_grammar
        )

        # Phase 2: Candidate generation & filtering
        (
            graphs,
            _judgment_scores,
            dev,
            dev_str,
            orchestrator,
            candidate_batch_size,
            t_start,
        ) = self._generate_and_filter_candidates(
            exp_id,
            config,
            nb,
            grammar,
            analytics,
            results,
            use_learned_grammar=use_learned_grammar,
        )

        # Early return for morphological box (already handled inside _generate_and_filter_candidates)
        if not graphs and config.model_source == "morphological_box":
            return results

        # Track ops from S0 failures for op_success_rates (not stored in DB)
        _s0_op_counts: Dict[str, Dict[str, int]] = {}  # op -> {n_used, n_s0, n_s05}

        # Pre-import outside graph loop to avoid per-node import overhead
        try:
            from ...synthesis.primitives import get_primitive as _get_primitive
        except ImportError:
            _get_primitive = None

        # Phase 3: Per-candidate screening loop
        for i, graph in enumerate(graphs):
            if self._stop_event.is_set():
                break

            results["funnel_counts"]["screening_considered"] += 1
            fp = graph.fingerprint()
            self._update_progress(
                current_program=i + 1,
                current_fingerprint=fp[:10],
                elapsed_seconds=time.time() - t_start,
            )

            # Gate checks: dedup, structural, toxic, proactive gating, validation
            gate_skip = self._screen_candidate_gates(
                graph=graph,
                fp=fp,
                failure_blocklist=failure_blocklist,
                config=config,
                results=results,
                nb=nb,
                _judgment_scores=_judgment_scores,
                _get_primitive=_get_primitive,
                i=i,
            )
            if gate_skip is not None:
                # gate_skip is (reason, program_metrics_or_None)
                continue

            # Collect metrics and estimate FLOPs (populated by gates on pass-through)
            program_metrics = self._last_gate_program_metrics
            graph_analysis = self._last_gate_graph_analysis

            # Compile, S0/S0.5 eval, op counts
            try:
                compile_result = self._screen_candidate_compile_eval(
                    graph=graph,
                    graph_analysis=graph_analysis,
                    config=config,
                    dev_str=dev_str,
                    results=results,
                    nb=nb,
                    exp_id=exp_id,
                    fp=fp,
                    program_metrics=program_metrics,
                    _s0_op_counts=_s0_op_counts,
                    i=i,
                )
                if compile_result is None:
                    continue

                # S0.75 through S1 queue submission
                skip = self._screen_candidate_pipeline(
                    i=i,
                    graph=graph,
                    model=compile_result["model"],
                    config=config,
                    nb=nb,
                    exp_id=exp_id,
                    fp=fp,
                    dev_str=dev_str,
                    results=results,
                    program_metrics=program_metrics,
                    sandbox_result=compile_result["sandbox_result"],
                    layer_graphs=compile_result["layer_graphs"],
                    _use_progressive=compile_result["use_progressive"],
                    _phase1_vocab=compile_result["phase1_vocab"],
                    stage1_config=stage1_config,
                    stage09_enabled=stage09_enabled,
                    orchestrator=orchestrator,
                    candidate_batch_size=candidate_batch_size,
                )
                if skip:
                    continue

            except Exception as e:
                logger.error("Error evaluating graph %d: %s", i, e)
                results["funnel_counts"]["dropped_runtime_error"] += 1
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
                            logger.info(
                                "CUDA context recovered after fatal error on graph %d",
                                i,
                            )
                        except (RuntimeError, OSError) as e_cuda:
                            logger.warning(
                                "CUDA context unrecoverable after fatal error on graph %d (reason: %s)",
                                i,
                                e_cuda,
                            )
                continue

            # Periodically process available results to keep the dashboard updated
            self._process_orchestrator_results(
                orchestrator, nb, exp_id, results, config
            )

        # Phase 4: Finalization
        self._finalize_experiment_results(
            exp_id, orchestrator, nb, results, config, t_start, _s0_op_counts
        )
        return results
