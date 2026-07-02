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
import random
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

from ..json_utils import fast_dumps, json_safe
from ._lifecycle import _LifecycleMixin


from ...synthesis.grammar import GrammarConfig, batch_generate
from ..refinement_scoring import rank_synthesis_candidates_by_stability
from .screening_candidate_rank import judgment_rerank
from .screening_signal_weights import (
    apply_insight_adjustments,
    build_signal_weight_maps,
)
from .failure_provenance import infer_graph_failure_provenance
from ..causal_attribution import causal_generation_adjustments
from ..construction_priors import (
    construction_prior_as_grammar_adjustments,
    get_active_construction_prior,
)
from ...meta_analysis.priors import (
    apply_meta_analysis_prior_to_grammar,
    load_latest_meta_analysis_prior,
)
from ..ml_influence_policy import component_is_allowed
from ..notebook import LabNotebook, ExperimentEntry
from ...synthesis.grammar_support import ROUTING_COMPRESSION_MOE_OPS

import logging

logger = logging.getLogger(__name__)

# Gate 5 constant: routing/MoE/sparse/compression ops required for efficiency scoring.
# Module-level to avoid re-instantiation per graph in the screening loop.
# UNIONED with the grammar's ROUTING_COMPRESSION_MOE_OPS so newly registered
# routing-class ops are never invisible to gate 5 — the 2026-07-02 overnight
# campaign measured gate5 killing graphs whose only routing op was a freshly
# wired novel mechanism (the hand-list had drifted from the registry).
_EFFICIENCY_OPS = ROUTING_COMPRESSION_MOE_OPS | frozenset(
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
    source_result_id: str | None = None,
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
        patch_kwargs = dict(
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
        if source_result_id:
            nb.merge_program_result_patch(
                result_id=source_result_id,
                graph_fingerprint=graph.fingerprint(),
                graph_json=json.dumps(graph.to_dict(), separators=(",", ":")),
                relabel_backfill_if_orphan=True,
                **patch_kwargs,
            )
            return
        nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=graph.fingerprint(),
            graph_json=json.dumps(graph.to_dict(), separators=(",", ":")),
            **patch_kwargs,
        )
    except Exception as exc:
        logger.debug("Failed to persist screening failure for %s: %s", exp_id, exc)


from ._types import RunConfig, LiveProgress
from ._helpers import (
    screening_probe_fields,
    screening_wikitext_fields,
)
from .execution_screening_graphs import (
    analyze_graph_for_screening as analyze_graph_for_screening,
    structural_gate_failure as structural_gate_failure,
    toxic_failure_ratio as toxic_failure_ratio,
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
    _publish_terminal_event = _LifecycleMixin._publish_terminal_event
    _publish_screening_terminal_event = _LifecycleMixin._publish_terminal_event
    _fail_experiment_compat = _LifecycleMixin._fail_experiment_compat
    _complete_experiment_compat = _LifecycleMixin._complete_experiment_compat
    _log_learning_event_compat = _LifecycleMixin._log_learning_event_compat

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

            self._publish_terminal_event(
                producer="runner.execution_screening",
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
            try:
                self._maybe_run_causal_ablation_loop(nb, exp_id, config, results)
            except Exception as exc:
                logger.warning("Causal ablation loop failed for %s: %s", exp_id, exc)
            # Auto-escalation pipeline (investigation/validation)
            results["experiment_id"] = exp_id
            self._auto_escalate(results, config, nb, phase="screening")

            # Auto-scale-up if criteria met (legacy, kept for backward compat)
            self._maybe_auto_scale_up(results, config, nb)

            # Auto-report for single experiments
            self._maybe_auto_report(config, nb, reason="experiment_complete")

            self._update_progress(
                status="completed",
                aria_message=(
                    summary.split("\n")[-1] if summary else "Experiment complete."
                ),
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
            self._publish_terminal_event(
                producer="runner.execution_screening",
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
                self._publish_terminal_event(
                    producer="runner.execution_screening",
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

    def _load_experiment_analytics(self, exp_id: str, nb: LabNotebook) -> Any:
        try:
            from ..analytics import ExperimentAnalytics

            return ExperimentAnalytics(nb)
        except Exception as e:
            logger.debug("Experiment analytics unavailable for %s: %s", exp_id, e)
            return None

    def _compute_gated_grammar_weights(
        self,
        *,
        exp_id: str,
        config: RunConfig,
        nb: LabNotebook,
        analytics: Any,
        use_learned_grammar: bool,
        learned_grammar_allowed: bool,
    ) -> tuple[Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
        if not use_learned_grammar:
            return None, None
        if not learned_grammar_allowed:
            logger.info(
                "Learned grammar weights requested but blocked by ML trust policy"
            )
            return None, None
        if analytics is None:
            logger.info(
                "Learned grammar weights requested for %s but analytics backend was unavailable",
                exp_id,
            )
            return None, None
        try:
            last_effective = nb.load_last_effective_weights()
            last_weights = last_effective[0] if last_effective else None
            grammar_weights = analytics.compute_grammar_weights(
                last_applied=last_weights,
                alpha=0.6,
            )
            if not grammar_weights:
                return grammar_weights, None
            grammar_gate = self._evaluate_grammar_update_gate(
                nb=nb,
                analytics=analytics,
                config=config,
            )
            if grammar_gate.get("gate_pass"):
                return grammar_weights, grammar_gate
            self._log_learning_event_compat(
                nb,
                "grammar_weights_blocked",
                f"Blocked grammar weight update for {exp_id}: weak attribution evidence",
                evidence=fast_dumps(json_safe(grammar_gate), safe=True),
            )
            return None, grammar_gate
        except Exception as e:
            logger.warning(
                "Failed computing learned grammar weights for %s: %s", exp_id, e
            )
            return None, None

    @staticmethod
    def _penalized_op_weight(op_info: Dict[str, Any], rehab: Any) -> float:
        if rehab and rehab.get("compile_passed") and rehab.get("forward_passed"):
            return 0.5
        if op_info.get("failure_stage") == "compilation":
            return 0.15
        return 0.1

    def _apply_negative_op_penalties(
        self,
        *,
        exp_id: str,
        nb: LabNotebook,
        analytics: Any,
        op_weights: Dict[str, float],
    ) -> None:
        try:
            rehab_cache = nb.get_op_rehabilitation_cache()
            if analytics is None:
                return
            neg = analytics.negative_results_synthesis()
            for op_info in neg.get("failed_ops", []):
                if (
                    op_info.get("s1_rate", 1) == 0
                    and op_info.get("n_used", 0) >= 5
                    and op_info.get("confidence", 0) >= 0.7
                ):
                    op_name = op_info["op_name"]
                    penalty = self._penalized_op_weight(
                        op_info, rehab_cache.get(op_name)
                    )
                    op_weights[op_name] = min(
                        float(op_weights.get(op_name, 1.0)), penalty
                    )
            for op_info in neg.get("weak_ops", []):
                op_name = op_info.get("op_name", "")
                if op_name:
                    op_weights[op_name] = min(
                        float(op_weights.get(op_name, 1.0)),
                        float(op_info.get("penalty_weight", 1.0)),
                    )
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

    def _load_failure_blocklist(
        self, *, exp_id: str, nb: LabNotebook
    ) -> Dict[str, float]:
        try:
            failure_blocklist = nb.get_failure_signature_blocklist()
            if failure_blocklist:
                self._log_learning_event_compat(
                    nb,
                    "failure_signatures_loaded",
                    f"Loaded {len(failure_blocklist)} toxic op-pair patterns",
                    signatures=sorted(failure_blocklist.keys())[:10],
                )
            return failure_blocklist
        except Exception as e:
            logger.warning("Failed loading failure signatures for %s: %s", exp_id, e)
            return {}

    @staticmethod
    def _reliable_recent_op(
        op_rates: Dict[str, Any],
        op_name: str,
        min_used: int = 10,
        min_s1: float = 0.25,
    ) -> bool:
        info = op_rates.get(op_name) or {}
        return (
            int(info.get("n_used") or 0) >= min_used
            and float(info.get("s1_rate") or 0.0) >= min_s1
        )

    def _compute_champion_bias(
        self, *, exp_id: str, analytics: Any
    ) -> Dict[str, float]:
        champion_bias: Dict[str, float] = {}
        try:
            if analytics is None:
                return champion_bias
            op_rates = analytics.op_success_rates(since_ts=time.time() - 604800) or {}
            if not op_rates:
                return champion_bias
            winning_ops = {"exp", "selective_scan", "tropical_center"}
            projection_ops = {"low_rank_proj", "shared_basis_proj", "tied_proj"}
            sparse_ops = {
                "nm_sparse_linear",
                "block_sparse_linear",
                "semi_structured_2_4_linear",
            }
            if any(self._reliable_recent_op(op_rates, op) for op in winning_ops):
                champion_bias["structural"] = max(
                    champion_bias.get("structural", 1.0), 1.2
                )
                champion_bias["sequence"] = max(champion_bias.get("sequence", 1.0), 1.2)
            if any(self._reliable_recent_op(op_rates, op) for op in projection_ops):
                champion_bias["parameterized"] = max(
                    champion_bias.get("parameterized", 1.0),
                    1.4,
                )
            if any(self._reliable_recent_op(op_rates, op) for op in sparse_ops):
                champion_bias["parameterized"] = max(
                    champion_bias.get("parameterized", 1.0),
                    1.5,
                )
                champion_bias["_structured_sparsity_bias"] = 0.8
        except Exception as e:
            logger.warning("Failed computing champion bias for %s: %s", exp_id, e)
        return champion_bias

    @staticmethod
    def _merge_slot_motif_multipliers(
        target: Dict[str, Dict[str, float]],
        source: Dict[str, Any],
    ) -> None:
        for slot_key, weights in source.items():
            merged = dict(target.get(str(slot_key), {}))
            for motif_name, weight in (weights or {}).items():
                current = float(merged.get(str(motif_name), 1.0))
                w = float(weight)
                merged[str(motif_name)] = (
                    max(current, w) if w >= 1.0 else min(current, w)
                )
            target[str(slot_key)] = merged

    @staticmethod
    def _merge_slot_motif_denylist(
        target: Dict[str, frozenset[str]],
        source: Dict[str, Any],
    ) -> None:
        for slot_key, denied in source.items():
            existing = set(target.get(str(slot_key), frozenset()))
            existing.update(str(name) for name in (denied or []) if str(name).strip())
            if existing:
                target[str(slot_key)] = frozenset(existing)

    def _apply_observability_priors(
        self,
        *,
        nb: LabNotebook,
        template_weights: Dict[str, float],
        motif_weights: Dict[str, float],
        slot_motif_multipliers: Dict[str, Dict[str, float]],
        slot_motif_denylist: Dict[str, frozenset[str]],
    ) -> None:
        observability_priors = nb.get_generation_observability_priors(
            max_rows=48,
            min_support=4,
        )
        for name, weight in (
            observability_priors.get("template_weights") or {}
        ).items():
            template_weights[name] = max(
                float(template_weights.get(name, 1.0)), float(weight)
            )
        for name, weight in (observability_priors.get("motif_weights") or {}).items():
            motif_weights[name] = max(
                float(motif_weights.get(name, 1.0)), float(weight)
            )
        for slot_key, weights in (
            observability_priors.get("slot_multipliers") or {}
        ).items():
            slot_motif_multipliers[str(slot_key)] = {
                str(name): float(weight) for name, weight in (weights or {}).items()
            }
        self._merge_slot_motif_denylist(
            slot_motif_denylist,
            observability_priors.get("slot_denylist") or {},
        )

    def _apply_generation_priors(
        self,
        *,
        nb: LabNotebook,
        op_weights: Dict[str, float],
        slot_motif_multipliers: Dict[str, Dict[str, float]],
        slot_motif_denylist: Dict[str, frozenset[str]],
    ) -> None:
        causal_adjustments = causal_generation_adjustments(nb)
        active_prior = get_active_construction_prior(nb)
        prior_adjustments = construction_prior_as_grammar_adjustments(active_prior)
        for layer in (causal_adjustments, prior_adjustments):
            for op_name, weight in (layer.get("op_weights") or {}).items():
                op_weights[op_name] = (
                    float(op_weights.get(op_name, 1.0)) * float(weight)
                ) ** 0.5
            self._merge_slot_motif_multipliers(
                slot_motif_multipliers,
                layer.get("slot_motif_multipliers") or {},
            )
            self._merge_slot_motif_denylist(
                slot_motif_denylist,
                layer.get("slot_motif_denylist") or {},
            )
        if prior_adjustments.get("version"):
            logger.info(
                "Applied construction prior snapshot %s (op_weights=%d slot_motifs=%d denylist=%d)",
                prior_adjustments.get("version"),
                len(prior_adjustments.get("op_weights") or {}),
                sum(
                    len(v)
                    for v in (
                        prior_adjustments.get("slot_motif_multipliers") or {}
                    ).values()
                ),
                sum(
                    len(v)
                    for v in (
                        prior_adjustments.get("slot_motif_denylist") or {}
                    ).values()
                ),
            )

    def _load_screening_signal_weights(
        self,
        *,
        nb: LabNotebook,
        screening_signal_allowed: bool,
        op_weights: Dict[str, float],
        template_weights: Dict[str, float],
        motif_weights: Dict[str, float],
        slot_motif_multipliers: Dict[str, Dict[str, float]],
        slot_motif_denylist: Dict[str, frozenset[str]],
    ) -> None:
        if not screening_signal_allowed:
            logger.info("Screening signal weight maps disabled or blocked for this run")
            return
        try:
            signal_template_weights, signal_motif_weights = build_signal_weight_maps(nb)
            template_weights.update(signal_template_weights)
            motif_weights.update(signal_motif_weights)
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Failed building signal weight maps: %s", e)
        try:
            self._apply_observability_priors(
                nb=nb,
                template_weights=template_weights,
                motif_weights=motif_weights,
                slot_motif_multipliers=slot_motif_multipliers,
                slot_motif_denylist=slot_motif_denylist,
            )
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Failed building observability priors: %s", e)
        try:
            self._apply_generation_priors(
                nb=nb,
                op_weights=op_weights,
                slot_motif_multipliers=slot_motif_multipliers,
                slot_motif_denylist=slot_motif_denylist,
            )
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Failed applying causal generation priors: %s", e)

    @staticmethod
    def _apply_learned_signal_weights(
        *,
        analytics: Any,
        screening_signal_allowed: bool,
        op_weights: Dict[str, float],
        template_weights: Dict[str, float],
        motif_weights: Dict[str, float],
    ) -> None:
        if analytics is None or not screening_signal_allowed:
            return
        window_cutoff = time.time() - 604800
        try:
            op_weights.update(analytics.compute_op_weights(since_ts=window_cutoff))
        except (TypeError, ValueError, KeyError) as e:
            logger.debug("Failed computing learned op weights: %s", e)
        try:
            learned_tpl_weights, learned_motif_weights = (
                analytics.compute_template_and_motif_weights(since_ts=window_cutoff)
            )
            template_weights.update(learned_tpl_weights or {})
            motif_weights.update(learned_motif_weights or {})
        except (TypeError, ValueError, KeyError) as e:
            logger.debug("Failed computing template/motif weights: %s", e)
        try:
            syn_motif_boosts, syn_tpl_boosts = analytics.compute_synergy_boosts()
            for name, boost in syn_motif_boosts.items():
                motif_weights[name] = motif_weights.get(name, 1.0) * boost
            for name, boost in syn_tpl_boosts.items():
                template_weights[name] = template_weights.get(name, 1.0) * boost
        except (TypeError, ValueError, KeyError) as e:
            logger.debug("Failed computing synergy boosts: %s", e)

    def _build_weighted_grammar(
        self,
        *,
        config: RunConfig,
        nb: LabNotebook,
        screening_signal_allowed: bool,
        op_weights: Dict[str, float],
        template_weights: Dict[str, float],
        motif_weights: Dict[str, float],
        slot_motif_multipliers: Dict[str, Dict[str, float]],
        slot_motif_denylist: Dict[str, frozenset[str]],
    ) -> GrammarConfig:
        op_weights = {**op_weights, **self._op_weights_overrides}
        grammar = self._build_grammar_config(config, op_weights=op_weights)
        if not getattr(config, "template_weights", None):
            if grammar.routing_mandatory:
                for key, value in template_weights.items():
                    grammar.template_weights.setdefault(key, value)
            else:
                grammar.template_weights = template_weights
        grammar.motif_weights = motif_weights
        grammar.slot_motif_weight_multipliers = slot_motif_multipliers
        grammar.slot_motif_denylist = slot_motif_denylist
        if screening_signal_allowed:
            try:
                apply_insight_adjustments(
                    nb,
                    grammar,
                    grammar.template_weights,
                    grammar.motif_weights,
                )
            except Exception as e:
                logger.debug("Insight grammar adjustment failed: %s", e)
        return grammar

    def _apply_gated_category_weights(
        self,
        *,
        exp_id: str,
        nb: LabNotebook,
        config: RunConfig,
        results: Dict,
        grammar: GrammarConfig,
        grammar_weights: Optional[Dict[str, float]],
        grammar_gate: Optional[Dict[str, Any]],
        analytics: Any,
    ) -> None:
        if not grammar_weights:
            return
        old_weights = dict(grammar.category_weights)
        grammar.category_weights.update(grammar_weights)
        n_changed = sum(
            1 for key, value in grammar_weights.items() if old_weights.get(key) != value
        )
        self._log_grammar_weight_application(
            nb,
            exp_id,
            old_weights,
            dict(grammar.category_weights),
            analytics=analytics,
        )
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

    def _apply_champion_bias_to_grammar(
        self,
        *,
        exp_id: str,
        nb: LabNotebook,
        results: Dict,
        grammar: GrammarConfig,
        champion_bias: Dict[str, float],
    ) -> None:
        if not champion_bias:
            return
        before_bias = dict(grammar.category_weights)
        for category, multiplier in champion_bias.items():
            if category == "_structured_sparsity_bias":
                grammar.structured_sparsity_bias = float(multiplier)
                continue
            base = float(grammar.category_weights.get(category, 1.0))
            grammar.category_weights[category] = round(
                max(0.5, min(8.0, base * multiplier)),
                2,
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

    def _apply_chat_or_default_weights(
        self,
        *,
        exp_id: str,
        nb: LabNotebook,
        config: RunConfig,
        results: Dict,
        grammar: GrammarConfig,
    ) -> None:
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

    @staticmethod
    def _apply_efficiency_bias(config: RunConfig, grammar: GrammarConfig) -> None:
        eff_weight = getattr(config, "selection_efficiency_weight", 0.25)
        if eff_weight < 0.3:
            return
        eff_boost = min(1.0 + eff_weight, 2.0)
        for category in ("structural", "parameterized"):
            base = float(grammar.category_weights.get(category, 1.0))
            grammar.category_weights[category] = round(min(8.0, base * eff_boost), 2)
        for op_name in (
            "moe_2expert",
            "moe_topk",
            "block_sparse_linear",
            "bottleneck_proj",
            "linear_proj_down",
            "selective_scan",
        ):
            grammar.op_weights[op_name] = (
                grammar.op_weights.get(op_name, 1.0) * eff_boost
            )

    @staticmethod
    def _apply_hierarchy_fitness(analytics: Any, grammar: GrammarConfig) -> None:
        if analytics is None:
            return
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

    def _apply_meta_analysis_priors(
        self,
        *,
        exp_id: str,
        nb: LabNotebook,
        config: RunConfig,
        results: Dict,
        grammar: GrammarConfig,
    ) -> None:
        if not getattr(config, "use_meta_analysis_priors", False):
            return
        try:
            prior_target = str(
                getattr(config, "meta_analysis_prior_target", "balanced") or "balanced"
            )
            prior_path = str(
                getattr(
                    config,
                    "meta_analysis_prior_path",
                    "research/artifacts/meta_analysis_priors",
                )
                or "research/artifacts/meta_analysis_priors"
            )
            prior = load_latest_meta_analysis_prior(prior_path, target=prior_target)
            if prior is None:
                logger.warning(
                    "Meta-analysis priors enabled for %s but no prior artifact was found in %s",
                    exp_id,
                    prior_path,
                )
                return
            counts = apply_meta_analysis_prior_to_grammar(grammar, prior)
            if config.category_weights:
                grammar.category_weights.update(config.category_weights)
            if config.op_weights:
                grammar.op_weights.update(config.op_weights)
            if config.template_weights:
                grammar.template_weights.update(config.template_weights)
            prior_record = {
                "version": prior.get("version"),
                "target": prior.get("target"),
                "counts": counts,
            }
            results["meta_analysis_prior"] = prior_record
            results["applied_grammar_weights"] = dict(grammar.category_weights)
            self._log_learning_event_compat(
                nb,
                "meta_analysis_prior_applied",
                f"Applied meta-analysis prior {prior.get('version')} for {exp_id}",
                **prior_record,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
            logger.warning("Failed applying meta-analysis priors for %s: %s", exp_id, e)

    @staticmethod
    def _maybe_use_synthesized_training_choices(config: RunConfig) -> None:
        if random.random() < 0.2:
            config.loss_type = "synthesized"
        if random.random() < 0.2:
            config.optimizer_type = "synthesized"

    def _prepare_grammar_config(
        self,
        exp_id: str,
        config: RunConfig,
        nb: LabNotebook,
        results: Dict,
        use_learned_grammar: bool = True,
    ) -> Tuple[GrammarConfig, Dict[str, float], Any]:
        """Build grammar config with learned weights, champion bias, and efficiency tuning."""
        # Telemetry emitted by the delegated helper includes:
        # "max_depth": int(config.max_depth)
        # "max_ops": int(config.max_ops)
        # "Applied learned grammar weights ("
        return self._prepare_grammar_config_impl(
            exp_id,
            config,
            nb,
            results,
            use_learned_grammar=use_learned_grammar,
        )

    def _prepare_grammar_config_impl(
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
        op_weights: Dict[str, float] = {}
        template_weights: Dict[str, float] = {}
        motif_weights: Dict[str, float] = {}
        slot_motif_multipliers: Dict[str, Dict[str, float]] = {}
        slot_motif_denylist: Dict[str, frozenset[str]] = {}
        learned_grammar_allowed = component_is_allowed(
            "learned_grammar_weights", config
        )
        screening_signal_allowed = component_is_allowed(
            "screening_signal_weights", config
        )
        analytics = self._load_experiment_analytics(exp_id, nb)
        grammar_weights, grammar_gate = self._compute_gated_grammar_weights(
            exp_id=exp_id,
            config=config,
            nb=nb,
            analytics=analytics,
            use_learned_grammar=use_learned_grammar,
            learned_grammar_allowed=learned_grammar_allowed,
        )
        self._apply_negative_op_penalties(
            exp_id=exp_id,
            nb=nb,
            analytics=analytics,
            op_weights=op_weights,
        )
        failure_blocklist = self._load_failure_blocklist(exp_id=exp_id, nb=nb)
        champion_bias = self._compute_champion_bias(exp_id=exp_id, analytics=analytics)
        self._load_screening_signal_weights(
            nb=nb,
            screening_signal_allowed=screening_signal_allowed,
            op_weights=op_weights,
            template_weights=template_weights,
            motif_weights=motif_weights,
            slot_motif_multipliers=slot_motif_multipliers,
            slot_motif_denylist=slot_motif_denylist,
        )
        self._apply_learned_signal_weights(
            analytics=analytics,
            screening_signal_allowed=screening_signal_allowed,
            op_weights=op_weights,
            template_weights=template_weights,
            motif_weights=motif_weights,
        )
        grammar = self._build_weighted_grammar(
            config=config,
            nb=nb,
            screening_signal_allowed=screening_signal_allowed,
            op_weights=op_weights,
            template_weights=template_weights,
            motif_weights=motif_weights,
            slot_motif_multipliers=slot_motif_multipliers,
            slot_motif_denylist=slot_motif_denylist,
        )
        self._apply_gated_category_weights(
            exp_id=exp_id,
            nb=nb,
            config=config,
            results=results,
            grammar=grammar,
            grammar_weights=grammar_weights,
            grammar_gate=grammar_gate,
            analytics=analytics,
        )
        self._apply_champion_bias_to_grammar(
            exp_id=exp_id,
            nb=nb,
            results=results,
            grammar=grammar,
            champion_bias=champion_bias,
        )
        self._apply_chat_or_default_weights(
            exp_id=exp_id,
            nb=nb,
            config=config,
            results=results,
            grammar=grammar,
        )
        self._apply_efficiency_bias(config, grammar)
        self._apply_hierarchy_fitness(analytics, grammar)
        self._apply_meta_analysis_priors(
            exp_id=exp_id,
            nb=nb,
            config=config,
            results=results,
            grammar=grammar,
        )
        self._maybe_use_synthesized_training_choices(config)
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
                log_event=self._log_learning_event_compat,
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


# ── Gate + pipeline methods moved to split module ─────────────────
# The per-candidate gate/compile/eval/pipeline/finalize methods and
# `_execute_experiment` live in `execution_screening_pipeline.py` to
# keep both files under the 1250-line file cap. They are composed
# onto ExperimentRunner via `_ExecutionScreeningPipelineMixin`.
from .execution_screening_pipeline import (  # noqa: E402,F401
    _ExecutionScreeningPipelineMixin,
)
