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
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..native_runner import compile_model_native_first as compile_model
from ...synthesis.serializer import graph_from_json
from ...eval.metrics import novelty_score
from ...eval.fingerprint import compute_fingerprint
from ...eval.perf_budget import evaluate_perf_budget_gate
from ...training.training_program import synthesize_training_program, synthesize_training_program_batch
from ...training.checkpointing import CheckpointManager
from ..notebook import LabNotebook, ExperimentEntry
from ..evidence import (
    build_evidence_pack,
    validate_selection_decision_log,
)
from ..llm.context import (
    build_investigation_context,
    build_validation_context,
    build_mode_selection_context,
    build_hypothesis_context,
)
from ..shared_utils import resolve_device

import logging
logger = logging.getLogger(__name__)

from ._types import RunConfig, LiveProgress
from .continuous_inline_validation_phase7 import _ContinuousInlineValidationPhase7Mixin

class _ContinuousMixin(_ContinuousInlineValidationPhase7Mixin):
    """Continuous mode thread, mode selection, inline phases."""

    def _run_continuous_thread(self, config: RunConfig):
        """Execute continuous experiments in background."""
        n_experiments = 0
        t_start = time.time()
        self.aria.reset_cost_tracking()
        # Skip per-cycle LLM calls — use rule-based paths to save API costs.
        # LLM is still available for user-initiated chat and campaign formulation.
        self.aria._continuous_mode = True
        self.aria._llm_decision_interval = config.llm_decision_interval

        # Knowledge distiller — background intelligence thread
        distiller = None
        try:
            from ..intelligence.distiller import KnowledgeDistiller
            from ..intelligence.digest import ExperimentDigest
            db_path = self.notebook_path
            distiller = KnowledgeDistiller(
                db_path=db_path,
                distill_interval_cycles=3,
            )
            # Recover last digest from DB
            try:
                init_nb = self._make_notebook()
                saved = init_nb.get_latest_digest()
                if saved:
                    distiller.set_digest(ExperimentDigest.from_dict(saved))
                    logger.info("Recovered knowledge digest from DB")
                init_nb.close()
            except Exception as e:
                logger.debug("Digest recovery failed: %s", e)
            distiller.start()
            self._knowledge_distiller = distiller
        except Exception as e:
            logger.warning("KnowledgeDistiller init failed (degrading gracefully): %s", e)
            distiller = None
            self._knowledge_distiller = None
        self._set_aria_cycle_phase(
            "planning",
            continuous_active=True,
            cycle_index=0,
            selected_mode=None,
            note="Preparing continuous research loop.",
        )

        # Initialize checkpoint manager
        ckpt = CheckpointManager(config.checkpoint_dir)
        resume_id = config.resume_experiment_id

        # Resume from checkpoint if requested
        if resume_id:
            ckpt_state = ckpt.load_continuous(resume_id)
            if ckpt_state:
                n_experiments = ckpt_state.get("n_experiments", 0)
                elapsed_prior = ckpt_state.get("elapsed_seconds", 0.0)
                t_start = time.time() - elapsed_prior
                logger.info("Resuming continuous session from checkpoint: "
                            "n_experiments=%d, elapsed=%.0fs",
                            n_experiments, elapsed_prior)
                self._emit_event("checkpoint_resumed", {
                    "experiment_id": resume_id,
                    "n_experiments": n_experiments,
                    "elapsed_seconds": elapsed_prior,
                })

        # Clean up stale experiments from previous interrupted runs
        try:
            cleanup_nb = self._make_notebook()
            n_cleaned = cleanup_nb.cleanup_stale_experiments()
            if n_cleaned:
                logger.info(f"Cleaned up {n_cleaned} stale running experiments")
            cleanup_nb.close()
        except Exception as e:
            logger.debug(f"Stale experiment cleanup failed: {e}")

        # Initialize campaign
        try:
            init_nb = self._make_notebook()
            self._ensure_campaign(config, init_nb)
            init_nb.close()
        except Exception as e:
            logger.debug(f"Campaign init failed: {e}")

        while not self._stop_event.is_set():
            self._wait_for_cycle_resume(n_experiments)
            if self._stop_event.is_set():
                break

            # Check for pending heal retry
            if self._pending_heal_retry:
                retry = self._pending_heal_retry
                self._pending_heal_retry = None
                logger.info("Retrying after successful heal: %s", retry.get("scope", "")[:100])
                try:
                    retry_nb = self._make_notebook()
                    retry_nb.log_learning_event(
                        "heal_retry",
                        f"Retrying after heal: {retry['scope'][:200]}",
                    )
                    retry_nb.close()
                except Exception:
                    pass

            # Check limits before starting next experiment
            stop_reason = self._check_continuous_limits(
                config, t_start, n_experiments)
            if stop_reason:
                self.aria._continuous_mode = False
                self._end_of_session_automation(
                    config, reason=f"continuous_session_end ({stop_reason})")
                self._set_aria_cycle_phase(
                    "completed",
                    continuous_active=False,
                    cycle_index=n_experiments,
                    note=f"Session ended: {stop_reason}",
                )

                with self._lock:
                    self._progress.status = "completed"
                    self._progress.aria_message = f"Session ended: {stop_reason}"
                self._emit_event("continuous_limit_reached", {
                    "reason": stop_reason,
                    "experiments_completed": n_experiments,
                    "elapsed_minutes": (time.time() - t_start) / 60,
                    "estimated_cost": self.aria.total_cost,
                })
                # Stop knowledge distiller
                if distiller is not None:
                    try:
                        distiller.stop()
                    except Exception:
                        pass
                # Launch queued auto-scale-up
                self._run_pending_scale_up()
                return

            n_experiments += 1
            nb = self._make_notebook()
            try:
                self.run_aria_cycle(config, nb, n_experiments, t_start)
                
                # Periodic Gate Performance Summary (Task 9)
                if n_experiments % 5 == 0:
                    try:
                        from ..analytics import ExperimentAnalytics
                        analytics = ExperimentAnalytics(nb)
                        stats = analytics.gate_performance_summary()
                        if stats:
                            logger.info(
                                "Gate Performance (Cycle %d): pass_rate=%.2f, violations=%d, corr=%s (n=%d)",
                                n_experiments, 
                                stats.get("stage05_pass_rate", 0),
                                stats.get("causality_violations", 0),
                                f"{stats.get('discovery_validation_correlation'):.2f}" if stats.get("discovery_validation_correlation") is not None else "N/A",
                                stats.get("n_correlation_samples", 0)
                            )
                    except Exception as e:
                        logger.debug("Failed to generate gate performance summary: %s", e)
            finally:
                nb.close()

            # Notify distiller that a cycle completed
            if distiller is not None:
                try:
                    distiller.notify_cycle_complete()
                except Exception:
                    pass

            # Update cost in progress
            with self._lock:
                self._progress.estimated_cost = self.aria.total_cost
                self._progress.total_tokens = self.aria.total_tokens

            # Save checkpoint after every checkpoint_interval experiments
            if (config.checkpoint_interval > 0
                    and n_experiments % config.checkpoint_interval == 0):
                try:
                    ckpt_exp_id = resume_id or "continuous"
                    ckpt.save_continuous(
                        experiment_id=ckpt_exp_id,
                        config_dict=config.to_dict(),
                        n_experiments=n_experiments,
                        elapsed_seconds=time.time() - t_start,
                        extra_state={
                            "estimated_cost": self.aria.total_cost,
                            "total_tokens": self.aria.total_tokens,
                        },
                    )
                except Exception as e:
                    logger.debug("Checkpoint save failed: %s", e)

            # Purge empty failed experiments between cycles to prevent DB bloat.
            try:
                self.notebook.purge_empty_experiments()
                self.notebook.compact_old_chat()
                self.notebook.backfill_failure_signatures()
            except Exception:
                pass

            if config.rest_between_experiments > 0 and not self._stop_event.is_set():
                time.sleep(config.rest_between_experiments)

        # Stop knowledge distiller
        if distiller is not None:
            try:
                distiller.stop()
            except Exception:
                pass

        # Re-enable LLM for interactive use after continuous mode ends.
        self.aria._continuous_mode = False

        # Session ending (user stopped) — auto-report and auto-scale-up
        if n_experiments > 0:
            self._end_of_session_automation(
                config,
                reason=f"continuous_session_stopped (after {n_experiments} experiments)")

        with self._lock:
            elapsed_min = (time.time() - t_start) / 60
            cost_str = f" | Est. cost: ${self.aria.total_cost:.2f}" if self.aria.total_cost > 0 else ""
            self._progress.status = "completed" if not self._stop_event.is_set() else "stopped"
            self._progress.estimated_cost = self.aria.total_cost
            self._progress.total_tokens = self.aria.total_tokens
            self._progress.aria_message = (
                f"Stopped after {n_experiments} experiments ({elapsed_min:.0f}min{cost_str})."
            )
        self._set_aria_cycle_phase(
            "completed" if not self._stop_event.is_set() else "idle",
            continuous_active=False,
            cycle_index=n_experiments,
            note=(
                f"Continuous run finished after {n_experiments} experiments."
                if not self._stop_event.is_set()
                else "Continuous run stopped by user."
            ),
        )

        # Clean up checkpoints on successful completion (unless keep_checkpoints)
        if not self._stop_event.is_set() and not config.keep_checkpoints:
            try:
                ckpt_exp_id = resume_id or "continuous"
                ckpt.cleanup(ckpt_exp_id)
            except Exception as e:
                logger.debug("Checkpoint cleanup failed: %s", e)

        # Launch queued auto-scale-up
        self._run_pending_scale_up()

    def _select_next_mode(self, config: RunConfig, nb: LabNotebook,
                          n_experiments: int, digest=None) -> Dict:
        """Have Aria decide the next experiment mode."""
        try:
            recent = nb.get_recent_experiments(10)
            leaderboard = nb.get_leaderboard(limit=50)
            analytics_data = self._gather_analytics_data(nb)

            context = build_mode_selection_context(
                recent_experiments=recent,
                leaderboard=leaderboard,
                analytics_data=analytics_data,
                current_mode="synthesis",
                n_experiments_in_session=n_experiments,
                cost_spent=self.aria.total_cost,
                budget=config.max_cost_dollars,
                digest=digest,
            )

            # Build fallback data for rule-based recommendation
            total_s1 = sum(e.get("n_stage1_passed", 0) for e in recent)
            novelty_scores = [
                e.get("best_novelty_score", 0) for e in recent
                if e.get("best_novelty_score") is not None
            ]
            avg_novelty = (sum(novelty_scores) / len(novelty_scores)
                           if novelty_scores else 0)

            # Count candidates ready for investigation, excluding those already
            # attempted and failed (checked via program_results in investigation experiments)
            _investigated_fps = set()
            try:
                _inv_rows = nb.conn.execute(
                    "SELECT DISTINCT pr.graph_fingerprint "
                    "FROM program_results pr "
                    "JOIN experiments e ON e.experiment_id = pr.experiment_id "
                    "WHERE e.experiment_type = 'investigation'"
                ).fetchall()
                _investigated_fps = {r[0] for r in _inv_rows if r[0]}
            except Exception:
                pass

            investigation_ready = len([
                e for e in leaderboard
                if e.get("tier") == "screening"
                and e.get("screening_loss_ratio") is not None
                and e["screening_loss_ratio"] < config.investigation_loss_ratio_threshold
                and e.get("result_id") not in _investigated_fps
                # Also check by fingerprint from the linked program_result
            ])
            # More robust: filter by fingerprint
            if _investigated_fps:
                _inv_candidates = []
                for e in leaderboard:
                    if (e.get("tier") == "screening"
                            and e.get("screening_loss_ratio") is not None
                            and e["screening_loss_ratio"] < config.investigation_loss_ratio_threshold):
                        # Look up the fingerprint for this result
                        try:
                            fp_row = nb.conn.execute(
                                "SELECT graph_fingerprint FROM program_results WHERE result_id = ?",
                                (e["result_id"],)
                            ).fetchone()
                            fp = fp_row[0] if fp_row else None
                        except Exception:
                            fp = None
                        if fp and fp not in _investigated_fps:
                            _inv_candidates.append(e)
                investigation_ready = len(_inv_candidates)
            validation_ready = len([
                e for e in leaderboard
                if e.get("tier") == "investigation"
                and e.get("investigation_robustness") is not None
                and e["investigation_robustness"] >= config.investigation_robustness_threshold
            ])

            # Gather richer analytics for data-driven rule-based recommendation
            recent_modes = [e.get("experiment_type", "synthesis") for e in recent]
            
            # Z16: Dynamic top-10% worthiness bar
            # Only investigate candidates whose loss_ratio is in the top 10%
            # of ALL unique fingerprints. This means investigation gives Aria
            # genuinely novel data about exceptional architectures, not noise.
            worth_it_investigation = []
            seen_fps = set()

            # Compute dynamic threshold: top 10% of all screening loss ratios
            all_screening_lrs = sorted([
                e.get("screening_loss_ratio") for e in leaderboard
                if e.get("tier") == "screening"
                and e.get("screening_loss_ratio") is not None
            ])
            if all_screening_lrs:
                # Top 10% = the 10th percentile value (lower is better)
                p10_idx = max(0, len(all_screening_lrs) // 10 - 1)
                dynamic_lr_threshold = all_screening_lrs[p10_idx]
            else:
                dynamic_lr_threshold = 0.05  # conservative default

            # Sort leaderboard by best loss to prioritize performance
            sorted_lb = sorted(leaderboard, key=lambda x: x.get("screening_loss_ratio") or 1.0)

            for e in sorted_lb:
                if e.get("tier") != "screening": continue
                lr = e.get("screening_loss_ratio") or 1.0
                fp = e.get("graph_fingerprint") or e.get("result_id")

                if fp in seen_fps: continue
                seen_fps.add(fp)

                # Worthiness Bar: must be in top 10% of all fingerprints
                pis = e.get("pre_inv_score")
                if pis is not None:
                    is_worth_it = float(pis) >= 20.0
                else:
                    is_worth_it = lr <= dynamic_lr_threshold

                if is_worth_it and fp not in _investigated_fps:
                    worth_it_investigation.append(e["result_id"])

            investigation_backlog = len(worth_it_investigation)

            # Enforce exploration-first policy:
            # - First 3 cycles of a session are always synthesis/exploration
            # - After that, at least 40% of recent experiments must be synthesis
            # - Without enough exploration, investigation just re-examines stale data
            synthesis_modes = {"synthesis", "novelty", "evolve"}
            recent_synthesis_count = sum(1 for m in recent_modes if m in synthesis_modes)
            synthesis_ratio = recent_synthesis_count / max(len(recent_modes), 1)
            exploration_first = n_experiments < 3
            synthesis_starved = exploration_first or synthesis_ratio < 0.4

            if not synthesis_starved:
                # Only force investigation/validation if we have enough synthesis going
                if investigation_backlog >= 5 and "investigation" not in recent_modes[:3]:
                    return {
                        "mode": "investigation",
                        "reasoning": (f"Top-10% investigation: {investigation_backlog} candidates with "
                                      f"loss_ratio <= {dynamic_lr_threshold:.4f} (top 10% threshold). "
                                      f"These are genuinely exceptional and worth deeper study."),
                        "confidence": 1.0,
                        "config": {"n_programs": min(investigation_backlog, 10)}
                    }

                if validation_ready >= 5 and "validation" not in recent_modes[:3]:
                    return {
                        "mode": "validation",
                        "reasoning": f"Pipeline bottleneck at validation: {validation_ready} candidates ready. Switching to multi-seed verification.",
                        "confidence": 1.0,
                        "config": {"n_programs": min(validation_ready, 10)}
                    }

            recent_failures = [e for e in recent if e.get("status") == "failed"]
            unique_fingerprints = set()
            for e in leaderboard:
                fp = e.get("graph_fingerprint") or ""
                if fp:
                    unique_fingerprints.add(fp[:8])

            # Optimizer diversity: count distinct optimizers used
            optimizer_counts = {}
            try:
                rows = nb.db.execute(
                    "SELECT optimizer_name, COUNT(*) as cnt "
                    "FROM program_results WHERE optimizer_name IS NOT NULL "
                    "GROUP BY optimizer_name"
                ).fetchall()
                for row in rows:
                    optimizer_counts[row[0]] = row[1]
            except Exception:
                pass  # Table/column may not exist yet

            fallback_data = {
                "total_s1_survivors": total_s1,
                "avg_novelty": avg_novelty,
                "n_experiments_in_session": n_experiments,
                "base_n_programs": config.n_programs,
                "investigation_ready": investigation_ready,
                "validation_ready": validation_ready,
                "analytics_data": analytics_data,
                "recent_modes": recent_modes,
                "recent_failure_count": len(recent_failures),
                "leaderboard_diversity": len(unique_fingerprints),
                "leaderboard_size": len(leaderboard),
                "optimizer_counts": optimizer_counts,
                "optimizer_diversity": len(optimizer_counts),
            }

            compression_coverage = analytics_data.get("compression_coverage") or {}
            compression_totals = compression_coverage.get("totals") or {}
            n_tested = int(compression_totals.get("n_tested") or 0)
            n_compressed_tested = int(compression_totals.get("n_compressed_tested") or 0)
            n_compressed_survived = int(compression_totals.get("n_compressed_survived") or 0)
            n_survived = int(compression_totals.get("n_survived") or 0)
            compressed_test_share = (
                n_compressed_tested / n_tested if n_tested > 0 else 0.0
            )
            fallback_data["compressed_test_share"] = compressed_test_share
            fallback_data["compression_summary"] = {
                "n_tested": n_tested,
                "n_compressed_tested": n_compressed_tested,
                "n_compressed_survived": n_compressed_survived,
                "n_survived": n_survived,
                "compressed_survival_rate": (
                    n_compressed_survived / n_compressed_tested
                    if n_compressed_tested > 0
                    else 0.0
                ),
                "overall_survival_rate": (
                    n_survived / n_tested if n_tested > 0 else 0.0
                ),
            }

            rec = self.aria.recommend_next_mode(
                context=context, fallback_data=fallback_data, digest=digest,
                op_success_rates=analytics_data.get("op_success_rates"),
                compression_coverage=analytics_data.get("compression_coverage"))

            compression_override = self._compression_focus_override(rec, fallback_data)
            if compression_override is not None:
                rec = compression_override

            trigger = self._selection_safety_valve(nb, config)
            if trigger and trigger.get("triggered"):
                self._invoke_code_healer(
                    nb=nb,
                    trigger_type="plateau",
                    experiment_id=None,
                    scope=f"Safety valve plateau trigger: {trigger.get('reason')}",
                    reproduction_steps=["python -m pytest tests/test_selection_policy.py -x --tb=short"],
                    acceptance_tests=["python -m pytest tests/test_selection_policy.py -x --tb=short"],
                    trigger_payload=trigger,
                )
                if trigger.get("mode") == "novelty":
                    rec["mode"] = "novelty"
                    rec["reasoning"] = (
                        f"{rec.get('reasoning', '')} | Safety valve: {trigger.get('reason')}"
                    ).strip(" |")
                    rec.setdefault("config", {})
                    rec["config"]["n_generations"] = max(4, int(config.n_generations))
                    rec["config"]["population_size"] = max(12, int(config.population_size))
                else:
                    rec["mode"] = "synthesis"
                    rec.setdefault("config", {})
                    rec["config"]["ablation_heavy"] = True
                    rec["config"]["n_programs"] = max(8, int(config.n_programs * 0.6))
                    rec["reasoning"] = (
                        f"{rec.get('reasoning', '')} | Safety valve(ablation-heavy): "
                        f"{trigger.get('reason')}"
                    ).strip(" |")
                rec["safety_valve"] = trigger

            refinement_plan = self._build_refinement_plan(nb, config)
            if (
                refinement_plan
                and rec.get("mode") not in {"investigation", "validation"}
            ):
                rec.setdefault("config", {})
                rec["mode"] = "refinement"
                rec["config"].update(refinement_plan.get("config", {}))
                rec["refinement_plan"] = {
                    "source_result_ids": refinement_plan.get("source_result_ids", []),
                    "source_count": refinement_plan.get("source_count", 0),
                    "generations": refinement_plan.get("generations", 1),
                    "budget_programs": refinement_plan.get("budget_programs", 0),
                }
                rec["confidence"] = max(float(rec.get("confidence", 0.5) or 0.5), 0.7)
                rec["reasoning"] = (
                    f"{rec.get('reasoning', '')} | "
                    f"Recursive refinement on {rec['refinement_plan']['source_count']} "
                    f"diverse Stage-1 winners for {rec['refinement_plan']['generations']} generation(s)."
                ).strip(" |")

            # Exploration-first override: force synthesis in the first 3 cycles
            # so Aria builds a data foundation before spending time on investigation.
            # Also enforce synthesis when recent mode history is investigation-heavy.
            if n_experiments < 3 and rec.get("mode") in ("investigation", "validation"):
                rec["mode"] = "synthesis"
                rec["reasoning"] = (
                    f"Exploration-first: cycle {n_experiments + 1}/3. "
                    f"Building data foundation before investigation. "
                    f"(Original recommendation: {rec.get('reasoning', '')})"
                )
                rec["confidence"] = 0.8

            evidence_pack = build_evidence_pack(
                nb,
                analytics=None,
                recommendation=rec,
                decision_type="mode_selection",
                recent_experiments=recent,
            )
            rec["evidence_pack"] = evidence_pack

            nb.add_entry(ExperimentEntry(
                entry_type="decision",
                title=f"Mode Selection: {rec.get('mode', 'synthesis')}",
                content=rec.get("reasoning", ""),
                metadata={
                    "mode": rec.get("mode"),
                    "confidence": rec.get("confidence"),
                    "experiment_number": n_experiments,
                    "evidence_pack": evidence_pack,
                },
            ))

            decision_log = {
                "decision_id": str(uuid.uuid4())[:12],
                "timestamp": time.time(),
                "context": "mode_selection",
                "experiment_id": None,
                "candidate_pool_summary": {
                    "recent_experiments": len(recent),
                    "leaderboard_candidates": len(leaderboard),
                    "total_s1_survivors": total_s1,
                    "avg_novelty": round(avg_novelty, 6),
                },
                "score_breakdown": [{
                    "mode": rec.get("mode"),
                    "confidence": rec.get("confidence"),
                    "quality_signal": total_s1,
                    "novelty_signal": round(avg_novelty, 6),
                }],
                "policy": {
                    "engine": "aria_mode_selection_with_refinement",
                    "safety_valve_triggered": bool(trigger),
                    "safety_valve": trigger,
                    "refinement_plan": rec.get("refinement_plan"),
                },
                "reason": rec.get("reasoning", ""),
                "chosen_experiments": [{
                    "mode": rec.get("mode"),
                    "config": rec.get("config", {}),
                }],
                "trigger": trigger,
            }
            try:
                validate_selection_decision_log(decision_log)
                nb.record_selection_decision(
                    context="mode_selection",
                    experiment_id=None,
                    candidate_pool_summary=decision_log["candidate_pool_summary"],
                    score_breakdown=decision_log["score_breakdown"],
                    policy=decision_log["policy"],
                    reason=decision_log["reason"],
                    chosen_experiments=decision_log["chosen_experiments"],
                    trigger=decision_log["trigger"],
                )
            except Exception as log_err:
                logger.debug("Mode selection decision log failed: %s", log_err)

            return rec
        except Exception as e:
            logger.debug(f"Mode selection failed, defaulting to synthesis: {e}")
            return {"mode": "synthesis", "reasoning": "Fallback", "confidence": 0.3,
                    "config": {}}

    def _run_continuous_synthesis(self, config: RunConfig, nb: LabNotebook,
                                  n_experiments: int, limit_str: str,
                                  mode_reasoning: str):
        """Run a single synthesis experiment within continuous mode."""
        is_control = self._is_control_experiment(config, n_experiments)

        # Build context so Aria's hypothesis is informed by recent results
        recent = nb.get_recent_experiments(5)
        leaderboard = nb.get_leaderboard(limit=20)
        context = build_mode_selection_context(
            recent_experiments=recent,
            leaderboard=leaderboard,
            current_mode="synthesis",
            n_experiments_in_session=n_experiments,
        )
        if config.max_cost_dollars > 0:
            context += (f"\n\nBudget: ${self.aria.total_cost:.2f} spent "
                        f"of ${config.max_cost_dollars:.2f}")

        # Populate refuted hypotheses cache for similarity gating
        self._populate_refuted_cache(nb)

        # Structured hypothesis (campaign-aware)
        structured_hyp = None
        hypothesis_id = None
        if config.enable_campaigns:
            try:
                knowledge = nb.get_knowledge()
                recent_hyps = []
                if self._active_campaign_id:
                    recent_hyps = nb.get_campaign_hypotheses(
                        self._active_campaign_id)[-5:]
                hyp_context = build_hypothesis_context(
                    campaign=nb.get_campaign(self._active_campaign_id) if self._active_campaign_id else None,
                    recent_hypotheses=recent_hyps,
                    knowledge=knowledge,
                    leaderboard=leaderboard,
                    recent_experiments=recent,
                )
                structured_hyp = self.aria.formulate_structured_hypothesis(
                    context=hyp_context)
                hypothesis = structured_hyp["prediction"]

                # Record structured hypothesis
                # Find parent: last unresolved hypothesis in chain
                parent_id = None
                nb.get_unresolved_hypotheses(self._active_campaign_id)
                # Also check if previous hypothesis suggested a follow-up
                if hasattr(self, '_next_follow_up_parent') and self._next_follow_up_parent:
                    parent_id = self._next_follow_up_parent
                    self._next_follow_up_parent = None

                hypothesis_id = nb.record_hypothesis(
                    campaign_id=self._active_campaign_id,
                    prediction=structured_hyp["prediction"],
                    reasoning=structured_hyp["reasoning"],
                    test_method=structured_hyp["test_method"],
                    success_metric=structured_hyp["success_metric"],
                    parent_id=parent_id,
                    confidence=structured_hyp["confidence"],
                    metadata={
                        "source": "structured_hypothesis",
                        "llm_used": True,
                        "fallback_used": False,
                        "used_context": True,
                        "review_status": "not_reviewed",
                        "confidence": structured_hyp.get("confidence"),
                        "critique": structured_hyp.get("critique"),
                    },
                )
                self._current_hypothesis_id = hypothesis_id

                self._emit_event("hypothesis_recorded", {
                    "hypothesis_id": hypothesis_id,
                    "prediction": structured_hyp["prediction"],
                    "confidence": structured_hyp["confidence"],
                    "campaign_id": self._active_campaign_id,
                })
            except Exception as e:
                logger.debug(f"Structured hypothesis failed, using basic: {e}")
                structured_hyp = None

        if structured_hyp is None:
            result = self.aria.formulate_hypothesis(
                context=context,
                return_metadata=True,
            )
            if isinstance(result, tuple):
                hypothesis, basic_hyp_meta = result
            else:
                hypothesis = result
                basic_hyp_meta = {
                    "source": "rule_based_fallback",
                    "llm_used": False,
                    "fallback_used": True,
                    "used_context": True,
                    "review_status": "not_reviewed",
                    "confidence": None,
                    "critique": None,
                }
            hypothesis_metadata = {
                **basic_hyp_meta,
                "context_char_count": len(context),
            }
        else:
            hypothesis_metadata = {
                "source": "structured_hypothesis",
                "llm_used": True,
                "fallback_used": False,
                "used_context": True,
                "review_status": "not_reviewed",
                "confidence": structured_hyp.get("confidence"),
                "critique": structured_hyp.get("critique"),
                "hypothesis_id": hypothesis_id,
            }

        exp_config = config.to_dict()
        if is_control:
            exp_config["control_experiment"] = True
            exp_config["use_learned_grammar_weights"] = False

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="synthesis",
            config=exp_config,
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
            created_by="continuous_synthesis",
        )
        logger.info(
            "Experiment %s started (synthesis, %d programs) — hypothesis: %s",
            exp_id[:8], config.n_programs,
            (hypothesis or "none")[:150],
        )

        if is_control:
            nb.log_learning_event(
                "grammar_control_experiment",
                f"Experiment {exp_id} is a control run using default grammar weights",
                evidence=f"interval={config.control_experiment_interval}, experiment_number={n_experiments}",
            )

        # Link experiment to campaign
        if config.enable_campaigns and self._active_campaign_id:
            try:
                nb.conn.execute(
                    "UPDATE experiments SET campaign_id = ? WHERE experiment_id = ?",
                    (self._active_campaign_id, exp_id),
                )
                nb.conn.commit()
            except Exception as e:
                logger.warning("Campaign linking failed for %s: %s", exp_id, e)

        # Link hypothesis to experiment
        if hypothesis_id:
            try:
                nb.conn.execute(
                    "UPDATE hypotheses SET experiment_id = ?, status = 'testing' "
                    "WHERE hypothesis_id = ?",
                    (exp_id, hypothesis_id),
                )
                nb.conn.commit()
            except Exception as e:
                logger.warning("Hypothesis linking failed for %s: %s", exp_id, e)

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=config.n_programs,
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=f"[{limit_str}|synthesis] {hypothesis}",
            )

        self._emit_event("experiment_started", {
            "experiment_id": exp_id,
            "experiment_number": n_experiments,
            "hypothesis": hypothesis,
            "mode": "synthesis",
            "is_control_experiment": is_control,
        })

        # Diversify grammar config based on experiment number
        synth_config = self._diversify_grammar_config(config, n_experiments)

        results = self._execute_experiment(
            exp_id,
            synth_config,
            nb,
            use_learned_grammar=not is_control,
        )
        self._persist_applied_grammar_weights(nb, exp_id, results)

        context = self._build_rich_context_for_experiment(
            results, config, hypothesis, nb)
        summary = self.aria.experiment_summary(results, context=context)
        insights = self._analyze_results(results, exp_id, nb, context=context)
        llm_analysis = self.aria.analyze_results(results, context=context)

        # Structured hypothesis validation
        if structured_hyp and hypothesis_id:
            try:
                validation = self.aria.validate_structured_hypothesis(
                    structured_hyp, results, context=context)
                nb.resolve_hypothesis(
                    hypothesis_id=hypothesis_id,
                    status=validation["status"],
                    evidence=validation["evidence"],
                    summary=validation["explanation"],
                    confidence_after=validation["confidence_after"],
                )
                nb.add_entry(ExperimentEntry(
                    entry_type="analysis",
                    title=f"Hypothesis {validation['status'].upper()}",
                    content=validation["explanation"],
                    experiment_id=exp_id,
                    metadata={
                        "hypothesis_id": hypothesis_id,
                        "status": validation["status"],
                        "confidence_after": validation["confidence_after"],
                    },
                ))
                self._emit_event("hypothesis_resolved", {
                    "hypothesis_id": hypothesis_id,
                    "status": validation["status"],
                    "evidence": validation["evidence"][:200],
                    "confidence_after": validation["confidence_after"],
                })
                # If follow-up suggested, queue it for next experiment
                if validation.get("follow_up"):
                    self._next_follow_up_parent = hypothesis_id
            except Exception as e:
                logger.debug(f"Structured validation failed: {e}")
        else:
            # Fallback to old-style validation
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
            experiment_id=exp_id, results=results,
            aria_summary=summary, aria_mood=self.aria.state.mood,
            insights=insights, llm_analysis=llm_analysis,
        )
        if summary:
            logger.info("Aria summary: %s", summary[:200])
        nb.update_op_success_rates(exp_id)
        s0_op_counts = results.pop("_s0_op_counts", None)
        if s0_op_counts:
            nb.merge_op_failure_counts(s0_op_counts)
        nb.strip_graph_json_for_failures(exp_id)
        nb.update_failure_signatures(exp_id)
        self._auto_recommend(results, config, hypothesis, nb)

        if (config.auto_report
                and config.auto_report_every_n > 0
                and n_experiments % config.auto_report_every_n == 0):
            self._maybe_auto_report(
                config, nb,
                reason=f"periodic (every {config.auto_report_every_n}, "
                       f"after exp #{n_experiments})")

        # Knowledge extraction
        self._maybe_extract_knowledge(config, nb, n_experiments)

        # Auto-escalation: promote S1 survivors to leaderboard and
        # queue investigation/validation if criteria met
        results["experiment_id"] = exp_id
        self._auto_escalate(results, config, nb, phase="screening")
        self._maybe_evaluate_campaign(config, nb)

        self._emit_event("experiment_completed", {
            "experiment_id": exp_id, "results": results, "mode": "synthesis",
        })

    def _run_continuous_evolution(self, config: RunConfig, nb: LabNotebook,
                                  n_experiments: int, limit_str: str,
                                  mode_reasoning: str):
        """Run evolution search within continuous mode (inline, not threaded)."""
        from ...search.evolution import evolutionary_search, EvolutionConfig

        hypothesis = f"Evolution search: {mode_reasoning}"
        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="evolution",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="runner_template",
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            created_by="continuous_evolution",
        )

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="evolving",
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=f"[{limit_str}|evolution] {hypothesis[:80]}",
            )

        self._emit_event("experiment_started", {
            "experiment_id": exp_id,
            "experiment_number": n_experiments,
            "hypothesis": hypothesis,
            "mode": "evolution",
        })

        # Cap depth/ops for evolution to prevent recursion overflow
        evo_config = EvolutionConfig(
            population_size=config.n_programs,
            n_generations=config.n_generations,
            grammar_config=self._build_grammar_config(config),
        )

        fitness_cache: dict = {}
        eval_counters = {"total": 0, "s0": 0, "s1": 0}

        def on_evaluate(graph, fitness, sandbox_result, s1_result):
            self._on_program_evaluated(graph, fitness, sandbox_result, s1_result,
                                       eval_counters, nb, exp_id, model_source="evolution")

        fitness_fn = self._make_fitness_fn(
            config, on_evaluate=on_evaluate, fitness_cache=fitness_cache)

        def novelty_fn(graph, all_graphs):
            nov = novelty_score(graph)
            my_fp = graph.fingerprint()
            dup_count = sum(1 for g in all_graphs
                            if g.fingerprint() == my_fp) - 1
            penalty = max(0, 1 - dup_count * 0.3)
            return nov.structural_novelty * penalty

        population = evolutionary_search(
            fitness_fn=fitness_fn,
            novelty_fn=novelty_fn,
            config=evo_config,
            stop_check=self._stop_event.is_set,
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
        context = self._build_rich_context_for_experiment(
            results, config, hypothesis, nb)
        summary = self.aria.experiment_summary(results, context=context)
        llm_analysis = self.aria.analyze_results(results, context=context)
        nb.complete_experiment(
            experiment_id=exp_id, results=results,
            aria_summary=summary, aria_mood=self.aria.state.mood,
            insights=self._analyze_results(results, exp_id, nb, context=context),
            llm_analysis=llm_analysis,
        )

        results["experiment_id"] = exp_id
        self._auto_escalate(results, config, nb, phase="screening")
        self._maybe_evaluate_campaign(config, nb)

        self._emit_event("experiment_completed", {
            "experiment_id": exp_id, "results": results, "mode": "evolution",
        })

    def _run_continuous_novelty(self, config: RunConfig, nb: LabNotebook,
                                 n_experiments: int, limit_str: str,
                                 mode_reasoning: str):
        """Run novelty search within continuous mode (inline, not threaded)."""
        from ...search.novelty_search import novelty_search, NoveltySearchConfig

        hypothesis = f"Novelty search: {mode_reasoning}"
        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="novelty",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="runner_template",
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            exploratory=True,
            created_by="continuous_novelty",
        )

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="novelty_search",
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=f"[{limit_str}|novelty] {hypothesis[:80]}",
            )

        self._emit_event("experiment_started", {
            "experiment_id": exp_id,
            "experiment_number": n_experiments,
            "hypothesis": hypothesis,
            "mode": "novelty",
        })

        # Cap depth/ops for novelty search to prevent recursion overflow
        min(config.max_depth, 12)
        min(config.max_ops, 20)

        grammar = self._build_grammar_config(config)
        ns_config = NoveltySearchConfig(
            population_size=config.n_programs,
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
                    stage_tag="novelty_fitness",
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

                # Compute behavioral fingerprint while model is still in memory
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

        ns_result = novelty_search(
            fitness_fn=combined_fitness_fn,
            fingerprint_fn=fingerprint_fn,
            config=ns_config,
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
            lr = 1.0 - ind.fitness if ind.fitness > 0 else None
            if lr is not None and (results["best_loss_ratio"] is None
                                    or lr < results["best_loss_ratio"]):
                results["best_loss_ratio"] = lr
            if ind.novelty and (results["best_novelty_score"] is None
                                 or ind.novelty > results["best_novelty_score"]):
                results["best_novelty_score"] = ind.novelty
            if ind.fitness > 0.2:
                results["survivors"].append({
                    "fingerprint": ind.fingerprint,
                    "novelty": ind.novelty,
                    "loss_ratio": 1.0 - ind.fitness,
                })

        nb.update_op_success_rates(exp_id)
        nb.update_failure_signatures(exp_id)
        context = self._build_rich_context_for_experiment(
            results, config, hypothesis, nb)
        summary = self.aria.experiment_summary(results, context=context)
        llm_analysis = self.aria.analyze_results(results, context=context)
        nb.complete_experiment(
            experiment_id=exp_id, results=results,
            aria_summary=summary, aria_mood=self.aria.state.mood,
            insights=self._analyze_results(results, exp_id, nb, context=context),
            llm_analysis=llm_analysis,
        )

        results["experiment_id"] = exp_id
        self._auto_escalate(results, config, nb, phase="screening")
        self._maybe_evaluate_campaign(config, nb)

        self._emit_event("experiment_completed", {
            "experiment_id": exp_id, "results": results, "mode": "novelty",
        })

    def _run_continuous_refinement(
        self,
        config: RunConfig,
        nb: LabNotebook,
        n_experiments: int,
        limit_str: str,
        mode_reasoning: str,
    ):
        """Run recursive local winner-tweak refinement with plateau stopping."""
        plan = self._build_refinement_plan(nb, config)
        if not plan:
            logger.info("Refinement requested but no eligible Stage-1 winners found; falling back to synthesis.")
            self._run_continuous_synthesis(config, nb, n_experiments, limit_str, mode_reasoning)
            return

        source_ids = list(plan.get("source_result_ids", []))
        total_generations = max(1, int(plan.get("generations") or config.refinement_generations or 1))
        budget_remaining = max(int(plan.get("budget_programs") or 0), int(config.n_programs))
        plateau_patience = max(1, int(config.refinement_plateau_patience or 1))
        mutation_radius = max(0.05, min(1.0, float(config.refinement_mutation_radius or 0.35)))
        novelty_pressure = max(0.0, min(1.0, float(config.refinement_novelty_pressure or 0.35)))

        best_loss_seen: Optional[float] = None
        plateau_count = 0
        executed_generations = 0
        history: List[Dict[str, Any]] = []

        for generation in range(total_generations):
            if self._stop_event.is_set() or budget_remaining <= 0 or not source_ids:
                break

            gen_cfg = RunConfig.from_dict(config.to_dict())
            gen_cfg.model_source = "fingerprint_refine"
            gen_cfg.refine_source_result_ids = ",".join(source_ids)
            gen_cfg.refine_mutations_per_source = max(1, int(round(2 + 4 * mutation_radius)))
            gen_cfg.refine_pool_multiplier = max(2, int(round(2 + 3 * novelty_pressure)))
            gen_cfg.mutation_rate = max(0.10, min(0.95, float(config.mutation_rate) * (0.5 + mutation_radius)))
            gen_cfg.n_programs = max(4, min(int(config.n_programs), budget_remaining))

            generation_reason = (
                f"{mode_reasoning} | recursive_refine gen {generation + 1}/{total_generations} "
                f"from {len(source_ids)} seed(s)"
            )
            self._run_continuous_synthesis(
                gen_cfg,
                nb,
                n_experiments,
                limit_str,
                generation_reason,
            )
            budget_remaining -= int(gen_cfg.n_programs)
            executed_generations += 1

            recent = nb.get_recent_experiments(1)
            if not recent:
                break
            current_exp_id = str(recent[0].get("experiment_id") or "")
            if not current_exp_id:
                break
            rows = nb.get_program_results(current_exp_id, limit=400)
            survivors = [row for row in rows if row.get("stage1_passed")]
            if not survivors:
                history.append({
                    "generation": generation + 1,
                    "experiment_id": current_exp_id,
                    "stage1_survivors": 0,
                    "best_loss_ratio": None,
                })
                break

            cur_best = min(
                (float(r.get("loss_ratio")) for r in survivors if isinstance(r.get("loss_ratio"), (int, float))),
                default=None,
            )
            history.append({
                "generation": generation + 1,
                "experiment_id": current_exp_id,
                "stage1_survivors": len(survivors),
                "best_loss_ratio": cur_best,
            })
            if cur_best is not None:
                if best_loss_seen is None or cur_best < best_loss_seen - 1e-4:
                    best_loss_seen = cur_best
                    plateau_count = 0
                else:
                    plateau_count += 1

            selected = self._select_diverse_refinement_sources(
                survivors,
                top_k=max(1, int(config.refinement_top_k or 1)),
                min_distance=max(0.01, float(config.refinement_min_distance or 0.01)),
                novelty_pressure=novelty_pressure,
            )
            source_ids = [str(r.get("result_id") or "") for r in selected if r.get("result_id")]
            if plateau_count >= plateau_patience:
                break

        nb.record_decision(
            campaign_id=self._active_campaign_id,
            decision_type="recursive_refinement",
            subject=f"cycle_{n_experiments}",
            rationale=(
                f"Executed recursive local refinement for {executed_generations}/{total_generations} generation(s); "
                f"plateau_count={plateau_count}, budget_remaining={budget_remaining}."
            ),
            evidence_pack={
                "seed_count": int(plan.get("source_count") or 0),
                "initial_source_result_ids": plan.get("source_result_ids", []),
                "history": history,
                "plateau_patience": plateau_patience,
                "min_distance": float(config.refinement_min_distance),
                "novelty_pressure": novelty_pressure,
            },
        )

    # ── Pre-investigation gate ─────────────────────────────────────────

    def _get_reference_baseline_lr(self, nb: LabNotebook) -> Optional[float]:
        """Fetch best screening_loss_ratio from registered reference architectures."""
        try:
            refs = nb.get_references()
            if not refs:
                return None
            lrs = [float(r["screening_loss_ratio"]) for r in refs
                   if r.get("screening_loss_ratio") is not None]
            return min(lrs) if lrs else None
        except Exception:
            return None

    def _pre_inv_probe(self, config: RunConfig, nb: LabNotebook,
                       result_id: str) -> Optional[float]:
        """Stage C: single-seed probe at reduced step count.

        Runs 1 training program at probe_steps_fraction of investigation_steps.
        Returns loss_ratio or None on failure.
        """
        try:
            details = nb.get_program_details([result_id])
            if not details or not details[0]:
                return None
            source = details[0]
            graph_json = source.get("graph_json")
            if not graph_json:
                return None

            probe_config = RunConfig.from_dict(config.to_dict())
            probe_config.stage1_steps = max(
                50, int(config.investigation_steps * config.pre_inv_probe_steps_fraction))
            probe_config.stage1_batch_size = config.investigation_batch_size
            probe_config.n_programs = 1

            dev = resolve_device(config.device)
            dev_str = str(dev)

            from research.synthesis.compiler import compile_model
            model = compile_model(graph_json, probe_config, device=dev)
            if model is None:
                return None

            from research.evaluator import evaluate_stage1
            result = evaluate_stage1(model, probe_config, device=dev)
            lr = result.get("loss_ratio") if result else None
            return float(lr) if lr is not None else None
        except Exception as e:
            logger.warning("Pre-inv probe failed for %s: %s", result_id[:8], e)
            return None

    def _pre_investigation_gate(self, config: RunConfig, nb: LabNotebook,
                                leaderboard: list) -> List[str]:
        """Orchestrate three-stage pre-investigation gate.

        Stage A: SQL hard reject (numerical health, stability, gradient path)
        Stage B: Composite readiness score, rank and take top-N
        Stage C: Optional single-seed probe

        Returns filtered, ranked result_ids ready for investigation.
        Falls back to legacy behavior when pre_inv_gate_enabled=False.
        """
        if not config.pre_inv_gate_enabled:
            # Legacy behavior: filter by loss_ratio threshold only
            investigated_fps = nb.get_investigated_fingerprints()
            candidates = [
                e for e in leaderboard
                if e.get("tier") == "screening"
                and e.get("screening_loss_ratio") is not None
                and e["screening_loss_ratio"] < config.investigation_loss_ratio_threshold
            ]
            if investigated_fps:
                candidates = [
                    c for c in candidates
                    if c.get("graph_fingerprint", c.get("architecture_desc", ""))
                    not in investigated_fps
                ]
            return [c["result_id"] for c in candidates[:config.auto_investigate_top_n]
                    if c.get("result_id")]

        # ── Stage A: Hard reject via SQL ──
        ref_lr = self._get_reference_baseline_lr(nb)
        ref_lr_ceiling = None
        if ref_lr is not None:
            ref_lr_ceiling = ref_lr * config.pre_inv_reference_margin

        eligible = nb.get_investigation_eligible(
            max_lr=config.investigation_loss_ratio_threshold,
            min_stability=config.pre_inv_min_stability,
            min_spectral_norm=config.pre_inv_min_spectral_norm,
            max_spectral_norm=config.pre_inv_max_spectral_norm,
            min_improvement_rate=config.pre_inv_min_improvement_rate,
            ref_lr_ceiling=ref_lr_ceiling,
        )

        # Filter out already-investigated fingerprints
        investigated_fps = nb.get_investigated_fingerprints()
        if investigated_fps:
            before = len(eligible)
            eligible = [e for e in eligible
                        if e.get("graph_fingerprint") not in investigated_fps]
            skipped = before - len(eligible)
            if skipped:
                logger.info("Pre-inv gate: skipped %d already-investigated candidates", skipped)

        if not eligible:
            logger.info("Pre-inv gate Stage A: no eligible candidates")
            return []

        logger.info("Pre-inv gate Stage A: %d candidates pass hard filters", len(eligible))

        # ── Stage B: Composite score + rank ──
        for row in eligible:
            row["_pre_inv_score"] = LabNotebook.compute_pre_investigation_score(
                row, best_ref_lr=ref_lr)

        eligible.sort(key=lambda r: r.get("_pre_inv_score", 0), reverse=True)
        top_n = eligible[:config.pre_inv_top_n]

        # Persist scores to leaderboard
        for row in eligible:
            try:
                nb.conn.execute(
                    "UPDATE leaderboard SET pre_inv_score = ? WHERE result_id = ?",
                    (row["_pre_inv_score"], row["result_id"]),
                )
            except Exception:
                pass
        try:
            nb.conn.commit()
        except Exception:
            pass

        logger.info("Pre-inv gate Stage B: top %d scored [%s]",
                     len(top_n),
                     ", ".join(f"{r['result_id'][:8]}={r['_pre_inv_score']:.1f}"
                               for r in top_n))

        # ── Stage C: Optional probe ──
        if config.pre_inv_probe_enabled:
            probed = []
            for row in top_n:
                probe_lr = self._pre_inv_probe(config, nb, row["result_id"])
                if probe_lr is not None and probe_lr > config.pre_inv_probe_max_lr:
                    logger.info("Pre-inv probe rejected %s (lr=%.3f > %.3f)",
                                row["result_id"][:8], probe_lr,
                                config.pre_inv_probe_max_lr)
                    continue
                probed.append(row)
            top_n = probed

        return [r["result_id"] for r in top_n if r.get("result_id")]

    def _run_continuous_phase(self, phase: str, config: RunConfig,
                               nb: LabNotebook, n_experiments: int,
                               limit_str: str, mode_reasoning: str):
        """Run investigation or validation phase inline within continuous mode."""
        leaderboard = nb.get_leaderboard(limit=50)

        if phase == "investigation":
            self._run_inline_investigation(
                config, nb, leaderboard, n_experiments, limit_str, mode_reasoning)
        elif phase == "validation":
            self._run_inline_validation(
                config, nb, leaderboard, n_experiments, limit_str, mode_reasoning)

    def _run_inline_investigation(self, config: RunConfig, nb: LabNotebook,
                                   leaderboard: list, n_experiments: int,
                                   limit_str: str, mode_reasoning: str):
        """Execute investigation phase inline (not threaded) for continuous mode."""
        # Use pre-investigation gate for candidate selection
        result_ids = self._pre_investigation_gate(config, nb, leaderboard)
        if not result_ids:
            self._run_continuous_synthesis(
                config, nb, n_experiments, limit_str, mode_reasoning)
            return

        # Build context for hypothesis formulation
        inv_details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
        inv_map = {d.get("result_id"): d for d in inv_details if d.get("result_id")}
        inv_context = build_investigation_context(inv_details, leaderboard)
        hypothesis = self.aria.formulate_investigation_hypothesis(
            context=inv_context)
        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="investigation",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="llm_context",
                llm_used=True,
                fallback_used=False,
                used_context=True,
            ),
            created_by="inline_investigation",
        )

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="investigating",
                total_programs=len(result_ids),
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=(f"[{limit_str}|investigation] "
                              f"Studying {len(result_ids)} candidates"),
            )

        self._emit_event("investigation_started", {
            "experiment_id": exp_id,
            "n_candidates": len(result_ids),
        })

        try:
            # ── Inline investigation logic (from _run_investigation_thread) ──
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
                if self._stop_event.is_set():
                    break

                # Cost check mid-investigation
                if config.max_cost_dollars > 0 and self.aria.total_cost >= config.max_cost_dollars:
                    logger.info("Cost limit reached during investigation")
                    break

                with self._lock:
                    self._progress.current_program = prog_idx + 1
                    self._progress.status = "investigating"
                    self._progress.aria_message = (
                        f"Investigating {prog_idx + 1}/{len(result_ids)}: "
                        f"{source_result_id[:8]}... "
                        f"({config.n_training_programs} training programs)"
                    )

                self._emit_event("investigation_progress", {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "starting",
                })

                # Fetch source program
                source = inv_map.get(source_result_id)
                if source is None:
                    continue

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
                        model = self._build_model_from_source(
                            model_source,
                            arch_spec_json_str,
                            graph_json_str,
                            config,
                            seq_len_override=config.max_seq_len,
                        )
                        if model is None:
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

                    tp_result = self._train_with_program(
                        model,
                        tp,
                        inv_config,
                        dev,
                        seed=self._stable_seed(exp_id, source_result_id, tp_i, "investigation"),
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
                        f"Investigation: skipping {source_result_id[:8]} — "
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

                # Benchmark evals (non-blocking) for inline investigation survivors
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

            # Complete experiment with LLM analysis
            results["perf_report"] = self._build_experiment_perf_report(results)
            results["perf_budget_gate"] = evaluate_perf_budget_gate(results["perf_report"])
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            nb.complete_experiment(
                experiment_id=exp_id, results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=self._analyze_results(results, exp_id, nb, context=context),
                llm_analysis=llm_analysis,
            )

            # Auto-escalate to validation if strong candidates found
            self._auto_escalate(results, config, nb, phase="investigation")

            # Knowledge extraction after investigation
            self._maybe_extract_knowledge(config, nb, n_experiments)

            self._emit_event("investigation_completed", {
                "experiment_id": exp_id, "results": results,
                "summary": summary,
            })

        except Exception as e:
            logger.warning(f"Inline investigation failed: {e}")
            nb.fail_experiment(exp_id, str(e))
            self._emit_event("investigation_completed", {
                "experiment_id": exp_id, "error": str(e),
            })

    def _validation_run_seeds(
        self,
        config, val_config,
        dev,
        exp_id: str,
        prog_idx: int,
        total_progs: int,
        source_result_id: str,
        source: dict,
        best_tp_json: str,
        model_source: str,
        arch_spec_json_str: str,
        graph_json_str: str,
    ):
        seed_results = []
        # Multi-seed evaluation
        seed_results = []
        for seed in range(config.validation_n_seeds):
            if self._stop_event.is_set():
                break

            torch.manual_seed(seed * 42 + 7)

            # Reconstruct model fresh
            init_scheme = "default"
            try:
                model = self._build_model_from_source(
                    model_source,
                    arch_spec_json_str,
                    graph_json_str,
                    config,
                    seq_len_override=config.validation_seq_len,
                )
                if model is None:
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
                        seed=self._stable_seed(exp_id, source_result_id, seed, "validation_tp"),
                    )
                except Exception:
                    s1_result = self._micro_train(
                        model,
                        val_config,
                        dev,
                        seed=self._stable_seed(exp_id, source_result_id, seed, "validation_micro"),
                    )
            else:
                s1_result = self._micro_train(
                    model,
                    val_config,
                    dev,
                    seed=self._stable_seed(exp_id, source_result_id, seed, "validation_micro"),
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

        return seed_results

    def _validation_compute_metrics(
        self, config, dev_str, source, seed_results
    ):
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

        return dict(
            val_loss_ratio=val_loss_ratio,
            multi_seed_std=multi_seed_std,
            init_sensitivity_std=init_sensitivity_std,
            val_baseline_ratio=val_baseline_ratio,
            val_normalized_ratio=val_normalized_ratio,
            val_param_efficiency=val_param_efficiency,
            passed_seeds=passed_seeds,
            best_seed=best_seed if 'best_seed' in locals() else None,
            source_params=source_params
        )

    def _validation_run_external_evals(
        self, config, dev, dev_str, best_seed, is_breakthrough,
        model_source, arch_spec_json_str, graph_json_str, source,
        source_result_id, exp_id, val_loss_ratio, val_baseline_ratio, val_normalized_ratio, multi_seed_std, passed_seeds, source_params
    ):
        best_seed_loss = best_seed.get("final_loss") if best_seed else 0
        # OOD robustness check (#54): test with reference recipes
        ood_result = None
        if len(passed_seeds) > 0:
            _gjs_ood = graph_json_str
            _asjs_ood = arch_spec_json_str
            _ms_ood = model_source
            _cfg_ood = config

            def _make_model_ood():
                if _ms_ood == "morphological_box" and _asjs_ood:
                    from ...morphological_box import ArchSpec
                    from ...arch_builder import build_model, BuildConfig
                    spec = ArchSpec(**json.loads(_asjs_ood))
                    bc = BuildConfig(
                        dim=_cfg_ood.model_dim,
                        n_layers=_cfg_ood.n_layers,
                        vocab_size=_cfg_ood.vocab_size,
                        max_seq_len=_cfg_ood.validation_seq_len)
                    return build_model(spec, bc)
                else:
                    g = graph_from_json(_gjs_ood)
                    return compile_model(
                        [g] * _cfg_ood.n_layers,
                        vocab_size=_cfg_ood.vocab_size,
                        max_seq_len=_cfg_ood.validation_seq_len)

            try:
                ood_result = self._ood_robustness_check(
                    _make_model_ood, config, dev,
                    n_steps=min(300, config.validation_steps // 3),
                    seed=self._stable_seed(
                        exp_id, source_result_id, 0, "ood"),
                )
                self._emit_event("ood_robustness", {
                    "experiment_id": exp_id,
                    "result_id": source_result_id,
                    "ood_robustness": ood_result.get("ood_robustness"),
                    "recipes_passed": ood_result.get("recipes_passed"),
                })
            except Exception as e:
                logger.debug("OOD robustness check failed: %s", e)

        # Hyperparameter sensitivity check (#57)
        sensitivity_result = None
        if len(passed_seeds) > 0 and val_loss_ratio is not None:
            try:
                sensitivity_result = self._sensitivity_check(
                    _make_model_ood, config, dev,
                    base_loss_ratio=val_loss_ratio,
                    n_steps=min(300, config.validation_steps // 3),
                    seed=self._stable_seed(
                        exp_id, source_result_id, 0, "sensitivity"),
                )
                self._emit_event("sensitivity_check", {
                    "experiment_id": exp_id,
                    "result_id": source_result_id,
                    "hp_robustness": sensitivity_result.get("hp_robustness"),
                    "avg_deviation": sensitivity_result.get("avg_deviation"),
                })
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
                # Build a fresh model for quant eval
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
                # Generate test batches
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
                    _asjs_lc = arch_spec_json_str
                    _cfg_lc = config
                    def _make_model_lc():
                        from ...morphological_box import ArchSpec
                        from ...arch_builder import build_model, BuildConfig
                        _sp = ArchSpec(**json.loads(_asjs_lc))
                        _bc2 = BuildConfig(
                            dim=_cfg_lc.model_dim, n_layers=_cfg_lc.n_layers,
                            vocab_size=_cfg_lc.vocab_size, max_seq_len=1024)
                        return build_model(_sp, _bc2)
                else:
                    _gjs_lc = graph_json_str
                    _cfg_lc = config
                    def _make_model_lc():
                        return compile_model(
                            [graph_from_json(_gjs_lc)] * _cfg_lc.n_layers,
                            vocab_size=_cfg_lc.vocab_size, max_seq_len=1024)
                from ...eval.long_context import run_long_context_sweep
                from ...eval.passkey import evaluate_long_context_retrieval
                
                lc_result = run_long_context_sweep(
                    _make_model_lc, config.vocab_size, dev,
                    base_loss=base_loss_val, seq_lens=(512, 1024),
                    n_steps=200, batch_size=2,
                )
                
                # Retrieval test (needle-in-a-haystack)
                # Use a small validation model for faster retrieval testing
                retr_model = _make_model_lc().to(dev)
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
                    "Long-context check: scaling=%.2f, assoc=%.2f, multi_hop=%.2f, passkey=%.2f, retrieval=%.2f, combined=%.2f",
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
                    _spec_ns = ArchSpec(**json.loads(arch_spec_json_str))
                    _bc_ns = BuildConfig(
                        dim=config.model_dim, n_layers=config.n_layers,
                        vocab_size=config.vocab_size,
                        max_seq_len=config.validation_seq_len)
                    ns_model = build_model(_spec_ns, _bc_ns).to(dev)
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

        return dict(
            is_breakthrough=is_breakthrough,
            flop_gated=flop_gated,
            quant_int8_retention=quant_int8_retention if 'quant_int8_retention' in locals() else None,
            quant_quality_per_byte=quant_quality_per_byte if 'quant_quality_per_byte' in locals() else None,
            long_context_score=long_context_score if 'long_context_score' in locals() else None,
            long_context_details=long_context_details if 'long_context_details' in locals() else None,
            noise_score=noise_score if 'noise_score' in locals() else None,
            ood_result=ood_result,
            sensitivity_result=sensitivity_result,
            activation_sparsity_score=activation_sparsity_score if 'activation_sparsity_score' in locals() else None,
            dead_neuron_ratio=dead_neuron_ratio if 'dead_neuron_ratio' in locals() else None,
            routing_collapse_score=routing_collapse_score if 'routing_collapse_score' in locals() else None,
            wikitext_perplexity=wikitext_perplexity if 'wikitext_perplexity' in locals() else None,
            wikitext_score=wikitext_score if 'wikitext_score' in locals() else None,
            tinystories_perplexity=tinystories_perplexity if 'tinystories_perplexity' in locals() else None,
            tinystories_score=tinystories_score if 'tinystories_score' in locals() else None,
            cross_task_score=cross_task_score if 'cross_task_score' in locals() else None,
            efficiency_wall_score=efficiency_wall_score if 'efficiency_wall_score' in locals() else None,
            max_viable_seq_len=max_viable_seq_len if 'max_viable_seq_len' in locals() else None,
            scaling_regime=scaling_regime if 'scaling_regime' in locals() else None,
            scaling_param_efficiency=scaling_param_efficiency if 'scaling_param_efficiency' in locals() else None,
            scaling_flop_efficiency=scaling_flop_efficiency if 'scaling_flop_efficiency' in locals() else None,
            scaling_gate_passed_val=scaling_gate_passed_val if 'scaling_gate_passed_val' in locals() else None,
            scaling_best_family=scaling_best_family if 'scaling_best_family' in locals() else None,
            scaling_confidence=scaling_confidence if 'scaling_confidence' in locals() else None,
            scaling_result=scaling_result if 'scaling_result' in locals() else None,
        )

    def _run_inline_validation(self, config: RunConfig, nb: LabNotebook,
                                leaderboard: list, n_experiments: int,
                                limit_str: str, mode_reasoning: str):
        """Execute validation phase inline (not threaded) for continuous mode."""
        result_ids = self._inline_validation_candidate_ids(config, leaderboard)
        if not result_ids:
            logger.info("No validation candidates, falling back to synthesis")
            self._run_continuous_synthesis(
                config, nb, n_experiments, limit_str, mode_reasoning)
            return

        exp_id, hypothesis = self._inline_validation_bootstrap(
            config=config,
            nb=nb,
            leaderboard=leaderboard,
            result_ids=result_ids,
            limit_str=limit_str,
        )

        try:
            # ── Inline validation logic (from _run_validation_thread) ──
            results, dev, dev_str, val_config, source_map = self._inline_validation_prepare_runtime(
                config=config,
                nb=nb,
                result_ids=result_ids,
            )

            for prog_idx, source_result_id in enumerate(result_ids):
                if self._stop_event.is_set():
                    break

                # Cost check mid-validation
                if config.max_cost_dollars > 0 and self.aria.total_cost >= config.max_cost_dollars:
                    logger.info("Cost limit reached during validation")
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
                best_tp_json = None
                for entry in leaderboard:
                    if entry.get("result_id") == source_result_id:
                        best_tp_json = entry.get("investigation_best_training")
                        break

                # Multi-seed evaluation
                seed_results = self._validation_run_seeds(
                    config, val_config, dev, exp_id, prog_idx, len(result_ids),
                    source_result_id, source, best_tp_json,
                    model_source, arch_spec_json_str, graph_json_str
                )

                # Skip candidates where no seed could reconstruct the model
                if not seed_results:
                    logger.debug(
                        f"Inline validation: skipping {source_result_id[:8]} — "
                        f"model failed to reconstruct for all {config.validation_n_seeds} seeds"
                    )
                    continue

                metrics = self._validation_compute_metrics(config, dev_str, source, seed_results)
                
                val_loss_ratio = metrics["val_loss_ratio"]
                multi_seed_std = metrics["multi_seed_std"]
                init_sensitivity_std = metrics["init_sensitivity_std"]
                val_baseline_ratio = metrics["val_baseline_ratio"]
                val_normalized_ratio = metrics["val_normalized_ratio"]
                val_param_efficiency = metrics["val_param_efficiency"]
                passed_seeds = metrics["passed_seeds"]
                best_seed = metrics["best_seed"]
                source_params = metrics["source_params"]

                if len(passed_seeds) > 0:
                    results["stage1_passed"] += 1
                results["stage0_passed"] += 1
                results["stage05_passed"] += 1

                # Extrapolate and Evals
                ev_res = self._validation_run_external_evals(
                    config, dev, dev_str, best_seed, True,
                    model_source, arch_spec_json_str, graph_json_str, source,
                    source_result_id, exp_id, val_loss_ratio, val_baseline_ratio, val_normalized_ratio, multi_seed_std, passed_seeds, source_params
                )
                
                is_breakthrough = ev_res["is_breakthrough"]
                flop_gated = ev_res["flop_gated"]
                quant_int8_retention = ev_res["quant_int8_retention"]
                quant_quality_per_byte = ev_res["quant_quality_per_byte"]
                long_context_score = ev_res["long_context_score"]
                noise_score = ev_res["noise_score"]
                ood_result = ev_res["ood_result"]
                sensitivity_result = ev_res.get("sensitivity_result")
                activation_sparsity_score = ev_res["activation_sparsity_score"]
                dead_neuron_ratio = ev_res["dead_neuron_ratio"]
                routing_collapse_score = ev_res["routing_collapse_score"]
                wikitext_perplexity = ev_res["wikitext_perplexity"]
                wikitext_score = ev_res["wikitext_score"]
                tinystories_perplexity = ev_res["tinystories_perplexity"]
                tinystories_score = ev_res["tinystories_score"]
                cross_task_score = ev_res["cross_task_score"]
                efficiency_wall_score = ev_res["efficiency_wall_score"]
                max_viable_seq_len = ev_res["max_viable_seq_len"]
                scaling_regime = ev_res["scaling_regime"]
                scaling_param_efficiency = ev_res["scaling_param_efficiency"]
                scaling_flop_efficiency = ev_res["scaling_flop_efficiency"]
                scaling_gate_passed_val = ev_res["scaling_gate_passed_val"]
                scaling_best_family = ev_res["scaling_best_family"]
                scaling_confidence = ev_res["scaling_confidence"]
                scaling_result = ev_res.get("scaling_result")
                long_context_details = ev_res.get("long_context_details")

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

                # Update leaderboard - find the actual entry for this result
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
                        # Store detailed benchmark payload in external_benchmarks_json
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

            # Complete experiment with LLM analysis
            results["perf_report"] = self._build_experiment_perf_report(results)
            results["perf_budget_gate"] = evaluate_perf_budget_gate(results["perf_report"])
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            nb.complete_experiment(
                experiment_id=exp_id, results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=self._analyze_results(results, exp_id, nb, context=context),
                llm_analysis=llm_analysis,
            )

            # Knowledge extraction after validation
            self._maybe_extract_knowledge(config, nb, n_experiments)

            self._emit_event("validation_completed", {
                "experiment_id": exp_id, "results": results,
                "summary": summary,
            })

        except Exception as e:
            logger.warning(f"Inline validation failed: {e}")
            nb.fail_experiment(exp_id, str(e))
            self._emit_event("validation_completed", {
                "experiment_id": exp_id, "error": str(e),
            })
        finally:
            self._live_training_context = None

    # ── Core Execution ──

    def _end_of_session_automation(self, config: RunConfig, reason: str):
        """Run end-of-session report and scale-up. Used by both limit-reached and user-stop paths."""
        nb = self._make_notebook()
        try:
            self._maybe_auto_report(config, nb, reason=reason)
            cumulative_results = {"stage1_passed": 0, "survivors": []}
            top = nb.get_top_programs(
                config.auto_scale_up_top_n, sort_by="loss_ratio")
            for p in top:
                if p.get("stage1_passed"):
                    cumulative_results["stage1_passed"] += 1
                    cumulative_results["survivors"].append({
                        "novelty": p.get("novelty_score", 0),
                    })
            self._maybe_auto_scale_up(cumulative_results, config, nb)
        except Exception as e:
            logger.debug(f"End-of-session automation failed: {e}")
        finally:
            nb.close()
