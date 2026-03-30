"""Continuous loop thread and mode selection, split from continuous.py."""

from __future__ import annotations

import time
import uuid
from typing import Dict


from ...training.checkpointing import CheckpointManager
from ..notebook import LabNotebook, ExperimentEntry
from ..evidence import (
    build_evidence_pack,
    validate_selection_decision_log,
)
from ..llm.context_experiment import (
    build_mode_selection_context,
)

import logging

logger = logging.getLogger(__name__)

from ._types import RunConfig


class _ContinuousLoopMixin:
    """Main continuous loop thread, mode selection, and session automation."""

    __slots__ = ()

    def _run_continuous_thread(self, config: RunConfig):
        """Execute continuous experiments in background."""
        try:
            self._run_continuous_thread_inner(config)
        except BaseException as e:
            import traceback

            logger.critical(
                "Continuous thread KILLED: %s\n%s", e, traceback.format_exc(),
            )
            try:
                self._update_progress(status="failed", error=f"FATAL: {e}")
                self._emit_event(
                    "experiment_failed",
                    {"experiment_id": "continuous", "error": f"FATAL: {e}"},
                )
            except Exception:
                logger.error("Failed to emit failure event after fatal error", exc_info=True)
            if not isinstance(e, Exception):
                raise

    def _run_continuous_thread_inner(self, config: RunConfig):
        """Inner continuous loop body."""
        n_experiments = 0
        t_start = time.time()
        self.aria.reset_cost_tracking()
        # Skip per-cycle LLM calls — use rule-based paths to save API costs.
        # LLM is still available for user-initiated chat and campaign formulation.
        self.aria._continuous_mode = True
        self.aria._llm_decision_interval = config.llm_decision_interval

        # Backfill replication aggregates so composite scores reflect true n_runs
        try:
            init_nb = self._make_notebook()
            n_backfilled = init_nb.backfill_replication_aggregates()
            if n_backfilled:
                logger.info(
                    "Backfilled replication data on %d leaderboard entries",
                    n_backfilled,
                )
            init_nb.close()
        except Exception as e:
            logger.warning("Replication backfill failed: %s", e)

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
                logger.warning("Digest recovery failed: %s", e)
            distiller.start()
            self._knowledge_distiller = distiller
        except Exception as e:
            logger.warning(
                "KnowledgeDistiller init failed (degrading gracefully): %s", e
            )
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
                logger.info(
                    "Resuming continuous session from checkpoint: "
                    "n_experiments=%d, elapsed=%.0fs",
                    n_experiments,
                    elapsed_prior,
                )
                self._emit_event(
                    "checkpoint_resumed",
                    {
                        "experiment_id": resume_id,
                        "n_experiments": n_experiments,
                        "elapsed_seconds": elapsed_prior,
                    },
                )

        # Clean up stale experiments from previous interrupted runs
        try:
            cleanup_nb = self._make_notebook()
            n_cleaned = cleanup_nb.cleanup_stale_experiments()
            if n_cleaned:
                logger.info(f"Cleaned up {n_cleaned} stale running experiments")
            cleanup_nb.close()
        except Exception as e:
            logger.warning("Stale experiment cleanup failed: %s", e)

        # Initialize campaign
        try:
            init_nb = self._make_notebook()
            self._ensure_campaign(config, init_nb)
            init_nb.close()
        except Exception as e:
            logger.warning("Campaign init failed: %s", e)

        while not self._stop_event.is_set():
            self._wait_for_cycle_resume(n_experiments)
            if self._stop_event.is_set():
                break

            # Check for pending heal retry
            if self._pending_heal_retry:
                retry = self._pending_heal_retry
                self._pending_heal_retry = None
                logger.info(
                    "Retrying after successful heal: %s", retry.get("scope", "")[:100]
                )
                try:
                    retry_nb = self._make_notebook()
                    retry_nb.log_learning_event(
                        "heal_retry",
                        f"Retrying after heal: {retry['scope'][:200]}",
                    )
                    retry_nb.close()
                except Exception as e:
                    logger.warning("Heal retry logging failed: %s", e)

            # Check limits before starting next experiment
            stop_reason = self._check_continuous_limits(config, t_start, n_experiments)
            if stop_reason:
                self.aria._continuous_mode = False
                self._end_of_session_automation(
                    config, reason=f"continuous_session_end ({stop_reason})"
                )
                self._set_aria_cycle_phase(
                    "completed",
                    continuous_active=False,
                    cycle_index=n_experiments,
                    note=f"Session ended: {stop_reason}",
                )

                self._update_progress(
                    status="completed",
                    aria_message=f"Session ended: {stop_reason}",
                )
                self._emit_event(
                    "continuous_limit_reached",
                    {
                        "reason": stop_reason,
                        "experiments_completed": n_experiments,
                        "elapsed_minutes": (time.time() - t_start) / 60,
                        "estimated_cost": self.aria.total_cost,
                    },
                )
                # Stop knowledge distiller
                if distiller is not None:
                    try:
                        distiller.stop()
                    except Exception as e:
                        logger.warning("Distiller stop failed at limit: %s", e)
                # Launch queued auto-scale-up
                self._run_pending_scale_up()
                return

            n_experiments += 1
            nb = self._make_notebook()
            self.aria._notebook = nb  # expose for data-driven hypotheses
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
                                f"{stats.get('discovery_validation_correlation'):.2f}"
                                if stats.get("discovery_validation_correlation")
                                is not None
                                else "N/A",
                                stats.get("n_correlation_samples", 0),
                            )
                    except Exception as e:
                        logger.warning(
                            "Failed to generate gate performance summary: %s", e
                        )
            finally:
                nb.close()

            # Notify distiller that a cycle completed
            if distiller is not None:
                try:
                    distiller.notify_cycle_complete()
                except Exception as e:
                    logger.warning("Distiller cycle notification failed: %s", e)

            # Update cost in progress
            self._update_progress(
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
            )

            # Save checkpoint after every checkpoint_interval experiments
            if (
                config.checkpoint_interval > 0
                and n_experiments % config.checkpoint_interval == 0
            ):
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
                    logger.warning("Checkpoint save failed: %s", e)

            # Purge empty failed experiments between cycles to prevent DB bloat.
            try:
                self.notebook.purge_empty_experiments()
                self.notebook.compact_old_chat()
                self.notebook.backfill_failure_signatures()
            except Exception as e:
                logger.warning("Inter-cycle cleanup failed: %s", e)

            if config.rest_between_experiments > 0 and not self._stop_event.is_set():
                time.sleep(config.rest_between_experiments)

        # Stop knowledge distiller
        if distiller is not None:
            try:
                distiller.stop()
            except Exception as e:
                logger.warning("Distiller stop failed: %s", e)

        # Re-enable LLM for interactive use after continuous mode ends.
        self.aria._continuous_mode = False

        # Session ending (user stopped) — auto-report and auto-scale-up
        if n_experiments > 0:
            self._end_of_session_automation(
                config,
                reason=f"continuous_session_stopped (after {n_experiments} experiments)",
            )

        with self._lock:
            elapsed_min = (time.time() - t_start) / 60
            cost_str = (
                f" | Est. cost: ${self.aria.total_cost:.2f}"
                if self.aria.total_cost > 0
                else ""
            )
            self._progress.status = (
                "completed" if not self._stop_event.is_set() else "stopped"
            )
            self._progress.estimated_cost = self.aria.total_cost
            self._progress.total_tokens = self.aria.total_tokens
            self._progress.aria_message = f"Stopped after {n_experiments} experiments ({elapsed_min:.0f}min{cost_str})."
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
                logger.warning("Checkpoint cleanup failed: %s", e)

        # Launch queued auto-scale-up
        self._run_pending_scale_up()

    def _select_next_mode(
        self, config: RunConfig, nb: LabNotebook, n_experiments: int, digest=None
    ) -> Dict:
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
                e.get("best_novelty_score", 0)
                for e in recent
                if e.get("best_novelty_score") is not None
            ]
            # Fallback: if experiments table has no novelty, compute from program_results
            if not novelty_scores:
                try:
                    _nov_rows = nb.conn.execute(
                        "SELECT AVG(novelty_score) as avg_nov FROM program_results "
                        "WHERE novelty_score IS NOT NULL AND stage1_passed = 1"
                    ).fetchone()
                    if _nov_rows and _nov_rows["avg_nov"] is not None:
                        novelty_scores = [float(_nov_rows["avg_nov"])]
                except Exception:
                    pass
            avg_novelty = (
                sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
            )

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

            investigation_ready = len(
                [
                    e
                    for e in leaderboard
                    if e.get("tier") == "screening"
                    and e.get("screening_loss_ratio") is not None
                    and e["screening_loss_ratio"]
                    < config.investigation_loss_ratio_threshold
                    and e.get("result_id") not in _investigated_fps
                    and "provisional_random_tokens" not in (e.get("tags") or "")
                ]
            )
            # More robust: filter by fingerprint
            if _investigated_fps:
                _inv_candidates = []
                for e in leaderboard:
                    if (
                        e.get("tier") == "screening"
                        and e.get("screening_loss_ratio") is not None
                        and e["screening_loss_ratio"]
                        < config.investigation_loss_ratio_threshold
                        and "provisional_random_tokens" not in (e.get("tags") or "")
                    ):
                        # Look up the fingerprint for this result
                        try:
                            fp_row = nb.conn.execute(
                                "SELECT graph_fingerprint FROM program_results WHERE result_id = ?",
                                (e["result_id"],),
                            ).fetchone()
                            fp = fp_row[0] if fp_row else None
                        except Exception:
                            fp = None
                        if fp and fp not in _investigated_fps:
                            _inv_candidates.append(e)
                investigation_ready = len(_inv_candidates)
            validation_ready = len(
                [
                    e
                    for e in leaderboard
                    if e.get("tier") == "investigation"
                    and e.get("investigation_robustness") is not None
                    and e["investigation_robustness"]
                    >= config.investigation_robustness_threshold
                ]
            )

            # Gather richer analytics for data-driven rule-based recommendation
            recent_modes = [e.get("experiment_type", "synthesis") for e in recent]

            # Z16: Composite-score worthiness bar
            # Investigate candidates whose composite_score is competitive with
            # the investigation tier — not just loss leaders.  A model with
            # exceptional efficiency/novelty/stability deserves study even if
            # its loss is only moderate.
            worth_it_investigation = []
            seen_fps: set = set()

            # Dynamic threshold: 25th percentile of investigation tier scores
            inv_scores = sorted(
                [
                    e.get("composite_score")
                    for e in leaderboard
                    if e.get("tier") in ("investigation", "validation")
                    and e.get("composite_score") is not None
                ]
            )
            score_threshold = inv_scores[len(inv_scores) // 4] if inv_scores else 50.0

            # Sort by composite_score descending — best overall candidates first
            sorted_lb = sorted(
                leaderboard, key=lambda x: x.get("composite_score") or 0, reverse=True
            )

            for e in sorted_lb:
                if e.get("tier") != "screening":
                    continue
                score = e.get("composite_score") or 0
                fp = e.get("graph_fingerprint") or e.get("result_id")

                if fp in seen_fps:
                    continue
                seen_fps.add(fp)

                if score >= score_threshold and fp not in _investigated_fps:
                    worth_it_investigation.append(e["result_id"])

            investigation_backlog = len(worth_it_investigation)

            # Enforce exploration-first policy:
            # - First 3 cycles of a session are always synthesis/exploration
            # - After that, at least 40% of recent experiments must be synthesis
            # - Without enough exploration, investigation just re-examines stale data
            synthesis_modes = {"synthesis", "novelty", "evolve"}
            recent_synthesis_count = sum(
                1 for m in recent_modes if m in synthesis_modes
            )
            synthesis_ratio = recent_synthesis_count / max(len(recent_modes), 1)
            exploration_first = n_experiments < 3
            synthesis_starved = exploration_first or synthesis_ratio < 0.4

            if not synthesis_starved:
                # Only force investigation/validation if we have enough synthesis going
                if (
                    investigation_backlog >= 5
                    and "investigation" not in recent_modes[:3]
                ):
                    return {
                        "mode": "investigation",
                        "reasoning": (
                            f"Score-based investigation: {investigation_backlog} candidates with "
                            f"composite_score >= {score_threshold:.1f} (investigation p25). "
                            f"These are competitive across loss, efficiency, novelty, and stability."
                        ),
                        "confidence": 1.0,
                        "config": {"n_programs": min(investigation_backlog, 15)},
                    }

                if validation_ready >= 5 and "validation" not in recent_modes[:3]:
                    return {
                        "mode": "validation",
                        "reasoning": f"Pipeline bottleneck at validation: {validation_ready} candidates ready. Switching to multi-seed verification.",
                        "confidence": 1.0,
                        "config": {"n_programs": min(validation_ready, 10)},
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
            n_compressed_tested = int(
                compression_totals.get("n_compressed_tested") or 0
            )
            n_compressed_survived = int(
                compression_totals.get("n_compressed_survived") or 0
            )
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
                context=context,
                fallback_data=fallback_data,
                digest=digest,
                op_success_rates=analytics_data.get("op_success_rates"),
                compression_coverage=analytics_data.get("compression_coverage"),
            )

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
                    reproduction_steps=[
                        "python -m pytest tests/test_selection_policy.py -x --tb=short"
                    ],
                    acceptance_tests=[
                        "python -m pytest tests/test_selection_policy.py -x --tb=short"
                    ],
                    trigger_payload=trigger,
                )
                if trigger.get("mode") == "novelty":
                    rec["mode"] = "novelty"
                    rec["reasoning"] = (
                        f"{rec.get('reasoning', '')} | Safety valve: {trigger.get('reason')}"
                    ).strip(" |")
                    rec.setdefault("config", {})
                    rec["config"]["n_generations"] = max(4, int(config.n_generations))
                    rec["config"]["population_size"] = max(
                        12, int(config.population_size)
                    )
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
            if refinement_plan and rec.get("mode") not in {
                "investigation",
                "validation",
            }:
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

            nb.add_entry(
                ExperimentEntry(
                    entry_type="decision",
                    title=f"Mode Selection: {rec.get('mode', 'synthesis')}",
                    content=rec.get("reasoning", ""),
                    metadata={
                        "mode": rec.get("mode"),
                        "confidence": rec.get("confidence"),
                        "experiment_number": n_experiments,
                        "evidence_pack": evidence_pack,
                    },
                )
            )

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
                "score_breakdown": [
                    {
                        "mode": rec.get("mode"),
                        "confidence": rec.get("confidence"),
                        "quality_signal": total_s1,
                        "novelty_signal": round(avg_novelty, 6),
                    }
                ],
                "policy": {
                    "engine": "aria_mode_selection_with_refinement",
                    "safety_valve_triggered": bool(trigger),
                    "safety_valve": trigger,
                    "refinement_plan": rec.get("refinement_plan"),
                },
                "reason": rec.get("reasoning", ""),
                "chosen_experiments": [
                    {
                        "mode": rec.get("mode"),
                        "config": rec.get("config", {}),
                    }
                ],
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
                logger.warning("Mode selection decision log failed: %s", log_err)

            return rec
        except Exception as e:
            logger.warning("Mode selection failed, defaulting to synthesis: %s", e)
            return {
                "mode": "synthesis",
                "reasoning": "Fallback",
                "confidence": 0.3,
                "config": {},
            }

    def _end_of_session_automation(self, config: RunConfig, reason: str):
        """Run end-of-session report and scale-up. Used by both limit-reached and user-stop paths."""
        nb = self._make_notebook()
        try:
            self._maybe_auto_report(config, nb, reason=reason)
            cumulative_results = {"stage1_passed": 0, "survivors": []}
            top = nb.get_top_programs(config.auto_scale_up_top_n, sort_by="loss_ratio")
            for p in top:
                if p.get("stage1_passed"):
                    cumulative_results["stage1_passed"] += 1
                    cumulative_results["survivors"].append(
                        {
                            "novelty": p.get("novelty_score", 0),
                        }
                    )
            self._maybe_auto_scale_up(cumulative_results, config, nb)
        except Exception as e:
            logger.warning("End-of-session automation failed: %s", e)
        finally:
            nb.close()
