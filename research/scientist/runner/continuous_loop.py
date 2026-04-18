"""Continuous loop thread and mode selection, split from continuous.py."""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Any, Dict, List


from ...training.checkpointing import CheckpointManager
from ..notebook import LabNotebook, ExperimentEntry
from ..runtime_events import publish_runtime_event
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

    def _log_learning_event_compat(self, nb: LabNotebook, *args, **kwargs) -> None:
        getattr(nb, "log_learning_event")(*args, **kwargs)

    def _publish_continuous_session_event(
        self,
        *,
        event_type: str,
        run_id: str | None,
        payload: dict,
    ) -> None:
        publish_runtime_event(
            notebook_path=self.notebook_path,
            event_type=event_type,
            producer="runner.continuous_loop",
            run_id=run_id,
            payload=payload,
        )

    def _run_continuous_thread(self, config: RunConfig):
        """Execute continuous experiments in background."""
        try:
            self._run_continuous_thread_inner(config)
        except BaseException as e:
            import traceback

            logger.critical(
                "Continuous thread KILLED: %s\n%s",
                e,
                traceback.format_exc(),
            )
            session_run_id = str(
                getattr(config, "resume_experiment_id", "") or "continuous"
            )
            self._publish_continuous_session_event(
                event_type="continuous_session_failed",
                run_id=session_run_id,
                payload={
                    "error": f"FATAL: {e}",
                    "mode": "continuous",
                    "failed_at": time.time(),
                },
            )
            try:
                self._update_progress(status="failed", error=f"FATAL: {e}")
                self._set_aria_cycle_phase(
                    "failed",
                    continuous_active=False,
                    note=f"Continuous session failed: {e}",
                )
                self._emit_event(
                    "continuous_session_failed",
                    {"session_id": session_run_id, "error": f"FATAL: {e}"},
                )
            except RuntimeError:
                logger.error(
                    "Failed to emit failure event after fatal error", exc_info=True
                )
            if not isinstance(e, Exception):
                raise

    def _backfill_continuous_replication(self) -> None:
        try:
            init_nb = self._make_notebook()
            n_backfilled = init_nb.backfill_replication_aggregates()
            if n_backfilled:
                logger.info(
                    "Backfilled replication data on %d leaderboard entries",
                    n_backfilled,
                )
            init_nb.close()
        except (RuntimeError, sqlite3.OperationalError) as e:
            logger.warning("Replication backfill failed: %s", e)

    def _start_knowledge_distiller(self):
        distiller = None
        try:
            from ..intelligence.distiller import KnowledgeDistiller
            from ..intelligence.digest import ExperimentDigest

            distiller = KnowledgeDistiller(
                db_path=self.notebook_path,
                distill_interval_cycles=3,
            )
            try:
                init_nb = self._make_notebook()
                saved = init_nb.get_latest_digest()
                if saved:
                    distiller.set_digest(ExperimentDigest.from_dict(saved))
                    logger.info("Recovered knowledge digest from DB")
                init_nb.close()
            except (RuntimeError, sqlite3.OperationalError, KeyError, ValueError) as e:
                logger.warning("Digest recovery failed: %s", e)
            distiller.start()
            self._knowledge_distiller = distiller
        except (ImportError, RuntimeError, OSError) as e:
            logger.warning(
                "KnowledgeDistiller init failed (degrading gracefully): %s", e
            )
            distiller = None
            self._knowledge_distiller = None
        return distiller

    def _resume_continuous_checkpoint(
        self, ckpt: CheckpointManager, resume_id: str | None
    ):
        n_experiments = 0
        t_start = time.time()
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
        return n_experiments, t_start

    def _run_continuous_startup_maintenance(self, config: RunConfig) -> None:
        try:
            cleanup_nb = self._make_notebook()
            n_cleaned = cleanup_nb.cleanup_stale_experiments()
            if n_cleaned:
                logger.info("Cleaned up %d stale running experiments", n_cleaned)
            cleanup_nb.close()
        except (RuntimeError, sqlite3.OperationalError) as e:
            logger.warning("Stale experiment cleanup failed: %s", e)

        try:
            sweep_nb = self._make_notebook()
            n_queued = self.sweep_backfill_candidates(config, sweep_nb)
            if n_queued:
                logger.info(
                    "Startup backfill sweep: queued %d candidates for investigation",
                    n_queued,
                )
            sweep_nb.close()
        except (RuntimeError, sqlite3.OperationalError) as e:
            logger.warning("Startup backfill sweep failed: %s", e)

        try:
            init_nb = self._make_notebook()
            self._ensure_campaign(config, init_nb)
            init_nb.close()
        except (RuntimeError, sqlite3.OperationalError, ValueError) as e:
            logger.warning("Campaign init failed: %s", e)

    def _maybe_log_pending_heal_retry(self) -> None:
        if not self._pending_heal_retry:
            return
        retry = self._pending_heal_retry
        self._pending_heal_retry = None
        logger.info("Retrying after successful heal: %s", retry.get("scope", "")[:100])
        try:
            retry_nb = self._make_notebook()
            self._log_learning_event_compat(
                retry_nb,
                "heal_retry",
                f"Retrying after heal: {retry['scope'][:200]}",
            )
            retry_nb.close()
        except (RuntimeError, sqlite3.OperationalError) as e:
            logger.warning("Heal retry logging failed: %s", e)

    def _try_make_notebook(self, *, purpose: str) -> LabNotebook | None:
        try:
            return self._make_notebook()
        except (RuntimeError, sqlite3.OperationalError) as e:
            logger.warning("%s skipped: %s", purpose, e)
            return None

    def _handle_continuous_limit_reached(
        self,
        *,
        config: RunConfig,
        stop_reason: str,
        n_experiments: int,
        t_start: float,
        distiller,
    ) -> None:
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
        self._publish_continuous_session_event(
            event_type="continuous_session_completed",
            run_id=str(getattr(config, "resume_experiment_id", "") or "continuous"),
            payload={
                "reason": stop_reason,
                "experiments_completed": n_experiments,
                "elapsed_minutes": (time.time() - t_start) / 60,
                "estimated_cost": self.aria.total_cost,
                "mode": "continuous",
                "completed_at": time.time(),
            },
        )
        if distiller is not None:
            try:
                distiller.stop()
            except RuntimeError as e:
                logger.warning("Distiller stop failed at limit: %s", e)
        self._run_pending_scale_up()

    def _run_periodic_cycle_maintenance(
        self, config: RunConfig, nb: LabNotebook, n_experiments: int
    ) -> None:
        if n_experiments % 5 != 0:
            return
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
                    if stats.get("discovery_validation_correlation") is not None
                    else "N/A",
                    stats.get("n_correlation_samples", 0),
                )
        except (ImportError, RuntimeError, sqlite3.OperationalError) as e:
            logger.warning("Failed to generate gate performance summary: %s", e)

        try:
            n_queued = self.sweep_backfill_candidates(config, nb)
            if n_queued:
                logger.info(
                    "Periodic backfill sweep (cycle %d): queued %d for investigation",
                    n_experiments,
                    n_queued,
                )
        except (RuntimeError, sqlite3.OperationalError) as e:
            logger.warning("Periodic backfill sweep failed: %s", e)

    def _run_continuous_cycle(
        self, config: RunConfig, n_experiments: int, t_start: float
    ) -> None:
        nb = self._make_notebook()
        self.aria._notebook = nb
        try:
            self.run_aria_cycle(config, nb, n_experiments, t_start)
            self._run_periodic_cycle_maintenance(config, nb, n_experiments)
        finally:
            nb.close()

    def _post_cycle_continuous_maintenance(
        self,
        *,
        config: RunConfig,
        ckpt: CheckpointManager,
        resume_id: str | None,
        n_experiments: int,
        t_start: float,
        distiller,
    ) -> None:
        if distiller is not None:
            try:
                distiller.notify_cycle_complete()
            except RuntimeError as e:
                logger.warning("Distiller cycle notification failed: %s", e)

        self._update_progress(
            estimated_cost=self.aria.total_cost,
            total_tokens=self.aria.total_tokens,
        )

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
            except (OSError, RuntimeError) as e:
                logger.warning("Checkpoint save failed: %s", e)

        try:
            cleanup_nb = self._try_make_notebook(purpose="Inter-cycle cleanup")
            if cleanup_nb is not None:
                try:
                    cleanup_nb.purge_empty_experiments()
                    cleanup_nb.compact_old_chat()
                    cleanup_nb.backfill_failure_signatures()
                except (RuntimeError, sqlite3.OperationalError) as e:
                    logger.warning("Inter-cycle cleanup failed: %s", e)
                finally:
                    cleanup_nb.close()
        except Exception as e:
            logger.warning("Inter-cycle cleanup failed unexpectedly: %s", e)

    def _finish_continuous_run(
        self,
        *,
        config: RunConfig,
        ckpt: CheckpointManager,
        resume_id: str | None,
        n_experiments: int,
        t_start: float,
        distiller,
    ) -> None:
        if distiller is not None:
            try:
                distiller.stop()
            except RuntimeError as e:
                logger.warning("Distiller stop failed: %s", e)

        self.aria._continuous_mode = False
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
            self._progress.aria_message = (
                f"Stopped after {n_experiments} experiments "
                f"({elapsed_min:.0f}min{cost_str})."
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
        if self._stop_event.is_set():
            self._publish_continuous_session_event(
                event_type="continuous_session_stopped",
                run_id=str(resume_id or "continuous"),
                payload={
                    "experiments_completed": n_experiments,
                    "elapsed_minutes": elapsed_min,
                    "estimated_cost": self.aria.total_cost,
                    "mode": "continuous",
                    "stopped_at": time.time(),
                },
            )

        if not self._stop_event.is_set() and not config.keep_checkpoints:
            try:
                ckpt.cleanup(resume_id or "continuous")
            except (OSError, RuntimeError) as e:
                logger.warning("Checkpoint cleanup failed: %s", e)
        self._run_pending_scale_up()

    @staticmethod
    def _build_cumulative_scale_up_results(
        top_programs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        survivors = [
            {"novelty": program.get("novelty_score", 0)}
            for program in top_programs
            if program.get("stage1_passed")
        ]
        return {
            "stage1_passed": len(survivors),
            "survivors": survivors,
        }

    def _compute_avg_recent_novelty(
        self, nb: LabNotebook, recent: List[Dict[str, Any]]
    ) -> float:
        novelty_scores = [
            e.get("best_novelty_score", 0)
            for e in recent
            if e.get("best_novelty_score") is not None
        ]
        if not novelty_scores:
            try:
                row = nb.conn.execute(
                    "SELECT AVG(novelty_score) as avg_nov FROM program_results "
                    "WHERE novelty_score IS NOT NULL AND stage1_passed = 1"
                ).fetchone()
                if row and row["avg_nov"] is not None:
                    novelty_scores = [float(row["avg_nov"])]
            except (sqlite3.OperationalError, ValueError, TypeError) as e:
                logger.debug("Novelty fallback query failed: %s", e)
        return sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

    def _fetch_investigated_fingerprints(self, nb: LabNotebook) -> set[str]:
        try:
            rows = nb.conn.execute(
                "SELECT DISTINCT pr.graph_fingerprint "
                "FROM program_results pr "
                "JOIN experiments e ON e.experiment_id = pr.experiment_id "
                "WHERE e.experiment_type = 'investigation'"
            ).fetchall()
            return {row[0] for row in rows if row[0]}
        except sqlite3.OperationalError as e:
            logger.debug("Investigated fingerprint query failed: %s", e)
            return set()

    def _count_investigation_ready_candidates(
        self,
        *,
        nb: LabNotebook,
        leaderboard: List[Dict[str, Any]],
        investigated_fps: set[str],
        threshold: float,
    ) -> int:
        ready_rows = [
            row
            for row in leaderboard
            if row.get("tier") == "screening"
            and row.get("screening_loss_ratio") is not None
            and row["screening_loss_ratio"] < threshold
            and "provisional_random_tokens" not in (row.get("tags") or "")
        ]
        if not investigated_fps:
            return len(
                [
                    row
                    for row in ready_rows
                    if row.get("result_id") not in investigated_fps
                ]
            )

        count = 0
        for row in ready_rows:
            try:
                fp_row = nb.conn.execute(
                    "SELECT graph_fingerprint FROM program_results WHERE result_id = ?",
                    (row["result_id"],),
                ).fetchone()
            except sqlite3.OperationalError as e:
                logger.debug("Fingerprint lookup failed: %s", e)
                continue
            fp = fp_row[0] if fp_row else None
            if fp and fp not in investigated_fps:
                count += 1
        return count

    @staticmethod
    def _count_validation_ready_candidates(
        leaderboard: List[Dict[str, Any]], threshold: float
    ) -> int:
        return len(
            [
                row
                for row in leaderboard
                if row.get("tier") == "investigation"
                and row.get("investigation_robustness") is not None
                and row["investigation_robustness"] >= threshold
            ]
        )

    @staticmethod
    def _compute_investigation_backlog(
        leaderboard: List[Dict[str, Any]],
        investigated_fps: set[str],
    ) -> tuple[int, float]:
        inv_scores = sorted(
            [
                row.get("composite_score")
                for row in leaderboard
                if row.get("tier") in ("investigation", "validation")
                and row.get("composite_score") is not None
            ]
        )
        score_threshold = inv_scores[len(inv_scores) // 4] if inv_scores else 50.0
        seen_fps: set[str] = set()
        backlog = 0
        for row in sorted(
            leaderboard, key=lambda item: item.get("composite_score") or 0, reverse=True
        ):
            if row.get("tier") != "screening":
                continue
            fp = row.get("graph_fingerprint") or row.get("result_id")
            if fp in seen_fps:
                continue
            seen_fps.add(fp)
            if (
                row.get("composite_score") or 0
            ) >= score_threshold and fp not in investigated_fps:
                backlog += 1
        return backlog, score_threshold

    @staticmethod
    def _collect_optimizer_counts(nb: LabNotebook) -> Dict[str, int]:
        optimizer_counts: Dict[str, int] = {}
        try:
            rows = nb.conn.execute(
                "SELECT optimizer_name, COUNT(*) as cnt "
                "FROM program_results WHERE optimizer_name IS NOT NULL "
                "GROUP BY optimizer_name"
            ).fetchall()
            for row in rows:
                optimizer_counts[row[0]] = row[1]
        except sqlite3.OperationalError as e:
            logger.debug(
                "Optimizer diversity query failed (table may not exist): %s", e
            )
        return optimizer_counts

    def _build_mode_selection_state(
        self,
        *,
        config: RunConfig,
        nb: LabNotebook,
        n_experiments: int,
        digest,
    ) -> Dict[str, Any]:
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
        investigated_fps = self._fetch_investigated_fingerprints(nb)
        avg_novelty = self._compute_avg_recent_novelty(nb, recent)
        investigation_ready = self._count_investigation_ready_candidates(
            nb=nb,
            leaderboard=leaderboard,
            investigated_fps=investigated_fps,
            threshold=config.investigation_loss_ratio_threshold,
        )
        validation_ready = self._count_validation_ready_candidates(
            leaderboard, config.investigation_robustness_threshold
        )
        investigation_backlog, score_threshold = self._compute_investigation_backlog(
            leaderboard, investigated_fps
        )
        recent_modes = [e.get("experiment_type", "synthesis") for e in recent]
        recent_failures = [e for e in recent if e.get("status") == "failed"]
        unique_fingerprints = {
            str(e.get("graph_fingerprint") or "")[:8]
            for e in leaderboard
            if e.get("graph_fingerprint")
        }
        optimizer_counts = self._collect_optimizer_counts(nb)
        total_s1 = sum(e.get("n_stage1_passed", 0) for e in recent)
        return {
            "recent": recent,
            "leaderboard": leaderboard,
            "analytics_data": analytics_data,
            "context": context,
            "avg_novelty": avg_novelty,
            "investigation_ready": investigation_ready,
            "validation_ready": validation_ready,
            "investigation_backlog": investigation_backlog,
            "score_threshold": score_threshold,
            "recent_modes": recent_modes,
            "recent_failures": recent_failures,
            "leaderboard_diversity": len(unique_fingerprints),
            "optimizer_counts": optimizer_counts,
            "total_s1": total_s1,
            "n_experiments": n_experiments,
        }

    @staticmethod
    def _forced_pipeline_mode(selection_state: Dict[str, Any]) -> Dict[str, Any] | None:
        recent_modes = selection_state["recent_modes"]
        synthesis_modes = {"synthesis", "novelty", "evolve"}
        recent_synthesis_count = sum(
            1 for mode in recent_modes if mode in synthesis_modes
        )
        synthesis_ratio = recent_synthesis_count / max(len(recent_modes), 1)
        synthesis_starved = (
            selection_state["n_experiments"] < 3 or synthesis_ratio < 0.4
        )
        if synthesis_starved:
            return None
        if (
            selection_state["investigation_backlog"] >= 5
            and "investigation" not in recent_modes[:3]
        ):
            return {
                "mode": "investigation",
                "reasoning": (
                    f"Score-based investigation: {selection_state['investigation_backlog']} candidates with "
                    f"composite_score >= {selection_state['score_threshold']:.1f} (investigation p25). "
                    "These are competitive across loss, efficiency, novelty, and stability."
                ),
                "confidence": 1.0,
                "config": {
                    "n_programs": min(selection_state["investigation_backlog"], 15)
                },
            }
        if (
            selection_state["validation_ready"] >= 5
            and "validation" not in recent_modes[:3]
        ):
            return {
                "mode": "validation",
                "reasoning": (
                    f"Pipeline bottleneck at validation: {selection_state['validation_ready']} "
                    "candidates ready. Switching to multi-seed verification."
                ),
                "confidence": 1.0,
                "config": {"n_programs": min(selection_state["validation_ready"], 10)},
            }
        return None

    def _build_mode_selection_fallback_data(
        self, config: RunConfig, selection_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        analytics_data = selection_state["analytics_data"]
        compression_coverage = analytics_data.get("compression_coverage") or {}
        compression_totals = compression_coverage.get("totals") or {}
        n_tested = int(compression_totals.get("n_tested") or 0)
        n_compressed_tested = int(compression_totals.get("n_compressed_tested") or 0)
        n_compressed_survived = int(
            compression_totals.get("n_compressed_survived") or 0
        )
        n_survived = int(compression_totals.get("n_survived") or 0)
        compressed_test_share = n_compressed_tested / n_tested if n_tested > 0 else 0.0
        return {
            "total_s1_survivors": selection_state["total_s1"],
            "avg_novelty": selection_state["avg_novelty"],
            "n_experiments_in_session": selection_state["n_experiments"],
            "base_n_programs": config.n_programs,
            "investigation_ready": selection_state["investigation_ready"],
            "validation_ready": selection_state["validation_ready"],
            "analytics_data": analytics_data,
            "recent_modes": selection_state["recent_modes"],
            "recent_failure_count": len(selection_state["recent_failures"]),
            "leaderboard_diversity": selection_state["leaderboard_diversity"],
            "leaderboard_size": len(selection_state["leaderboard"]),
            "optimizer_counts": selection_state["optimizer_counts"],
            "optimizer_diversity": len(selection_state["optimizer_counts"]),
            "compressed_test_share": compressed_test_share,
            "compression_summary": {
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
            },
        }

    def _apply_mode_selection_safety_valve(
        self, rec: Dict[str, Any], nb: LabNotebook, config: RunConfig
    ) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
        trigger = self._selection_safety_valve(nb, config)
        if not (trigger and trigger.get("triggered")):
            return rec, trigger
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
        return rec, trigger

    def _apply_mode_selection_refinement(
        self, rec: Dict[str, Any], nb: LabNotebook, config: RunConfig
    ) -> Dict[str, Any]:
        refinement_plan = self._build_refinement_plan(nb, config)
        if not refinement_plan or rec.get("mode") in {"investigation", "validation"}:
            return rec
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
        return rec

    @staticmethod
    def _apply_periodic_refinement_override(
        rec: Dict[str, Any], n_experiments: int
    ) -> Dict[str, Any]:
        """Force refinement mode every 5th experiment.

        Periodically re-exploiting proven winners prevents the pipeline from
        spending too long in pure exploration. Only triggers when the current
        mode is synthesis/novelty/evolve (never overrides investigation or
        validation).
        """
        if n_experiments > 0 and (n_experiments + 1) % 5 == 0:
            if rec.get("mode") not in {"investigation", "validation", "refinement"}:
                rec["mode"] = "refinement"
                rec["reasoning"] = (
                    f"Periodic refinement trigger (experiment {n_experiments + 1}, "
                    f"every 5th). Re-exploiting Stage-1 winners. "
                    f"(Original: {rec.get('reasoning', '')})"
                )
                rec["confidence"] = max(float(rec.get("confidence", 0.5) or 0.5), 0.75)
        return rec

    @staticmethod
    def _apply_exploration_first_override(
        rec: Dict[str, Any], n_experiments: int
    ) -> Dict[str, Any]:
        if n_experiments < 3 and rec.get("mode") in ("investigation", "validation"):
            rec["mode"] = "synthesis"
            rec["reasoning"] = (
                f"Exploration-first: cycle {n_experiments + 1}/3. "
                "Building data foundation before investigation. "
                f"(Original recommendation: {rec.get('reasoning', '')})"
            )
            rec["confidence"] = 0.8
        return rec

    def _record_mode_selection(
        self,
        nb: LabNotebook,
        rec: Dict[str, Any],
        selection_state: Dict[str, Any],
        trigger,
    ) -> None:
        evidence_pack = build_evidence_pack(
            nb,
            analytics=None,
            recommendation=rec,
            decision_type="mode_selection",
            recent_experiments=selection_state["recent"],
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
                    "experiment_number": selection_state["n_experiments"],
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
                "recent_experiments": len(selection_state["recent"]),
                "leaderboard_candidates": len(selection_state["leaderboard"]),
                "total_s1_survivors": selection_state["total_s1"],
                "avg_novelty": round(selection_state["avg_novelty"], 6),
            },
            "score_breakdown": [
                {
                    "mode": rec.get("mode"),
                    "confidence": rec.get("confidence"),
                    "quality_signal": selection_state["total_s1"],
                    "novelty_signal": round(selection_state["avg_novelty"], 6),
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
                {"mode": rec.get("mode"), "config": rec.get("config", {})}
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
        except (ValueError, sqlite3.OperationalError) as log_err:
            logger.warning("Mode selection decision log failed: %s", log_err)

    def _run_continuous_thread_inner(self, config: RunConfig):
        """Inner continuous loop body."""
        n_experiments = 0
        t_start = time.time()
        self.aria.reset_cost_tracking()
        self.aria._continuous_mode = True
        self.aria._llm_decision_interval = config.llm_decision_interval
        self._backfill_continuous_replication()
        distiller = self._start_knowledge_distiller()
        self._set_aria_cycle_phase(
            "planning",
            continuous_active=True,
            cycle_index=0,
            selected_mode=None,
            note="Preparing continuous research loop.",
        )
        ckpt = CheckpointManager(config.checkpoint_dir)
        resume_id = config.resume_experiment_id
        n_experiments, t_start = self._resume_continuous_checkpoint(ckpt, resume_id)
        self._run_continuous_startup_maintenance(config)

        while not self._stop_event.is_set():
            self._wait_for_cycle_resume(n_experiments)
            if self._stop_event.is_set():
                break
            self._maybe_log_pending_heal_retry()
            stop_reason = self._check_continuous_limits(config, t_start, n_experiments)
            if stop_reason:
                self._handle_continuous_limit_reached(
                    config=config,
                    stop_reason=stop_reason,
                    n_experiments=n_experiments,
                    t_start=t_start,
                    distiller=distiller,
                )
                return

            n_experiments += 1
            self._run_continuous_cycle(config, n_experiments, t_start)
            self._post_cycle_continuous_maintenance(
                config=config,
                ckpt=ckpt,
                resume_id=resume_id,
                n_experiments=n_experiments,
                t_start=t_start,
                distiller=distiller,
            )

            if config.rest_between_experiments > 0 and not self._stop_event.is_set():
                time.sleep(config.rest_between_experiments)

        self._finish_continuous_run(
            config=config,
            ckpt=ckpt,
            resume_id=resume_id,
            n_experiments=n_experiments,
            t_start=t_start,
            distiller=distiller,
        )

    def _select_next_mode(
        self, config: RunConfig, nb: LabNotebook, n_experiments: int, digest=None
    ) -> Dict:
        """Have Aria decide the next experiment mode."""
        try:
            selection_state = self._build_mode_selection_state(
                config=config,
                nb=nb,
                n_experiments=n_experiments,
                digest=digest,
            )
            forced = self._forced_pipeline_mode(selection_state)
            if forced is not None:
                return forced
            fallback_data = self._build_mode_selection_fallback_data(
                config, selection_state
            )
            rec = self.aria.recommend_next_mode(
                context=selection_state["context"],
                fallback_data=fallback_data,
                digest=digest,
                op_success_rates=selection_state["analytics_data"].get(
                    "op_success_rates"
                ),
                compression_coverage=selection_state["analytics_data"].get(
                    "compression_coverage"
                ),
            )
            compression_override = self._compression_focus_override(rec, fallback_data)
            if compression_override is not None:
                rec = compression_override
            rec, trigger = self._apply_mode_selection_safety_valve(rec, nb, config)
            rec = self._apply_mode_selection_refinement(rec, nb, config)
            rec = self._apply_periodic_refinement_override(rec, n_experiments)
            rec = self._apply_exploration_first_override(rec, n_experiments)
            self._record_mode_selection(nb, rec, selection_state, trigger)
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
        nb = self._try_make_notebook(purpose="End-of-session automation")
        if nb is None:
            return
        try:
            self._maybe_auto_report(config, nb, reason=reason)
            top = nb.get_top_programs(config.auto_scale_up_top_n, sort_by="loss_ratio")
            cumulative_results = self._build_cumulative_scale_up_results(top)
            self._maybe_auto_scale_up(cumulative_results, config, nb)
        except Exception as e:
            logger.warning("End-of-session automation failed: %s", e)
        finally:
            nb.close()
