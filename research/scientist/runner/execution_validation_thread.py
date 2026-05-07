"""Execution validation mixin — split from execution_validation."""

from __future__ import annotations

import time
import traceback
from typing import Dict, List
from ..shared_utils import resolve_device
from ._helpers import finalize_validation_results_summary
from ._types import RunConfig
from .execution_validation import _fail_loud
from ...training.checkpointing import CheckpointManager

import logging

logger = logging.getLogger(__name__)


class _ExecutionValidationThreadMixin:
    """Validation + scale-up threads, thread error handling."""

    __slots__ = ()

    def _run_validation_thread(
        self, exp_id: str, result_ids: List[str], config: RunConfig, hypothesis: str
    ):
        """Execute validation phase in background."""
        self._live_training_context = {"exp_id": exp_id, "phase": "validation"}
        nb = self._make_notebook()
        t_start = time.time()
        ckpt = CheckpointManager(config.checkpoint_dir)

        _outer_phase_index = 0
        _OUTER_TOTAL_PHASES = 5  # baseline, normalized baseline, external evals, leaderboard promotion, trajectory probe

        def _vstatus(phase: str, rid_short: str = "") -> None:
            """Emit validation sub-phase to dashboard + log."""
            nonlocal _outer_phase_index
            _outer_phase_index += 1
            label = (
                f"validation[{rid_short}]: {phase}"
                if rid_short
                else f"validation: {phase}"
            )
            logger.info(label)
            self._emit_event(
                "validation_phase",
                {
                    "experiment_id": exp_id,
                    "result_id": rid_short,
                    "phase": phase,
                    "outer_index": _outer_phase_index,
                    "outer_total": _OUTER_TOTAL_PHASES,
                },
            )
            self._update_progress(status=f"validation: {phase}")

        # Load phase checkpoint to find where we left off
        resume_from_candidate = 0
        ckpt_state = ckpt.load_phase(exp_id, "validation", -1, 0)
        if ckpt_state:
            resume_from_candidate = CheckpointManager.phase_resume_candidate_idx(
                ckpt_state
            )
            logger.info("Resuming validation from candidate %d", resume_from_candidate)

        try:
            results, dev, dev_str, val_config, source_map = (
                self._prepare_validation_state(
                    config=config,
                    result_ids=result_ids,
                    nb=nb,
                )
            )

            for prog_idx, source_result_id in enumerate(result_ids):
                if prog_idx < resume_from_candidate:
                    continue
                if self._stop_event.is_set():
                    break
                _outer_phase_index = 0  # reset per candidate

                self._run_single_validation_candidate(
                    exp_id=exp_id,
                    source_result_id=source_result_id,
                    prog_idx=prog_idx,
                    result_ids=result_ids,
                    config=config,
                    val_config=val_config,
                    dev=dev,
                    dev_str=dev_str,
                    nb=nb,
                    source_map=source_map,
                    results=results,
                    vstatus=_vstatus,
                    ckpt=ckpt,
                    t_start=t_start,
                )

            # Complete experiment
            _vstatus("generating experiment summary")
            finalize_validation_results_summary(results)
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb
            )
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)
            insights = self._analyze_results(results, exp_id, nb, context=context)

            self._publish_terminal_event(
                producer="runner.execution_validation",
                event_type="experiment_completed",
                exp_id=exp_id,
                payload={
                    "completed_at": time.time(),
                    "results": results,
                    "aria_summary": summary,
                    "aria_mood": self.aria.state.mood,
                    "insights": insights,
                    "llm_analysis": llm_analysis,
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

            # Clean up validation checkpoints on success
            if not config.keep_checkpoints:
                try:
                    ckpt.cleanup(exp_id)
                except (OSError, RuntimeError) as exc:
                    _fail_loud(
                        "validation",
                        f"checkpoint cleanup failed for {exp_id[:8]}",
                        exc,
                    )

            self._update_progress(
                status="completed",
                elapsed_seconds=time.time() - t_start,
                aria_message=summary.split("\n")[-1]
                if summary
                else "Validation complete.",
            )

            self._emit_event(
                "validation_completed",
                {
                    "experiment_id": exp_id,
                    "results": results,
                    "summary": summary,
                },
            )

        except Exception as e:
            self._handle_thread_error(phase="validation", exp_id=exp_id, nb=nb, exc=e)
        except BaseException as e:
            self._handle_thread_fatal(phase="validation", exp_id=exp_id, nb=nb, exc=e)
        finally:
            self._live_training_context = None
            nb.close()

    # ── Auto-Escalation Pipeline ──

    def _run_scale_up_thread(
        self, exp_id: str, result_ids: List[str], config: RunConfig, hypothesis: str
    ):
        """Execute scale-up training in background."""
        self._live_training_context = {"exp_id": exp_id, "phase": "scale_up"}
        nb = self._make_notebook()
        t_start = time.time()
        try:
            # graph_from_json already imported at module level
            results = {
                "total": len(result_ids),
                "stage0_passed": 0,
                "stage05_passed": 0,
                "stage1_passed": 0,
                "novel_count": 0,
                "confirmed_count": 0,
                "best_loss_ratio": None,
                "best_novelty_score": None,
                "survivors": [],
            }

            dev = resolve_device(config.device)
            dev_str = str(dev)

            # Create a modified config for scale-up training
            scale_config = config.copy()
            scale_config.stage1_steps = config.scale_up_steps
            scale_config.stage1_batch_size = config.scale_up_batch_size
            scale_config.max_seq_len = config.scale_up_seq_len

            for prog_idx, source_result_id in enumerate(result_ids):
                if self._stop_event.is_set():
                    break

                self._update_progress(
                    current_program=prog_idx + 1,
                    status="training",
                    aria_message=(
                        f"Scale-up {prog_idx + 1}/{len(result_ids)}: "
                        f"training {source_result_id[:8]}... "
                        f"({config.scale_up_steps} steps, batch={config.scale_up_batch_size})"
                    ),
                    elapsed_seconds=time.time() - t_start,
                )

                self._emit_event(
                    "scale_up_progress",
                    {
                        "experiment_id": exp_id,
                        "current_program": prog_idx + 1,
                        "total_programs": len(result_ids),
                        "source_result_id": source_result_id,
                        "status": "starting",
                    },
                )

                self._scale_up_candidate(
                    exp_id=exp_id,
                    source_result_id=source_result_id,
                    prog_idx=prog_idx,
                    total=len(result_ids),
                    config=config,
                    scale_config=scale_config,
                    dev=dev,
                    dev_str=dev_str,
                    nb=nb,
                    results=results,
                )

            if results["stage0_passed"] == 0 and results["total"] > 0:
                self._scale_up_fail_no_results(exp_id, nb, results, result_ids)
                return

            self._scale_up_complete(exp_id, nb, results, config, hypothesis, t_start)

        except Exception as e:
            self._handle_thread_error(phase="scale_up", exp_id=exp_id, nb=nb, exc=e)
        except BaseException as e:
            self._handle_thread_fatal(phase="scale_up", exp_id=exp_id, nb=nb, exc=e)
        finally:
            self._live_training_context = None
            nb.close()

    def _scale_up_fail_no_results(
        self,
        exp_id: str,
        nb,
        results: Dict,
        result_ids: List[str],
    ) -> None:
        reason = (
            f"All {results['total']} source programs were skipped "
            f"(not found or failed to compile). "
            f"Result IDs: {', '.join(r[:12] for r in result_ids)}"
        )
        logger.warning("Scale-up produced no results: %s", reason)
        self._publish_terminal_event(
            producer="runner.execution_validation",
            event_type="experiment_failed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "error": reason,
                "results": None,
            },
        )
        self._fail_experiment_compat(nb=nb, experiment_id=exp_id, error=reason)
        self._update_progress(
            status="failed",
            error=reason,
            aria_message=self.aria.react_to_failure(reason),
        )
        self._emit_event(
            "experiment_failed",
            {"experiment_id": exp_id, "error": reason},
        )

    def _scale_up_complete(
        self,
        exp_id: str,
        nb,
        results: Dict,
        config: RunConfig,
        hypothesis: str,
        t_start: float,
    ) -> None:
        context = self._build_rich_context_for_experiment(
            results, config, hypothesis, nb
        )
        summary = self.aria.experiment_summary(results, context=context)
        llm_analysis = self.aria.analyze_results(results, context=context)
        insights = self._analyze_results(results, exp_id, nb, context=context)

        self._publish_terminal_event(
            producer="runner.execution_validation",
            event_type="experiment_completed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "results": results,
                "aria_summary": summary,
                "aria_mood": self.aria.state.mood,
                "insights": insights,
                "llm_analysis": llm_analysis,
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
        self._auto_recommend(results, config, hypothesis, nb)
        self._update_progress(
            status="completed",
            elapsed_seconds=time.time() - t_start,
            aria_message=(summary.split("\n")[-1] if summary else "Scale-up complete."),
        )
        self._emit_event(
            "scale_up_completed",
            {"experiment_id": exp_id, "results": results, "summary": summary},
        )

    # ── Shared error handling for thread methods ──

    def _handle_thread_error(self, phase: str, exp_id: str, nb, exc: Exception) -> None:
        """Handle recoverable Exception in a background thread."""
        error = traceback.format_exc()
        logger.error("%s failed (%s): %s\n%s", phase.title(), exp_id, exc, error)
        try:
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"{phase.title()} failure: {str(exc)[:240]}",
                reproduction_steps=[
                    f'python -m pytest tests/test_integration.py -k "{phase}" -x --tb=short'
                ],
                acceptance_tests=[
                    f'python -m pytest tests/test_integration.py -k "{phase}" -x --tb=short'
                ],
                trigger_payload={"mode": phase, "error": str(exc)},
            )
        except (RuntimeError, OSError) as heal_err:
            logger.warning(
                "code_healer failed during %s error handling: %s",
                phase,
                heal_err,
                exc_info=True,
            )
        self._publish_terminal_event(
            producer="runner.execution_validation",
            event_type="experiment_failed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "error": str(exc),
                "results": None,
                "phase": phase,
            },
        )
        self._fail_experiment_compat(
            nb=nb,
            experiment_id=exp_id,
            error=str(exc),
        )
        self._update_progress(
            status="failed",
            error=str(exc),
            aria_message=self.aria.react_to_failure(str(exc)),
        )
        self._emit_event(
            "experiment_failed",
            {"experiment_id": exp_id, "error": str(exc)},
        )

    def _handle_thread_fatal(
        self, phase: str, exp_id: str, nb, exc: BaseException
    ) -> None:
        """Handle fatal BaseException (CUDA errors, KeyboardInterrupt, etc.)."""
        logger.critical(
            "%s thread KILLED (%s): %s\n%s",
            phase.title(),
            exp_id,
            exc,
            traceback.format_exc(),
        )
        try:
            self._publish_terminal_event(
                producer="runner.execution_validation",
                event_type="experiment_failed",
                exp_id=exp_id,
                payload={
                    "completed_at": time.time(),
                    "error": f"FATAL: {exc}",
                    "results": None,
                    "phase": phase,
                    "fatal": True,
                },
            )
            self._fail_experiment_compat(
                nb=nb,
                experiment_id=exp_id,
                error=f"FATAL: {exc}",
            )
            self._update_progress(status="failed", error=f"FATAL: {exc}")
            self._emit_event(
                "experiment_failed",
                {"experiment_id": exp_id, "error": f"FATAL: {exc}"},
            )
        except RuntimeError:
            logger.error(
                "Failed to emit failure event after fatal error", exc_info=True
            )
        raise

    # ── Extracted helpers for _run_validation_thread ──
