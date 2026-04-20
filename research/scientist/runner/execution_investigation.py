"""Execution mixin: investigation thread."""

from __future__ import annotations

import json
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch
from ...synthesis.serializer import graph_from_json
from ...training.training_program import synthesize_training_program_batch
from ...training.checkpointing import CheckpointManager
from ..native_runner import compile_model_native_first as compile_model
from ..shared_utils import resolve_device
from ._helpers import (
    _build_source_map,
    _record_investigation_result,
    _submit_benchmark_eval,
    clear_gpu_memory,
)
from .execution_investigation_scoring import (
    build_investigation_entry,
    summarize_investigation_program_runs,
    InvestigationProgramSummary,
)

import logging

logger = logging.getLogger(__name__)

from ._types import RunConfig

# Sentinel: _process_investigation_candidate returns this when all
# training programs for a candidate hit infrastructure errors.
_SKIP_INFRA = object()


def _fail_loud(phase: str, message: str, exc: BaseException) -> None:
    logger.exception("%s: %s", phase, message)
    raise RuntimeError(f"{phase}: {message}") from exc


class _ExecutionInvestigationMixin:
    """Investigation phase execution."""

    __slots__ = ()

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def _run_investigation_thread(
        self, exp_id: str, result_ids: List[str], config: RunConfig, hypothesis: str
    ):
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
                        (rid,),
                    ).fetchone()
                    if row and row[0] is not None:
                        logger.info(
                            "Investigation candidate %s pre_inv_score=%.1f",
                            rid[:8],
                            row[0],
                        )
                except Exception as exc:
                    _fail_loud(
                        "investigation",
                        f"failed to read pre-inv score for {rid[:8]}",
                        exc,
                    )

        # Load phase checkpoint to find where we left off
        resume_from_candidate = 0
        ckpt_state = ckpt.load_phase(exp_id, "investigation", -1, 0)
        if ckpt_state:
            resume_from_candidate = CheckpointManager.phase_resume_candidate_idx(
                ckpt_state
            )
            logger.info(
                "Resuming investigation from candidate %d", resume_from_candidate
            )

        try:
            self._execute_investigation_loop(
                exp_id=exp_id,
                result_ids=result_ids,
                config=config,
                hypothesis=hypothesis,
                resume_from_candidate=resume_from_candidate,
                nb=nb,
                ckpt=ckpt,
                t_start=t_start,
            )
        except RuntimeError as e:
            error = traceback.format_exc()
            logger.error("Investigation failed (%s): %s\n%s", exp_id, e, error)
            try:
                self._invoke_code_healer(
                    nb=nb,
                    trigger_type="repeated_exception",
                    experiment_id=exp_id,
                    scope=f"Investigation failure: {str(e)[:240]}",
                    reproduction_steps=[
                        'python -m pytest tests/test_integration.py -k "investigation" -x --tb=short'
                    ],
                    acceptance_tests=[
                        'python -m pytest tests/test_integration.py -k "investigation" -x --tb=short'
                    ],
                    trigger_payload={"mode": "investigation", "error": str(e)},
                )
            except Exception:  # noqa: BLE001 — error-path guardrail, must not raise
                logger.warning(
                    "code_healer failed during investigation error handling",
                    exc_info=True,
                )
            self._publish_terminal_event(
                producer="runner.execution_investigation",
                event_type="experiment_failed",
                exp_id=exp_id,
                payload={
                    "completed_at": time.time(),
                    "error": str(e),
                    "results": None,
                    "mode": "investigation",
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
                {"experiment_id": exp_id, "error": str(e)},
            )
        except BaseException as e:
            logger.critical(
                "Investigation thread KILLED (%s): %s\n%s",
                exp_id,
                e,
                traceback.format_exc(),
            )
            try:
                self._publish_terminal_event(
                    producer="runner.execution_investigation",
                    event_type="experiment_failed",
                    exp_id=exp_id,
                    payload={
                        "completed_at": time.time(),
                        "error": f"FATAL: {e}",
                        "results": None,
                        "mode": "investigation",
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
            except Exception:  # noqa: BLE001 — error-path guardrail, must not raise
                logger.error(
                    "Failed to emit failure event after fatal error", exc_info=True
                )
            raise
        finally:
            self._live_training_context = None
            nb.close()
            self._run_pending_scale_up()

    # ------------------------------------------------------------------
    # Build investigation-specific config
    # ------------------------------------------------------------------

    def _build_investigation_config(self, config: RunConfig) -> RunConfig:
        """Build a config copy with investigation-specific overrides."""
        inv_config = config.copy()
        inv_config.stage1_steps = config.investigation_steps
        inv_config.stage1_batch_size = config.investigation_batch_size
        # Scale early stopping for longer investigation runs.
        # Default patience (300) is calibrated for 500-step screening;
        # without scaling, investigation stops at ~step 400 (16% of 2500).
        step_ratio = config.investigation_steps / max(config.stage1_steps, 1)
        inv_config.early_stop_patience = int(config.early_stop_patience * step_ratio)
        inv_config.early_stop_min_steps = int(config.early_stop_min_steps * step_ratio)
        return inv_config

    # ------------------------------------------------------------------
    # Main investigation loop (happy path)
    # ------------------------------------------------------------------

    def _execute_investigation_loop(
        self,
        *,
        exp_id: str,
        result_ids: List[str],
        config: RunConfig,
        hypothesis: str,
        resume_from_candidate: int,
        nb: Any,
        ckpt: CheckpointManager,
        t_start: float,
    ) -> None:
        """Run the candidate loop, detect infra failure, complete experiment."""
        results: Dict[str, Any] = {
            "total": len(result_ids),
            "stage0_passed": 0,
            "stage05_passed": 0,
            "stage1_passed": 0,
            "novel_count": 0,
            "best_loss_ratio": None,
            "best_novelty_score": None,
            "survivors": [],
            "investigation_results": [],
        }

        dev = resolve_device(config.device)
        str(dev)

        inv_config = self._build_investigation_config(config)
        source_map = _build_source_map(nb, result_ids)

        for prog_idx, source_result_id in enumerate(result_ids):
            if prog_idx < resume_from_candidate:
                continue
            if self._stop_event.is_set():
                break

            self._update_progress(
                current_program=prog_idx + 1,
                status="investigating",
                aria_message=(
                    f"Investigating {prog_idx + 1}/{len(result_ids)}: "
                    f"{source_result_id[:8]}... "
                    f"({config.n_training_programs} training programs)"
                ),
                elapsed_seconds=time.time() - t_start,
            )
            self._emit_event(
                "investigation_progress",
                {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "starting",
                },
            )

            source = source_map.get(source_result_id)
            if source is None:
                continue

            status = self._process_investigation_candidate(
                exp_id=exp_id,
                source_result_id=source_result_id,
                source=source,
                config=config,
                inv_config=inv_config,
                dev=dev,
                prog_idx=prog_idx,
                total_candidates=len(result_ids),
                results=results,
                nb=nb,
                ckpt=ckpt,
            )
            if status is _SKIP_INFRA:
                continue

        # Detect all-infrastructure-failure
        if self._handle_investigation_infra_failure(
            exp_id=exp_id,
            results=results,
            nb=nb,
            t_start=t_start,
        ):
            return

        self._complete_investigation(
            exp_id=exp_id,
            results=results,
            config=config,
            hypothesis=hypothesis,
            nb=nb,
            ckpt=ckpt,
            t_start=t_start,
        )

    # ------------------------------------------------------------------
    # Process one investigation candidate end-to-end
    # ------------------------------------------------------------------

    def _process_investigation_candidate(
        self,
        *,
        exp_id: str,
        source_result_id: str,
        source: Dict[str, Any],
        config: RunConfig,
        inv_config: RunConfig,
        dev: torch.device,
        prog_idx: int,
        total_candidates: int,
        results: Dict[str, Any],
        nb: Any,
        ckpt: CheckpointManager,
    ) -> Optional[object]:
        """Run training, scoring, fingerprinting, recording for one candidate.

        Returns _SKIP_INFRA if all programs hit infrastructure errors,
        None on normal completion.
        """
        model_source = source.get("model_source") or "graph_synthesis"
        tp_max_seq = self._compute_tp_max_seq(config, dev)

        training_programs, tp_sched = synthesize_training_program_batch(
            n_programs=config.n_training_programs,
            n_steps=config.investigation_steps,
            max_seq_len=tp_max_seq,
            seed_offset=prog_idx * 1000,
        )
        results.setdefault("training_program_scheduling", []).append(
            {"result_id": source_result_id, **tp_sched}
        )

        # Per-candidate training
        tp_results, _best_inv_model, _best_inv_model_lr = (
            self._investigate_candidate_training(
                exp_id=exp_id,
                source_result_id=source_result_id,
                source=source,
                model_source=model_source,
                training_programs=training_programs,
                inv_config=inv_config,
                config=config,
                dev=dev,
                tp_max_seq=tp_max_seq,
                prog_idx=prog_idx,
                total_candidates=total_candidates,
                ckpt=ckpt,
            )
        )

        if not tp_results:
            raise RuntimeError(
                f"Investigation aborted for {source_result_id[:8]}: "
                f"model failed to reconstruct for all "
                f"{len(training_programs)} training programs"
            )

        summary = self._summarize_and_check_infra(
            source_result_id,
            tp_results,
            source,
            config,
            results,
        )
        if summary is _SKIP_INFRA:
            return _SKIP_INFRA

        if summary.n_passed > 0:
            results["stage1_passed"] += 1
        results["stage0_passed"] += 1
        results["stage05_passed"] += 1

        investigation_passed_early = summary.investigation_passed_early

        # Persist best model artifact
        self._save_best_model_artifact(
            ckpt=ckpt,
            exp_id=exp_id,
            source_result_id=source_result_id,
            source=source,
            best_tp=summary.best_tp,
            best_inv_model=_best_inv_model,
        )

        # Post-investigation fingerprint completion
        fp_completed, fp_attempted, source = self._investigate_fingerprint_completion(
            source_result_id=source_result_id,
            source=source,
            best_inv_model=_best_inv_model,
            config=config,
            dev=dev,
            nb=nb,
        )

        if fp_attempted and not fp_completed:
            investigation_passed_early = False
            logger.warning(
                "investigation_fingerprint_incomplete: "
                "result_id=%s — downgrading investigation_passed to False",
                source_result_id[:12],
            )

        # Free the retained model
        if _best_inv_model is not None:
            del _best_inv_model
            _best_inv_model = None
            clear_gpu_memory()

        _fp_incomplete = fp_attempted and not fp_completed

        # Record results, update leaderboard, checkpoint
        self._record_investigation_candidate(
            exp_id=exp_id,
            source_result_id=source_result_id,
            source=source,
            model_source=model_source,
            graph_json_str=source.get("graph_json"),
            arch_spec_json_str=source.get("arch_spec_json"),
            config=config,
            dev=dev,
            tp_results=tp_results,
            tp_sched=tp_sched,
            training_programs=training_programs,
            summary=summary,
            n_passed=summary.n_passed,
            robustness=summary.robustness,
            best_tp=summary.best_tp,
            best_lr=summary.best_lr,
            investigation_passed_early=investigation_passed_early,
            fp_incomplete=_fp_incomplete,
            results=results,
            nb=nb,
            ckpt=ckpt,
            prog_idx=prog_idx,
        )
        return None

    # ------------------------------------------------------------------
    # Summarize candidate and check for infra-only failures
    # ------------------------------------------------------------------

    def _summarize_and_check_infra(
        self,
        source_result_id: str,
        tp_results: List[Dict[str, Any]],
        source: Dict[str, Any],
        config: RunConfig,
        results: Dict[str, Any],
    ) -> Any:
        """Summarize training programs and check for all-infra failure.

        Returns _SKIP_INFRA sentinel if all programs failed with infra
        errors, otherwise returns the InvestigationProgramSummary.
        """
        summary = summarize_investigation_program_runs(
            tp_results=tp_results,
            screening_lr=source.get("loss_ratio"),
            investigation_max_loss_ratio_multiplier=float(
                config.investigation_max_loss_ratio_multiplier
            ),
            loss_multiplier_fn=self._investigation_loss_multiplier,
        )
        if summary.infra_failures > 0 and summary.infra_failures == len(tp_results):
            logger.warning(
                "Investigation of %s: all %d training programs failed "
                "with infrastructure errors (CUDA/OOM) — skipping. "
                "This is not an architecture failure.",
                source_result_id[:8],
                len(tp_results),
            )
            results.setdefault("infra_failures", []).append(
                {
                    "result_id": source_result_id,
                    "n_programs": len(tp_results),
                    "errors": [
                        r.get("error", "")[:200] for r in tp_results if r.get("error")
                    ],
                }
            )
            return _SKIP_INFRA
        return summary

    # ------------------------------------------------------------------
    # Complete investigation experiment
    # ------------------------------------------------------------------

    def _complete_investigation(
        self,
        *,
        exp_id: str,
        results: Dict[str, Any],
        config: RunConfig,
        hypothesis: str,
        nb: Any,
        ckpt: CheckpointManager,
        t_start: float,
    ) -> None:
        """Complete experiment, auto-escalate, emit events."""
        context = self._build_rich_context_for_experiment(
            results, config, hypothesis, nb
        )
        summary = self.aria.experiment_summary(results, context=context)
        llm_analysis = self.aria.analyze_results(results, context=context)
        insights = self._analyze_results(results, exp_id, nb, context=context)

        self._publish_terminal_event(
            producer="runner.execution_investigation",
            event_type="experiment_completed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "results": results,
                "aria_summary": summary,
                "aria_mood": self.aria.state.mood,
                "insights": insights,
                "llm_analysis": llm_analysis,
                "mode": "investigation",
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

        nb.flush_writes()
        self._auto_escalate(results, config, nb, phase="investigation")

        if not config.keep_checkpoints:
            try:
                ckpt.cleanup(exp_id)
            except Exception as exc:
                _fail_loud(
                    "investigation",
                    f"checkpoint cleanup failed for {exp_id[:8]}",
                    exc,
                )

        self._update_progress(
            status="completed",
            elapsed_seconds=time.time() - t_start,
            aria_message=summary.split("\n")[-1]
            if summary
            else "Investigation complete.",
        )

        self._emit_event(
            "investigation_completed",
            {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
            },
        )

    # ------------------------------------------------------------------
    # VRAM-aware seq_len cap
    # ------------------------------------------------------------------

    def _compute_tp_max_seq(self, config: RunConfig, dev: torch.device) -> int:
        """Compute VRAM-aware maximum sequence length for training programs."""
        _tp_cap = 512
        if dev.type == "cuda":
            try:
                free_mb = (
                    torch.cuda.get_device_properties(dev).total_memory
                    - torch.cuda.memory_allocated(dev)
                ) / (1024 * 1024)
                import math as _math

                _batch = int(getattr(config, "investigation_batch_size", 4) or 4)
                _nlayers = int(getattr(config, "n_layers", 4) or 4)
                _dim = int(getattr(config, "model_dim", 256) or 256)
                _budget = free_mb * 0.5 * 1024 * 1024
                _max_s = int(
                    _math.sqrt(
                        _budget
                        / (max(_batch, 1) * max(_dim, 1) * max(_nlayers, 1) * 12)
                    )
                )
                _tp_cap = min(_tp_cap, max(64, _max_s))
                if _tp_cap < config.max_seq_len:
                    logger.info(
                        "VRAM-capped curriculum seq_len: %d (free=%.0fMB)",
                        _tp_cap,
                        free_mb,
                    )
            except RuntimeError as exc:
                _fail_loud(
                    "investigation",
                    "failed to compute VRAM seq_len cap",
                    exc,
                )
        return min(config.max_seq_len, _tp_cap)

    # ------------------------------------------------------------------
    # Per-candidate training loop
    # ------------------------------------------------------------------

    def _investigate_candidate_training(
        self,
        *,
        exp_id: str,
        source_result_id: str,
        source: Dict[str, Any],
        model_source: str,
        training_programs: list,
        inv_config: RunConfig,
        config: RunConfig,
        dev: torch.device,
        tp_max_seq: int,
        prog_idx: int,
        total_candidates: int,
        ckpt: CheckpointManager,
    ) -> Tuple[List[Dict[str, Any]], Any, float]:
        """Run all training programs for one investigation candidate.

        Returns (tp_results, best_inv_model, best_inv_model_lr).
        """
        graph_json_str = source.get("graph_json")
        arch_spec_json_str = source.get("arch_spec_json")

        tp_results: List[Dict[str, Any]] = []
        _best_inv_model = None
        _best_inv_model_lr = float("inf")
        clear_gpu_memory()

        for tp_i, tp in enumerate(training_programs):
            if self._stop_event.is_set():
                break

            try:
                model = self._reconstruct_investigation_model(
                    source_result_id=source_result_id,
                    model_source=model_source,
                    graph_json_str=graph_json_str,
                    arch_spec_json_str=arch_spec_json_str,
                    config=config,
                    tp_max_seq=tp_max_seq,
                )
            except Exception as e:
                _fail_loud(
                    "investigation",
                    f"model reconstruction failed for {source_result_id[:8]} "
                    f"training program {tp_i + 1}/{len(training_programs)}",
                    e,
                )

            self._emit_event(
                "investigation_progress",
                {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": total_candidates,
                    "source_result_id": source_result_id,
                    "training_program": tp_i + 1,
                    "total_programs": len(training_programs),
                    "status": f"training with {tp.name}",
                },
            )

            tp_result = self._run_single_training_program(
                exp_id=exp_id,
                source_result_id=source_result_id,
                model=model,
                tp=tp,
                tp_i=tp_i,
                inv_config=inv_config,
                config=config,
                dev=dev,
                prog_idx=prog_idx,
                ckpt=ckpt,
            )
            tp_results.append(
                {
                    "training_program": tp.name,
                    "passed": tp_result.get("passed", False),
                    "loss_ratio": tp_result.get("loss_ratio"),
                    "initial_loss": tp_result.get("initial_loss"),
                    "final_loss": tp_result.get("final_loss"),
                    "min_loss": tp_result.get("min_loss"),
                    "n_train_steps": tp_result.get("n_train_steps"),
                    "training_curve": tp_result.get("training_curve") or [],
                    "training_program_json": tp_result.get("training_program_json"),
                    "error": tp_result.get("error"),
                    "artifact_path": None,
                }
            )

            self._save_training_program_artifact(
                ckpt=ckpt,
                exp_id=exp_id,
                source_result_id=source_result_id,
                source=source,
                tp=tp,
                tp_i=tp_i,
                tp_result=tp_result,
                tp_results=tp_results,
                n_programs=len(training_programs),
            )

            # CUDA fatal error recovery
            _tp_error = tp_result.get("error") or ""
            if "cuda" in _tp_error.lower() and dev.type == "cuda":
                from ...eval.sandbox import is_cuda_fatal

                if is_cuda_fatal(RuntimeError(_tp_error)):
                    logger.warning(
                        "CUDA fatal error during investigation of %s "
                        "program %d/%d — attempting context recovery",
                        source_result_id[:8],
                        tp_i + 1,
                        len(training_programs),
                    )
                    try:
                        del model
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        _probe = torch.zeros(1, device=dev)
                        del _probe
                        torch.cuda.synchronize()
                        logger.info("CUDA context recovered after fatal error")
                    except RuntimeError as exc:
                        _fail_loud(
                            "investigation",
                            f"CUDA context unrecoverable for {source_result_id[:8]}",
                            exc,
                        )
                    continue  # skip model retention, try next program

            # Retain the best-performing model
            _this_lr = tp_result.get("loss_ratio")
            if _this_lr is not None and (
                _best_inv_model is None or _this_lr < _best_inv_model_lr
            ):
                if _best_inv_model is not None:
                    del _best_inv_model
                _best_inv_model = model
                _best_inv_model_lr = _this_lr
            else:
                del model
            clear_gpu_memory()

        return tp_results, _best_inv_model, _best_inv_model_lr

    # ------------------------------------------------------------------
    # Run a single training program
    # ------------------------------------------------------------------

    def _run_single_training_program(
        self,
        *,
        exp_id: str,
        source_result_id: str,
        model: Any,
        tp: Any,
        tp_i: int,
        inv_config: RunConfig,
        config: RunConfig,
        dev: torch.device,
        prog_idx: int,
        ckpt: CheckpointManager,
    ) -> Dict[str, Any]:
        """Train model with one training program and return result dict."""
        resume_state = ckpt.load_phase(exp_id, "investigation", prog_idx, tp_i)
        base_ctx = {"exp_id": exp_id, "phase": "investigation"}
        self._live_training_context = {
            **base_ctx,
            "source_result_id": source_result_id,
            "checkpoint_manager": ckpt,
            "checkpoint_candidate_idx": prog_idx,
            "checkpoint_seed_idx": tp_i,
            "checkpoint_interval_steps": int(
                getattr(config, "phase_checkpoint_step_interval", 0) or 0
            ),
            "checkpoint_resume_state": (
                resume_state
                if resume_state and int(resume_state.get("step", 0) or 0) > 0
                else None
            ),
        }
        try:
            return self._train_with_program(
                model,
                tp,
                inv_config,
                dev,
                seed=self._stable_seed(
                    exp_id, source_result_id, tp_i, "investigation_inline"
                ),
            )
        finally:
            self._live_training_context = base_ctx

    # ------------------------------------------------------------------
    # Save training program artifact
    # ------------------------------------------------------------------

    def _save_training_program_artifact(
        self,
        *,
        ckpt: CheckpointManager,
        exp_id: str,
        source_result_id: str,
        source: Dict[str, Any],
        tp: Any,
        tp_i: int,
        tp_result: Dict[str, Any],
        tp_results: List[Dict[str, Any]],
        n_programs: int,
    ) -> None:
        """Persist a completed investigation program artifact."""
        try:
            _artifact_payload = {
                "source_result_id": source_result_id,
                "graph_fingerprint": source.get("graph_fingerprint"),
                "template_name": source.get("template_name"),
                "training_program_name": tp.name,
                "training_program_json": tp_result.get("training_program_json"),
                "loss_ratio": tp_result.get("loss_ratio"),
                "initial_loss": tp_result.get("initial_loss"),
                "final_loss": tp_result.get("final_loss"),
                "min_loss": tp_result.get("min_loss"),
                "n_train_steps": tp_result.get("n_train_steps"),
                "passed": tp_result.get("passed", False),
                "error": tp_result.get("error"),
                "training_curve": tp_result.get("training_curve") or [],
            }
            _artifact_path = ckpt.save_investigation_artifact(
                experiment_id=exp_id,
                source_result_id=source_result_id,
                training_program_idx=tp_i + 1,
                payload=_artifact_payload,
                artifact_kind="program",
            )
            tp_results[-1]["artifact_path"] = str(_artifact_path)
        except Exception as e:
            logger.warning(
                "investigation artifact save failed for %s program %d/%d: %s",
                source_result_id[:8],
                tp_i + 1,
                n_programs,
                e,
            )

    # ------------------------------------------------------------------
    # Model reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_investigation_model(
        self,
        *,
        source_result_id: str,
        model_source: str,
        graph_json_str: Optional[str],
        arch_spec_json_str: Optional[str],
        config: RunConfig,
        tp_max_seq: int,
    ):
        """Reconstruct a model from source for investigation training."""
        if model_source == "morphological_box" and arch_spec_json_str:
            from ...morphological_box import ArchSpec
            from ...arch_builder import build_model, BuildConfig

            spec_data = self._cached_json_load(arch_spec_json_str)
            spec = ArchSpec(**spec_data)
            build_cfg = BuildConfig(
                dim=config.model_dim,
                n_layers=config.n_layers,
                vocab_size=config.vocab_size,
                max_seq_len=tp_max_seq,
            )
            return build_model(spec, build_cfg)
        elif graph_json_str:
            graph = graph_from_json(graph_json_str)
            layer_graphs = [graph] * config.n_layers
            return compile_model(
                layer_graphs,
                vocab_size=config.vocab_size,
                max_seq_len=tp_max_seq,
            )
        else:
            raise RuntimeError(f"No model source available for {source_result_id[:8]}")

    # ------------------------------------------------------------------
    # Save best model artifact
    # ------------------------------------------------------------------

    def _save_best_model_artifact(
        self,
        *,
        ckpt: CheckpointManager,
        exp_id: str,
        source_result_id: str,
        source: Dict[str, Any],
        best_tp: Optional[Dict[str, Any]],
        best_inv_model: Any,
    ) -> None:
        """Persist the best reconstructed model before downstream steps."""
        if best_inv_model is None or best_tp is None:
            return
        try:
            _best_model_payload = {
                "source_result_id": source_result_id,
                "graph_fingerprint": source.get("graph_fingerprint"),
                "template_name": source.get("template_name"),
                "best_training_program": best_tp.get("training_program"),
                "best_training_program_json": best_tp.get("training_program_json"),
                "loss_ratio": best_tp.get("loss_ratio"),
                "final_loss": best_tp.get("final_loss"),
                "initial_loss": best_tp.get("initial_loss"),
                "min_loss": best_tp.get("min_loss"),
                "n_train_steps": best_tp.get("n_train_steps"),
                "training_curve": best_tp.get("training_curve") or [],
            }
            _best_model_path = ckpt.save_investigation_artifact(
                experiment_id=exp_id,
                source_result_id=source_result_id,
                training_program_idx=0,
                payload=_best_model_payload,
                model_state_dict=best_inv_model.state_dict(),
                artifact_kind="best_model",
            )
            logger.info(
                "Saved investigation best-model artifact for %s to %s",
                source_result_id[:8],
                _best_model_path,
            )
        except Exception as e:
            logger.warning(
                "best-model artifact save failed for %s: %s",
                source_result_id[:8],
                e,
            )

    # ------------------------------------------------------------------
    # Post-investigation fingerprint completion
    # ------------------------------------------------------------------

    def _investigate_fingerprint_completion(
        self,
        *,
        source_result_id: str,
        source: Dict[str, Any],
        best_inv_model: Any,
        config: RunConfig,
        dev: torch.device,
        nb: Any,
    ) -> Tuple[bool, bool, Dict[str, Any]]:
        """Run CKA + behavioral probes on the best converged model.

        Returns (fingerprint_completed, fingerprint_attempted, updated_source).
        """
        _fingerprint_completed = False
        _fingerprint_attempted = False
        _fp_dict = source.get("_behavioral_fingerprint")

        if best_inv_model is None or _fp_dict is None:
            return _fingerprint_completed, _fingerprint_attempted, source

        _fingerprint_attempted = True
        from ...eval.fingerprint import BehavioralFingerprint
        from ...eval.fingerprint_runtime import (
            complete_fingerprint_post_investigation,
        )

        _fp = BehavioralFingerprint(
            **{
                k: v
                for k, v in _fp_dict.items()
                if k
                in {f.name for f in BehavioralFingerprint.__dataclass_fields__.values()}
            }
        )
        if _fp.fingerprint_completed_post_investigation:
            _fingerprint_completed = True
        else:
            # Attempt fingerprint completion with one retry
            for _attempt in range(2):
                try:
                    _fp = complete_fingerprint_post_investigation(
                        _fp,
                        best_inv_model,
                        seq_len=min(64, config.max_seq_len),
                        model_dim=config.model_dim,
                        vocab_size=config.vocab_size,
                        device=str(dev),
                    )
                    if _fp.fingerprint_completed_post_investigation:
                        _fingerprint_completed = True
                        _fp_dict_updated = _fp.to_dict()
                        source["_behavioral_fingerprint"] = _fp_dict_updated
                        source["novelty_confidence"] = (
                            0.9
                            if _fp.quality == "full"
                            else 0.4 + (_fp.analyses_succeeded * 0.1)
                            if _fp.quality == "partial"
                            else 0.3
                        )
                        # Persist to DB so escalation gate can read it.
                        nb._submit_write(
                            "UPDATE program_results "
                            "SET fingerprint_json = ?, "
                            "    novelty_valid_for_promotion = ? "
                            "WHERE result_id = ?",
                            (
                                json.dumps(_fp_dict_updated),
                                int(_fp.novelty_valid_for_promotion),
                                source_result_id,
                            ),
                        )
                        logger.info(
                            "post_investigation_fingerprint_completed: "
                            "result_id=%s novelty_score=%.4f "
                            "novelty_valid=%s cka_source=%s attempt=%d",
                            source_result_id[:12],
                            _fp.novelty_score,
                            _fp.novelty_valid_for_promotion,
                            _fp.cka_source,
                            _attempt + 1,
                        )
                        break
                except Exception as e:
                    logger.error(
                        "post_investigation_fingerprint_failed: "
                        "result_id=%s attempt=%d error=%s",
                        source_result_id[:12],
                        _attempt + 1,
                        str(e),
                    )

        return _fingerprint_completed, _fingerprint_attempted, source

    # ------------------------------------------------------------------
    # Record investigation candidate results
    # ------------------------------------------------------------------

    def _record_investigation_candidate(
        self,
        *,
        exp_id: str,
        source_result_id: str,
        source: Dict[str, Any],
        model_source: str,
        graph_json_str: Optional[str],
        arch_spec_json_str: Optional[str],
        config: RunConfig,
        dev: torch.device,
        tp_results: List[Dict[str, Any]],
        tp_sched: Dict[str, Any],
        training_programs: list,
        summary: InvestigationProgramSummary,
        n_passed: int,
        robustness: float,
        best_tp: Optional[Dict[str, Any]],
        best_lr: Optional[float],
        investigation_passed_early: bool,
        fp_incomplete: bool,
        results: Dict[str, Any],
        nb: Any,
        ckpt: CheckpointManager,
        prog_idx: int,
    ) -> None:
        """Build entry, record to notebook, submit benchmarks, checkpoint."""
        investigation_entry = build_investigation_entry(
            source_result_id=source_result_id,
            config=config,
            source=source,
            tp_sched=tp_sched,
            n_programs_tested=len(tp_results),
            fingerprint_incomplete=fp_incomplete,
            summary=summary,
        )
        results["investigation_results"].append(investigation_entry)

        if best_lr and (
            results["best_loss_ratio"] is None or best_lr < results["best_loss_ratio"]
        ):
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

        investigation_passed = investigation_passed_early

        # Submit benchmark evals to background thread
        if n_passed > 0:
            _submit_benchmark_eval(
                nb=nb,
                exp_id=exp_id,
                source_result_id=source_result_id,
                source=source,
                model_source=model_source,
                graph_json_str=graph_json_str,
                arch_spec_json_str=arch_spec_json_str,
                n_passed=n_passed,
                best_lr=best_lr,
                best_tp_json=best_tp_json,
                robustness=robustness,
                investigation_passed=investigation_passed,
                config=config,
                dev=dev,
                cached_json_load=self._cached_json_load,
                fingerprint_incomplete=fp_incomplete,
            )
        else:
            _record_investigation_result(
                nb=nb,
                exp_id=exp_id,
                source_result_id=source_result_id,
                source=source,
                model_source=model_source,
                graph_json_str=graph_json_str,
                arch_spec_json_str=arch_spec_json_str,
                n_passed=n_passed,
                best_lr=best_lr,
                best_tp_json=best_tp_json,
                robustness=robustness,
                investigation_passed=investigation_passed,
                benchmark_result={},
                fingerprint_incomplete=fp_incomplete,
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
            _fail_loud(
                "investigation",
                f"checkpoint save failed for candidate {prog_idx + 1}",
                e,
            )

    # ------------------------------------------------------------------
    # All-infrastructure-failure detection
    # ------------------------------------------------------------------

    def _handle_investigation_infra_failure(
        self,
        *,
        exp_id: str,
        results: Dict[str, Any],
        nb: Any,
        t_start: float,
    ) -> bool:
        """Detect and handle all-infrastructure-failure case.

        Returns True if the experiment was marked as failed (caller should return).
        """
        infra_only = not results.get("investigation_results") and results.get(
            "infra_failures"
        )
        if not infra_only:
            return False

        n_infra = len(results["infra_failures"])
        err_summary = "; ".join(
            f.get("errors", ["unknown"])[0][:80] for f in results["infra_failures"]
        )
        logger.error(
            "Investigation %s: all %d candidate(s) failed with "
            "infrastructure errors — marking as failed, not completed. "
            "Candidates are NOT penalized.",
            exp_id[:8],
            n_infra,
        )
        error = (
            f"All {n_infra} candidate(s) failed with infrastructure "
            f"errors (CUDA/OOM): {err_summary}"
        )
        self._publish_terminal_event(
            producer="runner.execution_investigation",
            event_type="experiment_failed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "error": error,
                "results": results,
                "mode": "investigation",
                "infra_error": True,
            },
        )
        self._fail_experiment_compat(
            nb=nb,
            experiment_id=exp_id,
            error=error,
            results=results,
        )
        nb.flush_writes()
        self._update_progress(
            status="failed",
            elapsed_seconds=time.time() - t_start,
            aria_message=(
                "Investigation failed: all candidates hit infrastructure "
                "errors (CUDA/OOM). Architectures were not evaluated — "
                "retry when GPU is healthy."
            ),
        )
        self._emit_event(
            "investigation_completed",
            {
                "experiment_id": exp_id,
                "status": "infra_error",
                "infra_failures": results.get("infra_failures"),
            },
        )
        self._live_training_context = None
        return True
