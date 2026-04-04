"""Execution mixin: validation + scale-up threads."""

from __future__ import annotations

import json
import sqlite3
import time
import traceback
from typing import List

from ..json_utils import json_safe


from ..native_runner import compile_model_native_first as compile_model
from ...synthesis.serializer import graph_to_json, graph_from_json
from ...eval.metrics import novelty_score
from ...eval.fingerprint import compute_fingerprint
from ...eval.diagnostic_tasks import run_diagnostic_suite
from ...training.checkpointing import CheckpointManager
from ..shared_utils import resolve_device
from ._helpers import (
    clear_gpu_memory,
    compute_seed_metrics,
    run_baseline_comparison,
    build_validation_entry,
    promote_validation_candidate,
    run_trajectory_probe,
    handle_breakthrough,
    screening_probe_fields,
    screening_wikitext_fields,
)

import logging

logger = logging.getLogger(__name__)

from ._types import RunConfig


def _fail_loud(phase: str, message: str, exc: BaseException) -> None:
    logger.exception("%s: %s", phase, message)
    raise RuntimeError(f"{phase}: {message}") from exc


class _ExecutionValidationMixin:
    """Validation and scale-up thread execution."""

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
            resume_from_candidate = ckpt_state.get("candidate_idx", 0)
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

                self._update_progress(
                    current_program=prog_idx + 1,
                    status="validating",
                    aria_message=(
                        f"Validating {prog_idx + 1}/{len(result_ids)}: "
                        f"{source_result_id[:8]}... "
                        f"({config.validation_n_seeds} seeds, "
                        f"{config.validation_steps} steps)"
                    ),
                    elapsed_seconds=time.time() - t_start,
                )

                self._emit_event(
                    "validation_progress",
                    {
                        "experiment_id": exp_id,
                        "current": prog_idx + 1,
                        "total": len(result_ids),
                        "source_result_id": source_result_id,
                        "status": "starting",
                    },
                )

                # Fetch source and leaderboard entry
                source = source_map.get(source_result_id)
                if source is None:
                    continue

                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source") or "graph_synthesis"

                best_tp_json = self._get_validation_best_training_json(
                    nb, source_result_id
                )

                # B3: Validation tier requires artifact-backed CKA.
                # If the entry doesn't have it, attempt fingerprint
                # completion now. If it still fails, cap novelty at 50%.
                _novelty_cap = None
                _fp_data = source.get("_behavioral_fingerprint") or {}
                _cka_src = _fp_data.get("cka_source", "unknown")
                if _cka_src != "artifact":
                    logger.info(
                        "validation_cka_check: result_id=%s cka_source=%s "
                        "— attempting fingerprint completion",
                        source_result_id[:12],
                        _cka_src,
                    )
                    try:
                        from ...eval.fingerprint import (
                            BehavioralFingerprint,
                            complete_fingerprint_post_investigation,
                        )

                        _fp_fields = {
                            k: v
                            for k, v in _fp_data.items()
                            if k
                            in {
                                f.name
                                for f in BehavioralFingerprint.__dataclass_fields__.values()
                            }
                        }
                        if _fp_fields:
                            _fp = BehavioralFingerprint(**_fp_fields)
                            # Build a temporary model for fingerprinting
                            _tmp_model = self._build_model_from_source(
                                model_source,
                                arch_spec_json_str,
                                graph_json_str,
                                config,
                                seq_len_override=min(64, config.validation_seq_len),
                            )
                            if _tmp_model is not None:
                                _fp = complete_fingerprint_post_investigation(
                                    _fp,
                                    _tmp_model,
                                    seq_len=min(64, config.validation_seq_len),
                                    model_dim=config.model_dim,
                                    vocab_size=config.vocab_size,
                                    device=str(dev),
                                )
                                del _tmp_model
                                clear_gpu_memory()
                                if _fp.cka_source == "artifact":
                                    source["_behavioral_fingerprint"] = _fp.to_dict()
                                    logger.info(
                                        "validation_cka_completed: result_id=%s "
                                        "cka_source=artifact",
                                        source_result_id[:12],
                                    )
                                else:
                                    _novelty_cap = 0.5
                                    logger.warning(
                                        "validation_cka_still_missing: result_id=%s "
                                        "cka_source=%s — capping novelty at 50%%",
                                        source_result_id[:12],
                                        _fp.cka_source,
                                    )
                            else:
                                _novelty_cap = 0.5
                                logger.warning(
                                    "validation_cka_model_build_failed: result_id=%s "
                                    "— capping novelty at 50%%",
                                    source_result_id[:12],
                                )
                        else:
                            _novelty_cap = 0.5
                    except (RuntimeError, ValueError, TypeError, ImportError) as e:
                        _novelty_cap = 0.5
                        logger.warning(
                            "validation_cka_attempt_failed: result_id=%s error=%s "
                            "— capping novelty at 50%%",
                            source_result_id[:12],
                            str(e),
                        )

                seed_results = self._run_validation_seed_sweep(
                    exp_id=exp_id,
                    source_result_id=source_result_id,
                    model_source=model_source,
                    arch_spec_json_str=arch_spec_json_str,
                    graph_json_str=graph_json_str,
                    config=config,
                    val_config=val_config,
                    dev=dev,
                    best_tp_json=best_tp_json,
                    progress_payload={
                        "experiment_id": exp_id,
                        "current": prog_idx + 1,
                        "total": len(result_ids),
                        "source_result_id": source_result_id,
                    },
                )

                # Skip candidates where no seed could reconstruct the model
                if not seed_results:
                    raise RuntimeError(
                        f"Validation aborted for {source_result_id[:8]}: "
                        f"model failed to reconstruct for all "
                        f"{config.validation_n_seeds} seeds"
                    )

                # Compute validation metrics
                _sm = compute_seed_metrics(seed_results)
                passed_seeds = _sm["passed_seeds"]
                loss_ratios = _sm["loss_ratios"]
                val_loss_ratio = _sm["val_loss_ratio"]
                multi_seed_std = _sm["multi_seed_std"]
                robustness_score = _sm["robustness_score"]
                is_unstable = _sm["is_unstable"]
                init_sensitivity_std = _sm["init_sensitivity_std"]
                best_seed = _sm["best_seed"]

                _rid_short = source_result_id[:8]

                def _compare(loss, **kw):
                    return run_baseline_comparison(
                        get_baseline=self._get_baseline,
                        resolve_recipe=self._resolve_baseline_recipe,
                        make_data_fn=self._make_baseline_data_fn,
                        candidate_loss=loss,
                        train_result=best_seed,
                        config=config,
                        dev_str=dev_str,
                        **kw,
                    )

                # Baseline comparison at validation scale
                _vstatus("baseline comparison", _rid_short)
                val_baseline_ratio = None
                if best_seed is not None:
                    try:
                        val_baseline_ratio = _compare(best_seed["final_loss"])
                        v_loss = best_seed.get("validation_loss")
                        if v_loss is not None:
                            program_metrics["validation_baseline_loss_ratio"] = (
                                _compare(v_loss, split="val")
                            )
                    except (RuntimeError, ValueError, TypeError) as exc:
                        _fail_loud(
                            "validation",
                            f"baseline comparison failed for {source_result_id[:8]}",
                            exc,
                        )

                # Parameter-normalized baseline comparison
                _vstatus("normalized baseline comparison", _rid_short)
                val_normalized_ratio = None
                val_param_efficiency = None
                source_params = int(
                    (
                        source.get("param_count")
                        or source.get("graph_n_params_estimate")
                        or 0
                    )
                    if source
                    else 0
                )
                if loss_ratios and best_seed is not None and source_params > 0:
                    try:
                        norm_result = _compare(
                            best_seed["final_loss"],
                            normalized=True,
                            program_params=source_params,
                        )
                        val_normalized_ratio = norm_result.get("normalized_ratio")
                        val_param_efficiency = norm_result.get("param_efficiency")
                    except (RuntimeError, ValueError, TypeError) as exc:
                        _fail_loud(
                            "validation",
                            f"normalized baseline comparison failed for {source_result_id[:8]}",
                            exc,
                        )

                if len(passed_seeds) > 0:
                    results["stage1_passed"] += 1
                results["stage0_passed"] += 1
                results["stage05_passed"] += 1

                _vstatus("external evals", _rid_short)
                ev_res = self._run_external_evals(
                    config=config,
                    dev=dev,
                    dev_str=dev_str,
                    best_seed=best_seed,
                    model_source=model_source,
                    arch_spec_json_str=arch_spec_json_str,
                    graph_json_str=graph_json_str,
                    source=source,
                    source_result_id=source_result_id,
                    exp_id=exp_id,
                    val_loss_ratio=val_loss_ratio,
                    val_baseline_ratio=val_baseline_ratio,
                    val_normalized_ratio=val_normalized_ratio,
                    multi_seed_std=multi_seed_std,
                    passed_seeds=passed_seeds,
                    source_params=source_params,
                )

                nov_conf = source.get("novelty_confidence", 0) if source else 0

                # Build typed ValidationMetrics for helper consumption
                from ._types import ValidationMetrics

                _metrics = ValidationMetrics(
                    val_loss_ratio=val_loss_ratio,
                    multi_seed_std=multi_seed_std,
                    robustness_score=robustness_score,
                    is_unstable=is_unstable,
                    init_sensitivity_std=init_sensitivity_std,
                    val_baseline_ratio=val_baseline_ratio,
                    val_normalized_ratio=val_normalized_ratio,
                    val_param_efficiency=val_param_efficiency,
                    passed_seeds=passed_seeds,
                    best_seed=best_seed,
                    source_params=int(source_params),
                )

                validation_entry = build_validation_entry(
                    source_result_id=source_result_id,
                    metrics=_metrics,
                    ev_res=ev_res,
                    nov_conf=nov_conf,
                    config=config,
                )
                tier = "breakthrough" if ev_res.is_breakthrough else "validation"
                results["validation_results"].append(validation_entry.to_dict())

                if val_loss_ratio and (
                    results["best_loss_ratio"] is None
                    or val_loss_ratio < results["best_loss_ratio"]
                ):
                    results["best_loss_ratio"] = val_loss_ratio
                source_novelty = source.get("novelty_score")
                if source_novelty is not None and (
                    results["best_novelty_score"] is None
                    or source_novelty > results["best_novelty_score"]
                ):
                    results["best_novelty_score"] = source_novelty

                _vstatus("leaderboard promotion", _rid_short)
                promote_validation_candidate(
                    nb=nb,
                    source_result_id=source_result_id,
                    source=source,
                    tier=tier,
                    metrics=_metrics,
                    ev_res=ev_res,
                    novelty_cap=_novelty_cap,
                )

                _vstatus("trajectory probe (4000 steps)", _rid_short)
                trajectory_composite = run_trajectory_probe(
                    graph_json_str=graph_json_str,
                    config=config,
                    dev=dev,
                    dev_str=dev_str,
                    nb=nb,
                    source_result_id=source_result_id,
                    tier=tier,
                    passed_seeds=passed_seeds,
                )

                handle_breakthrough(
                    is_breakthrough=ev_res.is_breakthrough,
                    trajectory_composite=trajectory_composite,
                    aria=self.aria,
                    nb=nb,
                    exp_id=exp_id,
                    source_result_id=source_result_id,
                    source=source,
                    validation_entry=validation_entry,
                    val_loss_ratio=val_loss_ratio,
                    val_baseline_ratio=val_baseline_ratio,
                    multi_seed_std=multi_seed_std,
                    emit_event=self._emit_event,
                )

                # Compute capped novelty for record_program_result
                _raw_novelty = source.get("novelty_score")
                _raw_confidence = source.get("novelty_confidence")
                if _novelty_cap is not None:
                    if _raw_novelty is not None:
                        _raw_novelty = float(_raw_novelty) * _novelty_cap
                    if _raw_confidence is not None:
                        _raw_confidence = float(_raw_confidence) * _novelty_cap

                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=source.get("graph_fingerprint", source_result_id),
                    graph_json=graph_json_str or "{}",
                    stage0_passed=True,
                    stage05_passed=True,
                    stage1_passed=len(passed_seeds) > 0,
                    loss_ratio=val_loss_ratio,
                    baseline_loss_ratio=val_baseline_ratio,
                    novelty_score=_raw_novelty,
                    novelty_confidence=_raw_confidence,
                    novelty_raw_score=source.get("novelty_raw_score"),
                    novelty_z_score=source.get("novelty_z_score"),
                    novelty_reference_version=source.get("novelty_reference_version"),
                    novelty_valid_for_promotion=source.get(
                        "novelty_valid_for_promotion"
                    ),
                    novelty_validity_reason=source.get("novelty_validity_reason"),
                    novelty_requires_justification=source.get(
                        "novelty_requires_justification"
                    ),
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
                except (OSError, RuntimeError) as e:
                    _fail_loud(
                        "validation",
                        f"checkpoint save failed for candidate {prog_idx + 1}",
                        e,
                    )

            # Complete experiment
            _vstatus("generating experiment summary")
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb
            )
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
            error = traceback.format_exc()
            logger.error("Validation failed (%s): %s\n%s", exp_id, e, error)
            try:
                self._invoke_code_healer(
                    nb=nb,
                    trigger_type="repeated_exception",
                    experiment_id=exp_id,
                    scope=f"Validation failure: {str(e)[:240]}",
                    reproduction_steps=[
                        'python -m pytest tests/test_integration.py -k "validation" -x --tb=short'
                    ],
                    acceptance_tests=[
                        'python -m pytest tests/test_integration.py -k "validation" -x --tb=short'
                    ],
                    trigger_payload={"mode": "validation", "error": str(e)},
                )
            except (RuntimeError, OSError) as heal_err:
                logger.warning(
                    "code_healer failed during validation error handling: %s",
                    heal_err,
                    exc_info=True,
                )
            nb.fail_experiment(exp_id, str(e))
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
            # Catch everything — CUDA errors, KeyboardInterrupt, SystemExit.
            # A background thread must never die silent.
            logger.critical(
                "Validation thread KILLED (%s): %s\n%s",
                exp_id,
                e,
                traceback.format_exc(),
            )
            try:
                nb.fail_experiment(exp_id, f"FATAL: {e}")
                self._update_progress(status="failed", error=f"FATAL: {e}")
                self._emit_event(
                    "experiment_failed",
                    {"experiment_id": exp_id, "error": f"FATAL: {e}"},
                )
            except RuntimeError:
                logger.error(
                    "Failed to emit failure event after fatal error", exc_info=True
                )
            raise
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

                # Fetch source program
                source_program = nb.get_program_detail(source_result_id)
                if source_program is None:
                    self._emit_event(
                        "scale_up_progress",
                        {
                            "experiment_id": exp_id,
                            "current_program": prog_idx + 1,
                            "total_programs": len(result_ids),
                            "source_result_id": source_result_id,
                            "status": "skipped",
                            "error": "Source program not found",
                        },
                    )
                    continue

                # Reconstruct graph from stored JSON
                graph_json_str = source_program.get("graph_json")
                if not graph_json_str:
                    raise RuntimeError(
                        f"Scale-up source {source_result_id[:8]} has no graph_json"
                    )

                try:
                    graph = graph_from_json(graph_json_str)
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    self._emit_event(
                        "scale_up_progress",
                        {
                            "experiment_id": exp_id,
                            "current_program": prog_idx + 1,
                            "total_programs": len(result_ids),
                            "source_result_id": source_result_id,
                            "status": "error",
                            "error": f"Graph deserialization failed: {e}",
                        },
                    )
                    _fail_loud(
                        "scale_up",
                        f"graph deserialization failed for {source_result_id[:8]}",
                        e,
                    )

                # Compile model
                try:
                    layer_graphs = [graph] * config.n_layers
                    model = compile_model(
                        layer_graphs,
                        vocab_size=config.vocab_size,
                        max_seq_len=config.scale_up_seq_len,
                    )
                except (RuntimeError, ValueError, TypeError) as e:
                    self._emit_event(
                        "scale_up_progress",
                        {
                            "experiment_id": exp_id,
                            "current_program": prog_idx + 1,
                            "total_programs": len(result_ids),
                            "source_result_id": source_result_id,
                            "status": "error",
                            "error": f"Compilation failed: {e}",
                        },
                    )
                    _fail_loud(
                        "scale_up",
                        f"compilation failed for {source_result_id[:8]}",
                        e,
                    )

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
                for key in [
                    "initial_loss",
                    "min_loss",
                    "loss_improvement_rate",
                    "avg_step_time_ms",
                    "total_train_time_ms",
                    "max_grad_norm",
                    "mean_grad_norm",
                    "grad_norm_std",
                    "n_train_steps",
                    "final_lr",
                    "validation_loss",
                    "validation_loss_ratio",
                    "generalization_gap",
                    "discovery_loss",
                    "discovery_loss_ratio",
                ]:
                    program_metrics[key] = s1_result.get(key)
                program_metrics["train_budget_steps"] = config.scale_up_steps
                program_metrics.update(screening_wikitext_fields(s1_result))
                program_metrics.update(screening_probe_fields(s1_result))
                program_metrics.update(screening_probe_fields(program_metrics))
                self._merge_s1_telemetry(program_metrics, s1_result)

                if s1_passed:
                    results["stage1_passed"] += 1
                    # Baseline comparison at scale
                    if final_loss is not None:
                        try:
                            baseline = self._get_baseline()
                            baseline_steps = int(
                                s1_result.get("n_train_steps") or config.scale_up_steps
                            )
                            baseline_recipe = self._resolve_baseline_recipe(
                                s1_result, default_lr=config.stage1_lr
                            )
                            bl_data_fn, bl_data_tag, bl_cache = (
                                self._make_baseline_data_fn(config)
                            )
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
                                v_data_fn, v_data_tag, v_cache = (
                                    self._make_baseline_data_fn(config, split="val")
                                )
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
                                program_metrics["validation_baseline_loss_ratio"] = (
                                    v_baseline_ratio
                                )
                        except (RuntimeError, ValueError, TypeError) as exc:
                            _fail_loud(
                                "scale_up",
                                f"baseline comparison failed for {source_result_id[:8]}",
                                exc,
                            )

                program_metrics["stage_at_death"] = (
                    "survived" if s1_passed else "stage1"
                )

                # Diagnostic tasks for S1 survivors
                if s1_passed and model is not None:
                    try:
                        diag = run_diagnostic_suite(model, device=dev_str)
                        program_metrics["diagnostic_tasks_json"] = json.dumps(
                            json_safe(diag.to_dict())
                        )
                        program_metrics["diagnostic_score"] = diag.diagnostic_score
                    except (RuntimeError, ValueError) as exc:
                        _fail_loud(
                            "scale_up",
                            f"diagnostic suite failed for {source_result_id[:8]}",
                            exc,
                        )

                # Benchmark evals (non-blocking) for scale-up survivors
                if s1_passed and model is not None:
                    eval_seq_len = min(128, config.scale_up_seq_len)
                    try:
                        from ...eval.wikitext_eval import evaluate_wikitext_perplexity

                        wt_result = evaluate_wikitext_perplexity(
                            model,
                            config.vocab_size,
                            dev_str,
                            n_train_steps=200,
                            seq_len=eval_seq_len,
                        )
                        program_metrics["wikitext_perplexity"] = wt_result.get(
                            "wikitext_perplexity"
                        )
                        program_metrics["wikitext_score"] = wt_result.get(
                            "wikitext_score"
                        )
                        if program_metrics.get("wikitext_perplexity") is not None:
                            logger.info(
                                "Scale-up WikiText ppl=%.1f score=%.3f",
                                program_metrics["wikitext_perplexity"],
                                program_metrics.get("wikitext_score") or 0,
                            )
                    except (ImportError, RuntimeError, ValueError) as e:
                        logger.debug("Scale-up WikiText eval skipped: %s", e)
                    try:
                        from ...eval.tinystories_eval import evaluate_tinystories

                        ts_result = evaluate_tinystories(
                            model,
                            config.vocab_size,
                            dev_str,
                            n_train_steps=200,
                            seq_len=eval_seq_len,
                        )
                        program_metrics["tinystories_perplexity"] = ts_result.get(
                            "tinystories_perplexity"
                        )
                        program_metrics["tinystories_score"] = ts_result.get(
                            "tinystories_score"
                        )
                        if program_metrics.get("tinystories_perplexity") is not None:
                            logger.info(
                                "Scale-up TinyStories ppl=%.1f score=%.3f",
                                program_metrics["tinystories_perplexity"],
                                program_metrics.get("tinystories_score") or 0,
                            )
                    except (ImportError, RuntimeError, ValueError) as e:
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
                        program_metrics["cka_artifact_version"] = (
                            fp.cka_artifact_version
                        )
                        program_metrics["cka_probe_protocol_hash"] = (
                            fp.cka_probe_protocol_hash
                        )
                        program_metrics["cka_reference_quality"] = (
                            fp.cka_reference_quality
                        )
                        calibration_row = self._ensure_novelty_calibration(
                            nb, config, fp
                        )
                    except (RuntimeError, ValueError, TypeError) as exc:
                        _fail_loud(
                            "scale_up",
                            f"fingerprint computation failed for {source_result_id[:8]}",
                            exc,
                        )

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
                    results["survivors"].append(
                        {
                            "fingerprint": graph.fingerprint(),
                            "novelty": n_score,
                            "loss_ratio": loss_ratio,
                            "novelty_valid_for_promotion": novelty_valid,
                        }
                    )

                if loss_ratio and (
                    results["best_loss_ratio"] is None
                    or loss_ratio < results["best_loss_ratio"]
                ):
                    results["best_loss_ratio"] = loss_ratio
                if n_score and (
                    results["best_novelty_score"] is None
                    or n_score > results["best_novelty_score"]
                ):
                    results["best_novelty_score"] = n_score

                result_id = nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=graph.fingerprint(),
                    graph_json=graph_to_json(graph),
                    stage0_passed=True,
                    stage05_passed=True,
                    stage1_passed=s1_passed,
                    loss_ratio=loss_ratio,
                    final_loss=final_loss,
                    throughput=throughput,
                    novelty_score=n_score,
                    structural_novelty=nov.structural_novelty,
                    behavioral_novelty=nov.behavioral_novelty,
                    most_similar_to=nov.most_similar_to,
                    novelty_confidence=nov.novelty_confidence,
                    **program_metrics,
                )

                if training_curve and result_id:
                    try:
                        nb.store_training_curve(result_id, training_curve)
                    except (sqlite3.OperationalError, RuntimeError) as exc:
                        _fail_loud(
                            "scale_up",
                            f"training curve persistence failed for {result_id[:8]}",
                            exc,
                        )

                self._emit_event(
                    "scale_up_progress",
                    {
                        "experiment_id": exp_id,
                        "current_program": prog_idx + 1,
                        "total_programs": len(result_ids),
                        "source_result_id": source_result_id,
                        "status": "completed",
                        "passed": s1_passed,
                        "loss_ratio": round(loss_ratio, 4) if loss_ratio else None,
                        "final_loss": round(final_loss, 4) if final_loss else None,
                    },
                )

                # Cleanup
                del model
                clear_gpu_memory()

            # Guard: if no programs were processed at all, fail with clear reason
            if results["stage0_passed"] == 0 and results["total"] > 0:
                reason = (
                    f"All {results['total']} source programs were skipped "
                    f"(not found or failed to compile). "
                    f"Result IDs: {', '.join(r[:12] for r in result_ids)}"
                )
                logger.warning("Scale-up produced no results: %s", reason)
                nb.fail_experiment(exp_id, reason)
                self._update_progress(
                    status="failed",
                    error=reason,
                    aria_message=self.aria.react_to_failure(reason),
                )
                self._emit_event(
                    "experiment_failed",
                    {
                        "experiment_id": exp_id,
                        "error": reason,
                    },
                )
                return

            # Complete experiment
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb
            )
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

            self._update_progress(
                status="completed",
                elapsed_seconds=time.time() - t_start,
                aria_message=summary.split("\n")[-1]
                if summary
                else "Scale-up complete.",
            )

            self._emit_event(
                "scale_up_completed",
                {
                    "experiment_id": exp_id,
                    "results": results,
                    "summary": summary,
                },
            )

        except Exception as e:
            error = traceback.format_exc()
            logger.error("Scale-up failed (%s): %s\n%s", exp_id, e, error)
            try:
                self._invoke_code_healer(
                    nb=nb,
                    trigger_type="repeated_exception",
                    experiment_id=exp_id,
                    scope=f"Scale-up failure: {str(e)[:240]}",
                    reproduction_steps=[
                        'python -m pytest tests/test_integration.py -k "scale_up" -x --tb=short'
                    ],
                    acceptance_tests=[
                        'python -m pytest tests/test_integration.py -k "scale_up" -x --tb=short'
                    ],
                    trigger_payload={"mode": "scale_up", "error": str(e)},
                )
            except (RuntimeError, OSError) as heal_err:
                logger.warning(
                    "code_healer failed during scale-up error handling: %s",
                    heal_err,
                    exc_info=True,
                )
            nb.fail_experiment(exp_id, str(e))
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
                "Scale-up thread KILLED (%s): %s\n%s",
                exp_id,
                e,
                traceback.format_exc(),
            )
            try:
                nb.fail_experiment(exp_id, f"FATAL: {e}")
                self._update_progress(status="failed", error=f"FATAL: {e}")
                self._emit_event(
                    "experiment_failed",
                    {"experiment_id": exp_id, "error": f"FATAL: {e}"},
                )
            except RuntimeError:
                logger.error(
                    "Failed to emit failure event after fatal error", exc_info=True
                )
            raise
        finally:
            self._live_training_context = None
            nb.close()
