"""Execution mixin: investigation thread."""

from __future__ import annotations

import json
import time
import traceback
from typing import List

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

import logging

logger = logging.getLogger(__name__)

from ._types import RunConfig


class _ExecutionInvestigationMixin:
    """Investigation phase execution."""

    __slots__ = ()

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
                except Exception:
                    pass

        # Load phase checkpoint to find where we left off
        resume_from_candidate = 0
        ckpt_state = ckpt.load_phase(exp_id, "investigation", -1, 0)
        if ckpt_state:
            resume_from_candidate = ckpt_state.get("candidate_idx", 0)
            logger.info(
                "Resuming investigation from candidate %d", resume_from_candidate
            )

        try:
            results = {
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
            dev_str = str(dev)

            inv_config = RunConfig.from_dict(config.to_dict())
            inv_config.stage1_steps = config.investigation_steps
            inv_config.stage1_batch_size = config.investigation_batch_size
            # Scale early stopping for longer investigation runs.
            # Default patience (300) is calibrated for 500-step screening;
            # without scaling, investigation stops at ~step 400 (16% of 2500).
            step_ratio = config.investigation_steps / max(config.stage1_steps, 1)
            inv_config.early_stop_patience = int(
                config.early_stop_patience * step_ratio
            )
            inv_config.early_stop_min_steps = int(
                config.early_stop_min_steps * step_ratio
            )

            # Fetch all sources at once to avoid N+1 queries
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

                # Fetch source program
                source = source_map.get(source_result_id)
                if source is None:
                    continue

                # Reconstruct model
                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source") or "graph_synthesis"

                # Generate training programs (queue-level scheduling telemetry)
                # VRAM-aware seq_len cap for training program curriculum
                _tp_cap = 512
                if dev.type == "cuda":
                    try:
                        free_mb = (
                            torch.cuda.get_device_properties(dev).total_memory
                            - torch.cuda.memory_allocated(dev)
                        ) / (1024 * 1024)
                        import math as _math

                        _batch = int(
                            getattr(config, "investigation_batch_size", 4) or 4
                        )
                        _nlayers = int(getattr(config, "n_layers", 4) or 4)
                        _dim = int(getattr(config, "model_dim", 256) or 256)
                        _budget = free_mb * 0.5 * 1024 * 1024
                        _max_s = int(
                            _math.sqrt(
                                _budget
                                / (
                                    max(_batch, 1)
                                    * max(_dim, 1)
                                    * max(_nlayers, 1)
                                    * 12
                                )
                            )
                        )
                        _tp_cap = min(_tp_cap, max(64, _max_s))
                        if _tp_cap < config.max_seq_len:
                            logger.info(
                                "VRAM-capped curriculum seq_len: %d (free=%.0fMB)",
                                _tp_cap,
                                free_mb,
                            )
                    except Exception:
                        pass
                tp_max_seq = min(config.max_seq_len, _tp_cap)
                training_programs, tp_sched = synthesize_training_program_batch(
                    n_programs=config.n_training_programs,
                    n_steps=config.investigation_steps,
                    max_seq_len=tp_max_seq,
                    seed_offset=prog_idx * 1000,
                )
                results.setdefault("training_program_scheduling", []).append(
                    {
                        "result_id": source_result_id,
                        **tp_sched,
                    }
                )

                # Test each (model x training_program) pair
                tp_results = []
                _best_inv_model = None
                _best_inv_model_lr = float("inf")
                # Free GPU memory once before processing this candidate
                clear_gpu_memory()

                for tp_i, tp in enumerate(training_programs):
                    if self._stop_event.is_set():
                        break

                    # Reconstruct model fresh for each training program
                    # Use VRAM-capped seq_len for model construction too
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
                                max_seq_len=tp_max_seq,
                            )
                            model = build_model(spec, build_cfg)
                        elif graph_json_str:
                            graph = graph_from_json(graph_json_str)
                            layer_graphs = [graph] * config.n_layers
                            model = compile_model(
                                layer_graphs,
                                vocab_size=config.vocab_size,
                                max_seq_len=tp_max_seq,
                            )
                        else:
                            continue
                    except Exception as e:
                        logger.debug(f"Model reconstruction failed: {e}")
                        continue

                    self._emit_event(
                        "investigation_progress",
                        {
                            "experiment_id": exp_id,
                            "current": prog_idx + 1,
                            "total": len(result_ids),
                            "source_result_id": source_result_id,
                            "training_program": tp_i + 1,
                            "total_programs": len(training_programs),
                            "status": f"training with {tp.name}",
                        },
                    )

                    # Train with this program
                    tp_result = self._train_with_program(
                        model,
                        tp,
                        inv_config,
                        dev,
                        seed=self._stable_seed(
                            exp_id, source_result_id, tp_i, "investigation_inline"
                        ),
                    )
                    tp_results.append(
                        {
                            "training_program": tp.name,
                            "passed": tp_result.get("passed", False),
                            "loss_ratio": tp_result.get("loss_ratio"),
                            "final_loss": tp_result.get("final_loss"),
                            "error": tp_result.get("error"),
                        }
                    )

                    # CUDA fatal error recovery: after a device-side assert the
                    # entire CUDA context is poisoned. All subsequent operations
                    # will fail instantly with the same error. Attempt recovery
                    # before the next training program; if it fails, abort this
                    # candidate — continuing would waste time on instant failures.
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
                            except Exception:
                                logger.error(
                                    "CUDA context unrecoverable — aborting "
                                    "remaining training programs for %s",
                                    source_result_id[:8],
                                )
                                break
                            continue  # skip model retention, try next program

                    # Retain the best-performing model for post-investigation
                    # fingerprint completion (needs converged representations).
                    _this_lr = tp_result.get("loss_ratio")
                    if _this_lr is not None and (
                        _best_inv_model is None or _this_lr < _best_inv_model_lr
                    ):
                        # Free previous best before reassigning
                        if _best_inv_model is not None:
                            del _best_inv_model
                        _best_inv_model = model
                        _best_inv_model_lr = _this_lr
                    else:
                        del model
                    clear_gpu_memory()

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
                lr_multiplier = self._investigation_loss_multiplier(
                    screening_lr, best_lr
                )
                brittle_risk = lr_multiplier is not None and lr_multiplier > float(
                    config.investigation_max_loss_ratio_multiplier
                )

                if n_passed > 0:
                    results["stage1_passed"] += 1
                results["stage0_passed"] += 1
                results["stage05_passed"] += 1

                # Gate: pass investigation if loss quality is good enough.
                # [CALIBRATION] source: judgment — 0.5 and 0.3 are hardcoded; no config key
                #   last reviewed: unknown — flag for calibration sweep
                investigation_passed_early = (best_lr or 1.0) < 0.5 and (
                    not brittle_risk or (best_lr is not None and best_lr < 0.3)
                )

                # Post-investigation fingerprint completion: run CKA +
                # behavioral probes on the best converged model.
                _fp_dict = source.get("_behavioral_fingerprint")
                if _best_inv_model is not None and _fp_dict is not None:
                    try:
                        from ...eval.fingerprint import (
                            BehavioralFingerprint,
                            complete_fingerprint_post_investigation,
                        )

                        _fp = BehavioralFingerprint(
                            **{
                                k: v
                                for k, v in _fp_dict.items()
                                if k
                                in {
                                    f.name
                                    for f in BehavioralFingerprint.__dataclass_fields__.values()
                                }
                            }
                        )
                        if not _fp.fingerprint_completed_post_investigation:
                            _fp = complete_fingerprint_post_investigation(
                                _fp,
                                _best_inv_model,
                                seq_len=min(64, config.max_seq_len),
                                model_dim=config.model_dim,
                                vocab_size=config.vocab_size,
                                device=str(dev),
                            )
                            # Update the source fingerprint for downstream use
                            source["_behavioral_fingerprint"] = _fp.to_dict()
                            source["novelty_confidence"] = (
                                0.9
                                if _fp.quality == "full"
                                else 0.4 + (_fp.analyses_succeeded * 0.1)
                                if _fp.quality == "partial"
                                else 0.3
                            )
                            logger.info(
                                "post_investigation_fingerprint_completed: "
                                "result_id=%s novelty_score=%.4f "
                                "novelty_valid=%s cka_source=%s",
                                source_result_id[:12],
                                _fp.novelty_score,
                                _fp.novelty_valid_for_promotion,
                                _fp.cka_source,
                            )
                    except Exception as e:
                        logger.error(
                            "post_investigation_fingerprint_failed: "
                            "result_id=%s error=%s",
                            source_result_id[:12],
                            str(e),
                        )
                        # fp remains with cka_source="deferred",
                        # novelty_valid_for_promotion stays False

                # Free the retained model
                if _best_inv_model is not None:
                    del _best_inv_model
                    _best_inv_model = None
                    clear_gpu_memory()

                investigation_entry = {
                    "result_id": source_result_id,
                    "data_mode": str(config.data_mode or "random"),
                    "data_source": str(
                        config.hf_dataset or config.corpus_path or "random"
                    ),
                    "robustness": robustness,
                    "best_loss_ratio": best_lr,
                    "screening_loss_ratio": screening_lr,
                    "baseline_loss_ratio": source.get("baseline_loss_ratio"),
                    "novelty_confidence": source.get("novelty_confidence"),
                    "loss_ratio_multiplier": lr_multiplier,
                    "brittle_risk": brittle_risk,
                    "investigation_passed": investigation_passed_early,
                    "n_programs_passed": n_passed,
                    "n_programs_tested": len(tp_results),
                    "best_training_program": best_tp.get("training_program")
                    if best_tp
                    else None,
                    "training_program_scheduling_avg_ms": tp_sched.get(
                        "scheduling_avg_ms"
                    ),
                    "training_program_scheduling_max_ms": tp_sched.get(
                        "scheduling_max_ms"
                    ),
                    "training_errors": [
                        r["error"] for r in tp_results if r.get("error")
                    ],
                }
                results["investigation_results"].append(investigation_entry)

                if best_lr and (
                    results["best_loss_ratio"] is None
                    or best_lr < results["best_loss_ratio"]
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

                # Submit benchmark evals to background thread so the
                # investigation loop can proceed to the next candidate.
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

            nb.flush_writes()
            # Auto-escalate to validation
            self._auto_escalate(results, config, nb, phase="investigation")

            # Clean up investigation checkpoints on success
            if not config.keep_checkpoints:
                try:
                    ckpt.cleanup(exp_id)
                except Exception:
                    pass

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

        except Exception as e:
            error = traceback.format_exc()
            logger.error("Investigation failed (%s): %s\n%s", exp_id, e, error)
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
        finally:
            self._live_training_context = None
            nb.close()
            self._run_pending_scale_up()
