"""Execution mixin: investigation thread."""

from __future__ import annotations

import gc
import json
import time
import traceback
from typing import Any, Dict, List

import torch

from ...synthesis.serializer import graph_from_json
from ...training.training_program import synthesize_training_program_batch
from ...training.checkpointing import CheckpointManager
from ..native_runner import compile_model_native_first as compile_model
from ..notebook import LabNotebook, ExperimentEntry
from ..shared_utils import resolve_device
from ._helpers import (
    _record_investigation_result,
    _submit_benchmark_eval,
)

import logging
logger = logging.getLogger(__name__)

from ._types import RunConfig, LiveProgress


class _ExecutionInvestigationMixin:
    """Investigation phase execution."""

    __slots__ = ()

    def _run_investigation_thread(self, exp_id: str, result_ids: List[str],
                                   config: RunConfig, hypothesis: str):
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
                        (rid,)).fetchone()
                    if row and row[0] is not None:
                        logger.info("Investigation candidate %s pre_inv_score=%.1f",
                                    rid[:8], row[0])
                except Exception:
                    pass

        # Load phase checkpoint to find where we left off
        resume_from_candidate = 0
        ckpt_state = ckpt.load_phase(exp_id, "investigation", -1, 0)
        if ckpt_state:
            resume_from_candidate = ckpt_state.get("candidate_idx", 0)
            logger.info("Resuming investigation from candidate %d", resume_from_candidate)

        try:
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
                if prog_idx < resume_from_candidate:
                    continue
                if self._stop_event.is_set():
                    break

                with self._lock:
                    self._progress.current_program = prog_idx + 1
                    self._progress.status = "investigating"
                    self._progress.aria_message = (
                        f"Investigating {prog_idx + 1}/{len(result_ids)}: "
                        f"{source_result_id[:8]}... "
                        f"({config.n_training_programs} training programs)"
                    )
                    self._progress.elapsed_seconds = time.time() - t_start

                self._emit_event("investigation_progress", {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "starting",
                })

                # Fetch source program
                source = source_map.get(source_result_id)
                if source is None:
                    continue

                # Reconstruct model
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
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ...morphological_box import ArchSpec
                            from ...arch_builder import build_model, BuildConfig
                            spec_data = self._cached_json_load(arch_spec_json_str)
                            spec = ArchSpec(**spec_data)
                            build_cfg = BuildConfig(
                                dim=config.model_dim,
                                n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.max_seq_len,
                            )
                            model = build_model(spec, build_cfg)
                        elif graph_json_str:
                            graph = graph_from_json(graph_json_str)
                            layer_graphs = [graph] * config.n_layers
                            model = compile_model(
                                layer_graphs,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.max_seq_len,
                            )
                        else:
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

                    # Train with this program
                    tp_result = self._train_with_program(
                        model,
                        tp,
                        inv_config,
                        dev,
                        seed=self._stable_seed(exp_id, source_result_id, tp_i, "investigation_inline"),
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
                        inv_wikitext_ppl=None,
                        inv_wikitext_score=None,
                        inv_tinystories_ppl=None,
                        inv_tinystories_score=None,
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
                results, config, hypothesis, nb)
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

            with self._lock:
                self._progress.status = "completed"
                self._progress.elapsed_seconds = time.time() - t_start
                self._progress.aria_message = summary.split("\n")[-1] if summary else "Investigation complete."

            self._emit_event("investigation_completed", {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
            })

        except Exception as e:
            error = traceback.format_exc()
            logger.error("Investigation failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Investigation failure: {str(e)[:240]}",
                reproduction_steps=["python -m pytest tests/test_integration.py -k \"investigation\" -x --tb=short"],
                acceptance_tests=["python -m pytest tests/test_integration.py -k \"investigation\" -x --tb=short"],
                trigger_payload={"mode": "investigation", "error": str(e)},
            )
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
            self._emit_event("experiment_failed", {
                "experiment_id": exp_id,
                "error": str(e),
            })
        finally:
            self._live_training_context = None
            nb.close()
            self._run_pending_scale_up()
