"""Execution mixin: validation + scale-up threads."""

from __future__ import annotations

import gc
import json
import time
import traceback
from typing import List

import torch

from ..native_runner import compile_model_native_first as compile_model
from ...synthesis.serializer import graph_to_json, graph_from_json
from ...eval.metrics import novelty_score
from ...eval.fingerprint import compute_fingerprint
from ...eval.diagnostic_tasks import run_diagnostic_suite
from ...training.checkpointing import CheckpointManager
from ..notebook import ExperimentEntry
from ..llm.context_experiment import build_validation_context
from ..shared_utils import coerce_dict_payload, resolve_device

import logging

logger = logging.getLogger(__name__)

from ._types import RunConfig


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

                with self._lock:
                    self._progress.current_program = prog_idx + 1
                    self._progress.status = "validating"
                    self._progress.aria_message = (
                        f"Validating {prog_idx + 1}/{len(result_ids)}: "
                        f"{source_result_id[:8]}... "
                        f"({config.validation_n_seeds} seeds, "
                        f"{config.validation_steps} steps)"
                    )
                    self._progress.elapsed_seconds = time.time() - t_start

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
                    logger.debug(
                        f"Threaded validation: skipping {source_result_id[:8]} — "
                        f"model failed to reconstruct for all {config.validation_n_seeds} seeds"
                    )
                    continue

                # Compute validation metrics
                passed_seeds = [r for r in seed_results if r.get("passed")]
                loss_ratios = [
                    r["loss_ratio"]
                    for r in seed_results
                    if r.get("loss_ratio") is not None
                ]

                val_loss_ratio = (
                    sum(loss_ratios) / len(loss_ratios) if loss_ratios else None
                )
                multi_seed_std = 0.0
                robustness_score = 1.0
                is_unstable = False

                if len(loss_ratios) > 1:
                    mean_lr = sum(loss_ratios) / len(loss_ratios)
                    variance = sum((lr - mean_lr) ** 2 for lr in loss_ratios) / len(
                        loss_ratios
                    )
                    multi_seed_std = variance**0.5

                    # Task 3G: Check for instability and compute robustness_score
                    if variance > 0.15:
                        is_unstable = True
                    if mean_lr > 1e-6:
                        robustness_score = max(0.0, 1.0 - (multi_seed_std / mean_lr))

                # Init sensitivity: std between default and xavier seeds
                init_sensitivity_std = None
                default_losses = [
                    r["loss_ratio"]
                    for r in seed_results
                    if r.get("init_scheme") == "default"
                    and r.get("loss_ratio") is not None
                ]
                xavier_losses = [
                    r["loss_ratio"]
                    for r in seed_results
                    if r.get("init_scheme") == "xavier_uniform"
                    and r.get("loss_ratio") is not None
                ]
                if default_losses and xavier_losses:
                    default_mean = sum(default_losses) / len(default_losses)
                    xavier_mean = sum(xavier_losses) / len(xavier_losses)
                    init_sensitivity_std = abs(default_mean - xavier_mean)

                # Baseline comparison at validation scale
                val_baseline_ratio = None
                best_seed = None
                if loss_ratios:
                    best_seed = min(
                        (r for r in seed_results if r.get("final_loss") is not None),
                        key=lambda r: r["final_loss"],
                        default=None,
                    )
                    if best_seed is not None:
                        try:
                            baseline = self._get_baseline()
                            baseline_steps = int(
                                best_seed.get("n_train_steps")
                                or config.validation_steps
                            )
                            baseline_recipe = self._resolve_baseline_recipe(
                                best_seed, default_lr=config.stage1_lr
                            )
                            bl_data_fn, bl_data_tag, bl_cache = (
                                self._make_baseline_data_fn(config)
                            )
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
                                    v_data_fn, v_data_tag, v_cache = (
                                        self._make_baseline_data_fn(config, split="val")
                                    )
                                    v_baseline_ratio = baseline.compare(
                                        v_loss,
                                        d_model=config.model_dim,
                                        seq_len=min(
                                            128,
                                            int(
                                                getattr(
                                                    config, "validation_seq_len", 128
                                                )
                                            ),
                                        ),
                                        n_steps=max(1, baseline_steps),
                                        vocab_size=config.vocab_size,
                                        batch_size=int(
                                            getattr(config, "validation_batch_size", 4)
                                        ),
                                        lr=baseline_recipe["lr"],
                                        device=dev_str,
                                        n_layers=config.n_layers,
                                        optimizer_name=baseline_recipe[
                                            "optimizer_name"
                                        ],
                                        weight_decay=baseline_recipe["weight_decay"],
                                        momentum=baseline_recipe["momentum"],
                                        betas=baseline_recipe["betas"],
                                        data_fn=v_data_fn,
                                        data_tag=v_data_tag,
                                        cache_data_fn=v_cache,
                                    )
                                    program_metrics[
                                        "validation_baseline_loss_ratio"
                                    ] = v_baseline_ratio
                                except Exception:
                                    pass
                        except Exception:
                            pass

                # Parameter-normalized baseline comparison
                val_normalized_ratio = None
                val_param_efficiency = None
                source_params = (
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
                        baseline = self._get_baseline()
                        baseline_steps = int(
                            best_seed.get("n_train_steps") or config.validation_steps
                        )
                        baseline_recipe = self._resolve_baseline_recipe(
                            best_seed, default_lr=config.stage1_lr
                        )
                        bl_data_fn, bl_data_tag, bl_cache = self._make_baseline_data_fn(
                            config
                        )
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

                is_breakthrough = ev_res["is_breakthrough"]
                flop_gated = ev_res["flop_gated"]
                quant_int8_retention = ev_res["quant_int8_retention"]
                quant_quality_per_byte = ev_res["quant_quality_per_byte"]
                long_context_score = ev_res["long_context_score"]
                long_context_details = ev_res["long_context_details"]
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
                scaling_d512_param_efficiency = ev_res.get(
                    "scaling_d512_param_efficiency"
                )
                scaling_result = ev_res.get("scaling_result")
                nov_conf = source.get("novelty_confidence", 0) if source else 0

                tier = "breakthrough" if is_breakthrough else "validation"

                validation_entry = {
                    "result_id": source_result_id,
                    "val_loss_ratio": val_loss_ratio,
                    "val_baseline_ratio": val_baseline_ratio,
                    "val_normalized_ratio": val_normalized_ratio,
                    "param_efficiency": val_param_efficiency,
                    "multi_seed_std": multi_seed_std,
                    "robustness_score": robustness_score,
                    "is_unstable": is_unstable,
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

                # Update leaderboard — direct lookup by result_id
                entry = nb.get_leaderboard_entry(source_result_id)
                if entry:
                    nb.promote_to_tier(
                        entry_id=entry["entry_id"],
                        tier=tier,
                        validation_loss_ratio=val_loss_ratio,
                        validation_baseline_ratio=val_baseline_ratio,
                        validation_multi_seed_std=multi_seed_std,
                        validation_robustness_score=robustness_score,
                        validation_is_unstable=int(is_unstable),
                        validation_passed=len(passed_seeds) > 0,
                        normalized_baseline_ratio=val_normalized_ratio,
                        param_efficiency=val_param_efficiency,
                        quant_int8_retention=quant_int8_retention,
                        quant_quality_per_byte=quant_quality_per_byte,
                        robustness_long_ctx_score=long_context_score,
                        robustness_noise_score=noise_score,
                        init_sensitivity_std=init_sensitivity_std,
                        fp_jacobian_spectral_norm=source.get(
                            "fp_jacobian_spectral_norm"
                        ),
                        scaling_param_efficiency=scaling_param_efficiency,
                        scaling_d512_param_efficiency=scaling_d512_param_efficiency,
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
                    # Store detailed benchmark payload
                    external_benchmarks_payload = {}
                    scaling_payload = coerce_dict_payload(scaling_result)
                    if scaling_payload is not None:
                        external_benchmarks_payload.update(scaling_payload)
                        external_benchmarks_payload["scaling_comparison"] = (
                            scaling_payload
                        )
                    if long_context_details is not None:
                        external_benchmarks_payload["long_context"] = (
                            long_context_details
                        )
                    if external_benchmarks_payload:
                        nb.set_external_benchmarks(
                            source_result_id, external_benchmarks_payload
                        )

                # Trajectory probe — run after leaderboard update to get
                # peak_ppl / steps_to_divergence / ppl_500 for breakthrough
                # detection and composite scoring.
                trajectory_composite = None
                try:
                    if graph_json_str and len(passed_seeds) > 0:
                        from ...eval.wikitext_eval import evaluate_wikitext_trajectory

                        traj_graph = graph_from_json(graph_json_str)
                        traj_layers = [traj_graph] * config.n_layers
                        traj_model = compile_model(
                            traj_layers,
                            vocab_size=config.vocab_size,
                            max_seq_len=128,
                        )
                        traj_model = traj_model.to(dev)
                        traj_result = evaluate_wikitext_trajectory(
                            traj_model,
                            config.vocab_size,
                            dev_str,
                            checkpoints=(200, 500, 1000, 2000, 4000),
                            seq_len=128,
                        )
                        del traj_model
                        if dev.type == "cuda":
                            torch.cuda.empty_cache()

                        traj_peak_ppl = traj_result.get("peak_ppl")
                        traj_steps_div = traj_result.get("steps_to_divergence")
                        traj_ppl_500 = None
                        traj_ckpts = traj_result.get("checkpoints", {})
                        if 500 in traj_ckpts:
                            traj_ppl_500 = traj_ckpts[500].get("ppl")

                        # Update leaderboard with trajectory data
                        entry = nb.get_leaderboard_entry(source_result_id)
                        if entry:
                            traj_update = {}
                            if traj_peak_ppl is not None:
                                traj_update["peak_ppl"] = traj_peak_ppl
                                import math as _math

                                _vocab = config.vocab_size or 32000
                                _ws = max(
                                    0.0,
                                    _math.log(_vocab / traj_peak_ppl)
                                    / _math.log(_vocab),
                                )
                                traj_update["wikitext_score"] = round(_ws, 4)
                            if traj_result.get("peak_step") is not None:
                                traj_update["peak_step"] = traj_result["peak_step"]
                            if traj_steps_div is not None:
                                traj_update["steps_to_divergence"] = traj_steps_div
                            if traj_ppl_500 is not None:
                                traj_update["ppl_500"] = traj_ppl_500
                            if traj_update:
                                nb.promote_to_tier(
                                    entry_id=entry["entry_id"],
                                    tier=tier,
                                    **traj_update,
                                )
                                # Re-read composite for breakthrough check
                                updated = nb.conn.execute(
                                    "SELECT composite_score FROM leaderboard WHERE entry_id = ?",
                                    (entry["entry_id"],),
                                ).fetchone()
                                if updated:
                                    trajectory_composite = updated["composite_score"]
                        logger.info(
                            "Trajectory probe %s: peak_ppl=%.1f steps_to_div=%s ppl_500=%s composite=%.1f",
                            source_result_id[:8],
                            traj_peak_ppl or 0,
                            traj_steps_div,
                            traj_ppl_500,
                            trajectory_composite or 0,
                        )
                except Exception as e:
                    logger.warning(
                        "Trajectory probe failed for %s: %s", source_result_id[:8], e
                    )

                # Trajectory-aware breakthrough: composite > 300 or
                # never-diverging with frontier-quality PPL
                if not is_breakthrough and trajectory_composite is not None:
                    if trajectory_composite > 300.0:
                        is_breakthrough = True
                        logger.info(
                            "Trajectory-aware breakthrough: %s composite=%.1f",
                            source_result_id[:8],
                            trajectory_composite,
                        )

                # Breakthrough detection
                if is_breakthrough:
                    ctx = build_validation_context([source], [validation_entry])
                    announcement = self.aria.announce_breakthrough(ctx)
                    nb.add_entry(
                        ExperimentEntry(
                            entry_type="insight",
                            title="BREAKTHROUGH DETECTED",
                            content=announcement,
                            experiment_id=exp_id,
                            tags=["breakthrough"],
                        )
                    )
                    self._emit_event(
                        "breakthrough_detected",
                        {
                            "experiment_id": exp_id,
                            "result_id": source_result_id,
                            "val_loss_ratio": val_loss_ratio,
                            "val_baseline_ratio": val_baseline_ratio,
                            "multi_seed_std": multi_seed_std,
                            "announcement": announcement,
                        },
                    )

                # Record validation result
                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=source.get("graph_fingerprint", source_result_id),
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
                except Exception as e:
                    logger.debug("Validation checkpoint save failed: %s", e)

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

            # Clean up validation checkpoints on success
            if not config.keep_checkpoints:
                try:
                    ckpt.cleanup(exp_id)
                except Exception:
                    pass

            with self._lock:
                self._progress.status = "completed"
                self._progress.elapsed_seconds = time.time() - t_start
                self._progress.aria_message = (
                    summary.split("\n")[-1] if summary else "Validation complete."
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
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
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
            scale_config = RunConfig.from_dict(config.to_dict())
            scale_config.stage1_steps = config.scale_up_steps
            scale_config.stage1_batch_size = config.scale_up_batch_size
            scale_config.max_seq_len = config.scale_up_seq_len

            for prog_idx, source_result_id in enumerate(result_ids):
                if self._stop_event.is_set():
                    break

                with self._lock:
                    self._progress.current_program = prog_idx + 1
                    self._progress.status = "training"
                    self._progress.aria_message = (
                        f"Scale-up {prog_idx + 1}/{len(result_ids)}: "
                        f"training {source_result_id[:8]}... "
                        f"({config.scale_up_steps} steps, batch={config.scale_up_batch_size})"
                    )
                    self._progress.elapsed_seconds = time.time() - t_start

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
                    continue

                try:
                    graph = graph_from_json(graph_json_str)
                except Exception as e:
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
                    continue

                # Compile model
                try:
                    layer_graphs = [graph] * config.n_layers
                    model = compile_model(
                        layer_graphs,
                        vocab_size=config.vocab_size,
                        max_seq_len=config.scale_up_seq_len,
                    )
                except Exception as e:
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
                    continue

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
                        except Exception:
                            pass

                program_metrics["stage_at_death"] = (
                    "survived" if s1_passed else "stage1"
                )

                # Diagnostic tasks for S1 survivors
                if s1_passed and model is not None:
                    try:
                        diag = run_diagnostic_suite(model, device=dev_str)
                        program_metrics["diagnostic_tasks_json"] = json.dumps(
                            diag.to_dict()
                        )
                        program_metrics["diagnostic_score"] = diag.diagnostic_score
                    except Exception:
                        pass

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
                    except Exception as e:
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
                    except Exception as e:
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
                    except Exception:
                        pass

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
                    except Exception:
                        pass

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
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            # Guard: if no programs were processed at all, fail with clear reason
            if results["stage0_passed"] == 0 and results["total"] > 0:
                reason = (
                    f"All {results['total']} source programs were skipped "
                    f"(not found or failed to compile). "
                    f"Result IDs: {', '.join(r[:12] for r in result_ids)}"
                )
                logger.warning("Scale-up produced no results: %s", reason)
                nb.fail_experiment(exp_id, reason)
                with self._lock:
                    self._progress.status = "failed"
                    self._progress.error = reason
                    self._progress.aria_message = self.aria.react_to_failure(reason)
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

            with self._lock:
                self._progress.status = "completed"
                self._progress.elapsed_seconds = time.time() - t_start
                self._progress.aria_message = (
                    summary.split("\n")[-1] if summary else "Scale-up complete."
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
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
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
