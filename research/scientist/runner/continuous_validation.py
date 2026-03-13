"""Continuous validation methods (seeds, metrics, external evals, inline validation), split from continuous.py."""

from __future__ import annotations

import gc

import torch
import torch.nn as nn

from ...eval.perf_budget import evaluate_perf_budget_gate
from ...training.training_program import synthesize_training_program
from ..notebook import LabNotebook, ExperimentEntry
from ..llm.context_experiment import build_validation_context

import logging
logger = logging.getLogger(__name__)

from ._types import RunConfig


class _ContinuousValidationMixin:
    """Validation seed runs, metrics computation, external evals, and inline validation."""

    __slots__ = ()

    def _validation_run_seeds(
        self,
        config, val_config,
        dev,
        exp_id: str,
        prog_idx: int,
        _total_progs: int,
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

        self._live_training_context = {"exp_id": exp_id, "phase": "validation"}
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
                ev_res = self._run_external_evals(
                    config=config, dev=dev, dev_str=dev_str,
                    best_seed=best_seed, model_source=model_source,
                    arch_spec_json_str=arch_spec_json_str,
                    graph_json_str=graph_json_str, source=source,
                    source_result_id=source_result_id, exp_id=exp_id,
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
                nov_conf = source.get("novelty_confidence", 0) if source else 0

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
