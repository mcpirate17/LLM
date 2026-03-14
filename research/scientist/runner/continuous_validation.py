"""Continuous validation methods (seeds, metrics, external evals, inline validation), split from continuous.py."""

from __future__ import annotations

import gc
import torch
import torch.nn as nn

from ...eval.cross_task_eval import evaluate_cross_task_robustness
from ...eval.efficiency_wall import evaluate_efficiency_wall
from ...eval.long_context import run_long_context_sweep
from ...eval.noise_sensitivity import evaluate_noise_sensitivity
from ...eval.perf_budget import evaluate_perf_budget_gate
from ...eval.quantization import evaluate_sparse_quant_quality
from ...eval.routing_heatmap import evaluate_routing_heatmap
from ...eval.sparsity import evaluate_activation_sparsity
from ...training.training_program import synthesize_training_program
from ..shared_utils import coerce_dict_payload
from ..notebook import LabNotebook, ExperimentEntry
from ..llm.context_experiment import build_validation_context

import logging
logger = logging.getLogger(__name__)

from ._types import RunConfig


class _ContinuousValidationMixin:
    """Validation seed runs, metrics computation, external evals, and inline validation."""

    __slots__ = ()

    def _scaling_reference_families(self, config: RunConfig) -> tuple[str, ...]:
        raw = str(getattr(config, "scaling_reference_families", "gpt2") or "gpt2")
        families = tuple(part.strip() for part in raw.split(",") if part.strip())
        return families or ("gpt2",)

    def _run_scaling_reference_compare(
        self,
        *,
        config: RunConfig,
        dev_str: str,
        best_seed: dict | None,
        val_loss_ratio: float | None,
        source_params: float | int,
        source: dict | None,
        d_model: int,
    ) -> dict | None:
        if val_loss_ratio is None or source_params is None or int(source_params) <= 0:
            return None
        candidate_loss = best_seed.get("final_loss") if best_seed else None
        if candidate_loss is None:
            return None
        candidate_flops = int((source or {}).get("flops_forward") or 0)
        if candidate_flops <= 0:
            candidate_flops = max(1, int(source_params) * 2)
        baseline_steps = int(best_seed.get("n_train_steps") or config.validation_steps) if best_seed else int(config.validation_steps)
        baseline_recipe = self._resolve_baseline_recipe(best_seed, default_lr=config.stage1_lr)
        data_fn, data_tag, _cache = self._make_baseline_data_fn(config)
        comparison = self._get_scaling_reference_manager().compare_candidate(
            candidate_loss=float(candidate_loss),
            candidate_params=int(source_params),
            candidate_flops=candidate_flops,
            d_model=int(d_model),
            n_steps=max(1, baseline_steps),
            seq_len=min(128, int(getattr(config, "validation_seq_len", 128) or 128)),
            vocab_size=int(config.vocab_size),
            batch_size=max(1, min(4, int(getattr(config, "validation_batch_size", 4) or 4))),
            lr=float(baseline_recipe["lr"]),
            device=dev_str,
            data_fn=data_fn,
            data_tag=data_tag,
            families=self._scaling_reference_families(config),
            param_efficiency_target=float(config.scaling_param_efficiency_target),
            flop_ceiling=float(config.scaling_flop_ceiling),
        )
        payload = comparison.to_dict()
        payload["d_model"] = int(d_model)
        payload["proxy_only"] = int(d_model) != int(config.model_dim)
        return payload

    def _make_validation_model_factory(
        self,
        model_source: str,
        arch_spec_json_str: str,
        graph_json_str: str,
        config: RunConfig,
    ):
        seq_len = int(getattr(config, "validation_seq_len", 128) or 128)

        def _factory():
            return self._build_model_from_source(
                model_source,
                arch_spec_json_str,
                graph_json_str,
                config,
                seq_len_override=seq_len,
            )

        return _factory

    def _make_validation_input_batches(
        self,
        config: RunConfig,
        dev: torch.device,
        source_result_id: str,
        n_batches: int = 2,
    ):
        seq_len = min(128, int(getattr(config, "validation_seq_len", 128) or 128))
        batch_size = max(1, min(4, int(getattr(config, "validation_batch_size", 2) or 2)))
        return [
            self._sample_training_input_ids(
                config=config,
                dev=dev,
                batch_size=batch_size,
                seq_len=seq_len,
                seed=self._stable_seed(source_result_id, "validation_eval", idx),
            )
            for idx in range(n_batches)
        ]

    def _run_external_evals(
        self,
        *,
        config: RunConfig,
        dev: torch.device,
        dev_str: str,
        best_seed: dict | None,
        model_source: str,
        arch_spec_json_str: str | None,
        graph_json_str: str | None,
        source: dict | None,
        source_result_id: str,
        exp_id: str,
        val_loss_ratio: float | None,
        val_baseline_ratio: float | None,
        val_normalized_ratio: float | None,
        multi_seed_std: float,
        passed_seeds: list,
        source_params: float | int,
    ) -> dict:
        del exp_id, multi_seed_std, passed_seeds
        result = {
            "is_breakthrough": False,
            "flop_gated": False,
            "quant_int8_retention": None,
            "quant_quality_per_byte": None,
            "long_context_score": None,
            "long_context_details": None,
            "noise_score": None,
            "ood_result": None,
            "sensitivity_result": None,
            "activation_sparsity_score": None,
            "dead_neuron_ratio": None,
            "routing_collapse_score": None,
            "wikitext_perplexity": None,
            "wikitext_score": None,
            "tinystories_perplexity": None,
            "tinystories_score": None,
            "cross_task_score": None,
            "efficiency_wall_score": None,
            "max_viable_seq_len": None,
            "scaling_regime": None,
            "scaling_param_efficiency": val_normalized_ratio,
            "scaling_flop_efficiency": None,
            "scaling_gate_passed_val": None,
            "scaling_best_family": None,
            "scaling_confidence": None,
            "scaling_result": None,
            "scaling_d512_param_efficiency": None,
        }
        scaling_enabled = bool(getattr(config, "enable_scaling_comparison", True))
        model_factory = self._make_validation_model_factory(
            model_source,
            arch_spec_json_str,
            graph_json_str,
            config,
        )
        base_final_loss = float(best_seed.get("final_loss")) if best_seed and best_seed.get("final_loss") is not None else None
        input_batches = self._make_validation_input_batches(config, dev, source_result_id)
        if val_loss_ratio is not None:
            try:
                result["ood_result"] = self._ood_robustness_check(
                    model_factory, config, dev, n_steps=min(100, max(20, int(config.validation_steps) // 50))
                )
            except Exception as exc:
                logger.debug("OOD robustness check failed for %s: %s", source_result_id[:8], exc)
            try:
                result["sensitivity_result"] = self._sensitivity_check(
                    model_factory,
                    config,
                    dev,
                    base_loss_ratio=float(val_loss_ratio),
                    n_steps=min(100, max(20, int(config.validation_steps) // 50)),
                )
            except Exception as exc:
                logger.debug("Sensitivity check failed for %s: %s", source_result_id[:8], exc)
        model = None
        try:
            model = model_factory()
            if model is None:
                return result
            model = model.to(dev)
            eval_seq_len = min(128, int(getattr(config, "validation_seq_len", 128) or 128))
            try:
                from ...eval.wikitext_eval import evaluate_wikitext_perplexity
                wt_result = evaluate_wikitext_perplexity(model, config.vocab_size, dev_str, n_train_steps=200, seq_len=eval_seq_len)
                result["wikitext_perplexity"] = wt_result.get("wikitext_perplexity")
                result["wikitext_score"] = wt_result.get("wikitext_score")
            except Exception as exc:
                logger.debug("WikiText eval skipped for %s: %s", source_result_id[:8], exc)
            try:
                from ...eval.tinystories_eval import evaluate_tinystories
                ts_result = evaluate_tinystories(model, config.vocab_size, dev_str, n_train_steps=200, seq_len=eval_seq_len)
                result["tinystories_perplexity"] = ts_result.get("tinystories_perplexity")
                result["tinystories_score"] = ts_result.get("tinystories_score")
            except Exception as exc:
                logger.debug("TinyStories eval skipped for %s: %s", source_result_id[:8], exc)
            try:
                long_context = run_long_context_sweep(
                    model_factory,
                    config.vocab_size,
                    dev,
                    base_loss=base_final_loss or max(float(val_loss_ratio or 1.0), 1e-6),
                    seq_lens=(512, 1024),
                    n_steps=min(60, max(20, int(config.validation_steps) // 100)),
                    batch_size=max(1, min(2, int(getattr(config, "validation_batch_size", 2) or 2))),
                    lr=float(best_seed.get("optimizer_lr") or config.stage1_lr) if best_seed else float(config.stage1_lr),
                )
                result["long_context_score"] = long_context.get("long_context_score")
                result["long_context_details"] = long_context
                result["max_viable_seq_len"] = long_context.get("max_viable_len")
            except Exception as exc:
                logger.debug("Long-context sweep skipped for %s: %s", source_result_id[:8], exc)
            try:
                noise_result = evaluate_noise_sensitivity(model, input_batches, dev, vocab_size=int(config.vocab_size))
                result["noise_score"] = noise_result.get("noise_sensitivity_score")
            except Exception as exc:
                logger.debug("Noise sensitivity skipped for %s: %s", source_result_id[:8], exc)
            try:
                sparsity_result = evaluate_activation_sparsity(model, input_batches, dev)
                result["activation_sparsity_score"] = sparsity_result.get("activation_sparsity_score")
                result["dead_neuron_ratio"] = sparsity_result.get("dead_neuron_ratio")
            except Exception as exc:
                logger.debug("Activation sparsity skipped for %s: %s", source_result_id[:8], exc)
            try:
                routing_result = evaluate_routing_heatmap(model, input_batches, dev)
                result["routing_collapse_score"] = routing_result.get("routing_collapse_score")
            except Exception as exc:
                logger.debug("Routing heatmap skipped for %s: %s", source_result_id[:8], exc)
            try:
                quant_result = evaluate_sparse_quant_quality(model, input_batches, dev)
                if quant_result:
                    result["quant_int8_retention"] = quant_result.get("full_retention")
                    result["quant_quality_per_byte"] = quant_result.get("quality_per_byte")
            except Exception as exc:
                logger.debug("Sparse+quant eval skipped for %s: %s", source_result_id[:8], exc)
            if scaling_enabled:
                try:
                    wall_result = evaluate_efficiency_wall(model, int(config.vocab_size), dev)
                    result["efficiency_wall_score"] = wall_result.get("efficiency_wall_score")
                    result["scaling_regime"] = wall_result.get("scaling_regime")
                    result["max_viable_seq_len"] = max(
                        result["max_viable_seq_len"] or 0,
                        int(wall_result.get("max_viable_seq_len") or 0),
                    ) or None
                    result["scaling_flop_efficiency"] = wall_result.get("time_scaling_factor")
                except Exception as exc:
                    logger.debug("Efficiency-wall eval skipped for %s: %s", source_result_id[:8], exc)
        finally:
            if model is not None:
                del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        try:
            cross_task = evaluate_cross_task_robustness(
                model_factory,
                vocab_size=int(config.vocab_size),
                device=dev,
                n_train_steps=min(80, max(20, int(config.validation_steps) // 100)),
                batch_size=max(1, min(4, int(getattr(config, "validation_batch_size", 4) or 4))),
                seq_len=min(128, int(getattr(config, "validation_seq_len", 128) or 128)),
            )
            result["cross_task_score"] = cross_task.get("cross_task_score")
        except Exception as exc:
            logger.debug("Cross-task eval skipped for %s: %s", source_result_id[:8], exc)
        scaling_gate_passed = (
            not scaling_enabled
            or (
                result["scaling_param_efficiency"] is not None
                and float(result["scaling_param_efficiency"]) >= float(config.scaling_param_efficiency_target)
                and (
                    result["scaling_flop_efficiency"] is None
                    or float(result["scaling_flop_efficiency"]) <= float(config.scaling_flop_ceiling)
                )
            )
        )
        if scaling_enabled:
            try:
                scaling_payload = self._run_scaling_reference_compare(
                    config=config,
                    dev_str=dev_str,
                    best_seed=best_seed,
                    val_loss_ratio=val_loss_ratio,
                    source_params=source_params,
                    source=source,
                    d_model=int(config.model_dim),
                )
                if scaling_payload is not None:
                    result["scaling_result"] = scaling_payload
                    result["scaling_param_efficiency"] = scaling_payload.get("best_param_efficiency")
                    result["scaling_flop_efficiency"] = scaling_payload.get("flop_efficiency")
                    result["scaling_best_family"] = scaling_payload.get("best_param_efficiency_family")
                    scaling_gate_passed = bool(scaling_payload.get("scaling_gate_passed"))
                    result["scaling_confidence"] = str(scaling_payload.get("confidence") or "local_reference")
            except Exception as exc:
                logger.debug("Scaling reference comparison skipped for %s: %s", source_result_id[:8], exc)
            if bool(getattr(config, "scaling_d512_enabled", True)):
                try:
                    d512_payload = self._run_scaling_reference_compare(
                        config=config,
                        dev_str=dev_str,
                        best_seed=best_seed,
                        val_loss_ratio=val_loss_ratio,
                        source_params=source_params,
                        source=source,
                        d_model=512,
                    )
                    if d512_payload is not None:
                        result["scaling_d512_param_efficiency"] = d512_payload.get("best_param_efficiency")
                        if isinstance(result.get("scaling_result"), dict):
                            result["scaling_result"]["d512_result"] = d512_payload
                except Exception as exc:
                    logger.debug("d512 scaling comparison skipped for %s: %s", source_result_id[:8], exc)
        raw_breakthrough_passed = (
            val_loss_ratio is not None
            and float(val_loss_ratio) <= float(getattr(config, "breakthrough_raw_threshold", 0.70) or 0.70)
        )
        normalized_breakthrough_passed = (
            val_normalized_ratio is not None
            and float(val_normalized_ratio) >= float(
                getattr(config, "breakthrough_normalized_threshold", 0.85) or 0.85
            )
        )
        result["scaling_gate_passed_val"] = scaling_gate_passed
        if result["scaling_confidence"] is None:
            result["scaling_confidence"] = "disabled" if not scaling_enabled else "high" if scaling_gate_passed else "low"
        if result["scaling_best_family"] is None:
            result["scaling_best_family"] = str((source or {}).get("most_similar_to") or "reference")
        if result["scaling_result"] is None:
            result["scaling_result"] = {
                "param_efficiency": result["scaling_param_efficiency"],
                "flop_efficiency": result["scaling_flop_efficiency"],
                "gate_passed": scaling_gate_passed,
                "confidence": result["scaling_confidence"],
                "enabled": scaling_enabled,
            }
        else:
            result["scaling_result"]["enabled"] = scaling_enabled
            result["scaling_result"]["gate_passed"] = scaling_gate_passed
        result["is_breakthrough"] = bool(
            raw_breakthrough_passed
            and normalized_breakthrough_passed
            and scaling_gate_passed
            and (val_baseline_ratio is None or float(val_baseline_ratio) < 1.0)
        )
        result["flop_gated"] = bool(not scaling_gate_passed and result["scaling_flop_efficiency"] is not None)
        return result

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
                "total": _total_progs,
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
                scaling_d512_param_efficiency = ev_res.get("scaling_d512_param_efficiency")
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

                # Update leaderboard — direct lookup by result_id
                entry = nb.get_leaderboard_entry(source_result_id)
                if entry:
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
                    # Store detailed benchmark payload in external_benchmarks_json
                    external_benchmarks_payload = {}
                    scaling_payload = coerce_dict_payload(scaling_result)
                    if scaling_payload is not None:
                        external_benchmarks_payload.update(scaling_payload)
                        external_benchmarks_payload["scaling_comparison"] = scaling_payload
                    if long_context_details is not None:
                        external_benchmarks_payload["long_context"] = long_context_details
                    if external_benchmarks_payload:
                        nb.set_external_benchmarks(source_result_id, external_benchmarks_payload)

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

                # Trajectory probe — peak_ppl / steps_to_divergence / ppl_500
                trajectory_composite = None
                try:
                    if graph_json_str and len(passed_seeds) > 0:
                        from ...eval.wikitext_eval import evaluate_wikitext_trajectory
                        from ...synthesis.serializer import graph_from_json
                        from ..native_runner import compile_model_native_first as _compile_model
                        traj_graph = graph_from_json(graph_json_str)
                        traj_layers = [traj_graph] * config.n_layers
                        traj_model = _compile_model(
                            traj_layers,
                            vocab_size=config.vocab_size,
                            max_seq_len=128,
                        )
                        traj_model = traj_model.to(dev)
                        traj_result = evaluate_wikitext_trajectory(
                            traj_model, config.vocab_size, dev_str,
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

                        entry = nb.get_leaderboard_entry(source_result_id)
                        if entry:
                            traj_update = {}
                            if traj_peak_ppl is not None:
                                traj_update["peak_ppl"] = traj_peak_ppl
                                import math as _math
                                _vocab = config.vocab_size or 32000
                                _ws = max(0.0, _math.log(_vocab / traj_peak_ppl) / _math.log(_vocab))
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
                    logger.warning("Trajectory probe failed for %s: %s", source_result_id[:8], e)

                # Trajectory-aware breakthrough
                if not is_breakthrough and trajectory_composite is not None:
                    if trajectory_composite > 300.0:
                        is_breakthrough = True
                        logger.info(
                            "Trajectory-aware breakthrough: %s composite=%.1f",
                            source_result_id[:8], trajectory_composite,
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
