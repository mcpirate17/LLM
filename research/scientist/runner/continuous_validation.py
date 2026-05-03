"""Continuous validation methods (seeds, metrics, external evals, inline validation), split from continuous.py."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from ...eval.perf_budget import evaluate_perf_budget_gate
from ...training.checkpointing import CheckpointManager
from ...training.training_program import synthesize_training_program
from ._helpers import (
    clear_gpu_memory,
    compute_seed_metrics,
    run_baseline_comparison,
    build_validation_entry,
    promote_validation_candidate,
    run_trajectory_probe,
    handle_breakthrough,
)
from ._eval_registry import (
    EvalContext,
    EVAL_SPECS,
    run_eval_suite,
    apply_breakthrough_logic,
)
from ._lifecycle import _LifecycleMixin
from ..notebook import LabNotebook

import logging

logger = logging.getLogger(__name__)

from ._types import RunConfig, ExternalEvalResult


class _ContinuousValidationMixin:
    """Validation seed runs, metrics computation, external evals, and inline validation."""

    __slots__ = ()
    _publish_terminal_event = _LifecycleMixin._publish_terminal_event
    _publish_continuous_validation_terminal_event = (
        _LifecycleMixin._publish_terminal_event
    )
    _fail_experiment_compat = _LifecycleMixin._fail_experiment_compat
    _complete_experiment_compat = _LifecycleMixin._complete_experiment_compat
    _log_learning_event_compat = _LifecycleMixin._log_learning_event_compat

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
        baseline_steps = (
            int(best_seed.get("n_train_steps") or config.validation_steps)
            if best_seed
            else int(config.validation_steps)
        )
        baseline_recipe = self._resolve_baseline_recipe(
            best_seed, default_lr=config.stage1_lr
        )
        data_fn, data_tag, cache_data_fn = self._make_baseline_data_fn(config)
        comparison = self._get_scaling_reference_manager().compare_candidate(
            candidate_loss=float(candidate_loss),
            candidate_params=int(source_params),
            candidate_flops=candidate_flops,
            d_model=int(d_model),
            n_steps=max(1, baseline_steps),
            seq_len=min(128, config.validation_seq_len),
            vocab_size=int(config.vocab_size),
            batch_size=max(1, min(4, config.validation_batch_size)),
            lr=float(baseline_recipe["lr"]),
            device=dev_str,
            data_fn=data_fn,
            data_tag=data_tag,
            families=self._scaling_reference_families(config),
            param_efficiency_target=float(config.scaling_param_efficiency_target),
            flop_ceiling=float(config.scaling_flop_ceiling),
            cacheable=bool(cache_data_fn),
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
        seq_len = config.validation_seq_len

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
        seq_len = min(128, config.validation_seq_len)
        batch_size = max(1, min(4, config.validation_batch_size))
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
    ) -> ExternalEvalResult:
        del multi_seed_std
        _rid_short = source_result_id[:8]
        _eval_test_index = 0
        _EVAL_TOTAL_TESTS = len(EVAL_SPECS)

        def _vstatus(phase: str) -> None:
            nonlocal _eval_test_index
            _eval_test_index += 1
            logger.info(
                "validation[%s]: %s (%d/%d)",
                _rid_short,
                phase,
                _eval_test_index,
                _EVAL_TOTAL_TESTS,
            )
            self._emit_event(
                "validation_phase",
                {
                    "experiment_id": exp_id,
                    "result_id": source_result_id,
                    "phase": phase,
                    "test_index": _eval_test_index,
                    "total_tests": _EVAL_TOTAL_TESTS,
                },
            )
            self._update_progress(
                status=f"validation: {phase} ({_eval_test_index}/{_EVAL_TOTAL_TESTS})"
            )

        scaling_enabled = bool(getattr(config, "enable_scaling_comparison", True))
        result = ExternalEvalResult(scaling_param_efficiency=val_normalized_ratio)

        ctx = EvalContext(
            config=config,
            dev=dev,
            dev_str=dev_str,
            model=None,
            model_factory=self._make_validation_model_factory(
                model_source,
                arch_spec_json_str,
                graph_json_str,
                config,
            ),
            input_batches=self._make_validation_input_batches(
                config, dev, source_result_id
            ),
            best_seed=best_seed,
            base_final_loss=(
                float(best_seed.get("final_loss"))
                if best_seed and best_seed.get("final_loss") is not None
                else None
            ),
            val_loss_ratio=val_loss_ratio,
            val_baseline_ratio=val_baseline_ratio,
            val_normalized_ratio=val_normalized_ratio,
            source=source,
            source_params=int(source_params),
            source_result_id=source_result_id,
            scaling_enabled=scaling_enabled,
            ood_check=self._ood_robustness_check,
            sensitivity_check=self._sensitivity_check,
            scaling_compare=self._run_scaling_reference_compare,
            stop_event=self._stop_event,
        )

        run_eval_suite(ctx=ctx, result=result, vstatus=_vstatus)

        apply_breakthrough_logic(
            result,
            config,
            val_loss_ratio=val_loss_ratio,
            val_baseline_ratio=val_baseline_ratio,
            val_normalized_ratio=val_normalized_ratio,
            passed_seeds=passed_seeds,
            source=source,
            scaling_enabled=scaling_enabled,
            source_result_id=source_result_id,
        )

        return result

    def _validation_run_seeds(
        self,
        config,
        val_config,
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
        checkpoint_manager: CheckpointManager | None = None,
    ):
        seed_results = []
        checkpoint_interval = int(
            getattr(config, "phase_checkpoint_step_interval", 0) or 0
        )
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
                logger.warning("Model reconstruction FAILED for seed %d: %s", seed, e)
                continue

            self._emit_event(
                "validation_progress",
                {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": _total_progs,
                    "source_result_id": source_result_id,
                    "seed": seed + 1,
                    "total_seeds": config.validation_n_seeds,
                    "status": f"seed {seed + 1}/{config.validation_n_seeds}",
                },
            )

            # Train (use best training program if available)
            resume_state = None
            if checkpoint_manager is not None:
                loaded_state = checkpoint_manager.load_phase(
                    exp_id, "validation", prog_idx, seed
                )
                if loaded_state and int(loaded_state.get("step", 0) or 0) > 0:
                    resume_state = loaded_state
            base_ctx = {"exp_id": exp_id, "phase": "validation"}
            self._live_training_context = {
                **base_ctx,
                "source_result_id": source_result_id,
                "candidate_index": prog_idx + 1,
                "total_candidates": _total_progs,
                "training_program_index": seed + 1,
                "total_training_programs": config.validation_n_seeds,
                "training_program_label": f"seed {seed + 1}",
                "training_seed": seed,
                "run_kind": "validation",
                "checkpoint_manager": checkpoint_manager,
                "checkpoint_phase": "validation",
                "checkpoint_candidate_idx": prog_idx,
                "checkpoint_seed_idx": seed,
                "checkpoint_interval_steps": checkpoint_interval,
                "checkpoint_resume_state": resume_state,
            }
            if best_tp_json:
                try:
                    tp_data = self._cached_json_load(best_tp_json)
                    tp = synthesize_training_program(
                        n_steps=config.validation_steps,
                        max_seq_len=config.validation_seq_len,
                        seed=tp_data.get("seed", seed),
                    )
                    try:
                        s1_result = self._train_with_program(
                            model,
                            tp,
                            val_config,
                            dev,
                            seed=self._stable_seed(
                                exp_id, source_result_id, seed, "validation_tp"
                            ),
                        )
                    finally:
                        self._live_training_context = base_ctx
                except Exception:  # noqa: BLE001 — fallback from TP training to basic
                    self._live_training_context = {
                        **base_ctx,
                        "source_result_id": source_result_id,
                        "candidate_index": prog_idx + 1,
                        "total_candidates": _total_progs,
                        "training_program_index": seed + 1,
                        "total_training_programs": config.validation_n_seeds,
                        "training_program_label": f"seed {seed + 1}",
                        "training_seed": seed,
                        "run_kind": "validation",
                        "checkpoint_manager": checkpoint_manager,
                        "checkpoint_phase": "validation",
                        "checkpoint_candidate_idx": prog_idx,
                        "checkpoint_seed_idx": seed,
                        "checkpoint_interval_steps": checkpoint_interval,
                        "checkpoint_resume_state": resume_state,
                    }
                    try:
                        s1_result = self._micro_train(
                            model,
                            val_config,
                            dev,
                            seed=self._stable_seed(
                                exp_id, source_result_id, seed, "validation_micro"
                            ),
                        )
                    finally:
                        self._live_training_context = base_ctx
            else:
                try:
                    s1_result = self._micro_train(
                        model,
                        val_config,
                        dev,
                        seed=self._stable_seed(
                            exp_id, source_result_id, seed, "validation_micro"
                        ),
                    )
                finally:
                    self._live_training_context = base_ctx

            seed_results.append(
                {
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
                }
            )

            del model
            clear_gpu_memory()

        return seed_results

    def _validation_compute_metrics(self, config, dev_str, source, seed_results):
        from ._types import ValidationMetrics

        _sm = compute_seed_metrics(seed_results)
        passed_seeds = _sm["passed_seeds"]
        loss_ratios = _sm["loss_ratios"]
        best_seed = _sm["best_seed"]

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

        val_baseline_ratio = None
        if best_seed is not None:
            try:
                val_baseline_ratio = _compare(best_seed["final_loss"])
                v_loss = best_seed.get("validation_loss")
                if v_loss is not None:
                    try:
                        _compare(v_loss, split="val")
                    except Exception as exc:
                        logger.warning("Val-split baseline FAILED: %s", exc)
            except Exception as exc:
                logger.warning("Baseline comparison FAILED: %s", exc)

        val_normalized_ratio = None
        val_param_efficiency = None
        source_params = int(
            (source.get("param_count") or source.get("graph_n_params_estimate") or 0)
            if source
            else 0
        )
        if loss_ratios and best_seed is not None and source_params > 0:
            try:
                norm = _compare(
                    best_seed["final_loss"],
                    normalized=True,
                    program_params=source_params,
                )
                val_normalized_ratio = norm.get("normalized_ratio")
                val_param_efficiency = norm.get("param_efficiency")
            except Exception as exc:
                logger.warning("Param-normalized baseline FAILED: %s", exc)

        return ValidationMetrics(
            val_loss_ratio=_sm["val_loss_ratio"],
            multi_seed_std=_sm["multi_seed_std"],
            robustness_score=_sm["robustness_score"],
            is_unstable=_sm["is_unstable"],
            init_sensitivity_std=_sm["init_sensitivity_std"],
            val_baseline_ratio=val_baseline_ratio,
            val_normalized_ratio=val_normalized_ratio,
            val_param_efficiency=val_param_efficiency,
            passed_seeds=passed_seeds,
            loss_ratios=loss_ratios,
            best_seed=best_seed,
            source_params=source_params,
        )

    def _run_inline_validation(
        self,
        config: RunConfig,
        nb: LabNotebook,
        leaderboard: list,
        n_experiments: int,
        limit_str: str,
        mode_reasoning: str,
    ):
        """Execute validation phase inline (not threaded) for continuous mode."""
        result_ids = self._inline_validation_candidate_ids(config, leaderboard)
        if not result_ids:
            logger.info("No validation candidates, falling back to synthesis")
            self._run_continuous_synthesis(
                config, nb, n_experiments, limit_str, mode_reasoning
            )
            return

        exp_id, hypothesis = self._inline_validation_bootstrap(
            config=config,
            nb=nb,
            leaderboard=leaderboard,
            result_ids=result_ids,
            limit_str=limit_str,
        )
        ckpt = CheckpointManager(config.checkpoint_dir)

        self._live_training_context = {"exp_id": exp_id, "phase": "validation"}
        try:
            resume_from_candidate = 0
            ckpt_state = ckpt.load_phase(exp_id, "validation", -1, 0)
            if ckpt_state:
                resume_from_candidate = CheckpointManager.phase_resume_candidate_idx(
                    ckpt_state
                )
                logger.info(
                    "Resuming continuous validation from candidate %d",
                    resume_from_candidate,
                )
            results, dev, dev_str, val_config, source_map = (
                self._inline_validation_prepare_runtime(
                    config=config,
                    nb=nb,
                    result_ids=result_ids,
                )
            )

            for prog_idx, source_result_id in enumerate(result_ids):
                if prog_idx < resume_from_candidate:
                    continue
                if self._stop_event.is_set():
                    break
                if (
                    config.max_cost_dollars > 0
                    and self.aria.total_cost >= config.max_cost_dollars
                ):
                    logger.info("Cost limit reached during validation")
                    break

                self._update_progress(
                    current_program=prog_idx + 1,
                    status="validating",
                    aria_message=(
                        f"Validating {prog_idx + 1}/{len(result_ids)}: "
                        f"{source_result_id[:8]}... "
                        f"({config.validation_n_seeds} seeds, {config.validation_steps} steps)"
                    ),
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

                source = source_map.get(source_result_id)
                if source is None:
                    continue

                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source") or "graph_synthesis"

                best_tp_json = None
                for entry in leaderboard:
                    if entry.get("result_id") == source_result_id:
                        best_tp_json = entry.get("investigation_best_training")
                        break

                seed_results = self._validation_run_seeds(
                    config,
                    val_config,
                    dev,
                    exp_id,
                    prog_idx,
                    len(result_ids),
                    source_result_id,
                    source,
                    best_tp_json,
                    model_source,
                    arch_spec_json_str,
                    graph_json_str,
                    checkpoint_manager=ckpt,
                )
                if not seed_results:
                    logger.warning(
                        "Inline validation: skipping %s — model failed for all %d seeds",
                        source_result_id[:8],
                        config.validation_n_seeds,
                    )
                    continue

                metrics = self._validation_compute_metrics(
                    config, dev_str, source, seed_results
                )

                if len(metrics.passed_seeds) > 0:
                    results["stage1_passed"] += 1
                results["stage0_passed"] += 1
                results["stage05_passed"] += 1

                ev_res = self._run_external_evals(
                    config=config,
                    dev=dev,
                    dev_str=dev_str,
                    best_seed=metrics.best_seed,
                    model_source=model_source,
                    arch_spec_json_str=arch_spec_json_str,
                    graph_json_str=graph_json_str,
                    source=source,
                    source_result_id=source_result_id,
                    exp_id=exp_id,
                    val_loss_ratio=metrics.val_loss_ratio,
                    val_baseline_ratio=metrics.val_baseline_ratio,
                    val_normalized_ratio=metrics.val_normalized_ratio,
                    multi_seed_std=metrics.multi_seed_std,
                    passed_seeds=metrics.passed_seeds,
                    source_params=metrics.source_params,
                )

                nov_conf = source.get("novelty_confidence", 0) if source else 0
                validation_entry = build_validation_entry(
                    source_result_id=source_result_id,
                    metrics=metrics,
                    ev_res=ev_res,
                    nov_conf=nov_conf,
                    config=config,
                )
                tier = "breakthrough" if ev_res.is_breakthrough else "validation"
                results["validation_results"].append(validation_entry.to_dict())

                if metrics.val_loss_ratio and (
                    results["best_loss_ratio"] is None
                    or metrics.val_loss_ratio < results["best_loss_ratio"]
                ):
                    results["best_loss_ratio"] = metrics.val_loss_ratio
                source_novelty = source.get("novelty_score")
                if source_novelty is not None and (
                    results["best_novelty_score"] is None
                    or source_novelty > results["best_novelty_score"]
                ):
                    results["best_novelty_score"] = source_novelty

                promote_validation_candidate(
                    nb=nb,
                    source_result_id=source_result_id,
                    source=source,
                    tier=tier,
                    metrics=metrics,
                    ev_res=ev_res,
                )

                trajectory_composite = run_trajectory_probe(
                    graph_json_str=graph_json_str,
                    config=config,
                    dev=dev,
                    dev_str=dev_str,
                    nb=nb,
                    source_result_id=source_result_id,
                    tier=tier,
                    passed_seeds=metrics.passed_seeds,
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
                    val_loss_ratio=metrics.val_loss_ratio,
                    val_baseline_ratio=metrics.val_baseline_ratio,
                    multi_seed_std=metrics.multi_seed_std,
                    emit_event=self._emit_event,
                )

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
                    logger.warning(
                        "Continuous validation checkpoint save failed for candidate %d: %s",
                        prog_idx + 1,
                        e,
                    )

            # Complete experiment with LLM analysis
            results["perf_report"] = self._build_experiment_perf_report(results)
            results["perf_budget_gate"] = evaluate_perf_budget_gate(
                results["perf_report"]
            )
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb
            )
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            insights = self._analyze_results(results, exp_id, nb, context=context)
            self._publish_terminal_event(
                producer="runner.continuous_validation",
                event_type="experiment_completed",
                exp_id=exp_id,
                payload={
                    "completed_at": time.time(),
                    "results": results,
                    "aria_summary": summary,
                    "aria_mood": self.aria.state.mood,
                    "insights": insights,
                    "llm_analysis": llm_analysis,
                    "mode": "continuous_validation",
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
            self._maybe_extract_knowledge(config, nb, n_experiments)
            self._emit_event(
                "validation_completed",
                {
                    "experiment_id": exp_id,
                    "results": results,
                    "summary": summary,
                },
            )

        except Exception as e:
            logger.warning(f"Inline validation failed: {e}")
            self._publish_terminal_event(
                producer="runner.continuous_validation",
                event_type="experiment_failed",
                exp_id=exp_id,
                payload={
                    "completed_at": time.time(),
                    "error": str(e),
                    "results": None,
                    "mode": "continuous_validation",
                },
            )
            self._fail_experiment_compat(
                nb=nb,
                experiment_id=exp_id,
                error=str(e),
            )
            self._emit_event(
                "validation_completed",
                {
                    "experiment_id": exp_id,
                    "error": str(e),
                },
            )
        finally:
            self._live_training_context = None
