"""Execution training mixin — split from execution_training."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ._helpers import (
    InflightState,
    apply_adaptive_grad_clip,
    check_inflight_health,
)
from ._types import RunConfig
from .execution_training import (
    _EntropyGateSampler,
    _MicroTrainContext,
    _allow_synthesized_training,
    _maybe_save_phase_training_state,
    _micro_train_attribute_error,
    _restore_phase_training_state,
    _smoke_test_graph_structure,
)
from .execution_training_native_boundary import (
    _MicroTrainLoopProgress,
    _apply_training_aux_losses,
    _backward_loss,
    _build_training_step_event,
    _collect_aux_modules,
    _compute_micro_train_forward_loss,
    _maybe_extend_training_budget,
    _optimizer_step,
    _training_step_error,
)
from ...eval.pruning import apply_one_shot_pruning, estimate_lm_ce_loss
from ...eval.utils import clip_grad_norm, language_model_loss
from ...training.profiling import TrainingRunProfiler

import logging

logger = logging.getLogger(__name__)


class _ExecutionTrainingMicroMixin:
    """The _micro_train loop and all its helpers."""

    __slots__ = ()

    def _micro_train(
        self,
        model: nn.Module,
        config: RunConfig,
        dev: torch.device,
        seed: int = 42,
        graph_json: str = "",
    ) -> Dict:
        """Run Stage 1 micro-training with comprehensive metric capture.

        Uses deterministic seeding per step so all candidates see the same
        training data in the same order, enabling fair comparison (#56).
        """
        ctx = self._micro_train_build_context(model, config, dev, seed, graph_json)
        result = ctx.result
        run_profiler = ctx.run_profiler
        entropy_gate_sampler = None

        try:
            run_profiler.__enter__()
            optimizer, opt_type = self._micro_train_setup_optimizer(
                model,
                config,
                dev,
                seed,
                result,
                ctx.use_synthesized_training,
                ctx.tracer,
                ctx.trace_totals_ms,
                run_profiler,
            )
            ctx.optimizer = optimizer

            if ctx.graph_data:
                smoke = _smoke_test_graph_structure(ctx.graph_data)
                if not smoke.get("ok"):
                    result["passed"] = False
                    result["smoke_test_failure"] = smoke.get("reason", "unknown")
                    result["smoke_test_result"] = smoke
                    return result

            self._micro_train_record_optimizer_info(result, optimizer, opt_type, config)
            ctx.model_params = tuple(model.parameters())
            ctx.routing_modules, ctx.early_exit_modules, ctx.lm_head, ctx.norm = (
                _collect_aux_modules(model)
            )

            discovery_loss_fast = self._micro_train_discovery_eval(
                model=model,
                config=config,
                dev=dev,
                seed_int=ctx.seed_int,
                seq_len=ctx.seq_len,
            )
            if discovery_loss_fast is not None:
                result["discovery_loss"] = discovery_loss_fast

            ctx.total_steps = self._micro_train_adaptive_budget(
                config, ctx.graph_data, result
            )

            use_cuda_graph = self._micro_train_should_use_cuda_graph(ctx)
            resume_state = _restore_phase_training_state(
                self,
                model=model,
                optimizer=optimizer,
                device=dev,
            )
            if resume_state is not None:
                ctx.progress = resume_state["progress"]
                ctx.total_steps = max(
                    ctx.total_steps, int(resume_state.get("total_steps") or 0)
                )
                ctx.t_start = time.perf_counter() - (
                    float(resume_state.get("elapsed_ms", 0.0) or 0.0) / 1000.0
                )
                result["checkpoint_resumed"] = True
                result["checkpoint_resume_step"] = int(resume_state["step"])
                use_cuda_graph = False
                ctx.resume_state = resume_state

            ran_cuda_graph = use_cuda_graph and self._micro_train_cuda_graph_loop(ctx)

            entropy_gate_sampler = _EntropyGateSampler(model)
            if not ran_cuda_graph:
                early_return = self._micro_train_standard_loop(
                    ctx, entropy_gate_sampler
                )
                if early_return is not None:
                    return early_return

            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            loop_state = ctx.progress.to_loop_state(
                total_time_ms=(time.perf_counter() - ctx.t_start) * 1000,
                collect_curve=ctx.collect_curve,
                seq_len=ctx.seq_len,
                seed=seed,
            )
            self._collect_post_training_metrics(
                model,
                result,
                config,
                dev,
                loop_state,
                ctx.tracer,
                ctx.trace_totals_ms,
                ctx.starvation_detector,
                ctx.kernel_profiles,
                run_profiler,
                graph_json,
                ctx.graph_data,
                ctx.use_synthesized_training,
            )
            self._run_post_s1_screening_probes(
                model,
                result,
                config,
                dev,
                graph_json,
                ctx.graph_data,
            )

        except Exception as e:
            _micro_train_attribute_error(e, result)
        finally:
            if entropy_gate_sampler is not None:
                entropy_gate_sampler.close()
            run_profiler.__exit__(None, None, None)

        self._micro_train_pruning_eval(model, config, dev, seed, result)
        self._micro_train_finalize_perf(
            result,
            ctx.tracer,
            ctx.trace_totals_ms,
            ctx.starvation_detector,
            model,
        )
        return result

    def _micro_train_build_context(
        self,
        model: nn.Module,
        config: RunConfig,
        dev: torch.device,
        seed: int,
        graph_json: str,
    ) -> _MicroTrainContext:
        """Build shared context for _micro_train sub-methods."""
        from research.scientist.perf import (
            PerfTracer,
            GPUStarvationDetector,
            OpKernelProfiler,
        )

        graph_data = (
            (
                json.loads(graph_json)
                if isinstance(graph_json, str) and graph_json
                else graph_json
            )
            if graph_json
            else None
        )
        trace_enabled = bool(getattr(config, "enable_perf_tracing", False))
        tracer = PerfTracer() if trace_enabled else None
        collect_curve = bool(getattr(config, "collect_training_curve", False))
        grad_clip_norm = float(getattr(config, "gradient_clip_norm", 1.0) or 0.0)
        if grad_clip_norm < 0.0:
            grad_clip_norm = 0.0

        grad_clip_norm = apply_adaptive_grad_clip(model, grad_clip_norm)

        return _MicroTrainContext(
            model=model,
            config=config,
            dev=dev,
            seed=seed,
            graph_json=graph_json,
            graph_data=graph_data,
            result={"passed": False},
            progress=_MicroTrainLoopProgress(),
            optimizer=None,  # filled in by _micro_train_setup_optimizer
            model_params=(),
            routing_modules=[],
            early_exit_modules=[],
            lm_head=None,
            norm=None,
            tracer=tracer,
            trace_totals_ms={
                "model_setup": 0.0,
                "data_sampling": 0.0,
                "forward_pass": 0.0,
                "backward_pass": 0.0,
                "optimizer_step": 0.0,
            },
            starvation_detector=GPUStarvationDetector(threshold_ms=2.0),
            op_profiler=OpKernelProfiler(
                enabled=bool(getattr(config, "enable_kernel_profiling", False)),
                top_k=max(1, int(getattr(config, "kernel_profile_top_k", 20) or 20)),
            ),
            run_profiler=TrainingRunProfiler(config, dev),
            use_synthesized_training=_allow_synthesized_training(self, config),
            collect_curve=collect_curve,
            grad_clip_norm=grad_clip_norm,
            total_steps=int(config.stage1_steps),
            seq_len=min(128, config.max_seq_len),
            random_mode=str(config.data_mode or "random").strip().lower() == "random",
            seed_int=int(seed),
            t_start=time.perf_counter(),
            starvation_interval=max(
                1, int(getattr(config, "starvation_check_interval", 8) or 8)
            ),
            starvation_monitoring=bool(
                getattr(config, "enable_starvation_monitoring", False)
                or trace_enabled
                or bool(getattr(config, "profile_enabled", False))
            ),
        )

    @staticmethod
    def _micro_train_record_optimizer_info(
        result: Dict[str, Any],
        optimizer: Any,
        opt_type: str,
        config: RunConfig,
    ) -> None:
        """Record optimizer metadata into result dict."""
        result["optimizer_class"] = optimizer.__class__.__name__.lower()
        result["optimizer_type"] = opt_type
        if optimizer.param_groups:
            pg0 = optimizer.param_groups[0]
            result["optimizer_lr"] = float(pg0.get("lr", config.stage1_lr))
            result["optimizer_weight_decay"] = float(pg0.get("weight_decay", 0.01))
            result["optimizer_momentum"] = float(pg0.get("momentum", 0.0))
            betas = pg0.get("betas")
            if isinstance(betas, tuple) and len(betas) == 2:
                result["optimizer_beta1"] = float(betas[0])
                result["optimizer_beta2"] = float(betas[1])

    @staticmethod
    def _micro_train_should_use_cuda_graph(ctx: _MicroTrainContext) -> bool:
        """Decide whether to use CUDA graph path for training."""
        return bool(
            ctx.dev.type == "cuda"
            and bool(getattr(ctx.config, "enable_cuda_graphs", True))
            and ctx.random_mode
            and not ctx.op_profiler.enabled
            and ctx.tracer is None
            and not ctx.collect_curve
            and not bool(getattr(ctx.config, "profile_enabled", False))
            and ctx.total_steps >= 8
        )

    def _micro_train_setup_optimizer(
        self,
        model: nn.Module,
        config: RunConfig,
        dev: torch.device,
        seed: int,
        result: Dict[str, Any],
        use_synthesized_training: bool,
        tracer: Any,
        trace_totals_ms: Dict[str, float],
        run_profiler: Any,
    ) -> Tuple[Any, str]:
        """Set up model, optimizer, and return (optimizer, opt_type)."""

        def _trace_ctx(name: str, use_gpu: bool = True):
            return (
                tracer.trace(name, use_gpu=use_gpu)
                if tracer is not None
                else nullcontext()
            )

        setup_t0 = time.perf_counter()
        with _trace_ctx("model_setup"), run_profiler.trace("model_setup_ms"):
            model.to(dev)
            model.train()
            model_params = tuple(model.parameters())
            from ...training.optimizer_synthesis import build_optimizer

            phase_opt = getattr(config, "screening_optimizer", "") or ""
            opt_type = (
                phase_opt or getattr(config, "optimizer_type", "adamw") or "adamw"
            )
            phase_lr = getattr(config, "screening_lr", 0.0) or 0.0
            effective_lr = phase_lr if phase_lr > 0 else config.stage1_lr

            if use_synthesized_training and opt_type == "synthesized":
                from ...training.optimizer_synthesis import synthesize_optimizer

                synth_opt = synthesize_optimizer(seed=seed)
                optimizer = synth_opt.create(model_params, lr=effective_lr)
                result["optimizer_synthesized"] = synth_opt.name
            else:
                resolved_type = opt_type if opt_type != "synthesized" else "adamw"
                native_optimizer_flag = (
                    os.getenv("MICRO_TRAIN_NATIVE_OPTIMIZER", "1").strip().lower()
                )
                prefer_native_optimizer = native_optimizer_flag not in {
                    "0",
                    "false",
                    "no",
                    "off",
                }
                if (
                    prefer_native_optimizer
                    and dev.type == "cpu"
                    and resolved_type in {"adamw", "sgd"}
                ):
                    from ...eval.training_core import make_optimizer

                    optimizer = make_optimizer(
                        model_params,
                        optimizer_name=resolved_type,
                        lr=effective_lr,
                        weight_decay=getattr(config, "optimizer_weight_decay", 0.01),
                        momentum=getattr(config, "optimizer_momentum", 0.95),
                        betas=getattr(config, "optimizer_betas", (0.9, 0.95)),
                        prefer_native=True,
                    )
                    result["native_optimizer_requested"] = True
                    result["native_optimizer_active"] = optimizer.__class__.__name__
                else:
                    optimizer = build_optimizer(
                        model_params,
                        optimizer_type=resolved_type,
                        lr=effective_lr,
                        weight_decay=getattr(config, "optimizer_weight_decay", 0.01),
                        betas=getattr(config, "optimizer_betas", (0.9, 0.95)),
                        fused=(
                            dev.type == "cuda"
                            and bool(getattr(config, "optimizer_fused", True))
                        ),
                        foreach=(
                            dev.type == "cuda"
                            and bool(getattr(config, "optimizer_foreach", True))
                        ),
                    )
                    if prefer_native_optimizer:
                        result["native_optimizer_requested"] = True
                        result["native_optimizer_active"] = False
                        result["native_optimizer_skip_reason"] = "cpu_adamw_sgd_only"
        trace_totals_ms["model_setup"] += (time.perf_counter() - setup_t0) * 1000.0
        return optimizer, opt_type

    def _micro_train_adaptive_budget(
        self,
        config: RunConfig,
        graph_data: Any,
        result: Dict[str, Any],
    ) -> int:
        """Compute total training steps, applying adaptive budget for exotic ops."""
        total_steps = int(config.stage1_steps)
        if graph_data:
            try:
                from ...synthesis.primitives import OpCategory, get_primitive

                _nodes = graph_data.get("nodes", [])
                exotic_categories = {
                    OpCategory.MATH_SPACE,
                    OpCategory.SPIKING,
                    OpCategory.FUNCTIONAL,
                }
                exotic_count = 0
                for n in _nodes:
                    op_name = n.get("op_name", n.get("op"))
                    if op_name:
                        try:
                            if get_primitive(op_name).category in exotic_categories:
                                exotic_count += 1
                        except (KeyError, ValueError, AttributeError):
                            pass
                if exotic_count >= 2:
                    total_steps *= 2
                    result["adaptive_budget_novelty_bonus"] = True
                    result["exotic_op_count"] = exotic_count
                    logger.debug(
                        "    Novelty bonus: granting 2x budget (%d steps) for %d exotic ops",
                        total_steps,
                        exotic_count,
                    )
            except (KeyError, ValueError, AttributeError, ImportError) as e_novel:
                logger.debug("Adaptive budget novel check failed: %s", e_novel)
        return total_steps

    def _micro_train_cuda_graph_capture(
        self,
        ctx: _MicroTrainContext,
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Warmup and capture a CUDA graph for the training step.

        Returns (graph, static_input_ids, captured_loss, captured_grad_norm).
        """
        config = ctx.config
        dev = ctx.dev
        static_input_ids = torch.empty(
            (config.stage1_batch_size, ctx.seq_len),
            dtype=torch.long,
            device=dev,
        )
        captured_loss = torch.zeros((), device=dev)
        captured_grad_norm = torch.zeros((), device=dev)
        warmup_steps = max(1, int(getattr(config, "cuda_graph_warmup_steps", 3) or 3))

        def _graph_step() -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.amp.autocast(
                device_type=dev.type, dtype=torch.bfloat16, enabled=True
            ):
                logits = ctx.model(static_input_ids)
                loss_t = language_model_loss(
                    logits,
                    static_input_ids,
                    min(config.vocab_size, int(logits.shape[-1])),
                )
            ctx.optimizer.zero_grad(set_to_none=True)
            loss_t.backward()
            if ctx.grad_clip_norm > 0.0:
                grad_norm_t = clip_grad_norm(ctx.model_params, ctx.grad_clip_norm)
            else:
                grad_norm_t = torch.zeros((), device=dev)
            ctx.optimizer.step()
            return loss_t, grad_norm_t

        for wi in range(min(warmup_steps, ctx.total_steps)):
            static_input_ids.copy_(
                self._micro_train_make_random_batch(
                    seed_int=ctx.seed_int,
                    step=wi,
                    batch_size=config.stage1_batch_size,
                    seq_len=ctx.seq_len,
                    vocab_size=config.vocab_size,
                    dev=dev,
                ),
                non_blocking=True,
            )
            loss_t, grad_norm_t = _graph_step()
            captured_loss.copy_(loss_t.detach())
            captured_grad_norm.copy_(torch.as_tensor(grad_norm_t, device=dev).detach())
            if not torch.isfinite(captured_loss):
                break

        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss_t, grad_norm_t = _graph_step()
            captured_loss.copy_(loss_t.detach())
            captured_grad_norm.copy_(torch.as_tensor(grad_norm_t, device=dev).detach())
        return graph, static_input_ids, captured_loss, captured_grad_norm

    def _micro_train_cuda_graph_loop(self, ctx: _MicroTrainContext) -> bool:
        """Run the CUDA graph training loop. Returns True if successful."""
        config = ctx.config
        dev = ctx.dev
        progress = ctx.progress
        result = ctx.result

        try:
            graph, static_input_ids, captured_loss, captured_grad_norm = (
                self._micro_train_cuda_graph_capture(ctx)
            )

            check_interval = max(1, int(getattr(config, "loss_check_interval", 8) or 8))
            step = 0
            _es_best_loss_cg = progress.min_loss
            _es_no_improve_cg = 0
            while step < ctx.total_steps:
                if self._stop_event.is_set():
                    break
                t_step = time.perf_counter()
                static_input_ids.copy_(
                    self._micro_train_make_random_batch(
                        seed_int=ctx.seed_int,
                        step=step,
                        batch_size=config.stage1_batch_size,
                        seq_len=ctx.seq_len,
                        vocab_size=config.vocab_size,
                        dev=dev,
                    ),
                    non_blocking=True,
                )
                graph.replay()
                step_time_ms = (time.perf_counter() - t_step) * 1000.0
                progress.record_cuda_graph_step(
                    token_count=static_input_ids.numel(),
                    step_time_ms=step_time_ms,
                )

                should_check = (
                    (step == 0)
                    or (step == ctx.total_steps - 1)
                    or (step % check_interval == 0)
                )
                is_milestone = step == 250 or step == 500
                if not should_check and not is_milestone:
                    step += 1
                    continue

                loss_val = float(captured_loss.item())
                grad_norm = float(captured_grad_norm.item())

                prev_total_steps = ctx.total_steps
                ctx.total_steps = _maybe_extend_training_budget(
                    progress,
                    result,
                    step=step,
                    loss_val=loss_val,
                    total_steps=ctx.total_steps,
                )
                if ctx.total_steps != prev_total_steps:
                    logger.debug(
                        "    Step 500: improvement detected. Extending budget to %d steps.",
                        ctx.total_steps,
                    )

                if not should_check:
                    step += 1
                    continue

                step_error = _training_step_error(
                    step=step,
                    loss_val=loss_val,
                    grad_norm=grad_norm,
                )
                if step_error is not None:
                    result.update(step_error)
                    return True
                if progress.initial_loss is None:
                    progress.initial_loss = loss_val
                    _es_best_loss_cg = loss_val
                    _es_no_improve_cg = 0
                progress.record_loss_snapshot(loss_val=loss_val)
                progress.grad_norm_sum += grad_norm
                progress.grad_norm_sq_sum += grad_norm * grad_norm
                progress.grad_norm_max = max(progress.grad_norm_max, grad_norm)
                progress.grad_norm_count += 1

                if loss_val < _es_best_loss_cg - config.early_stop_min_delta:
                    _es_best_loss_cg = loss_val
                    _es_no_improve_cg = 0
                else:
                    _es_no_improve_cg += check_interval
                if (
                    step >= config.early_stop_min_steps
                    and _es_no_improve_cg >= config.early_stop_patience
                ):
                    result["early_stopped"] = True
                    result["early_stop_step"] = step
                    progress.step_count = step + 1
                    break

                ctx.run_profiler.record_step(
                    step=step,
                    loss=loss_val,
                    grad_norm=grad_norm,
                )
                _maybe_save_phase_training_state(
                    self,
                    model=ctx.model,
                    optimizer=ctx.optimizer,
                    completed_steps=step + 1,
                    total_steps=ctx.total_steps,
                    progress=progress,
                    inflight_state=None,
                    early_stop_best_loss=_es_best_loss_cg,
                    early_stop_steps_since_improve=_es_no_improve_cg,
                    elapsed_ms=(time.perf_counter() - ctx.t_start) * 1000.0,
                )
                step += 1
            return True
        except RuntimeError as e:
            logger.debug("CUDA graph capture failed, falling back: %s", e)
            result["cuda_graph_fallback_reason"] = str(e)
            return False

    def _micro_train_standard_loop(
        self,
        ctx: _MicroTrainContext,
        entropy_gate_sampler: _EntropyGateSampler,
    ) -> Optional[Dict]:
        """Run the standard (non-CUDA-graph) training loop.

        Returns a result dict for early termination (NaN/Inf), or None on normal completion.
        """
        progress = ctx.progress
        result = ctx.result
        _ENTROPY_GATE_SAMPLE_STEPS = frozenset({10, 25, 50, 75, 100})

        step = 0
        _inflight_state = InflightState()
        _es_best_loss = float("inf")
        _es_steps_since_improve = 0
        if ctx.resume_state is not None:
            step = int(ctx.resume_state["step"])
            _inflight_state = ctx.resume_state["inflight_state"]
            restored_best = ctx.resume_state.get("early_stop_best_loss")
            if restored_best is not None:
                _es_best_loss = float(restored_best)
            elif math.isfinite(progress.min_loss):
                _es_best_loss = float(progress.min_loss)
            restored_wait = ctx.resume_state.get("early_stop_steps_since_improve")
            _es_steps_since_improve = int(restored_wait or 0)

        while step < ctx.total_steps:
            if self._stop_event.is_set():
                break

            input_ids, t_step = self._micro_train_sample_data(ctx, step)

            should_sample_entropy = (
                entropy_gate_sampler.available and step in _ENTROPY_GATE_SAMPLE_STEPS
            )
            if should_sample_entropy:
                entropy_gate_sampler.begin_sample()

            step_state = self._micro_train_execute_step(ctx, input_ids, step)

            if should_sample_entropy:
                _eg_val = entropy_gate_sampler.finish_sample()
                if _eg_val is not None:
                    progress.entropy_gate_trajectory.append(_eg_val)
                    if _eg_val < 0.05:
                        logger.warning(
                            "entropy_gate_collapse_detected at step %d: value=%.4f",
                            step,
                            _eg_val,
                        )

            loss = step_state.get("loss")
            grad_norm = float(step_state.get("grad_norm", 0.0))

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                result["error"] = f"NaN/Inf loss at step {step}"
                result["n_train_steps"] = step
                return result

            loss_val = loss.item()
            _raux_t = step_state.get("routing_aux_loss_tensor")
            routing_aux_loss = float(_raux_t.item()) if _raux_t is not None else None
            progress.record_routing_aux_loss(routing_aux_loss)

            prev_total_steps = ctx.total_steps
            ctx.total_steps = _maybe_extend_training_budget(
                progress,
                result,
                step=step,
                loss_val=loss_val,
                total_steps=ctx.total_steps,
            )
            if ctx.total_steps != prev_total_steps:
                logger.debug(
                    "    Step 500: improvement detected. Extending budget to %d steps.",
                    ctx.total_steps,
                )

            step_error = _training_step_error(
                step=step,
                loss_val=loss_val,
                grad_norm=grad_norm,
            )
            if step_error is not None:
                result.update(step_error)
                return result

            if ctx.dev.type == "cuda" and (
                ctx.tracer is not None or ctx.op_profiler.enabled
            ):
                torch.cuda.synchronize(ctx.dev)

            step_time_ms = (time.perf_counter() - t_step) * 1000
            action = self._micro_train_post_step(
                ctx,
                step,
                loss_val,
                grad_norm,
                step_time_ms,
                input_ids.numel(),
                routing_aux_loss,
                _inflight_state,
                _es_best_loss,
                _es_steps_since_improve,
            )
            if action == "break":
                break
            _es_best_loss, _es_steps_since_improve = action
            step += 1
        return None

    def _micro_train_sample_data(
        self,
        ctx: _MicroTrainContext,
        step: int,
    ) -> Tuple[torch.Tensor, float]:
        """Sample a training batch and return (input_ids, step_start_time)."""
        starvation_sample = (
            ctx.starvation_monitoring
            and (not ctx.random_mode)
            and ((step % ctx.starvation_interval) == 0)
        )
        if starvation_sample:
            ctx.starvation_detector.start_wait()
        data_t0 = time.perf_counter()
        with ctx.trace_ctx("data_sampling"), ctx.run_profiler.trace("data_sampling_ms"):
            if ctx.random_mode:
                input_ids = self._micro_train_make_random_batch(
                    seed_int=ctx.seed_int,
                    step=step,
                    batch_size=ctx.config.stage1_batch_size,
                    seq_len=ctx.seq_len,
                    vocab_size=ctx.config.vocab_size,
                    dev=ctx.dev,
                )
            else:
                input_ids = self._sample_training_input_ids(
                    config=ctx.config,
                    dev=ctx.dev,
                    batch_size=ctx.config.stage1_batch_size,
                    seq_len=ctx.seq_len,
                    seed=ctx.seed + step,
                    timer=ctx.run_profiler.record_timing,
                )
        if starvation_sample:
            ctx.starvation_detector.end_wait()
        ctx.trace_totals_ms["data_sampling"] += (time.perf_counter() - data_t0) * 1000.0
        return input_ids, time.perf_counter()

    def _micro_train_execute_step(
        self,
        ctx: _MicroTrainContext,
        input_ids: torch.Tensor,
        step: int,
    ) -> Dict[str, Any]:
        """Execute one forward-backward-optimize step. Returns step_state dict."""
        step_state: Dict[str, Any] = {}

        def _run_step() -> None:
            fwd_t0 = time.perf_counter()
            with (
                ctx.trace_ctx("forward_pass"),
                ctx.run_profiler.trace("forward_pass_ms"),
            ):
                loss = _compute_micro_train_forward_loss(
                    self,
                    ctx.model,
                    input_ids,
                    config=ctx.config,
                    dev=ctx.dev,
                    use_synthesized_training=ctx.use_synthesized_training,
                    seed=ctx.seed,
                )
            ctx.trace_totals_ms["forward_pass"] += (
                time.perf_counter() - fwd_t0
            ) * 1000.0
            loss, aux_loss, ee_loss = _apply_training_aux_losses(
                loss,
                routing_modules=ctx.routing_modules,
                early_exit_modules=ctx.early_exit_modules,
                lm_head=ctx.lm_head,
                norm=ctx.norm,
                input_ids=input_ids,
            )
            step_state["loss"] = loss
            if aux_loss is not None:
                step_state["routing_aux_loss_tensor"] = aux_loss.detach()
            if ee_loss is not None:
                step_state["early_exit_aux_loss_tensor"] = ee_loss.detach()

            bwd_t0 = time.perf_counter()
            native_backward_step = getattr(
                ctx.optimizer, "backward_step_with_grad_clip", None
            )
            use_native_backward_step = callable(native_backward_step) and (
                os.getenv("MICRO_TRAIN_NATIVE_BACKWARD_STEP", "1").strip().lower()
                not in {"0", "false", "no", "off"}
            )
            fused_step = getattr(ctx.optimizer, "step_with_grad_clip", None)
            use_fused_native_step = (
                callable(fused_step) and not use_native_backward_step
            )
            if use_native_backward_step:
                with (
                    ctx.trace_ctx("backward_optimizer_step"),
                    ctx.run_profiler.trace("backward_optimizer_step_ms"),
                ):
                    step_state["grad_norm"] = native_backward_step(
                        loss, ctx.grad_clip_norm
                    )
                ctx.trace_totals_ms["backward_optimizer_step"] = (
                    ctx.trace_totals_ms.get("backward_optimizer_step", 0.0)
                    + (time.perf_counter() - bwd_t0) * 1000.0
                )
                return
            if use_fused_native_step:
                with (
                    ctx.trace_ctx("backward_pass"),
                    ctx.run_profiler.trace("backward_pass_ms"),
                ):
                    ctx.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
            else:
                with (
                    ctx.trace_ctx("backward_pass"),
                    ctx.run_profiler.trace("backward_pass_ms"),
                ):
                    step_state["grad_norm"] = _backward_loss(
                        loss,
                        optimizer=ctx.optimizer,
                        grad_clip_norm=ctx.grad_clip_norm,
                        model_params=ctx.model_params,
                    )
            ctx.trace_totals_ms["backward_pass"] += (
                time.perf_counter() - bwd_t0
            ) * 1000.0

            opt_t0 = time.perf_counter()
            with (
                ctx.trace_ctx("optimizer_step"),
                ctx.run_profiler.trace("optimizer_step_ms"),
            ):
                if use_fused_native_step:
                    step_state["grad_norm"] = fused_step(ctx.grad_clip_norm)
                else:
                    _optimizer_step(ctx.optimizer)
            ctx.trace_totals_ms["optimizer_step"] += (
                time.perf_counter() - opt_t0
            ) * 1000.0

        if step == 0 and ctx.op_profiler.enabled:
            kernel_summary = ctx.op_profiler.profile_callable(_run_step)
            if kernel_summary:
                ctx.kernel_profiles.append({"step": step, **kernel_summary})
            else:
                _run_step()
        else:
            _run_step()
        return step_state

    def _micro_train_post_step(
        self,
        ctx: _MicroTrainContext,
        step: int,
        loss_val: float,
        grad_norm: float,
        step_time_ms: float,
        token_count: int,
        routing_aux_loss: Optional[float],
        inflight_state: InflightState,
        es_best_loss: float,
        es_steps_since_improve: int,
    ) -> Any:
        """Process post-step metrics, inflight checks, early stopping.

        Returns "break" to stop the loop, or (es_best_loss, es_steps_since_improve) to continue.
        """
        config = ctx.config
        progress = ctx.progress
        result = ctx.result

        if progress.initial_loss is None:
            progress.initial_loss = loss_val
            es_best_loss = loss_val
            es_steps_since_improve = 0
        progress.record_loss_snapshot(loss_val=loss_val)
        progress.total_tokens += token_count

        _inflight_fail = None
        if not bool(getattr(config, "profile_disable_inflight_checks", False)):
            _inflight_fail = check_inflight_health(
                step=step,
                loss_val=loss_val,
                grad_norm=grad_norm,
                min_loss=progress.min_loss,
                initial_loss=progress.initial_loss,
                total_steps=ctx.total_steps,
                state=inflight_state,
                spike_ratio=getattr(config, "inflight_spike_ratio", 2.0),
                spike_window=getattr(config, "inflight_spike_window", 10),
                grad_norm_limit=getattr(config, "inflight_grad_norm_limit", 100.0),
                grad_norm_strikes=getattr(config, "inflight_grad_norm_strikes", 3),
            )
        if _inflight_fail is not None:
            result.update(_inflight_fail)
            result["n_train_steps"] = step
            progress.step_count += 1
            return "break"

        if loss_val < es_best_loss - config.early_stop_min_delta:
            es_best_loss = loss_val
            es_steps_since_improve = 0
        else:
            es_steps_since_improve += 1
        if (
            step >= config.early_stop_min_steps
            and es_steps_since_improve >= config.early_stop_patience
        ):
            result["early_stopped"] = True
            result["early_stop_step"] = step
            logger.debug(
                "    early stop at step %d/%d: loss=%.4f plateau for %d steps",
                step,
                ctx.total_steps,
                loss_val,
                config.early_stop_patience,
            )
            progress.step_count += 1
            return "break"

        progress.commit_eager_step(
            step=step,
            loss_val=loss_val,
            grad_norm=grad_norm,
            step_time_ms=step_time_ms,
            token_count=0,
            collect_curve=ctx.collect_curve,
        )

        step_event = _build_training_step_event(
            getattr(self, "_live_training_context", None),
            step=step,
            total_steps=ctx.total_steps,
            loss_val=loss_val,
            grad_norm=grad_norm,
            routing_aux_loss=routing_aux_loss,
        )
        if step_event is not None:
            self._emit_event("training_step", step_event)

        if step == 0 or step == ctx.total_steps // 2 or step == ctx.total_steps - 1:
            logger.debug(
                "    train step %d/%d: loss=%.4f, grad_norm=%.3f, step_time=%.1fms",
                step + 1,
                ctx.total_steps,
                loss_val,
                grad_norm,
                step_time_ms,
            )

        ctx.run_profiler.record_step(step=step, loss=loss_val, grad_norm=grad_norm)
        _maybe_save_phase_training_state(
            self,
            model=ctx.model,
            optimizer=ctx.optimizer,
            completed_steps=step + 1,
            total_steps=ctx.total_steps,
            progress=progress,
            inflight_state=inflight_state,
            early_stop_best_loss=es_best_loss,
            early_stop_steps_since_improve=es_steps_since_improve,
            elapsed_ms=(time.perf_counter() - ctx.t_start) * 1000.0,
        )
        return es_best_loss, es_steps_since_improve

    def _micro_train_pruning_eval(
        self,
        model: nn.Module,
        config: RunConfig,
        dev: torch.device,
        seed: int,
        result: Dict[str, Any],
    ) -> None:
        """Run one-shot pruning evaluation if configured."""
        if result.get("final_loss") is None or not bool(
            getattr(config, "one_shot_pruning_baseline", False)
        ):
            return
        try:
            seq_len = min(128, int(config.max_seq_len))
            eval_batches = max(
                1, int(getattr(config, "one_shot_pruning_eval_batches", 4))
            )
            eval_batch_size = max(
                1, int(getattr(config, "one_shot_pruning_batch_size", 2))
            )

            eval_inputs = [
                self._sample_training_input_ids(
                    config=config,
                    dev=dev,
                    batch_size=eval_batch_size,
                    seq_len=seq_len,
                    seed=seed + 100_000 + i,
                )
                for i in range(eval_batches)
            ]

            dense_eval_loss = estimate_lm_ce_loss(model, eval_inputs, dev)

            pruned_model = copy.deepcopy(model).to(dev)
            prune_info = apply_one_shot_pruning(
                pruned_model,
                target_sparsity=float(
                    getattr(config, "one_shot_pruning_sparsity", 0.5)
                ),
                method=str(getattr(config, "one_shot_pruning_method", "wanda")),
            )
            pruned_eval_loss = estimate_lm_ce_loss(pruned_model, eval_inputs, dev)

            quality_retention = None
            if (
                dense_eval_loss is not None
                and pruned_eval_loss is not None
                and pruned_eval_loss > 0
            ):
                quality_retention = max(
                    0.0, min(1.5, dense_eval_loss / pruned_eval_loss)
                )

            result["pruning_method"] = prune_info.method
            result["pruning_target_sparsity"] = prune_info.target_sparsity
            result["pruning_actual_sparsity"] = prune_info.actual_sparsity
            result["pruning_n_params_total"] = prune_info.n_params_total
            result["pruning_n_params_pruned"] = prune_info.n_params_pruned
            result["pruning_dense_eval_loss"] = dense_eval_loss
            result["pruning_pruned_eval_loss"] = pruned_eval_loss
            result["pruning_quality_retention"] = quality_retention
            if prune_info.n_params_total > 0:
                result["pruning_active_params_estimate"] = (
                    prune_info.n_params_total - prune_info.n_params_pruned
                )

            del pruned_model
        except (RuntimeError, ValueError) as e:
            logger.debug("Pruning eval failed: %s", e)
            result["pruning_error"] = str(e)

    def _micro_train_finalize_perf(
        self,
        result: Dict[str, Any],
        tracer: Any,
        trace_totals_ms: Dict[str, float],
        starvation_detector: Any,
        model: nn.Module,
    ) -> None:
        """Finalize performance reports and architecture telemetry."""
        try:
            if tracer is not None:
                fallback_perf = tracer.get_report()
            else:
                fallback_perf = {
                    "summary_ms": {k: round(v, 4) for k, v in trace_totals_ms.items()},
                    "traces": [],
                }
            result["perf_report"] = result.get("perf_traces", fallback_perf)
            if isinstance(result.get("throughput"), (int, float)):
                result["perf_report"]["avg_throughput_tok_s"] = float(
                    result["throughput"]
                )

            result["starvation_report"] = result.get(
                "gpu_starvation", starvation_detector.get_summary()
            )
            if "kernel_timing" in result:
                result["kernel_timings_ms"] = result["kernel_timing"]
        except (RuntimeError, KeyError, TypeError) as e:
            logger.debug("Perf report finalization failed: %s", e)
            result["perf_error"] = str(e)

        try:
            result.update(self._extract_architecture_telemetry(model))
        except (RuntimeError, AttributeError) as e:
            logger.debug("Architecture telemetry extract failed: %s", e)
