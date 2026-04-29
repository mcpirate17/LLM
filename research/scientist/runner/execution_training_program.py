"""Execution training mixin — split from execution_training."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..json_utils import json_safe
from ._helpers import (
    InflightState,
    apply_adaptive_grad_clip,
    check_inflight_health,
    normalized_loss_ratio,
)
from ._types import RunConfig
from .execution_training import (
    _candidate_perf_budget_verdict,
    _maybe_save_phase_training_state,
    _restore_phase_training_state,
)
from .execution_training_native_boundary import _MicroTrainLoopProgress
from ._curriculum_schedule import precompute_curriculum_seq_lens
from ...eval.utils import clip_grad_norm

import logging

logger = logging.getLogger(__name__)


class _ExecutionTrainingProgramMixin:
    """Train-with-program, data sampling, baseline data."""

    __slots__ = ()

    def _micro_train_async(
        self, model: nn.Module, config: RunConfig, seed: int, dev: torch.device
    ) -> Dict:
        """Async worker entry point for training a pre-compiled model."""
        try:
            return self._micro_train(model, config, dev, seed=seed)
        except Exception as e:
            logger.debug("Async micro-train failed (%s): %s", type(e).__name__, e)
            return {
                "error": str(e),
                "error_type": "training_exception",
                "passed": False,
            }

    def _train_init_model_and_optimizer(
        self,
        model: nn.Module,
        program,
        config: RunConfig,
        dev: torch.device,
        result: Dict[str, Any],
        tracer,
    ) -> Tuple[nn.Module, Any, Tuple, int, int, float]:
        """Set up model, apply init scheme, create optimizer, extract hyperparams.

        Returns (model, optimizer, model_params, n_steps, batch_size,
        max_grad_norm_val).
        """
        with tracer.trace("model_setup"):
            model = model.to(dev)
            model.train()
            model_params = tuple(model.parameters())

        # Apply init scheme
        if program.init_scheme == "small":
            for p in model_params:
                if p.dim() >= 2:
                    nn.init.normal_(p, std=program.init_scale)
        elif program.init_scheme == "orthogonal":
            for m in model.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d)):
                    nn.init.orthogonal_(m.weight, gain=program.init_scale)
        elif program.init_scheme == "spectral":
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)

        # Create optimizer from program
        opt_fallback = False
        try:
            optimizer = program.optimizer.create(model_params)
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning(
                "program.optimizer.create() failed (%s); "
                "falling back to AdamW via build_optimizer",
                exc,
            )
            from ...training.optimizer_synthesis import build_optimizer

            optimizer = build_optimizer(
                model_params,
                optimizer_type="adamw",
                lr=3e-4,
                weight_decay=getattr(config, "optimizer_weight_decay", 0.01),
                betas=getattr(config, "optimizer_betas", (0.9, 0.95)),
            )
            opt_fallback = True

        result["optimizer_class"] = optimizer.__class__.__name__.lower()
        result["optimizer_fallback"] = opt_fallback
        if optimizer.param_groups:
            pg0 = optimizer.param_groups[0]
            result["optimizer_lr"] = float(pg0.get("lr", 3e-4))
            result["optimizer_weight_decay"] = float(pg0.get("weight_decay", 0.01))
            result["optimizer_momentum"] = float(pg0.get("momentum", 0.0))
            betas = pg0.get("betas")
            if isinstance(betas, tuple) and len(betas) == 2:
                result["optimizer_beta1"] = float(betas[0])
                result["optimizer_beta2"] = float(betas[1])

        n_steps = program.n_steps
        batch_size = program.batch_size
        max_grad_norm_val = program.max_grad_norm
        max_grad_norm_val = apply_adaptive_grad_clip(model, max_grad_norm_val)

        return model, optimizer, model_params, n_steps, batch_size, max_grad_norm_val

    def _train_compute_safe_seq_len(
        self,
        config: RunConfig,
        dev: torch.device,
        program,
        n_steps: int,
    ) -> Tuple[int, int]:
        """Compute VRAM-safe seq_len with curriculum schedule.

        Returns (seq_len, safe_max_seq).
        """
        _static_cap = 512
        if dev.type == "cuda":
            try:
                free_mb = (
                    torch.cuda.get_device_properties(dev).total_memory
                    - torch.cuda.memory_allocated(dev)
                ) / (1024 * 1024)
                _batch = int(getattr(config, "stage1_batch_size", 4) or 4)
                _nlayers = int(getattr(config, "n_layers", 4) or 4)
                _dim = int(getattr(config, "model_dim", 256) or 256)
                import math as _math

                _budget = free_mb * 0.5 * 1024 * 1024  # bytes
                _max_s = int(
                    _math.sqrt(
                        _budget
                        / (max(_batch, 1) * max(_dim, 1) * max(_nlayers, 1) * 12)
                    )
                )
                _static_cap = min(_static_cap, max(64, _max_s))
                if _static_cap < config.max_seq_len:
                    logger.info(
                        "VRAM-capped seq_len: %d (free=%.0fMB, B=%d, L=%d)",
                        _static_cap,
                        free_mb,
                        _batch,
                        _nlayers,
                    )
            except RuntimeError as e:
                logger.debug("VRAM cap estimation failed: %s", e)
        safe_max_seq = min(config.max_seq_len, _static_cap)
        seq_len = min(128, safe_max_seq)
        # Apply curriculum seq_len schedule
        try:
            base_seq = program.curriculum.get_seq_len(0, n_steps)
            if base_seq and base_seq > 0:
                seq_len = min(base_seq, safe_max_seq)
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Curriculum seq_len lookup failed: %s", e)

        return seq_len, safe_max_seq

    def _train_finalize_metrics(
        self,
        result: Dict[str, Any],
        model: nn.Module,
        optimizer,
        program,
        config: RunConfig,
        step_times: List[float],
        grad_norms: List[float],
        training_curve: List[Dict],
        initial_loss: Optional[float],
        final_loss: Optional[float],
        min_loss: float,
        total_tokens: int,
        total_time_ms: float,
    ) -> None:
        """Compute post-training metrics and update result dict in-place."""
        if initial_loss is None or final_loss is None:
            return

        _raw = final_loss / max(initial_loss, 1e-6)
        _norm = normalized_loss_ratio(final_loss, config.vocab_size)
        result["loss_ratio"] = _raw
        result["loss_ratio_raw"] = _raw
        result["loss_ratio_norm"] = _norm
        result["final_loss"] = final_loss
        result["initial_loss"] = initial_loss
        result["min_loss"] = min_loss
        result["throughput"] = total_tokens / (total_time_ms / 1000)
        # Adaptive S1 gate: use loss_ratio threshold but scale for
        # graphs with low initial_loss (complex architectures start
        # closer to the entropy floor, so loss_ratio is harder).
        raw_ratio = _raw
        _base_thr = config.stage1_loss_ratio_threshold
        _init = initial_loss if initial_loss and initial_loss > 0 else 100.0
        _scale = max(0.0, 1.0 - _init / 50.0)
        _adaptive_thr = _base_thr + (1.0 - _base_thr) * _scale
        result["passed"] = raw_ratio < _adaptive_thr
        # Validation loss gate
        _vlr = result.get("validation_loss_ratio")
        if result["passed"] and _vlr is not None and _vlr > 0.6:
            result["passed"] = False
            result["error_type"] = "insufficient_learning"
            result["error"] = (
                f"Validation loss ratio {_vlr:.4f} > 0.60 — "
                f"model memorized training but failed to generalize"
            )
        # Inflight checks already flagged this run; override pass.
        if result.get("error_type", "").startswith("inflight_"):
            result["passed"] = False
        if not result["passed"] and result.get("error_type") is None:
            result["error_type"] = "failed_convergence"
            result["error"] = (
                f"Insufficient loss reduction during investigation: {result['loss_ratio']:.4f}"
            )
            result["loss_improvement_rate"] = (initial_loss - final_loss) / initial_loss

        result["avg_step_time_ms"] = (
            sum(step_times) / len(step_times) if step_times else 0
        )
        result["total_train_time_ms"] = total_time_ms

        if grad_norms:
            result["max_grad_norm"] = max(grad_norms)
            result["mean_grad_norm"] = sum(grad_norms) / len(grad_norms)
            mean_gn = result["mean_grad_norm"]
            result["grad_norm_std"] = (
                sum((g - mean_gn) ** 2 for g in grad_norms) / len(grad_norms)
            ) ** 0.5

        result["n_train_steps"] = len(step_times)
        result["final_lr"] = getattr(optimizer, "defaults", {}).get("lr", 3e-4)
        result["training_curve"] = training_curve
        result["training_program_json"] = json.dumps(json_safe(program.to_dict()))

        # Extract architecture-specific telemetry (MoE, MoD, MoR, etc.)
        arch_telemetry = self._extract_architecture_telemetry(model)
        result.update(arch_telemetry)

    def _train_restore_checkpoint(
        self,
        model: nn.Module,
        optimizer,
        dev: torch.device,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Restore training state from checkpoint if available.

        Returns a dict with keys: step_start, initial_loss, final_loss, min_loss,
        total_tokens, step_times, grad_norms, training_curve, t_start,
        inflight_state, es_best_loss, es_steps_since_improve.
        """
        t_start = time.perf_counter()
        state = {
            "step_start": 0,
            "initial_loss": None,
            "final_loss": None,
            "min_loss": float("inf"),
            "total_tokens": 0,
            "step_times": [],
            "grad_norms": [],
            "training_curve": [],
            "t_start": t_start,
            "inflight_state": None,
            "es_best_loss": None,
            "es_steps_since_improve": None,
        }

        resume_state = _restore_phase_training_state(
            self,
            model=model,
            optimizer=optimizer,
            device=dev,
        )
        if resume_state is None:
            return state

        progress = resume_state["progress"]
        state["initial_loss"] = progress.initial_loss
        state["final_loss"] = progress.final_loss
        state["min_loss"] = progress.min_loss
        state["total_tokens"] = progress.total_tokens
        state["step_times"] = [
            float(point.get("step_time_ms", 0.0))
            for point in progress.training_curve
            if point.get("step_time_ms") is not None
        ]
        state["grad_norms"] = [
            float(point.get("grad_norm", 0.0))
            for point in progress.training_curve
            if point.get("grad_norm") is not None
        ]
        state["training_curve"] = list(progress.training_curve)
        state["step_start"] = int(resume_state["step"])
        state["t_start"] = time.perf_counter() - (
            float(resume_state.get("elapsed_ms", 0.0) or 0.0) / 1000.0
        )
        state["inflight_state"] = resume_state["inflight_state"]
        state["es_best_loss"] = float(
            resume_state.get("early_stop_best_loss")
            or progress.min_loss
            or float("inf")
        )
        state["es_steps_since_improve"] = int(
            resume_state.get("early_stop_steps_since_improve", 0) or 0
        )
        result["checkpoint_resumed"] = True
        result["checkpoint_resume_step"] = state["step_start"]
        return state

    def _train_with_program(
        self,
        model: nn.Module,
        program,
        config: RunConfig,
        dev: torch.device,
        seed: int = 42,
    ) -> Dict:
        """Train a model using a synthesized TrainingProgram.

        Returns same metrics dict as _micro_train() plus training_program_json.
        """
        from research.scientist.perf import (
            PerfTracer,
            GPUStarvationDetector,
            KernelTimer,
        )

        tracer = PerfTracer()
        starvation_detector = GPUStarvationDetector(threshold_ms=2.0)
        kernel_timer = KernelTimer(
            model, enabled=bool(getattr(config, "enable_kernel_profiling", False))
        )

        result: Dict[str, Any] = {"passed": False}

        try:
            model, optimizer, model_params, n_steps, batch_size, max_grad_norm_val = (
                self._train_init_model_and_optimizer(
                    model, program, config, dev, result, tracer
                )
            )

            seq_len, safe_max_seq = self._train_compute_safe_seq_len(
                config, dev, program, n_steps
            )
            curriculum_seq_lens = precompute_curriculum_seq_lens(
                getattr(program, "curriculum", None), n_steps
            )

            ckpt = self._train_restore_checkpoint(model, optimizer, dev, result)
            initial_loss = ckpt["initial_loss"]
            final_loss = ckpt["final_loss"]
            min_loss = ckpt["min_loss"]
            total_tokens = ckpt["total_tokens"]
            step_times: List[float] = ckpt["step_times"]
            grad_norms: List[float] = ckpt["grad_norms"]
            training_curve: List[Dict] = ckpt["training_curve"]
            t_start = ckpt["t_start"]
            step_start = ckpt["step_start"]
            _inflight_state_inv = ckpt["inflight_state"]
            _es_best_loss = ckpt["es_best_loss"]
            _es_steps_since_improve = ckpt["es_steps_since_improve"]

            # ID Collapse snapshots — capture hidden-state participation
            # ratio at 20% and 100% of training so execution_training_post
            # can compute the rate. Adaptive to n_steps so short runs
            # still get a signal. Stored on self so the post-eval mixin
            # can read them after this method returns.
            #
            # Probe-id stash and "early" capture happen on the first loop
            # iteration we actually run, not strictly step 0 — that way
            # checkpoint resumes still produce snapshots. The "late"
            # capture is also re-attempted post-loop so early-stop /
            # NaN-abort runs still populate id_collapse.
            _id_collapse_early_at = max(1, int(n_steps * 0.2))
            _id_collapse_late_at = max(_id_collapse_early_at + 1, n_steps - 1)
            self._id_collapse_early_snap = None
            self._id_collapse_late_snap = None
            self._id_collapse_probe_ids = None
            last_completed_step: Optional[int] = None

            for step in range(step_start, n_steps):
                if self._stop_event.is_set():
                    break

                # Update seq_len from curriculum
                if curriculum_seq_lens is not None:
                    curr_seq = curriculum_seq_lens[step]
                    if curr_seq and curr_seq > 0:
                        seq_len = min(curr_seq, safe_max_seq)
                else:
                    curr_seq = program.curriculum.get_seq_len(step, n_steps)
                    if curr_seq and curr_seq > 0:
                        seq_len = min(curr_seq, safe_max_seq)

                starvation_detector.start_wait()
                with tracer.trace("data_sampling"):
                    input_ids = self._sample_training_input_ids(
                        config=config,
                        dev=dev,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        seed=seed + step,
                    )
                starvation_detector.end_wait()

                t_step = time.perf_counter()

                with tracer.trace("forward_pass"):
                    with torch.amp.autocast(
                        device_type=dev.type,
                        dtype=torch.bfloat16,
                        enabled=(dev.type == "cuda"),
                    ):
                        logits = model(input_ids)
                        loss = program.loss.compute(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            input_ids[:, 1:].reshape(-1),
                        )

                if torch.isnan(loss) or torch.isinf(loss):
                    result["error"] = f"NaN/Inf loss at step {step}"
                    result["n_train_steps"] = step
                    return result

                with tracer.trace("backward_pass"):
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = clip_grad_norm(model_params, max_grad_norm_val).item()
                    optimizer.step()

                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)

                t_step_end = time.perf_counter()
                step_time_ms = (t_step_end - t_step) * 1000

                loss_val = loss.item()
                if step == 0:
                    initial_loss = loss_val
                    _es_best_loss = loss_val
                    _es_steps_since_improve = 0
                    _inflight_state_inv = InflightState()
                # Stash a fixed probe batch on the first iteration we
                # actually execute (handles checkpoint resumes too) so the
                # early/late ID-collapse snapshots use identical inputs
                # (otherwise the PR delta confounds with different input
                # distributions). When resuming past _id_collapse_early_at
                # the original early target is unreachable, so retarget
                # early to the resume step — id_collapse rate then covers
                # "from resume to end of training", a weaker but still
                # meaningful signal.
                if self._id_collapse_probe_ids is None:
                    self._id_collapse_probe_ids = (
                        input_ids[: min(8, batch_size)].detach().clone()
                    )
                    if step > _id_collapse_early_at:
                        _id_collapse_early_at = step
                        _id_collapse_late_at = max(
                            _id_collapse_early_at + 1, _id_collapse_late_at
                        )
                final_loss = loss_val
                min_loss = min(min_loss, loss_val)

                # ID Collapse hidden-state snapshots. Cheap (~50 ms each:
                # one fwd-pass + one 256x256 eigvalsh) compared to the
                # training step itself, so unconditional capture is fine.
                if (
                    self._id_collapse_probe_ids is not None
                    and (step == _id_collapse_early_at or step == _id_collapse_late_at)
                ):
                    try:
                        from research.eval.intrinsic_dim_collapse import (
                            capture_hidden_state_snapshot,
                        )

                        snap = capture_hidden_state_snapshot(
                            model,
                            self._id_collapse_probe_ids,
                            step=step,
                            device=str(dev),
                        )
                        if step == _id_collapse_early_at:
                            self._id_collapse_early_snap = snap
                        else:
                            self._id_collapse_late_snap = snap
                        model.train()
                    except (RuntimeError, ValueError):
                        # Snapshot is opportunistic — never let it break
                        # training itself.
                        pass
                last_completed_step = step
                total_tokens += input_ids.numel()

                # Inflight health checks — abort hopeless runs early
                _inflight_fail = check_inflight_health(
                    step=step,
                    loss_val=loss_val,
                    grad_norm=grad_norm,
                    min_loss=min_loss,
                    initial_loss=initial_loss,
                    total_steps=n_steps,
                    state=_inflight_state_inv,
                    spike_ratio=getattr(config, "inflight_spike_ratio", 2.0),
                    spike_window=getattr(config, "inflight_spike_window", 10),
                    grad_norm_limit=getattr(config, "inflight_grad_norm_limit", 100.0),
                    grad_norm_strikes=getattr(config, "inflight_grad_norm_strikes", 3),
                )
                if _inflight_fail is not None:
                    result.update(_inflight_fail)
                    result["n_train_steps"] = step
                    break

                # Early stopping: break if loss plateaus
                if loss_val < _es_best_loss - config.early_stop_min_delta:
                    _es_best_loss = loss_val
                    _es_steps_since_improve = 0
                else:
                    _es_steps_since_improve += 1
                if (
                    step >= config.early_stop_min_steps
                    and _es_steps_since_improve >= config.early_stop_patience
                ):
                    result["early_stopped"] = True
                    result["early_stop_step"] = step
                    logger.debug(
                        "    early stop at step %d/%d: loss=%.4f plateau for %d steps",
                        step,
                        n_steps,
                        loss_val,
                        config.early_stop_patience,
                    )
                    break

                step_times.append(step_time_ms)
                grad_norms.append(grad_norm)

                training_curve.append(
                    {
                        "step": step,
                        "loss": loss_val,
                        "grad_norm": grad_norm,
                        "step_time_ms": step_time_ms,
                    }
                )

                # Emit live training step events for dashboard
                ctx = getattr(self, "_live_training_context", None)
                if ctx and step % 10 == 0:
                    step_event = {
                        "experiment_id": ctx.get("exp_id", ""),
                        "step": step,
                        "loss": round(loss_val, 6),
                        "total_steps": n_steps,
                        "phase": ctx.get("phase", ""),
                        "run_kind": ctx.get("run_kind") or ctx.get("phase", ""),
                    }
                    for source_key, event_key in (
                        ("source_result_id", "source_result_id"),
                        ("candidate_index", "candidate_index"),
                        ("total_candidates", "total_candidates"),
                        ("training_program_index", "training_program_index"),
                        ("total_training_programs", "total_training_programs"),
                        ("training_program_label", "training_program_label"),
                        ("training_seed", "training_seed"),
                    ):
                        value = ctx.get(source_key)
                        if value is not None:
                            step_event[event_key] = value
                    if grad_norm > 0:
                        step_event["grad_norm"] = round(grad_norm, 4)
                    self._emit_event("training_step", step_event)

                progress_snapshot = _MicroTrainLoopProgress(
                    initial_loss=initial_loss,
                    final_loss=final_loss,
                    min_loss=min_loss,
                    total_tokens=total_tokens,
                    step_count=len(step_times),
                    step_time_sum_ms=sum(step_times),
                    grad_norm_sum=sum(grad_norms),
                    grad_norm_sq_sum=sum(g * g for g in grad_norms),
                    grad_norm_max=max(grad_norms) if grad_norms else 0.0,
                    grad_norm_count=len(grad_norms),
                    training_curve=list(training_curve),
                )
                _maybe_save_phase_training_state(
                    self,
                    model=model,
                    optimizer=optimizer,
                    completed_steps=step + 1,
                    total_steps=n_steps,
                    progress=progress_snapshot,
                    inflight_state=_inflight_state_inv,
                    early_stop_best_loss=_es_best_loss,
                    early_stop_steps_since_improve=_es_steps_since_improve,
                    elapsed_ms=(time.perf_counter() - t_start) * 1000.0,
                )

            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000

            # Fallback late-snapshot: if training broke out before reaching
            # _id_collapse_late_at (early-stop, inflight gate, NaN return
            # would have already exited via `return`), use whichever step
            # we last completed. id_collapse_rate is then computed over
            # the actual training span, not the planned one. Without this
            # fallback, ~13% of screening_750 rows have early_snap but no
            # late_snap and id_collapse stays NULL.
            if (
                self._id_collapse_early_snap is not None
                and self._id_collapse_late_snap is None
                and self._id_collapse_probe_ids is not None
                and last_completed_step is not None
                and last_completed_step > _id_collapse_early_at
            ):
                try:
                    from research.eval.intrinsic_dim_collapse import (
                        capture_hidden_state_snapshot,
                    )

                    self._id_collapse_late_snap = capture_hidden_state_snapshot(
                        model,
                        self._id_collapse_probe_ids,
                        step=last_completed_step,
                        device=str(dev),
                    )
                    model.train()
                except (RuntimeError, ValueError):
                    pass

            self._train_finalize_metrics(
                result=result,
                model=model,
                optimizer=optimizer,
                program=program,
                config=config,
                step_times=step_times,
                grad_norms=grad_norms,
                training_curve=training_curve,
                initial_loss=initial_loss,
                final_loss=final_loss,
                min_loss=min_loss,
                total_tokens=total_tokens,
                total_time_ms=total_time_ms,
            )

        except Exception as e:
            logger.debug("Program training failed (%s): %s", type(e).__name__, e)
            result["error"] = str(e)

        # Finalize performance reports
        try:
            result["perf_report"] = tracer.get_report()
            # Ensure throughput is included in perf_report for experiment-level aggregation
            if isinstance(result.get("throughput"), (int, float)):
                result["perf_report"]["avg_throughput_tok_s"] = float(
                    result["throughput"]
                )
            perf_gate = _candidate_perf_budget_verdict(result["perf_report"])
            if perf_gate is not None:
                result["perf_budget_gate"] = perf_gate
                if result.get("passed") and not perf_gate.get("passed", True):
                    result["passed"] = False
                    result["error_type"] = "perf_budget_exceeded"
                    result["error"] = (
                        "Stage-1 candidate exceeded screening perf budget: "
                        + ", ".join(
                            check["metric"]
                            for check in perf_gate.get("checks", [])
                            if not check.get("passed", False)
                        )
                    )

            result["starvation_report"] = starvation_detector.get_summary()
            if kernel_timer.enabled:
                result["kernel_timings_ms"] = kernel_timer.synchronize_and_get_timings()
        except (RuntimeError, KeyError, TypeError) as e:
            logger.debug("Perf report finalization failed: %s", e)
            result["perf_error"] = str(e)

        return result

    # ── OOD Robustness Testing (#54) ──

    # Hand-designed reference training recipes for out-of-distribution testing.
    # Each recipe exercises a different optimizer/LR/schedule to test whether
    # a candidate's learnability is robust or just an artifact of one recipe.
    def _sample_training_input_ids(
        self,
        config: RunConfig,
        dev: torch.device,
        batch_size: int,
        seq_len: int,
        seed: int,
        split: str = "train",
        timer=None,
    ) -> torch.Tensor:
        """Sample input IDs from configured data source with deterministic seed."""
        mode = str(config.data_mode or "random").strip().lower()
        # Generator on CPU: corpus batchers use CPU randint for start indices.
        # Data is moved to the target device after sampling.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        if mode == "huggingface":
            batcher = self._get_hf_batcher(config)
            if batcher is not None:
                batch = batcher.sample_batch(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    generator=generator,
                    device=dev,
                    split=split,
                    timer=timer,
                )
                if batch is not None:
                    return batch
            # Fall through to random on failure

        if mode == "hydra":
            batch = self._get_hydra_batch(config, batch_size, seq_len, dev)
            if batch is not None:
                return batch
            # Fall through to random on failure

        if mode == "corpus":
            batcher = self._get_corpus_batcher(config)
            if batcher is not None:
                batch = batcher.sample_batch(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    generator=generator,
                    device=dev,
                    split=split,
                    timer=timer,
                )
                if batch is not None:
                    return batch

        return torch.randint(
            0,
            int(config.vocab_size),
            (batch_size, seq_len),
            device=dev,
        )

    def _make_baseline_data_fn(self, config: RunConfig, split: str = "train"):
        """Build a data_fn for baseline training when using real data.

        Returns (data_fn, data_tag, cache_data_fn) tuple. data_fn is None for
        random mode (baseline uses its own random tokens). data_tag is a cache
        key suffix. cache_data_fn indicates safe caching for data_fn.
        """
        mode = str(config.data_mode or "random").strip().lower()
        if mode == "huggingface":
            ds_name = str(config.hf_dataset or "").strip()
            subset = str(config.hf_subset or "").strip()
            data_tag = f"hf:{ds_name}:{subset}:{config.hf_split}:{split}"
            step_state = {"step": 0}

            def data_fn(batch_size, seq_len, dev):
                step = step_state["step"]
                step_state["step"] = step + 1
                generator = torch.Generator(device="cpu")
                generator.manual_seed(1337 + step)
                batcher = self._get_hf_batcher(config)
                if batcher is not None:
                    batch = batcher.sample_batch(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        generator=generator,
                        device=dev,
                        split=str(split or "train").lower(),
                    )
                    if batch is not None:
                        return batch
                return torch.randint(
                    0,
                    config.vocab_size,
                    (batch_size, seq_len),
                    device=dev,
                    generator=generator,
                )

            return data_fn, data_tag, True
        if mode == "hydra":

            def data_fn(batch_size, seq_len, dev):
                batch = self._get_hydra_batch(config, batch_size, seq_len, dev)
                if batch is not None:
                    return batch
                return torch.randint(
                    0, config.vocab_size, (batch_size, seq_len), device=dev
                )

            return data_fn, "hydra", False
        if mode == "corpus":
            path = str(config.corpus_path or "").strip()
            version = self._corpus_version_tag(path)
            train_frac = float(getattr(config, "corpus_train_fraction", 0.9) or 0.9)
            val_frac = float(getattr(config, "corpus_val_fraction", 0.1) or 0.1)
            fmt = str(config.corpus_format or "auto")
            text_key = str(config.corpus_text_key or "text")
            tok = str(config.tokenizer_mode or "tiktoken")
            max_chars = int(config.corpus_max_chars)
            split_tag = str(split or "train").lower()
            data_tag = (
                f"corpus:{version}:{fmt}:{text_key}:{tok}:{max_chars}:"
                f"train{train_frac:.3f}:val{val_frac:.3f}:split{split_tag}"
            )
            step_state = {"step": 0}

            def data_fn(batch_size, seq_len, dev):
                step = step_state["step"]
                step_state["step"] = step + 1
                generator = torch.Generator(device="cpu")
                generator.manual_seed(1337 + step)
                batcher = self._get_corpus_batcher(config)
                if batcher is not None:
                    batch = batcher.sample_batch(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        generator=generator,
                        device=dev,
                        split=split_tag,
                    )
                    if batch is not None:
                        return batch
                return torch.randint(
                    0,
                    config.vocab_size,
                    (batch_size, seq_len),
                    device=dev,
                    generator=generator,
                )

            return data_fn, data_tag, True
        return None, "random", False
