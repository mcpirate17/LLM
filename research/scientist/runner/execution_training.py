"""Execution mixin: micro-train, train-with-program, data sampling, baseline."""

from __future__ import annotations

import copy
import json
import math
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...eval.fingerprint import compute_gated_fingerprint
from ...eval.pruning import apply_one_shot_pruning, estimate_lm_ce_loss
from ._helpers import normalized_loss_ratio

import logging
logger = logging.getLogger(__name__)


def _smoke_test_graph_structure(graph_json: str) -> Dict[str, Any]:
    """Run fast C++ structural smoke test on a computation graph.

    Checks gradient flow, parameter presence, and unsafe op absence.
    Returns dict with keys: ok, has_params, grad_flows, no_unsafe, reason.
    ~0.01ms per graph.
    """
    try:
        import aria_core
        smoke_fn = getattr(aria_core, "smoke_test_graph", None)
        if smoke_fn is None:
            return {"ok": True, "reason": "smoke_test unavailable"}
    except ImportError:
        return {"ok": True, "reason": "aria_core unavailable"}

    try:
        from ...synthesis.op_roles import get_role, OpRole

        graph_data = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
        nodes_raw = graph_data.get("nodes", [])
        if not nodes_raw:
            return {"ok": False, "reason": "empty graph"}

        # Sort nodes by id for stable indexing
        nodes_sorted = sorted(nodes_raw, key=lambda n: n["id"])
        id_to_idx = {n["id"]: i for i, n in enumerate(nodes_sorted)}
        n_nodes = len(nodes_sorted)

        # Role code mapping
        _ROLE_CODES = {
            OpRole.PROJECT: 0, OpRole.NORMALIZE: 1, OpRole.ACTIVATE: 2,
            OpRole.MIX: 3, OpRole.ROUTE: 4, OpRole.GATE: 5,
            OpRole.POSITION: 6, OpRole.REDUCE: 7, OpRole.RESIDUAL: 8,
            OpRole.UNSAFE: 9,
        }

        edges = []
        op_roles = []
        has_params_flag = []
        preserves_grad = []
        output_idx = n_nodes - 1

        for i, node in enumerate(nodes_sorted):
            # Edges: up to 2 inputs, -1 if none
            input_ids = node.get("input_ids", [])
            e0 = id_to_idx.get(input_ids[0], -1) if len(input_ids) > 0 else -1
            e1 = id_to_idx.get(input_ids[1], -1) if len(input_ids) > 1 else -1
            edges.extend([e0, e1])

            op_name = node.get("op_name", "input")
            if node.get("is_input"):
                op_roles.append(10)  # input role code
                has_params_flag.append(0)
                preserves_grad.append(1)
            else:
                role = get_role(op_name)
                op_roles.append(_ROLE_CODES.get(role, 9))
                from ...synthesis.primitives import PRIMITIVE_REGISTRY
                prim = PRIMITIVE_REGISTRY.get(op_name)
                has_params_flag.append(1 if (prim and prim.has_params) else 0)
                preserves_grad.append(
                    1 if (prim is None or prim.preserves_gradient) else 0
                )

            if node.get("is_output"):
                output_idx = i

        result = smoke_fn(n_nodes, edges, op_roles, has_params_flag,
                          preserves_grad, output_idx)
        if not result["ok"]:
            reasons = []
            if not result["has_params"]:
                reasons.append("no learnable parameters")
            if not result["grad_flows"]:
                reasons.append("gradient cannot flow input→output")
            if not result["no_unsafe"]:
                reasons.append("contains standalone unsafe ops")
            result["reason"] = "; ".join(reasons) if reasons else "unknown"
        else:
            result["reason"] = ""
        return result

    except Exception as exc:
        logger.debug("Smoke test failed with exception: %s", exc)
        return {"ok": True, "reason": f"smoke_test error: {exc}"}

from ._types import RunConfig
from ._helpers import InflightState, check_inflight_health


def _training_phase(owner: Any) -> str:
    """Return the current training phase name, if one was set by the runner."""
    context = getattr(owner, "_live_training_context", None)
    if not isinstance(context, dict):
        return ""
    phase = context.get("phase")
    return str(phase).strip().lower() if phase is not None else ""


def _allow_synthesized_training(owner: Any, config: RunConfig) -> bool:
    """Restrict synthesized loss/optimizer exploration to screening runs."""
    if not (
        getattr(config, "loss_type", "cross_entropy") != "cross_entropy"
        or getattr(config, "optimizer_type", "adamw") != "adamw"
    ):
        return False
    return _training_phase(owner) in {"screening", "candidate_screening", "synthesis"}


def _collect_routing_aux_loss(
    model: nn.Module, weight: float = 0.01,
) -> "torch.Tensor | None":
    """Collect load-balance auxiliary loss from routing telemetry on model layers.

    After a forward pass, routing ops attach ``routing_telemetry`` dicts to
    their modules with ``expert_counts`` tensors.  This function computes a
    squared-deviation load-balance loss encouraging uniform expert utilisation
    and returns it (or ``None`` if no routing telemetry was found).
    """
    aux = torch.tensor(0.0)
    found = False

    for module in model.modules():
        rt = getattr(module, "routing_telemetry", None)
        if rt is None:
            continue
        ec = rt.get("expert_counts")
        if not isinstance(ec, torch.Tensor) or ec.numel() < 2:
            continue
        found = True
        # Compute load-balance loss: (actual_frac - uniform)^2 summed over experts
        total = ec.sum().clamp(min=1.0)
        fracs = ec.float() / total
        uniform = 1.0 / ec.numel()
        aux = aux + ((fracs - uniform) ** 2).sum()

    if not found:
        return None
    return aux * weight


class _ExecutionTrainingMixin:
    """Micro-training, train-with-program, data sampling, baseline data."""

    __slots__ = ()

    # ── Scale-Up Mode ──

    def _micro_train(self, model: nn.Module, config: RunConfig,
                     dev: torch.device, seed: int = 42,
                     graph_json: str = "") -> Dict:
        """Run Stage 1 micro-training with comprehensive metric capture.

        Uses deterministic seeding per step so all candidates see the same
        training data in the same order, enabling fair comparison (#56).
        """
        from research.scientist.perf import PerfTracer, GPUStarvationDetector, OpKernelProfiler
        trace_enabled = bool(getattr(config, "enable_perf_tracing", False))
        tracer = PerfTracer() if trace_enabled else None
        starvation_detector = GPUStarvationDetector(threshold_ms=2.0)
        op_profiler = OpKernelProfiler(
            enabled=bool(getattr(config, "enable_kernel_profiling", False)),
            top_k=max(1, int(getattr(config, "kernel_profile_top_k", 20) or 20)),
        )

        result: Dict[str, Any] = {"passed": False}
        use_synthesized_training = _allow_synthesized_training(self, config)
        collect_curve = bool(getattr(config, "collect_training_curve", False))
        grad_clip_norm = float(getattr(config, "gradient_clip_norm", 1.0) or 0.0)
        if grad_clip_norm < 0.0:
            grad_clip_norm = 0.0
        # Adaptive clip for math-space architectures
        from ._helpers import apply_adaptive_grad_clip
        grad_clip_norm = apply_adaptive_grad_clip(model, grad_clip_norm)

        trace_totals_ms: Dict[str, float] = {
            "model_setup": 0.0,
            "data_sampling": 0.0,
            "forward_pass": 0.0,
            "backward_pass": 0.0,
            "optimizer_step": 0.0,
        }

        def _trace_ctx(name: str, use_gpu: bool = True):
            return tracer.trace(name, use_gpu=use_gpu) if tracer is not None else nullcontext()

        try:
            setup_t0 = time.perf_counter()
            with _trace_ctx("model_setup"):
                model = model.to(dev)
                model.train()
                from ...training.optimizer_synthesis import build_optimizer

                # Resolve phase-specific optimizer: screening_optimizer overrides optimizer_type
                phase_opt = getattr(config, "screening_optimizer", "") or ""
                opt_type = phase_opt or getattr(config, "optimizer_type", "adamw") or "adamw"
                phase_lr = getattr(config, "screening_lr", 0.0) or 0.0
                effective_lr = phase_lr if phase_lr > 0 else config.stage1_lr

                # Synthesized optimizer support (screening exploration only)
                if use_synthesized_training and opt_type == "synthesized":
                    from ...training.optimizer_synthesis import synthesize_optimizer
                    synth_opt = synthesize_optimizer(seed=seed)
                    optimizer = synth_opt.create(model.parameters(), lr=effective_lr)
                    result["optimizer_synthesized"] = synth_opt.name
                else:
                    # Non-synthesized: use build_optimizer (no silent fallbacks)
                    resolved_type = opt_type if opt_type != "synthesized" else "adamw"
                    optimizer = build_optimizer(
                        model.parameters(),
                        optimizer_type=resolved_type,
                        lr=effective_lr,
                        weight_decay=getattr(config, "optimizer_weight_decay", 0.01),
                        betas=getattr(config, "optimizer_betas", (0.9, 0.95)),
                        fused=(dev.type == "cuda" and bool(getattr(config, "optimizer_fused", True))),
                        foreach=(dev.type == "cuda" and bool(getattr(config, "optimizer_foreach", True))),
                    )
            trace_totals_ms["model_setup"] += (time.perf_counter() - setup_t0) * 1000.0

            # ── Structural smoke test (Phase 6.6) ──
            if graph_json:
                smoke = _smoke_test_graph_structure(graph_json)
                if not smoke.get("ok"):
                    result["passed"] = False
                    result["smoke_test_failure"] = smoke.get("reason", "unknown")
                    result["smoke_test_result"] = smoke
                    return result

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

            # ── RigL dynamic sparse training for sparse architectures ──
            rigl_scheduler = None
            if graph_json:
                _SPARSE_OPS = {"nm_sparse_linear", "block_sparse_linear",
                               "semi_structured_2_4_linear", "ternary_projection"}
                try:
                    _gj = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
                    _nodes = _gj.get("nodes", [])
                    _sparse_count = sum(1 for n in _nodes
                                       if n.get("op_name", "") in _SPARSE_OPS)
                    if _sparse_count >= 1:
                        from ...training.sparse_training import RigLScheduler
                        rigl_scheduler = RigLScheduler(
                            model, sparsity=0.8,
                            update_freq=max(50, config.stage1_steps // 10),
                            total_steps=config.stage1_steps,
                        )
                        result["rigl_enabled"] = True
                        result["rigl_sparse_op_count"] = _sparse_count
                except Exception:
                    rigl_scheduler = None

            initial_loss = None
            final_loss = None
            min_loss = float("inf")
            total_tokens = 0
            t_start = time.perf_counter()

            step_time_sum_ms = 0.0
            step_count = 0
            grad_norm_sum = 0.0
            grad_norm_sq_sum = 0.0
            grad_norm_max = 0.0
            grad_norm_count = 0
            training_curve: List[Dict] = [] if collect_curve else []
            kernel_profiles: List[Dict[str, Any]] = []

            seq_len = min(128, config.max_seq_len)
            random_mode = str(config.data_mode or "random").strip().lower() == "random"
            _seed_int = int(seed)

            # --- Part 1: Discovery Evaluation (Fast) ---
            discovery_loss_fast = self._micro_train_discovery_eval(
                model=model,
                config=config,
                dev=dev,
                seed_int=_seed_int,
                seq_len=seq_len,
            )
            if discovery_loss_fast is not None:
                result["discovery_loss"] = discovery_loss_fast
                # Note: discovery_loss_ratio needs a baseline; we'll compute it in _execute_experiment
            # --- Part 2: Main Training (Validation Channel) ---

            # Adaptive Budget for Novel Architectures (Task 2G)
            total_steps = int(config.stage1_steps)
            if graph_json:
                try:
                    from ...synthesis.primitives import OpCategory
                    _gj = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
                    _nodes = _gj.get("nodes", [])
                    exotic_categories = {OpCategory.MATH_SPACE, OpCategory.SPIKING, OpCategory.FUNCTIONAL}
                    exotic_count = 0
                    for n in _nodes:
                        op_name = n.get("op_name", n.get("op"))
                        if op_name:
                            try:
                                from ...synthesis.primitives import get_primitive
                                if get_primitive(op_name).category in exotic_categories:
                                    exotic_count += 1
                            except Exception:
                                pass
                    if exotic_count >= 2:
                        total_steps *= 2
                        result["adaptive_budget_novelty_bonus"] = True
                        result["exotic_op_count"] = exotic_count
                        logger.debug("    Novelty bonus: granting 2x budget (%d steps) for %d exotic ops", total_steps, exotic_count)
                except Exception as e_novel:
                    logger.debug("Adaptive budget novel check failed: %s", e_novel)

            # Implementation of train/val split for Stage 1
            train_steps = int(total_steps * 0.8)
            total_steps - train_steps

            starvation_interval = max(1, int(getattr(config, "starvation_check_interval", 8) or 8))

            use_cuda_graph = bool(
                dev.type == "cuda"
                and bool(getattr(config, "enable_cuda_graphs", True))
                and random_mode
                and not op_profiler.enabled
                and not trace_enabled
                and not collect_curve
                and int(total_steps) >= 8
            )

            ran_cuda_graph = False
            if use_cuda_graph:
                try:
                    static_input_ids = torch.empty(
                        (config.stage1_batch_size, seq_len), dtype=torch.long, device=dev
                    )
                    captured_loss = torch.zeros((), device=dev)
                    captured_grad_norm = torch.zeros((), device=dev)
                    warmup_steps = max(1, int(getattr(config, "cuda_graph_warmup_steps", 3) or 3))

                    def _graph_step() -> Tuple[torch.Tensor, torch.Tensor]:
                        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=True):
                            logits = model(static_input_ids)
                            loss_t = F.cross_entropy(
                                logits[:, :-1].reshape(-1, logits.shape[-1]),
                                static_input_ids[:, 1:].reshape(-1),
                            )
                        optimizer.zero_grad(set_to_none=True)
                        loss_t.backward()
                        if grad_clip_norm > 0.0:
                            grad_norm_t = nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip_norm, foreach=True
                            )
                        else:
                            grad_norm_t = torch.zeros((), device=dev)
                        optimizer.step()
                        return loss_t, grad_norm_t

                    for wi in range(min(warmup_steps, int(total_steps))):
                        static_input_ids.copy_(
                            self._micro_train_make_random_batch(
                                seed_int=_seed_int,
                                step=wi,
                                batch_size=config.stage1_batch_size,
                                seq_len=seq_len,
                                vocab_size=config.vocab_size,
                                dev=dev,
                            ),
                            non_blocking=True,
                        )
                        loss_t, grad_norm_t = _graph_step()
                        captured_loss.copy_(loss_t.detach())
                        captured_grad_norm.copy_(torch.as_tensor(grad_norm_t, device=dev).detach())

                    torch.cuda.synchronize(dev)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        loss_t, grad_norm_t = _graph_step()
                        captured_loss.copy_(loss_t.detach())
                        captured_grad_norm.copy_(torch.as_tensor(grad_norm_t, device=dev).detach())

                    check_interval = max(1, int(getattr(config, "loss_check_interval", 8) or 8))
                    
                    # Budget extension tracking
                    extension_check_step = 500
                    has_extended = False

                    step = 0
                    while step < total_steps:
                        if self._stop_event.is_set():
                            break
                        t_step = time.perf_counter()
                        static_input_ids.copy_(
                            self._micro_train_make_random_batch(
                                seed_int=_seed_int,
                                step=step,
                                batch_size=config.stage1_batch_size,
                                seq_len=seq_len,
                                vocab_size=config.vocab_size,
                                dev=dev,
                            ),
                            non_blocking=True,
                        )
                        graph.replay()
                        t_step_end = time.perf_counter()
                        step_time_ms = (t_step_end - t_step) * 1000.0
                        step_count += 1
                        step_time_sum_ms += step_time_ms
                        total_tokens += static_input_ids.numel()

                        should_check = (step == 0) or (step == total_steps - 1) or (step % check_interval == 0)
                        
                        loss_val = float(captured_loss.item())
                        grad_norm = float(captured_grad_norm.item())

                        if step == 250:
                            loss_at_250 = loss_val
                        if step == 500:
                            loss_at_500 = loss_val
                            # Task 2G: If still improving at step 500, extend to 1000
                            improvement_rate = (loss_at_250 - loss_at_500) / max(loss_at_250, 1e-6)
                            if improvement_rate > 0 and total_steps < 1000:
                                total_steps = 1000
                                has_extended = True
                                result["adaptive_budget_extension"] = True
                                logger.debug("    Step 500: improvement_rate=%.4f > 0. Extending budget to 1000 steps.", improvement_rate)

                        if not should_check:
                            step += 1
                            continue

                        if not math.isfinite(loss_val):
                            result["error"] = f"NaN/Inf loss at step {step}"
                            result["n_train_steps"] = step
                            return result
                        if step == 0 and (not math.isfinite(grad_norm) or grad_norm <= 1e-10):
                            result["error"] = "zero_grad_precheck_failed"
                            result["n_train_steps"] = 0
                            result["max_grad_norm"] = grad_norm
                            result["mean_grad_norm"] = grad_norm
                            result["grad_norm_std"] = 0.0
                            return result
                        if step == 0:
                            initial_loss = loss_val
                            _es_best_loss_cg = loss_val
                            _es_no_improve_cg = 0
                        final_loss = loss_val
                        min_loss = min(min_loss, loss_val)
                        grad_norm_sum += grad_norm
                        grad_norm_sq_sum += grad_norm * grad_norm
                        grad_norm_max = max(grad_norm_max, grad_norm)
                        grad_norm_count += 1

                        # Early stopping (CUDA graph path)
                        if loss_val < _es_best_loss_cg - config.early_stop_min_delta:
                            _es_best_loss_cg = loss_val
                            _es_no_improve_cg = 0
                        else:
                            _es_no_improve_cg += check_interval
                        if (step >= config.early_stop_min_steps
                                and _es_no_improve_cg >= config.early_stop_patience):
                            result["early_stopped"] = True
                            result["early_stop_step"] = step
                            step_count = step + 1
                            break
                        
                        step += 1
                    ran_cuda_graph = True
                except Exception as e:
                    result["cuda_graph_fallback_reason"] = str(e)

            if not ran_cuda_graph:
                # Budget extension tracking
                loss_at_250 = None
                loss_at_500 = None
                routing_aux_loss_sum = 0.0
                routing_aux_loss_count = 0

                step = 0
                while step < total_steps:
                    if self._stop_event.is_set():
                        break

                    starvation_sample = (not random_mode) and ((step % starvation_interval) == 0)
                    if starvation_sample:
                        starvation_detector.start_wait()
                    data_t0 = time.perf_counter()
                    with _trace_ctx("data_sampling"):
                        if random_mode:
                            input_ids = self._micro_train_make_random_batch(
                                seed_int=_seed_int,
                                step=step,
                                batch_size=config.stage1_batch_size,
                                seq_len=seq_len,
                                vocab_size=config.vocab_size,
                                dev=dev,
                            )
                        else:
                            input_ids = self._sample_training_input_ids(
                                config=config,
                                dev=dev,
                                batch_size=config.stage1_batch_size,
                                seq_len=seq_len,
                                seed=seed + step,
                            )
                    if starvation_sample:
                        starvation_detector.end_wait()
                    trace_totals_ms["data_sampling"] += (time.perf_counter() - data_t0) * 1000.0

                    t_step = time.perf_counter()

                    step_state: Dict[str, Any] = {}

                    def _run_step() -> None:
                        fwd_t0 = time.perf_counter()
                        with _trace_ctx("forward_pass"):
                            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                                    enabled=(dev.type == "cuda")):
                                logits = model(input_ids)
                                if use_synthesized_training and getattr(config, 'loss_type', 'cross_entropy') != 'cross_entropy':
                                    try:
                                        if not hasattr(self, '_synth_loss'):
                                            from ...training.loss_synthesis import synthesize_loss
                                            self._synth_loss = synthesize_loss(seed=seed)
                                        loss = self._synth_loss.compute(
                                            logits[:, :-1], input_ids[:, 1:],
                                        )
                                    except Exception:
                                        loss = F.cross_entropy(
                                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                                            input_ids[:, 1:].reshape(-1),
                                        )
                                else:
                                    loss = F.cross_entropy(
                                        logits[:, :-1].reshape(-1, logits.shape[-1]),
                                        input_ids[:, 1:].reshape(-1),
                                    )
                        trace_totals_ms["forward_pass"] += (time.perf_counter() - fwd_t0) * 1000.0
                        step_state["loss"] = loss

                        # Collect routing load-balance auxiliary loss from
                        # routing telemetry attached during forward pass.
                        aux_loss = _collect_routing_aux_loss(model)
                        if aux_loss is not None:
                            loss = loss + aux_loss
                            step_state["routing_aux_loss"] = aux_loss.item()

                        bwd_t0 = time.perf_counter()
                        with _trace_ctx("backward_pass"):
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            if grad_clip_norm > 0.0:
                                step_state["grad_norm"] = nn.utils.clip_grad_norm_(
                                    model.parameters(), grad_clip_norm, foreach=(dev.type == "cuda")
                                ).item()
                            else:
                                step_state["grad_norm"] = 0.0
                        trace_totals_ms["backward_pass"] += (time.perf_counter() - bwd_t0) * 1000.0

                        opt_t0 = time.perf_counter()
                        with _trace_ctx("optimizer_step"):
                            optimizer.step()
                        trace_totals_ms["optimizer_step"] += (time.perf_counter() - opt_t0) * 1000.0

                    if step == 0 and op_profiler.enabled:
                        kernel_summary = op_profiler.profile_callable(_run_step)
                        if kernel_summary:
                            kernel_profiles.append({"step": step, **kernel_summary})
                        else:
                            _run_step()
                    else:
                        _run_step()

                    # RigL mask update (outside closure to avoid scoping issues)
                    if rigl_scheduler is not None:
                        try:
                            rigl_scheduler.step()
                        except Exception:
                            rigl_scheduler = None

                    loss = step_state.get("loss")
                    grad_norm = float(step_state.get("grad_norm", 0.0))

                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        result["error"] = f"NaN/Inf loss at step {step}"
                        result["n_train_steps"] = step
                        return result

                    loss_val = loss.item()

                    _raux = step_state.get("routing_aux_loss")
                    if _raux is not None:
                        routing_aux_loss_sum += _raux
                        routing_aux_loss_count += 1

                    if step == 250:
                        loss_at_250 = loss_val
                    if step == 500:
                        loss_at_500 = loss_val
                        # Task 2G: If still improving at step 500, extend to 1000
                        if loss_at_250 is not None:
                            improvement_rate = (loss_at_250 - loss_at_500) / max(loss_at_250, 1e-6)
                            if improvement_rate > 0 and total_steps < 1000:
                                total_steps = 1000
                                result["adaptive_budget_extension"] = True
                                logger.debug("    Step 500: improvement_rate=%.4f > 0. Extending budget to 1000 steps.", improvement_rate)

                    if step == 0 and (not math.isfinite(grad_norm) or grad_norm <= 1e-10):
                        result["error"] = "zero_grad_precheck_failed"
                        result["n_train_steps"] = 0
                        result["max_grad_norm"] = grad_norm
                        result["mean_grad_norm"] = grad_norm
                        result["grad_norm_std"] = 0.0
                        return result

                    if dev.type == "cuda" and (trace_enabled or op_profiler.enabled):
                        torch.cuda.synchronize(dev)

                    t_step_end = time.perf_counter()
                    step_time_ms = (t_step_end - t_step) * 1000

                    if step == 0:
                        initial_loss = loss_val
                        _es_best_loss = loss_val
                        _es_steps_since_improve = 0
                        _inflight_state = InflightState()
                    final_loss = loss_val
                    min_loss = min(min_loss, loss_val)
                    total_tokens += input_ids.numel()

                    # Inflight health checks — abort hopeless runs early
                    _inflight_fail = check_inflight_health(
                        step=step, loss_val=loss_val, grad_norm=grad_norm,
                        min_loss=min_loss, initial_loss=initial_loss,
                        total_steps=total_steps, state=_inflight_state,
                        spike_ratio=getattr(config, "inflight_spike_ratio", 2.0),
                        spike_window=getattr(config, "inflight_spike_window", 10),
                        grad_norm_limit=getattr(config, "inflight_grad_norm_limit", 100.0),
                        grad_norm_strikes=getattr(config, "inflight_grad_norm_strikes", 3),
                    )
                    if _inflight_fail is not None:
                        result.update(_inflight_fail)
                        result["n_train_steps"] = step
                        step_count += 1
                        break

                    # Early stopping: break if loss plateaus
                    if loss_val < _es_best_loss - config.early_stop_min_delta:
                        _es_best_loss = loss_val
                        _es_steps_since_improve = 0
                    else:
                        _es_steps_since_improve += 1
                    if (step >= config.early_stop_min_steps
                            and _es_steps_since_improve >= config.early_stop_patience):
                        result["early_stopped"] = True
                        result["early_stop_step"] = step
                        logger.debug(
                            "    early stop at step %d/%d: loss=%.4f plateau for %d steps",
                            step, total_steps, loss_val, config.early_stop_patience,
                        )
                        step_count += 1
                        break

                    step_count += 1
                    step_time_sum_ms += step_time_ms
                    grad_norm_sum += grad_norm
                    grad_norm_sq_sum += grad_norm * grad_norm
                    grad_norm_max = max(grad_norm_max, grad_norm)
                    grad_norm_count += 1

                    # Record per-step data
                    if collect_curve:
                        training_curve.append({
                            "step": step,
                            "loss": loss_val,
                            "grad_norm": grad_norm,
                            "step_time_ms": step_time_ms,
                        })

                    # Emit live training step events for dashboard
                    ctx = getattr(self, "_live_training_context", None)
                    if ctx and step % 10 == 0:
                        step_event = {
                            "experiment_id": ctx.get("exp_id", ""),
                            "step": step,
                            "loss": round(loss_val, 6),
                            "total_steps": total_steps,
                            "phase": ctx.get("phase", ""),
                        }
                        # Append per-step routing telemetry when available
                        _raux_step = step_state.get("routing_aux_loss")
                        if _raux_step is not None:
                            step_event["routing_aux_loss"] = round(_raux_step, 6)
                        if grad_norm > 0:
                            step_event["grad_norm"] = round(grad_norm, 4)
                        self._emit_event("training_step", step_event)

                    # Log training progress at start, midpoint, and end
                    if step == 0 or step == total_steps // 2 or step == total_steps - 1:
                        logger.debug(
                            "    train step %d/%d: loss=%.4f, grad_norm=%.3f, "
                            "step_time=%.1fms",
                            step + 1, total_steps, loss_val, grad_norm, step_time_ms,
                        )
                    
                    step += 1

            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000

            # Optional validation loss on heldout corpus split
            validation_loss = None
            validation_loss_ratio = None
            generalization_gap = None
            try:
                validation_loss = self._micro_train_optional_validation_loss(
                    model=model,
                    config=config,
                    dev=dev,
                    seq_len=seq_len,
                    seed=seed,
                )
            except Exception as e:
                result["validation_loss_error"] = str(e)

            # Optional discovery loss on random tokens (fast triage signal)
            discovery_loss = None
            discovery_loss_ratio = None
            try:
                discovery_loss = self._micro_train_optional_discovery_loss(
                    model=model,
                    config=config,
                    dev=dev,
                    seq_len=seq_len,
                    seed=seed,
                )
            except Exception as e:
                result["discovery_loss_error"] = str(e)

            if validation_loss is not None and initial_loss:
                validation_loss_ratio = validation_loss / max(initial_loss, 1e-6)
            if validation_loss is not None and final_loss is not None:
                generalization_gap = validation_loss - final_loss
            if discovery_loss is not None and initial_loss:
                discovery_loss_ratio = discovery_loss / max(initial_loss, 1e-6)

            # Collect perf results
            if tracer is not None:
                result["perf_traces"] = tracer.get_report()
            else:
                result["perf_traces"] = {
                    "summary_ms": {k: round(v, 4) for k, v in trace_totals_ms.items()},
                    "traces": [],
                }
            result["gpu_starvation"] = starvation_detector.get_summary()
            if kernel_profiles:
                result["kernel_timing"] = {
                    "sample_count": len(kernel_profiles),
                    "samples": kernel_profiles,
                    "top_ops": kernel_profiles[0].get("top_ops", []),
                }

            if initial_loss and final_loss:
                result["loss_ratio"] = normalized_loss_ratio(final_loss, config.vocab_size)
                result["final_loss"] = final_loss
                result["initial_loss"] = initial_loss
                result["min_loss"] = min_loss
                if validation_loss is not None:
                    result["validation_loss"] = validation_loss
                if validation_loss_ratio is not None:
                    result["validation_loss_ratio"] = validation_loss_ratio
                if generalization_gap is not None:
                    result["generalization_gap"] = generalization_gap
                if discovery_loss is not None:
                    result["discovery_loss"] = discovery_loss
                if discovery_loss_ratio is not None:
                    result["discovery_loss_ratio"] = discovery_loss_ratio
                result["throughput"] = total_tokens / (total_time_ms / 1000)
                result["passed"] = result["loss_ratio"] < config.stage1_loss_ratio_threshold
                # Inflight checks already flagged this run — override pass
                if result.get("error_type", "").startswith("inflight_"):
                    result["passed"] = False
                if not result["passed"] and result.get("error_type") is None:
                    result["error_type"] = "failed_convergence"
                    result["error"] = f"Insufficient loss reduction: {result['loss_ratio']:.4f} >= {config.stage1_loss_ratio_threshold}"
                if initial_loss > 0:
                    result["loss_improvement_rate"] = (initial_loss - final_loss) / initial_loss

                # Timing stats
                result["avg_step_time_ms"] = (step_time_sum_ms / step_count) if step_count > 0 else 0.0
                result["total_train_time_ms"] = total_time_ms

                # Gradient norm stats
                if grad_norm_count > 0:
                    result["max_grad_norm"] = grad_norm_max
                    result["mean_grad_norm"] = grad_norm_sum / grad_norm_count
                    mean_gn = result["mean_grad_norm"]
                    var = max((grad_norm_sq_sum / grad_norm_count) - (mean_gn * mean_gn), 0.0)
                    result["grad_norm_std"] = var ** 0.5

                result["n_train_steps"] = step_count
                result["final_lr"] = config.stage1_lr  # constant for now
                if collect_curve:
                    result["training_curve"] = training_curve

                # Extract architecture-specific telemetry (MoE, MoD, MoR, etc.)
                arch_telemetry = self._extract_architecture_telemetry(model)
                result.update(arch_telemetry)

                # Routing training metrics: load-balance aux loss + derived stats
                if routing_aux_loss_count > 0:
                    result["routing_aux_loss_mean"] = (
                        routing_aux_loss_sum / routing_aux_loss_count
                    )
                rt_total = result.get("routing_tokens_total", 0)
                rt_processed = result.get("routing_tokens_processed", 0)
                if rt_total > 0:
                    result["routing_fast_fraction"] = max(
                        0.0, 1.0 - (rt_processed / rt_total),
                    )
                    eu_json = result.get("routing_expert_utilization_json")
                    if eu_json:
                        try:
                            counts = json.loads(eu_json)
                            if counts:
                                total_c = sum(counts)
                                if total_c > 0:
                                    fracs = [c / total_c for c in counts]
                                    uniform = 1.0 / len(fracs)
                                    # Balance = 1 - normalized MSE (1=uniform, 0=collapsed)
                                    mse = sum((f - uniform) ** 2 for f in fracs) / len(fracs)
                                    result["routing_balance_score"] = max(
                                        0.0, 1.0 - mse * len(fracs),
                                    )
                        except (json.JSONDecodeError, TypeError):
                            pass

                # Behavioral fingerprint for S1 survivors (novelty scoring)
                if result.get("passed") and model is not None:
                    try:
                        # Task 4I: skip full fingerprint for poor performers (Investigation Gating)
                        _lr = result.get("loss_ratio", 1.0)
                        _perf_gate = float(getattr(config, "fingerprint_perf_gate", 0.85) or 0.85)
                        _force_lightning = _lr > _perf_gate
                        
                        if _force_lightning:
                            logger.debug("    Investigation gating: skipping full fingerprint for poor performer (LR=%.4f > %.2f)", _lr, _perf_gate)

                        _fp, full_ran = compute_gated_fingerprint(
                            model,
                            seq_len=min(64, config.max_seq_len),
                            model_dim=config.model_dim,
                            vocab_size=config.vocab_size,
                            device=str(dev),
                            full_gate_enabled=True,
                            force_lightning_only=_force_lightning,
                        )
                        result["_behavioral_fingerprint"] = _fp.to_dict()
                        result["fingerprint_full_ran"] = full_ran
                    except Exception as e_fp:
                        logger.debug("Fingerprint failed in S1 worker: %s", e_fp)

                # Fast WikiText perplexity at screening time
                if not getattr(config, "skip_screening_wikitext", False):
                    try:
                        from ...eval.wikitext_eval import screening_wikitext_eval

                        wt = screening_wikitext_eval(
                            model, config.vocab_size, str(dev),
                            seq_len=min(128, config.max_seq_len),
                        )
                        result["screening_wikitext_status"] = wt.get("screening_wikitext_status")
                        result["screening_wikitext_metric_version"] = wt.get("screening_wikitext_metric_version")
                        if wt.get("wikitext_perplexity") is not None:
                            result["wikitext_perplexity"] = wt["wikitext_perplexity"]
                            result["wikitext_score"] = wt.get("wikitext_score")
                            result["wikitext_pre_perplexity"] = wt.get("wikitext_pre_perplexity")
                            result["wikitext_ppl_improvement"] = wt.get("wikitext_ppl_improvement")
                            logger.info(
                                "    Screening WikiText ppl=%.1f score=%.3f (%.0fms)",
                                wt["wikitext_perplexity"],
                                wt.get("wikitext_score") or 0,
                                wt.get("elapsed_ms") or 0,
                            )
                    except Exception as e_wt:
                        logger.debug("Screening WikiText eval skipped: %s", e_wt)

        except Exception as e:
            result["error"] = str(e)

        if result.get("final_loss") is not None and bool(getattr(config, "one_shot_pruning_baseline", False)):
            try:
                seq_len = min(128, int(config.max_seq_len))
                eval_batches = max(1, int(getattr(config, "one_shot_pruning_eval_batches", 4)))
                eval_batch_size = max(1, int(getattr(config, "one_shot_pruning_batch_size", 2)))

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
                    target_sparsity=float(getattr(config, "one_shot_pruning_sparsity", 0.5)),
                    method=str(getattr(config, "one_shot_pruning_method", "wanda")),
                )
                pruned_eval_loss = estimate_lm_ce_loss(pruned_model, eval_inputs, dev)

                quality_retention = None
                if dense_eval_loss is not None and pruned_eval_loss is not None and pruned_eval_loss > 0:
                    quality_retention = max(0.0, min(1.5, dense_eval_loss / pruned_eval_loss))

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
            except Exception as e:
                result["pruning_error"] = str(e)

        # Finalize performance reports
        try:
            if tracer is not None:
                fallback_perf = tracer.get_report()
            else:
                fallback_perf = {
                    "summary_ms": {k: round(v, 4) for k, v in trace_totals_ms.items()},
                    "traces": [],
                }
            result["perf_report"] = result.get("perf_traces", fallback_perf)
            # Ensure throughput is included in perf_report for experiment-level aggregation
            if isinstance(result.get("throughput"), (int, float)):
                result["perf_report"]["avg_throughput_tok_s"] = float(result["throughput"])

            result["starvation_report"] = result.get("gpu_starvation", starvation_detector.get_summary())
            if "kernel_timing" in result:
                result["kernel_timings_ms"] = result["kernel_timing"]
        except Exception as e:
            result["perf_error"] = str(e)

        try:
            result.update(self._extract_architecture_telemetry(model))
        except Exception as e:
            logger.debug("Architecture telemetry extract failed: %s", e)

        return result

    def _micro_train_async(self, model: nn.Module, config: RunConfig, seed: int, dev: torch.device) -> Dict:
        """Async worker entry point for training a pre-compiled model."""
        try:
            return self._micro_train(model, config, dev, seed=seed)
        except Exception as e:
            return {"error": str(e), "passed": False}

    def _train_with_program(self, model: nn.Module, program,
                            config: RunConfig,
                            dev: torch.device,
                            seed: int = 42) -> Dict:
        """Train a model using a synthesized TrainingProgram.

        Returns same metrics dict as _micro_train() plus training_program_json.
        """
        from research.scientist.perf import PerfTracer, GPUStarvationDetector, KernelTimer
        tracer = PerfTracer()
        starvation_detector = GPUStarvationDetector(threshold_ms=2.0)
        kernel_timer = KernelTimer(model, enabled=bool(getattr(config, "enable_kernel_profiling", False)))

        result: Dict[str, Any] = {"passed": False}

        try:
            with tracer.trace("model_setup"):
                model = model.to(dev)
                model.train()

            # Apply init scheme
            if program.init_scheme == "small":
                for p in model.parameters():
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
                optimizer = program.optimizer.create(model.parameters())
            except Exception as exc:
                logger.warning(
                    "program.optimizer.create() failed (%s); "
                    "falling back to AdamW via build_optimizer",
                    exc,
                )
                from ...training.optimizer_synthesis import build_optimizer
                optimizer = build_optimizer(
                    model.parameters(),
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
            # Adaptive clip for math-space architectures
            from ._helpers import apply_adaptive_grad_clip
            max_grad_norm_val = apply_adaptive_grad_clip(model, max_grad_norm_val)

            initial_loss = None
            final_loss = None
            min_loss = float("inf")
            total_tokens = 0
            t_start = time.perf_counter()

            step_times: List[float] = []
            grad_norms: List[float] = []
            training_curve: List[Dict] = []

            # VRAM-aware seq_len cap: probe free memory and scale down
            # to avoid OOM with quadratic-attention ops like ultrametric_attention
            _static_cap = 512
            if dev.type == "cuda":
                try:
                    free_mb = (torch.cuda.get_device_properties(dev).total_memory
                               - torch.cuda.memory_allocated(dev)) / (1024 * 1024)
                    # Rough heuristic: quadratic ops need ~B*S*S*D*4 bytes per layer
                    # At dim=256, batch=B, n_layers=L: budget ≈ free * 0.5 (leave headroom)
                    _batch = int(getattr(config, 'stage1_batch_size', 4) or 4)
                    _nlayers = int(getattr(config, 'n_layers', 4) or 4)
                    _dim = int(getattr(config, 'model_dim', 256) or 256)
                    # max_seq where B*S^2*D*L*12 (fwd+bwd+optim) < free*0.5
                    import math as _math
                    _budget = free_mb * 0.5 * 1024 * 1024  # bytes
                    _max_s = int(_math.sqrt(_budget / (max(_batch, 1) * max(_dim, 1) * max(_nlayers, 1) * 12)))
                    _static_cap = min(_static_cap, max(64, _max_s))
                    if _static_cap < config.max_seq_len:
                        logger.info("VRAM-capped seq_len: %d (free=%.0fMB, B=%d, L=%d)",
                                    _static_cap, free_mb, _batch, _nlayers)
                except Exception:
                    pass
            safe_max_seq = min(config.max_seq_len, _static_cap)
            seq_len = min(128, safe_max_seq)
            # Apply curriculum seq_len schedule
            try:
                base_seq = program.curriculum.get_seq_len(0, n_steps)
                if base_seq and base_seq > 0:
                    seq_len = min(base_seq, safe_max_seq)
            except Exception:
                pass

            for step in range(n_steps):
                if self._stop_event.is_set():
                    break

                # Update seq_len from curriculum
                try:
                    curr_seq = program.curriculum.get_seq_len(step, n_steps)
                    if curr_seq and curr_seq > 0:
                        seq_len = min(curr_seq, safe_max_seq)
                except Exception:
                    pass

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
                    with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                            enabled=(dev.type == "cuda")):
                        logits = model(input_ids)
                        # Use synthesized loss if possible
                        try:
                            loss = program.loss.compute(
                                logits[:, :-1].reshape(-1, logits.shape[-1]),
                                input_ids[:, 1:].reshape(-1),
                            )
                        except Exception:
                            loss = F.cross_entropy(
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
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm_val).item()
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
                final_loss = loss_val
                min_loss = min(min_loss, loss_val)
                total_tokens += input_ids.numel()

                # Inflight health checks — abort hopeless runs early
                _inflight_fail = check_inflight_health(
                    step=step, loss_val=loss_val, grad_norm=grad_norm,
                    min_loss=min_loss, initial_loss=initial_loss,
                    total_steps=n_steps, state=_inflight_state_inv,
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
                if (step >= config.early_stop_min_steps
                        and _es_steps_since_improve >= config.early_stop_patience):
                    result["early_stopped"] = True
                    result["early_stop_step"] = step
                    logger.debug(
                        "    early stop at step %d/%d: loss=%.4f plateau for %d steps",
                        step, n_steps, loss_val, config.early_stop_patience,
                    )
                    break

                step_times.append(step_time_ms)
                grad_norms.append(grad_norm)

                training_curve.append({
                    "step": step,
                    "loss": loss_val,
                    "grad_norm": grad_norm,
                    "step_time_ms": step_time_ms,
                })

                # Emit live training step events for dashboard
                ctx = getattr(self, "_live_training_context", None)
                if ctx and step % 10 == 0:
                    step_event = {
                        "experiment_id": ctx.get("exp_id", ""),
                        "step": step,
                        "loss": round(loss_val, 6),
                        "total_steps": n_steps,
                        "phase": ctx.get("phase", ""),
                    }
                    if grad_norm > 0:
                        step_event["grad_norm"] = round(grad_norm, 4)
                    self._emit_event("training_step", step_event)

            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000

            if initial_loss and final_loss:
                result["loss_ratio"] = normalized_loss_ratio(final_loss, config.vocab_size)
                result["final_loss"] = final_loss
                result["initial_loss"] = initial_loss
                result["min_loss"] = min_loss
                result["throughput"] = total_tokens / (total_time_ms / 1000)
                result["passed"] = result["loss_ratio"] < config.stage1_loss_ratio_threshold
                # Inflight checks already flagged this run — override pass
                if result.get("error_type", "").startswith("inflight_"):
                    result["passed"] = False
                if not result["passed"] and result.get("error_type") is None:
                    result["error_type"] = "failed_convergence"
                    result["error"] = f"Insufficient loss reduction during investigation: {result['loss_ratio']:.4f}"
                    result["loss_improvement_rate"] = (initial_loss - final_loss) / initial_loss

                result["avg_step_time_ms"] = sum(step_times) / len(step_times) if step_times else 0
                result["total_train_time_ms"] = total_time_ms

                if grad_norms:
                    result["max_grad_norm"] = max(grad_norms)
                    result["mean_grad_norm"] = sum(grad_norms) / len(grad_norms)
                    mean_gn = result["mean_grad_norm"]
                    result["grad_norm_std"] = (
                        sum((g - mean_gn) ** 2 for g in grad_norms) / len(grad_norms)
                    ) ** 0.5

                result["n_train_steps"] = len(step_times)
                result["final_lr"] = getattr(optimizer, 'defaults', {}).get('lr', 3e-4)
                result["training_curve"] = training_curve
                result["training_program_json"] = json.dumps(program.to_dict())

                # Extract architecture-specific telemetry (MoE, MoD, MoR, etc.)
                arch_telemetry = self._extract_architecture_telemetry(model)
                result.update(arch_telemetry)

        except Exception as e:
            result["error"] = str(e)

        # Finalize performance reports
        try:
            result["perf_report"] = tracer.get_report()
            # Ensure throughput is included in perf_report for experiment-level aggregation
            if isinstance(result.get("throughput"), (int, float)):
                result["perf_report"]["avg_throughput_tok_s"] = float(result["throughput"])

            result["starvation_report"] = starvation_detector.get_summary()
            if kernel_timer.enabled:
                result["kernel_timings_ms"] = kernel_timer.synchronize_and_get_timings()
        except Exception as e:
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
    ) -> torch.Tensor:
        """Sample input IDs from configured data source with deterministic seed."""
        mode = str(config.data_mode or "random").strip().lower()
        generator = torch.Generator(device=dev)
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
                )
                if batch is not None:
                    return batch

        return torch.randint(
            0,
            int(config.vocab_size),
            (batch_size, seq_len),
            device=dev,
            generator=generator,
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
                generator = torch.Generator(device=dev)
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
                return torch.randint(0, config.vocab_size, (batch_size, seq_len), device=dev, generator=generator)

            return data_fn, data_tag, True
        if mode == "hydra":
            def data_fn(batch_size, seq_len, dev):
                batch = self._get_hydra_batch(config, batch_size, seq_len, dev)
                if batch is not None:
                    return batch
                return torch.randint(0, config.vocab_size, (batch_size, seq_len), device=dev)
            return data_fn, "hydra", False
        if mode == "corpus":
            path = str(config.corpus_path or "").strip()
            version = self._corpus_version_tag(path)
            train_frac = float(getattr(config, "corpus_train_fraction", 0.9) or 0.9)
            val_frac = float(getattr(config, "corpus_val_fraction", 0.1) or 0.1)
            fmt = str(config.corpus_format or "auto")
            text_key = str(config.corpus_text_key or "text")
            tok = str(config.tokenizer_mode or "byte")
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
                generator = torch.Generator(device=dev)
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
                return torch.randint(0, config.vocab_size, (batch_size, seq_len), device=dev, generator=generator)

            return data_fn, data_tag, True
        return None, "random", False
