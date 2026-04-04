"""Execution mixin: micro-train, train-with-program, data sampling, baseline."""

from __future__ import annotations

import copy
import json
import math
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

from ..json_utils import json_safe

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...eval.fingerprint import compute_gated_fingerprint
from ...eval.pruning import apply_one_shot_pruning, estimate_lm_ce_loss
from ...training.profiling import TrainingRunProfiler
from ._helpers import (
    normalized_loss_ratio,
    stage1_learning_gate,
    get_reference_losses,
    _corpus_type_from_config,
)

import logging

logger = logging.getLogger(__name__)


class _EntropyGateSampler:
    """Capture token-entropy telemetry from the main training forward pass."""

    __slots__ = ("_enabled", "_handles", "_values")

    def __init__(self, model: nn.Module):
        self._enabled = False
        self._values: List[float] = []
        self._handles = []
        for mod in model.modules():
            op_name = getattr(mod, "_op_name", None)
            if op_name and "token_entropy" in str(op_name):
                self._handles.append(mod.register_forward_hook(self._hook))

    def _hook(self, module: nn.Module, inp: Any, out: Any) -> None:  # noqa: ARG002
        if not self._enabled or not isinstance(out, torch.Tensor):
            return
        self._values.append(float(out.detach().abs().mean().item()))

    @property
    def available(self) -> bool:
        return bool(self._handles)

    def begin_sample(self) -> None:
        self._values.clear()
        self._enabled = True

    def finish_sample(self) -> float | None:
        self._enabled = False
        if not self._values:
            return None
        return sum(self._values) / len(self._values)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _smoke_test_graph_structure(graph_json) -> Dict[str, Any]:
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

        graph_data = (
            json.loads(graph_json) if isinstance(graph_json, str) else graph_json
        )
        nodes_raw = graph_data.get("nodes", [])
        if not nodes_raw:
            return {"ok": False, "reason": "empty graph"}

        # Sort nodes by id for stable indexing
        nodes_sorted = sorted(nodes_raw, key=lambda n: n["id"])
        id_to_idx = {n["id"]: i for i, n in enumerate(nodes_sorted)}
        n_nodes = len(nodes_sorted)

        # Role code mapping
        _ROLE_CODES = {
            OpRole.PROJECT: 0,
            OpRole.NORMALIZE: 1,
            OpRole.ACTIVATE: 2,
            OpRole.MIX: 3,
            OpRole.ROUTE: 4,
            OpRole.GATE: 5,
            OpRole.POSITION: 6,
            OpRole.REDUCE: 7,
            OpRole.RESIDUAL: 8,
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

        result = smoke_fn(
            n_nodes, edges, op_roles, has_params_flag, preserves_grad, output_idx
        )
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

    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
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
    model: nn.Module,
    weight: float = 0.01,
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


def _collect_early_exit_loss(
    model: nn.Module,
    targets: torch.Tensor,
    weight: float = 0.1,
) -> "torch.Tensor | None":
    """Collect early-exit auxiliary losses from early_exit ops.

    Early-exit ops store hidden states and gate values during the forward pass.
    This function projects those hidden states through the model's shared
    lm_head to produce early logits, then computes gate-weighted cross-entropy
    against the training targets.  This gives the confidence gate real gradient
    signal to learn which tokens are easy vs hard.
    """
    lm_head = getattr(model, "lm_head", None)
    norm = getattr(model, "norm", None)
    if lm_head is None:
        return None

    aux = torch.tensor(0.0)
    found = False

    for module in model.modules():
        ee_aux = getattr(module, "_early_exit_aux", None)
        if ee_aux is None:
            continue
        found = True
        hidden = ee_aux["hidden"]  # (B, S, D)
        gate = ee_aux["gate"]  # (B, S) — high = easy
        module._early_exit_aux = None  # free memory

        # Project through shared lm_head (with optional norm)
        normed = norm(hidden) if norm is not None else hidden
        early_logits = lm_head(normed)  # (B, S, V)

        B, S, V = early_logits.shape
        # Shift: predict next token (same as main loss)
        logits_shifted = early_logits[:, :-1].reshape(-1, V)
        gate_shifted = gate[:, :-1].reshape(-1)

        # Align targets shape
        tgt = targets
        if tgt.numel() != logits_shifted.shape[0]:
            tgt = tgt[: logits_shifted.shape[0]]

        per_token_ce = F.cross_entropy(logits_shifted, tgt, reduction="none")
        # Gate-weighted mean: easy tokens (high gate) contribute more
        weighted_ce = (gate_shifted * per_token_ce).sum() / gate_shifted.sum().clamp(
            min=1.0
        )
        aux = aux + weighted_ce

    if not found:
        return None
    return aux * weight


class _ExecutionTrainingMixin:
    """Micro-training, train-with-program, data sampling, baseline data."""

    __slots__ = ()

    # ── Scale-Up Mode ──

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
        from research.scientist.perf import (
            PerfTracer,
            GPUStarvationDetector,
            OpKernelProfiler,
        )

        # Parse graph JSON once upfront to avoid repeated deserialization
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
        starvation_detector = GPUStarvationDetector(threshold_ms=2.0)
        op_profiler = OpKernelProfiler(
            enabled=bool(getattr(config, "enable_kernel_profiling", False)),
            top_k=max(1, int(getattr(config, "kernel_profile_top_k", 20) or 20)),
        )
        run_profiler = TrainingRunProfiler(config, dev)

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
            return (
                tracer.trace(name, use_gpu=use_gpu)
                if tracer is not None
                else nullcontext()
            )

        try:
            run_profiler.__enter__()
            setup_t0 = time.perf_counter()
            with _trace_ctx("model_setup"), run_profiler.trace("model_setup_ms"):
                model = model.to(dev)
                model.train()
                from ...training.optimizer_synthesis import build_optimizer

                # Resolve phase-specific optimizer: screening_optimizer overrides optimizer_type
                phase_opt = getattr(config, "screening_optimizer", "") or ""
                opt_type = (
                    phase_opt or getattr(config, "optimizer_type", "adamw") or "adamw"
                )
                phase_lr = getattr(config, "screening_lr", 0.0) or 0.0
                effective_lr = phase_lr if phase_lr > 0 else config.stage1_lr

                # Synthesized optimizer support (screening exploration only)
                if use_synthesized_training and opt_type == "synthesized":
                    from ...training.optimizer_synthesis import synthesize_optimizer

                    synth_opt = synthesize_optimizer(seed=seed)
                    optimizer = synth_opt.create(
                        model.parameters(),
                        lr=effective_lr,
                    )
                    result["optimizer_synthesized"] = synth_opt.name
                else:
                    # Non-synthesized: use build_optimizer (no silent fallbacks)
                    resolved_type = opt_type if opt_type != "synthesized" else "adamw"
                    optimizer = build_optimizer(
                        model.parameters(),
                        optimizer_type=resolved_type,
                        lr=effective_lr,
                        weight_decay=getattr(
                            config,
                            "optimizer_weight_decay",
                            0.01,
                        ),
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
            trace_totals_ms["model_setup"] += (time.perf_counter() - setup_t0) * 1000.0

            # ── Structural smoke test (Phase 6.6) ──
            if graph_data:
                smoke = _smoke_test_graph_structure(graph_data)
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
            if graph_data:
                try:
                    from ...synthesis.primitives import OpCategory

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
                                from ...synthesis.primitives import get_primitive

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

            starvation_interval = max(
                1, int(getattr(config, "starvation_check_interval", 8) or 8)
            )
            starvation_monitoring = bool(
                getattr(config, "enable_starvation_monitoring", False)
                or trace_enabled
                or bool(getattr(config, "profile_enabled", False))
            )

            use_cuda_graph = bool(
                dev.type == "cuda"
                and bool(getattr(config, "enable_cuda_graphs", True))
                and random_mode
                and not op_profiler.enabled
                and not trace_enabled
                and not collect_curve
                and not bool(getattr(config, "profile_enabled", False))
                and int(total_steps) >= 8
            )

            ran_cuda_graph = False
            if use_cuda_graph:
                try:
                    static_input_ids = torch.empty(
                        (config.stage1_batch_size, seq_len),
                        dtype=torch.long,
                        device=dev,
                    )
                    captured_loss = torch.zeros((), device=dev)
                    captured_grad_norm = torch.zeros((), device=dev)
                    warmup_steps = max(
                        1, int(getattr(config, "cuda_graph_warmup_steps", 3) or 3)
                    )

                    def _graph_step() -> Tuple[torch.Tensor, torch.Tensor]:
                        with torch.amp.autocast(
                            device_type=dev.type, dtype=torch.bfloat16, enabled=True
                        ):
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
                        captured_grad_norm.copy_(
                            torch.as_tensor(grad_norm_t, device=dev).detach()
                        )
                        # Bail early if model produces NaN/Inf during warmup
                        if not torch.isfinite(captured_loss):
                            break

                    torch.cuda.synchronize(dev)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        loss_t, grad_norm_t = _graph_step()
                        captured_loss.copy_(loss_t.detach())
                        captured_grad_norm.copy_(
                            torch.as_tensor(grad_norm_t, device=dev).detach()
                        )

                    check_interval = max(
                        1, int(getattr(config, "loss_check_interval", 8) or 8)
                    )

                    # Budget extension tracking

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

                        should_check = (
                            (step == 0)
                            or (step == total_steps - 1)
                            or (step % check_interval == 0)
                        )

                        # Milestone steps need Python floats; all other steps
                        # skip to `continue` below, so defer .item() to avoid
                        # CPU-GPU sync on every step.
                        is_milestone = step == 250 or step == 500
                        if not should_check and not is_milestone:
                            run_profiler.step()
                            step += 1
                            continue

                        loss_val = float(captured_loss.item())
                        grad_norm = float(captured_grad_norm.item())

                        if step == 250:
                            loss_at_250 = loss_val
                        if step == 500:
                            loss_at_500 = loss_val
                            # Task 2G: If still improving at step 500, extend to 1000
                            improvement_rate = (loss_at_250 - loss_at_500) / max(
                                loss_at_250, 1e-6
                            )
                            if improvement_rate > 0 and total_steps < 1000:
                                total_steps = 1000
                                result["adaptive_budget_extension"] = True
                                logger.debug(
                                    "    Step 500: improvement_rate=%.4f > 0. Extending budget to 1000 steps.",
                                    improvement_rate,
                                )

                        if not should_check:
                            run_profiler.step()
                            step += 1
                            continue

                        if not math.isfinite(loss_val):
                            result["error"] = f"NaN/Inf loss at step {step}"
                            result["n_train_steps"] = step
                            return result
                        if step == 0 and (
                            not math.isfinite(grad_norm) or grad_norm <= 1e-10
                        ):
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
                        if (
                            step >= config.early_stop_min_steps
                            and _es_no_improve_cg >= config.early_stop_patience
                        ):
                            result["early_stopped"] = True
                            result["early_stop_step"] = step
                            step_count = step + 1
                            break

                        run_profiler.record_step(
                            step=step, loss=loss_val, grad_norm=grad_norm
                        )
                        run_profiler.step()
                        step += 1
                    ran_cuda_graph = True
                except RuntimeError as e:
                    logger.debug("CUDA graph capture failed, falling back: %s", e)
                    result["cuda_graph_fallback_reason"] = str(e)

            # Entropy gate trajectory (sampled at key steps during training)
            _ENTROPY_GATE_SAMPLE_STEPS = frozenset({10, 25, 50, 75, 100})
            _entropy_gate_trajectory: List[float] = []
            entropy_gate_sampler = _EntropyGateSampler(model)

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

                    starvation_sample = (
                        starvation_monitoring
                        and (not random_mode)
                        and ((step % starvation_interval) == 0)
                    )
                    if starvation_sample:
                        starvation_detector.start_wait()
                    data_t0 = time.perf_counter()
                    with (
                        _trace_ctx("data_sampling"),
                        run_profiler.trace("data_sampling_ms"),
                    ):
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
                                timer=run_profiler.record_timing,
                            )
                    if starvation_sample:
                        starvation_detector.end_wait()
                    trace_totals_ms["data_sampling"] += (
                        time.perf_counter() - data_t0
                    ) * 1000.0

                    t_step = time.perf_counter()
                    should_sample_entropy = (
                        entropy_gate_sampler.available
                        and step in _ENTROPY_GATE_SAMPLE_STEPS
                    )
                    if should_sample_entropy:
                        entropy_gate_sampler.begin_sample()

                    step_state: Dict[str, Any] = {}

                    def _run_step() -> None:
                        fwd_t0 = time.perf_counter()
                        with (
                            _trace_ctx("forward_pass"),
                            run_profiler.trace("forward_pass_ms"),
                        ):
                            with torch.amp.autocast(
                                device_type=dev.type,
                                dtype=torch.bfloat16,
                                enabled=(dev.type == "cuda"),
                            ):
                                logits = model(input_ids)
                                if (
                                    use_synthesized_training
                                    and getattr(config, "loss_type", "cross_entropy")
                                    != "cross_entropy"
                                ):
                                    try:
                                        if not hasattr(self, "_synth_loss"):
                                            from ...training.loss_synthesis import (
                                                synthesize_loss,
                                            )

                                            self._synth_loss = synthesize_loss(
                                                seed=seed
                                            )
                                        loss = self._synth_loss.compute(
                                            logits[:, :-1],
                                            input_ids[:, 1:],
                                        )
                                    except (RuntimeError, ValueError, TypeError) as e:
                                        logger.debug(
                                            "Synthesized loss failed, falling back to CE: %s",
                                            e,
                                        )
                                        loss = F.cross_entropy(
                                            logits[:, :-1].reshape(
                                                -1, logits.shape[-1]
                                            ),
                                            input_ids[:, 1:].reshape(-1),
                                        )
                                else:
                                    loss = F.cross_entropy(
                                        logits[:, :-1].reshape(-1, logits.shape[-1]),
                                        input_ids[:, 1:].reshape(-1),
                                    )
                        trace_totals_ms["forward_pass"] += (
                            time.perf_counter() - fwd_t0
                        ) * 1000.0
                        step_state["loss"] = loss

                        # Collect routing load-balance auxiliary loss from
                        # routing telemetry attached during forward pass.
                        aux_loss = _collect_routing_aux_loss(model)
                        if aux_loss is not None:
                            loss = loss + aux_loss
                            # Keep tensor in step_state; extract at
                            # checkpoint only to avoid per-step GPU sync.
                            step_state["routing_aux_loss_tensor"] = aux_loss.detach()

                        # Collect early-exit auxiliary loss: projects
                        # intermediate hidden states through shared lm_head.
                        ee_loss = _collect_early_exit_loss(
                            model, input_ids[:, 1:].reshape(-1)
                        )
                        if ee_loss is not None:
                            loss = loss + ee_loss
                            step_state["early_exit_aux_loss_tensor"] = ee_loss.detach()

                        bwd_t0 = time.perf_counter()
                        with (
                            _trace_ctx("backward_pass"),
                            run_profiler.trace("backward_pass_ms"),
                        ):
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            if grad_clip_norm > 0.0:
                                step_state["grad_norm"] = nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    grad_clip_norm,
                                    foreach=(dev.type == "cuda"),
                                ).item()
                            else:
                                step_state["grad_norm"] = 0.0
                        trace_totals_ms["backward_pass"] += (
                            time.perf_counter() - bwd_t0
                        ) * 1000.0

                        opt_t0 = time.perf_counter()
                        with (
                            _trace_ctx("optimizer_step"),
                            run_profiler.trace("optimizer_step_ms"),
                        ):
                            optimizer.step()
                        trace_totals_ms["optimizer_step"] += (
                            time.perf_counter() - opt_t0
                        ) * 1000.0

                    if step == 0 and op_profiler.enabled:
                        kernel_summary = op_profiler.profile_callable(_run_step)
                        if kernel_summary:
                            kernel_profiles.append({"step": step, **kernel_summary})
                        else:
                            _run_step()
                    else:
                        _run_step()

                    if should_sample_entropy:
                        _eg_val = entropy_gate_sampler.finish_sample()
                        if _eg_val is not None:
                            _entropy_gate_trajectory.append(_eg_val)
                            if _eg_val < 0.05:
                                logger.warning(
                                    "entropy_gate_collapse_detected at step %d: "
                                    "value=%.4f",
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
                    if _raux_t is not None:
                        routing_aux_loss_sum += float(_raux_t.item())
                        routing_aux_loss_count += 1

                    if step == 250:
                        loss_at_250 = loss_val
                    if step == 500:
                        loss_at_500 = loss_val
                        # Task 2G: If still improving at step 500, extend to 1000
                        if loss_at_250 is not None:
                            improvement_rate = (loss_at_250 - loss_at_500) / max(
                                loss_at_250, 1e-6
                            )
                            if improvement_rate > 0 and total_steps < 1000:
                                total_steps = 1000
                                result["adaptive_budget_extension"] = True
                                logger.debug(
                                    "    Step 500: improvement_rate=%.4f > 0. Extending budget to 1000 steps.",
                                    improvement_rate,
                                )

                    if step == 0 and (
                        not math.isfinite(grad_norm) or grad_norm <= 1e-10
                    ):
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
                    _inflight_fail = None
                    if not bool(
                        getattr(config, "profile_disable_inflight_checks", False)
                    ):
                        _inflight_fail = check_inflight_health(
                            step=step,
                            loss_val=loss_val,
                            grad_norm=grad_norm,
                            min_loss=min_loss,
                            initial_loss=initial_loss,
                            total_steps=total_steps,
                            state=_inflight_state,
                            spike_ratio=getattr(config, "inflight_spike_ratio", 2.0),
                            spike_window=getattr(config, "inflight_spike_window", 10),
                            grad_norm_limit=getattr(
                                config,
                                "inflight_grad_norm_limit",
                                100.0,
                            ),
                            grad_norm_strikes=getattr(
                                config,
                                "inflight_grad_norm_strikes",
                                3,
                            ),
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
                    if (
                        step >= config.early_stop_min_steps
                        and _es_steps_since_improve >= config.early_stop_patience
                    ):
                        result["early_stopped"] = True
                        result["early_stop_step"] = step
                        logger.debug(
                            "    early stop at step %d/%d: loss=%.4f plateau for %d steps",
                            step,
                            total_steps,
                            loss_val,
                            config.early_stop_patience,
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
                            "total_steps": total_steps,
                            "phase": ctx.get("phase", ""),
                        }
                        # Append per-step routing telemetry when available
                        _raux_step_t = step_state.get("routing_aux_loss_tensor")
                        if _raux_step_t is not None:
                            step_event["routing_aux_loss"] = round(
                                float(_raux_step_t.item()), 6
                            )
                        if grad_norm > 0:
                            step_event["grad_norm"] = round(grad_norm, 4)
                        self._emit_event("training_step", step_event)

                    # Log training progress at start, midpoint, and end
                    if step == 0 or step == total_steps // 2 or step == total_steps - 1:
                        logger.debug(
                            "    train step %d/%d: loss=%.4f, grad_norm=%.3f, "
                            "step_time=%.1fms",
                            step + 1,
                            total_steps,
                            loss_val,
                            grad_norm,
                            step_time_ms,
                        )

                    run_profiler.record_step(
                        step=step, loss=loss_val, grad_norm=grad_norm
                    )
                    run_profiler.step()
                    step += 1

            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000

            # Optional validation loss on heldout corpus split
            validation_loss = None
            validation_loss_ratio = None
            generalization_gap = None
            if not bool(getattr(config, "profile_disable_post_eval", False)):
                try:
                    with run_profiler.trace("validation_eval_ms"):
                        validation_loss = self._micro_train_optional_validation_loss(
                            model=model,
                            config=config,
                            dev=dev,
                            seq_len=seq_len,
                            seed=seed,
                        )
                except RuntimeError as e:
                    logger.debug("Validation loss eval failed: %s", e)
                    result["validation_loss_error"] = str(e)

            # Optional discovery loss on random tokens (fast triage signal)
            discovery_loss = None
            discovery_loss_ratio = None
            if not bool(getattr(config, "profile_disable_post_eval", False)):
                try:
                    with run_profiler.trace("discovery_eval_ms"):
                        discovery_loss = self._micro_train_optional_discovery_loss(
                            model=model,
                            config=config,
                            dev=dev,
                            seq_len=seq_len,
                            seed=seed,
                        )
                except RuntimeError as e:
                    logger.debug("Discovery loss eval failed: %s", e)
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

            if initial_loss is not None and final_loss is not None:
                # Store both loss ratio formulas with unambiguous names:
                #   loss_ratio_raw  = final_loss / initial_loss  (relative improvement)
                #   loss_ratio_norm = final_loss / ln(vocab_size) (absolute position)
                # The auto-escalation threshold (0.18) is calibrated against RAW.
                # loss_ratio keeps RAW for backward compatibility.
                _raw = final_loss / max(initial_loss, 1e-6)
                _norm = normalized_loss_ratio(final_loss, config.vocab_size)
                result["loss_ratio"] = _raw
                result["loss_ratio_raw"] = _raw
                result["loss_ratio_norm"] = _norm
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

                # Corpus-aware learning gate (replaces fixed threshold)
                corpus_type = _corpus_type_from_config(config)
                tokenizer = str(config.tokenizer_mode or "byte")
                try:
                    ref_losses = get_reference_losses(
                        str(getattr(self, "notebook_path", "research/lab_notebook.db"))
                    )
                except (OSError, ValueError, KeyError) as e:
                    logger.debug("Reference loss lookup failed: %s", e)
                    ref_losses = {}
                # Gate uses raw final/initial ratio for divergence checks,
                # NOT normalized_loss_ratio which measures a different thing.
                raw_ratio = final_loss / max(initial_loss, 1e-6)
                gate_passed, gate_reason = stage1_learning_gate(
                    final_loss=final_loss,
                    loss_ratio=raw_ratio,
                    initial_loss=initial_loss,
                    n_steps=step_count,
                    corpus_type=corpus_type,
                    tokenizer=tokenizer,
                    reference_losses=ref_losses,
                )
                result["passed"] = gate_passed
                result["gate_reason"] = gate_reason

                # Validation loss gate: if val loss didn't improve, fail.
                if (
                    result["passed"]
                    and validation_loss_ratio is not None
                    and validation_loss_ratio > 0.6
                ):
                    result["passed"] = False
                    result["error_type"] = "insufficient_learning"
                    result["error"] = (
                        f"Validation loss ratio {validation_loss_ratio:.4f} > 0.60 — "
                        f"model memorized training but failed to generalize"
                    )
                # Inflight checks already flagged this run — override pass
                if result.get("error_type", "").startswith("inflight_"):
                    result["passed"] = False
                if not result["passed"] and result.get("error_type") is None:
                    result["error_type"] = "failed_convergence"
                    result["error"] = gate_reason
                if initial_loss > 0:
                    result["loss_improvement_rate"] = (
                        initial_loss - final_loss
                    ) / initial_loss

                # Timing stats
                result["avg_step_time_ms"] = (
                    (step_time_sum_ms / step_count) if step_count > 0 else 0.0
                )
                result["total_train_time_ms"] = total_time_ms

                # Gradient norm stats
                if grad_norm_count > 0:
                    result["max_grad_norm"] = grad_norm_max
                    result["mean_grad_norm"] = grad_norm_sum / grad_norm_count
                    mean_gn = result["mean_grad_norm"]
                    var = max(
                        (grad_norm_sq_sum / grad_norm_count) - (mean_gn * mean_gn), 0.0
                    )
                    result["grad_norm_std"] = var**0.5

                result["n_train_steps"] = step_count
                result["final_lr"] = config.stage1_lr  # constant for now
                if collect_curve:
                    result["training_curve"] = training_curve
                artifacts = run_profiler.artifacts()
                if artifacts is not None:
                    result["profile_artifacts"] = {
                        "output_dir": artifacts.output_dir,
                        "summary_json": artifacts.summary_json,
                        "trace_json": artifacts.trace_json,
                    }
                    run_profiler.event("avg_step_time_ms", result["avg_step_time_ms"])
                    run_profiler.event("throughput_tok_s", result["throughput"])
                    run_profiler.event("n_train_steps", result["n_train_steps"])

                # Extract architecture-specific telemetry (MoE, MoD, MoR, etc.)
                arch_telemetry = self._extract_architecture_telemetry(model)
                result.update(arch_telemetry)

                # Entropy gate trajectory (sampled during training)
                if _entropy_gate_trajectory:
                    result["entropy_gate_trajectory_json"] = json.dumps(
                        json_safe(_entropy_gate_trajectory)
                    )
                    if any(v < 0.05 for v in _entropy_gate_trajectory):
                        result["routing_collapse_score"] = 1.0

                # Routing training metrics: load-balance aux loss + derived stats
                if routing_aux_loss_count > 0:
                    result["routing_aux_loss_mean"] = (
                        routing_aux_loss_sum / routing_aux_loss_count
                    )
                rt_total = result.get("routing_tokens_total", 0)
                rt_processed = result.get("routing_tokens_processed", 0)
                if rt_total > 0:
                    result["routing_fast_fraction"] = max(
                        0.0,
                        1.0 - (rt_processed / rt_total),
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
                                    mse = sum((f - uniform) ** 2 for f in fracs) / len(
                                        fracs
                                    )
                                    result["routing_balance_score"] = max(
                                        0.0,
                                        1.0 - mse * len(fracs),
                                    )
                        except (json.JSONDecodeError, TypeError):
                            pass

                # Behavioral fingerprint for S1 survivors (structural-only at
                # screening; CKA + behavioral probes deferred to post-investigation)
                if (
                    result.get("passed")
                    and model is not None
                    and not bool(getattr(config, "skip_post_s1_triage", False))
                    and not bool(getattr(config, "profile_disable_post_eval", False))
                ):
                    try:
                        _lr = result.get("loss_ratio", 1.0)
                        _perf_gate = float(
                            getattr(config, "fingerprint_perf_gate", 0.85) or 0.85
                        )
                        _force_lightning = _lr > _perf_gate

                        if _force_lightning:
                            logger.debug(
                                "    Investigation gating: skipping full fingerprint for poor performer (LR=%.4f > %.2f)",
                                _lr,
                                _perf_gate,
                            )

                        # Parse graph for structural novelty computation
                        _graph_obj = None
                        if graph_json:
                            try:
                                from ...synthesis.serializer import graph_from_json

                                _graph_obj = graph_from_json(graph_json)
                            except (ValueError, KeyError, json.JSONDecodeError) as e:
                                logger.debug(
                                    "Graph deserialization failed for fingerprint: %s",
                                    e,
                                )

                        _fp, full_ran = compute_gated_fingerprint(
                            model,
                            seq_len=min(64, config.max_seq_len),
                            model_dim=config.model_dim,
                            vocab_size=config.vocab_size,
                            device=str(dev),
                            full_gate_enabled=True,
                            force_lightning_only=_force_lightning,
                            graph=_graph_obj,
                            structural_floor=float(
                                getattr(config, "lightning_structural_floor", 0.10)
                                or 0.10
                            ),
                        )
                        result["_behavioral_fingerprint"] = _fp.to_dict()
                        result["fingerprint_full_ran"] = full_ran
                    except (RuntimeError, ValueError, TypeError) as e_fp:
                        logger.debug("Fingerprint failed in S1 worker: %s", e_fp)

                # Fast WikiText perplexity at screening time
                if not getattr(config, "skip_screening_wikitext", False) and not bool(
                    getattr(config, "profile_disable_post_eval", False)
                ):
                    try:
                        from ...eval.wikitext_eval import screening_wikitext_eval

                        wt = screening_wikitext_eval(
                            model,
                            config.vocab_size,
                            str(dev),
                            seq_len=min(128, config.max_seq_len),
                        )
                        result["screening_wikitext_status"] = wt.get(
                            "screening_wikitext_status"
                        )
                        result["screening_wikitext_metric_version"] = wt.get(
                            "screening_wikitext_metric_version"
                        )
                        if wt.get("wikitext_perplexity") is not None:
                            result["wikitext_perplexity"] = wt["wikitext_perplexity"]
                            result["wikitext_score"] = wt.get("wikitext_score")
                            result["wikitext_pre_perplexity"] = wt.get(
                                "wikitext_pre_perplexity"
                            )
                            result["wikitext_ppl_improvement"] = wt.get(
                                "wikitext_ppl_improvement"
                            )
                        # Slope trajectory fields (for slope reprieve)
                        for _slope_key in (
                            "screening_loss_10",
                            "screening_loss_25",
                            "screening_loss_50",
                            "screening_slope",
                            "screening_slope_consistent",
                        ):
                            if wt.get(_slope_key) is not None:
                                result[_slope_key] = wt[_slope_key]
                        logger.info(
                            "    Screening WikiText ppl=%.1f score=%.3f (%.0fms)",
                            wt["wikitext_perplexity"],
                            wt.get("wikitext_score") or 0,
                            wt.get("elapsed_ms") or 0,
                        )
                    except (RuntimeError, ValueError, OSError, ImportError) as e_wt:
                        logger.debug("Screening WikiText eval skipped: %s", e_wt)

                # Fast HellaSwag commonsense reasoning probe at screening time
                if not getattr(config, "skip_screening_hellaswag", False) and not bool(
                    getattr(config, "profile_disable_post_eval", False)
                ):
                    try:
                        from ...eval.hellaswag_eval import screening_hellaswag_eval

                        hs = screening_hellaswag_eval(
                            model,
                            config.vocab_size,
                            str(dev),
                        )
                        result["hellaswag_acc"] = hs.get("hellaswag_acc")
                        result["hellaswag_status"] = hs.get("hellaswag_status")
                        result["hellaswag_n_examples"] = hs.get("hellaswag_total")
                        result["screening_hellaswag_correct"] = hs.get(
                            "hellaswag_correct"
                        )
                        result["screening_hellaswag_total"] = hs.get("hellaswag_total")
                        result["screening_hellaswag_elapsed_ms"] = hs.get("elapsed_ms")
                        if hs.get("hellaswag_acc") is not None:
                            logger.info(
                                "    Screening HellaSwag acc=%.1f%% (%d/%d, %.0fms)",
                                hs["hellaswag_acc"] * 100,
                                hs.get("hellaswag_correct", 0),
                                hs.get("hellaswag_total", 0),
                                hs.get("elapsed_ms", 0),
                            )
                    except (RuntimeError, ValueError, OSError, ImportError) as e_hs:
                        logger.debug("Screening HellaSwag eval skipped: %s", e_hs)

                # Binding probes: induction head + binding range (post micro-train)
                if not getattr(config, "skip_binding_probes", False) and not bool(
                    getattr(config, "profile_disable_post_eval", False)
                ):
                    try:
                        from ...eval.induction_probe import induction_score
                        from ...eval.binding_range import binding_range_profile

                        ind = induction_score(
                            model,
                            gaps=(4, 8, 16, 32, 64),
                            n_train_steps=1000,
                            n_eval=100,
                            batch_size=16,
                            device=str(dev),
                        )
                        result["induction_auc"] = ind.auc
                        result["induction_gap_accuracies"] = ind.gap_accuracies
                        result["induction_probe_train_steps"] = 1000
                        result["induction_probe_eval_examples"] = 100
                        result["induction_probe_batch_size"] = 16
                        result["induction_probe_gaps"] = [4, 8, 16, 32, 64]
                        result["induction_probe_elapsed_ms"] = ind.elapsed_ms

                        br = binding_range_profile(
                            model,
                            distances=(2, 4, 8, 16, 32, 64),
                            n_eval=100,
                            device=str(dev),
                        )
                        result["binding_auc"] = br.auc
                        result["binding_distance_accuracies"] = br.distance_accuracies
                        result["binding_probe_eval_examples"] = 100
                        result["binding_probe_distances"] = [2, 4, 8, 16, 32, 64]
                        result["binding_probe_elapsed_ms"] = br.elapsed_ms

                        # AR probe skipped at screening (too slow, signal is weak)
                        result["ar_auc"] = None

                        # Binding composite: 3-signal weighted average
                        # (ar_auc=None at screening, so only induction + binding_auc)
                        bc = 0.3 * ind.auc + 0.3 * br.auc
                        result["binding_composite"] = round(bc, 4)

                        logger.info(
                            "    Binding probes: induction_auc=%.3f binding_auc=%.3f bc=%.3f (%.0f+%.0fms)",
                            ind.auc,
                            br.auc,
                            bc,
                            ind.elapsed_ms,
                            br.elapsed_ms,
                        )

                        # HIGH PRIORITY DISCOVERY: induction_auc > 0.20 without
                        # standard causal attention. This would be a novel mechanism
                        # for exact token retrieval across gaps.
                        if ind.auc > 0.20 and graph_data:
                            graph_nodes = []
                            if isinstance(graph_data, dict):
                                graph_nodes = [
                                    node
                                    for node in graph_data.get("nodes", [])
                                    if not node.get("is_input", False)
                                ]
                            _has_attention = any(
                                n.get("op_name", n.get("op"))
                                in (
                                    "softmax_attention",
                                    "diff_attention",
                                    "graph_attention",
                                    "linear_attention",
                                )
                                for n in graph_nodes
                            )
                            if not _has_attention:
                                logger.warning(
                                    "*** HIGH PRIORITY DISCOVERY: %s induction_auc=%.3f "
                                    "WITHOUT standard attention ops! Investigate immediately. "
                                    "Graph ops: %s",
                                    result.get("graph_fingerprint", "?")[:10],
                                    ind.auc,
                                    [
                                        n.get("op_name", n.get("op"))
                                        for n in graph_nodes
                                    ],
                                )
                    except (RuntimeError, ValueError, TypeError, ImportError) as e_bp:
                        logger.debug("Binding probes skipped: %s", e_bp)

                # Post-S1 triage: cheap evals for composite score dimensions
                if (
                    result.get("passed")
                    and model is not None
                    and not bool(getattr(config, "skip_post_s1_fingerprint", False))
                    and not bool(getattr(config, "profile_disable_post_eval", False))
                ):
                    try:
                        from .execution_triage import run_triage

                        _graph_for_triage = None
                        if graph_json:
                            try:
                                from ...synthesis.serializer import graph_from_json

                                _graph_for_triage = graph_from_json(graph_json)
                            except (ValueError, KeyError, json.JSONDecodeError) as e:
                                logger.debug(
                                    "Graph deserialization failed for triage: %s", e
                                )
                        triage = run_triage(
                            model,
                            _graph_for_triage,
                            result,
                            config.model_dim,
                        )
                        if triage:
                            result.update(triage)
                            _n_rt = triage.get("n_routing_ops", 0)
                            _n_sp = triage.get("n_sparse_ops", 0)
                            _n_mo = triage.get("n_moe_ops", 0)
                            _qpp = triage.get("param_efficiency", 0)
                            logger.info(
                                "    Triage: %d fields (qpp=%.2f, route=%d sparse=%d moe=%d)",
                                len(triage),
                                _qpp,
                                _n_rt,
                                _n_sp,
                                _n_mo,
                            )
                    except (RuntimeError, ValueError, TypeError) as e_tri:
                        logger.debug("Triage eval skipped: %s", e_tri)

        except Exception as e:
            logger.debug("Training failed (%s): %s", type(e).__name__, e)
            result["error"] = str(e)
            result["error_type"] = type(e).__name__
            # Op attribution: parse the traceback for the failing op
            import traceback as _tb
            import re as _re

            tb_lines = _tb.format_exc().strip().split("\n")
            for line in reversed(tb_lines):
                if "_op_" in line and "in _op_" in line:
                    m = _re.search(r"in (_op_\w+)", line)
                    if m:
                        result["failure_op"] = m.group(1).removeprefix("_op_")
                        break
            if "failure_op" not in result:
                err_str = str(e)
                if "kv_compress" in err_str:
                    result["failure_op"] = "latent_attention_compressor"
                elif "conv_weight" in err_str:
                    result["failure_op"] = "conv1d_seq"
        finally:
            if "entropy_gate_sampler" in locals():
                entropy_gate_sampler.close()
            run_profiler.__exit__(None, None, None)

        if result.get("final_loss") is not None and bool(
            getattr(config, "one_shot_pruning_baseline", False)
        ):
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

        return result

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
            except (RuntimeError, ValueError, TypeError) as exc:
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
                    free_mb = (
                        torch.cuda.get_device_properties(dev).total_memory
                        - torch.cuda.memory_allocated(dev)
                    ) / (1024 * 1024)
                    # Rough heuristic: quadratic ops need ~B*S*S*D*4 bytes per layer
                    # At dim=256, batch=B, n_layers=L: budget ≈ free * 0.5 (leave headroom)
                    _batch = int(getattr(config, "stage1_batch_size", 4) or 4)
                    _nlayers = int(getattr(config, "n_layers", 4) or 4)
                    _dim = int(getattr(config, "model_dim", 256) or 256)
                    # max_seq where B*S^2*D*L*12 (fwd+bwd+optim) < free*0.5
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

            for step in range(n_steps):
                if self._stop_event.is_set():
                    break

                # Update seq_len from curriculum
                try:
                    curr_seq = program.curriculum.get_seq_len(step, n_steps)
                    if curr_seq and curr_seq > 0:
                        seq_len = min(curr_seq, safe_max_seq)
                except (AttributeError, TypeError, ValueError):
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
                    with torch.amp.autocast(
                        device_type=dev.type,
                        dtype=torch.bfloat16,
                        enabled=(dev.type == "cuda"),
                    ):
                        logits = model(input_ids)
                        # Use synthesized loss if possible
                        try:
                            loss = program.loss.compute(
                                logits[:, :-1].reshape(-1, logits.shape[-1]),
                                input_ids[:, 1:].reshape(-1),
                            )
                        except (RuntimeError, ValueError, TypeError) as e:
                            logger.debug(
                                "Program loss failed, falling back to CE: %s", e
                            )
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
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm_val
                    ).item()
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
                    }
                    if grad_norm > 0:
                        step_event["grad_norm"] = round(grad_norm, 4)
                    self._emit_event("training_step", step_event)

            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000

            if initial_loss is not None and final_loss is not None:
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
                # Formula: threshold = base + (1 - base) * max(0, 1 - init_loss / 50)
                # At init_loss=190: threshold = 0.4 (unchanged)
                # At init_loss=50:  threshold = 0.4 (unchanged)
                # At init_loss=20:  threshold = 0.76
                # At init_loss=12:  threshold = 0.86
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
                # Inflight checks already flagged this run — override pass
                if result.get("error_type", "").startswith("inflight_"):
                    result["passed"] = False
                if not result["passed"] and result.get("error_type") is None:
                    result["error_type"] = "failed_convergence"
                    result["error"] = (
                        f"Insufficient loss reduction during investigation: {result['loss_ratio']:.4f}"
                    )
                    result["loss_improvement_rate"] = (
                        initial_loss - final_loss
                    ) / initial_loss

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
                result["training_program_json"] = json.dumps(
                    json_safe(program.to_dict())
                )

                # Extract architecture-specific telemetry (MoE, MoD, MoR, etc.)
                arch_telemetry = self._extract_architecture_telemetry(model)
                result.update(arch_telemetry)

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
