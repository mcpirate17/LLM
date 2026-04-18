"""Execution mixin: micro-train, train-with-program, data sampling, baseline."""

from __future__ import annotations

import copy
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..json_utils import json_safe

import torch
import torch.nn as nn

from ...eval.fingerprint import compute_gated_fingerprint
from ...eval.perf_budget import DEFAULT_PERF_BUDGETS, evaluate_perf_budget_gate
from ...eval.pruning import apply_one_shot_pruning, estimate_lm_ce_loss
from ...eval.utils import clip_grad_norm, language_model_loss
from ...training.profiling import TrainingRunProfiler
from ._helpers import (
    normalized_loss_ratio,
    stage1_learning_gate,
    resolve_stage1_gate_metrics,
    get_reference_losses,
    _corpus_type_from_config,
)
from .execution_training_native_boundary import (
    _build_training_step_event,
    _MicroTrainLoopProgress,
    _TrainingLoopState,
    _apply_training_aux_losses,
    _backward_loss,
    _collect_aux_modules,
    _compute_micro_train_forward_loss,
    _maybe_extend_training_budget,
    _optimizer_step,
    _training_step_error,
)

import logging

logger = logging.getLogger(__name__)


def _nested_metric_present(payload: Dict[str, Any], dotted_key: str) -> bool:
    node: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return node is not None


def _candidate_perf_budget_verdict(
    perf_report: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    """Evaluate the screening perf budget against metrics this run produced.

    Stage-1 micro-trains do not always emit the full experiment-level perf
    report, so avoid failing candidates only because a metric family is absent.
    When at least one screening_default budget metric is present, evaluate the
    available subset and return the verdict.
    """
    report = perf_report or {}
    screening_budgets = DEFAULT_PERF_BUDGETS.get("screening_default", {})
    available = {
        key: limit
        for key, limit in screening_budgets.items()
        if _nested_metric_present(report, key)
    }
    if not available:
        return None
    verdict = evaluate_perf_budget_gate(
        report,
        budget_profile="screening_default",
        budgets=available,
    )
    verdict["partial"] = len(available) < len(screening_budgets)
    verdict["checked_metrics"] = sorted(available)
    return verdict


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
from ...training.checkpointing import CheckpointManager


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


def _serialize_inflight_state(state: InflightState | None) -> Dict[str, Any]:
    if state is None:
        return {}
    return {
        "recent_losses": list(state.recent_losses),
        "grad_strikes": int(state.grad_strikes),
        "window": int(state.window),
    }


def _restore_inflight_state(payload: Dict[str, Any] | None) -> InflightState:
    payload = payload or {}
    state = InflightState(window=int(payload.get("window", 20) or 20))
    state.recent_losses = [float(v) for v in payload.get("recent_losses", [])]
    state.grad_strikes = int(payload.get("grad_strikes", 0) or 0)
    return state


def _serialize_progress(progress: _MicroTrainLoopProgress) -> Dict[str, Any]:
    return {
        "initial_loss": progress.initial_loss,
        "final_loss": progress.final_loss,
        "min_loss": progress.min_loss,
        "total_tokens": progress.total_tokens,
        "step_count": progress.step_count,
        "step_time_sum_ms": progress.step_time_sum_ms,
        "grad_norm_sum": progress.grad_norm_sum,
        "grad_norm_sq_sum": progress.grad_norm_sq_sum,
        "grad_norm_max": progress.grad_norm_max,
        "grad_norm_count": progress.grad_norm_count,
        "training_curve": list(progress.training_curve),
        "entropy_gate_trajectory": list(progress.entropy_gate_trajectory),
        "routing_aux_loss_sum": progress.routing_aux_loss_sum,
        "routing_aux_loss_count": progress.routing_aux_loss_count,
        "loss_at_250": progress.loss_at_250,
        "loss_at_500": progress.loss_at_500,
    }


def _restore_progress(payload: Dict[str, Any] | None) -> _MicroTrainLoopProgress:
    payload = payload or {}
    progress = _MicroTrainLoopProgress()
    progress.initial_loss = payload.get("initial_loss")
    progress.final_loss = payload.get("final_loss")
    progress.min_loss = float(payload.get("min_loss", float("inf")))
    progress.total_tokens = int(payload.get("total_tokens", 0) or 0)
    progress.step_count = int(payload.get("step_count", 0) or 0)
    progress.step_time_sum_ms = float(payload.get("step_time_sum_ms", 0.0) or 0.0)
    progress.grad_norm_sum = float(payload.get("grad_norm_sum", 0.0) or 0.0)
    progress.grad_norm_sq_sum = float(payload.get("grad_norm_sq_sum", 0.0) or 0.0)
    progress.grad_norm_max = float(payload.get("grad_norm_max", 0.0) or 0.0)
    progress.grad_norm_count = int(payload.get("grad_norm_count", 0) or 0)
    progress.training_curve = list(payload.get("training_curve", []) or [])
    progress.entropy_gate_trajectory = list(
        payload.get("entropy_gate_trajectory", []) or []
    )
    progress.routing_aux_loss_sum = float(
        payload.get("routing_aux_loss_sum", 0.0) or 0.0
    )
    progress.routing_aux_loss_count = int(payload.get("routing_aux_loss_count", 0) or 0)
    progress.loss_at_250 = payload.get("loss_at_250")
    progress.loss_at_500 = payload.get("loss_at_500")
    return progress


def _phase_checkpoint_context(owner: Any) -> Dict[str, Any] | None:
    context = getattr(owner, "_live_training_context", None)
    if not isinstance(context, dict):
        return None
    manager = context.get("checkpoint_manager")
    if not isinstance(manager, CheckpointManager):
        return None
    checkpoint_phase = context.get("checkpoint_phase", context.get("phase"))
    if not context.get("exp_id") or not checkpoint_phase:
        return None
    if context.get("checkpoint_candidate_idx") is None:
        return None
    if context.get("checkpoint_seed_idx") is None:
        return None
    return context


def _restore_phase_training_state(
    owner: Any,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any] | None:
    context = _phase_checkpoint_context(owner)
    if context is None:
        return None
    checkpoint_state = context.pop("checkpoint_resume_state", None)
    if not checkpoint_state:
        return None
    restored = context["checkpoint_manager"].restore_phase_state(
        checkpoint_state,
        model=model,
        optimizer=optimizer,
        device=device,
    )
    metrics = restored.get("metrics") or {}
    return {
        "step": int(restored.get("step", 0) or 0),
        "total_steps": int(metrics.get("total_steps", 0) or 0),
        "elapsed_ms": float(metrics.get("elapsed_ms", 0.0) or 0.0),
        "progress": _restore_progress(metrics.get("progress")),
        "inflight_state": _restore_inflight_state(metrics.get("inflight_state")),
        "early_stop_best_loss": metrics.get("early_stop_best_loss"),
        "early_stop_steps_since_improve": int(
            metrics.get("early_stop_steps_since_improve", 0) or 0
        ),
    }


def _maybe_save_phase_training_state(
    owner: Any,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    total_steps: int,
    progress: _MicroTrainLoopProgress,
    inflight_state: InflightState | None,
    early_stop_best_loss: float | None,
    early_stop_steps_since_improve: int,
    elapsed_ms: float,
) -> None:
    context = _phase_checkpoint_context(owner)
    if context is None:
        return
    interval = int(context.get("checkpoint_interval_steps", 0) or 0)
    if interval <= 0 or completed_steps <= 0 or (completed_steps % interval) != 0:
        return
    context["checkpoint_manager"].save_phase(
        experiment_id=str(context["exp_id"]),
        phase=str(context.get("checkpoint_phase", context["phase"])),
        candidate_idx=int(context["checkpoint_candidate_idx"]),
        seed_idx=int(context["checkpoint_seed_idx"]),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        step=completed_steps,
        metrics={
            "source_result_id": context.get("source_result_id"),
            "total_steps": int(total_steps),
            "elapsed_ms": float(elapsed_ms),
            "progress": _serialize_progress(progress),
            "inflight_state": _serialize_inflight_state(inflight_state),
            "early_stop_best_loss": early_stop_best_loss,
            "early_stop_steps_since_improve": int(early_stop_steps_since_improve),
        },
    )


@dataclass
class _MicroTrainContext:
    """Shared state passed between _micro_train sub-methods."""

    model: Any  # nn.Module
    config: Any  # RunConfig
    dev: Any  # torch.device
    seed: int
    graph_json: str
    graph_data: Any
    result: Dict[str, Any]
    progress: Any  # _MicroTrainLoopProgress
    optimizer: Any  # torch.optim.Optimizer
    model_params: tuple
    routing_modules: list
    early_exit_modules: list
    lm_head: Any
    norm: Any
    tracer: Any
    trace_totals_ms: Dict[str, float]
    starvation_detector: Any
    op_profiler: Any
    run_profiler: Any  # TrainingRunProfiler
    use_synthesized_training: bool
    collect_curve: bool
    grad_clip_norm: float
    total_steps: int
    seq_len: int
    random_mode: bool
    seed_int: int
    t_start: float
    kernel_profiles: List[Dict[str, Any]] = field(default_factory=list)
    resume_state: Optional[Dict] = None
    starvation_interval: int = 8
    starvation_monitoring: bool = False

    def trace_ctx(self, name: str, use_gpu: bool = True):
        return (
            self.tracer.trace(name, use_gpu=use_gpu)
            if self.tracer is not None
            else nullcontext()
        )


def _micro_train_attribute_error(e: Exception, result: Dict[str, Any]) -> None:
    """Handle training exceptions: log, attribute failing op."""
    import re as _re
    import traceback as _tb

    logger.debug("Training failed (%s): %s", type(e).__name__, e)
    result["error"] = str(e)
    result["error_type"] = type(e).__name__
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




# ── Mixin composition ─────────────────────────────────────────────
# Method bodies live in three split modules to stay under the 1250-
# line file cap. Composing them here preserves the external symbol
# `_ExecutionTrainingMixin` referenced by `runner/__init__.py`.

from .execution_training_post import _ExecutionTrainingPostMixin  # noqa: E402
from .execution_training_micro import _ExecutionTrainingMicroMixin  # noqa: E402
from .execution_training_program import _ExecutionTrainingProgramMixin  # noqa: E402


class _ExecutionTrainingMixin(
    _ExecutionTrainingPostMixin,
    _ExecutionTrainingMicroMixin,
    _ExecutionTrainingProgramMixin,
):
    """Execution-training mixin (composed)."""

    __slots__ = ()
