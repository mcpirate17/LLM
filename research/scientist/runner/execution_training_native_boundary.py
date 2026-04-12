"""Native-oriented helpers for Stage 1 training loops.

This module clusters compute-adjacent helpers that are good candidates for a
future C/C++ or Rust replacement while leaving orchestration in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, List, Tuple
import logging

import torch
import torch.nn as nn

from ...eval.utils import clip_grad_norm, language_model_loss

logger = logging.getLogger(__name__)


def _native_training_summary(loop_state: "_TrainingLoopState") -> dict[str, Any]:
    """Summarize per-step training telemetry with a native fast path."""

    try:
        from ...eval._runner_native import load_runner_native

        return dict(
            load_runner_native().summarize_training_loop(
                int(loop_state.total_tokens),
                float(loop_state.total_time_ms),
                int(loop_state.step_count),
                float(loop_state.step_time_sum_ms),
                float(loop_state.grad_norm_sum),
                float(loop_state.grad_norm_sq_sum),
                float(loop_state.grad_norm_max),
                int(loop_state.grad_norm_count),
            )
        )
    except Exception:
        throughput = (
            loop_state.total_tokens / (loop_state.total_time_ms / 1000.0)
            if loop_state.total_time_ms > 0.0
            else 0.0
        )
        avg_step_time_ms = (
            loop_state.step_time_sum_ms / loop_state.step_count
            if loop_state.step_count > 0
            else 0.0
        )
        summary: dict[str, Any] = {
            "throughput": throughput,
            "avg_step_time_ms": avg_step_time_ms,
            "n_train_steps": loop_state.step_count,
            "max_grad_norm": None,
            "mean_grad_norm": None,
            "grad_norm_std": None,
        }
        if loop_state.grad_norm_count > 0:
            mean = loop_state.grad_norm_sum / loop_state.grad_norm_count
            var = max(
                (loop_state.grad_norm_sq_sum / loop_state.grad_norm_count)
                - (mean * mean),
                0.0,
            )
            summary["max_grad_norm"] = loop_state.grad_norm_max
            summary["mean_grad_norm"] = mean
            summary["grad_norm_std"] = var**0.5
        return summary


def _collect_routing_aux_loss(
    routing_modules: Tuple[nn.Module, ...],
    weight: float = 0.01,
) -> "torch.Tensor | None":
    """Collect load-balance auxiliary loss from routing telemetry."""

    aux = torch.tensor(0.0)
    found = False

    for module in routing_modules:
        rt = getattr(module, "routing_telemetry", None)
        if rt is None:
            continue
        expert_counts = rt.get("expert_counts")
        if not isinstance(expert_counts, torch.Tensor) or expert_counts.numel() < 2:
            continue
        found = True
        total = expert_counts.sum().clamp(min=1.0)
        fracs = expert_counts.float() / total
        uniform = 1.0 / expert_counts.numel()
        aux = aux + ((fracs - uniform) ** 2).sum()

    if not found:
        return None
    return aux * weight


def _collect_early_exit_loss(
    early_exit_modules: Tuple[nn.Module, ...],
    lm_head: nn.Module | None,
    norm: nn.Module | None,
    targets: torch.Tensor,
    weight: float = 0.1,
) -> "torch.Tensor | None":
    """Collect gate-weighted auxiliary loss from early-exit modules."""

    if lm_head is None:
        return None

    aux = torch.tensor(0.0)
    found = False

    for module in early_exit_modules:
        early_exit_aux = getattr(module, "_early_exit_aux", None)
        if early_exit_aux is None:
            continue
        found = True
        hidden = early_exit_aux["hidden"]
        gate = early_exit_aux["gate"]
        module._early_exit_aux = None

        normed = norm(hidden) if norm is not None else hidden
        early_logits = lm_head(normed)
        gate_shifted = gate[:, :-1].reshape(-1)

        tgt = targets
        if tgt.dim() != 2:
            continue
        if tgt.shape[1] != early_logits.shape[1]:
            tgt = tgt[:, : early_logits.shape[1]]

        per_token_ce = language_model_loss(
            early_logits,
            tgt,
            int(early_logits.shape[-1]),
            reduction="none",
        )
        if per_token_ce.numel() != gate_shifted.numel():
            min_tokens = min(per_token_ce.numel(), gate_shifted.numel())
            per_token_ce = per_token_ce[:min_tokens]
            gate_shifted = gate_shifted[:min_tokens]
        weighted_ce = (gate_shifted * per_token_ce).sum() / gate_shifted.sum().clamp(
            min=1.0
        )
        aux = aux + weighted_ce

    if not found:
        return None
    return aux * weight


def _collect_aux_modules(
    model: nn.Module,
) -> tuple[
    Tuple[nn.Module, ...], Tuple[nn.Module, ...], nn.Module | None, nn.Module | None
]:
    """Collect routing and early-exit modules once before the training loop."""

    routing_modules: List[nn.Module] = []
    early_exit_modules: List[nn.Module] = []
    for module in model.modules():
        if hasattr(module, "routing_telemetry"):
            routing_modules.append(module)
        if hasattr(module, "_early_exit_aux"):
            early_exit_modules.append(module)
    return (
        tuple(routing_modules),
        tuple(early_exit_modules),
        getattr(model, "lm_head", None),
        getattr(model, "norm", None),
    )


def _compute_micro_train_forward_loss(
    owner: Any,
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    config: Any,
    dev: torch.device,
    use_synthesized_training: bool,
    seed: int,
) -> torch.Tensor:
    """Compute the micro-train forward loss for one eager step."""

    with torch.amp.autocast(
        device_type=dev.type,
        dtype=torch.bfloat16,
        enabled=(dev.type == "cuda"),
    ):
        logits = model(input_ids)
        if (
            use_synthesized_training
            and getattr(config, "loss_type", "cross_entropy") != "cross_entropy"
        ):
            try:
                if not hasattr(owner, "_synth_loss"):
                    from ...training.loss_synthesis import synthesize_loss

                    owner._synth_loss = synthesize_loss(seed=seed)
                return owner._synth_loss.compute(
                    logits[:, :-1],
                    input_ids[:, 1:],
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.debug("Synthesized loss failed, falling back to CE: %s", exc)
        return language_model_loss(
            logits,
            input_ids,
            min(
                int(getattr(config, "vocab_size", logits.shape[-1])),
                int(logits.shape[-1]),
            ),
        )


def _apply_training_aux_losses(
    loss: torch.Tensor,
    *,
    routing_modules: Tuple[nn.Module, ...],
    early_exit_modules: Tuple[nn.Module, ...],
    lm_head: nn.Module | None,
    norm: nn.Module | None,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Attach routing and early-exit auxiliary losses to the main step loss."""

    routing_aux_loss = _collect_routing_aux_loss(routing_modules)
    if routing_aux_loss is not None:
        loss = loss + routing_aux_loss

    early_exit_aux_loss = _collect_early_exit_loss(
        early_exit_modules,
        lm_head,
        norm,
        input_ids,
    )
    if early_exit_aux_loss is not None:
        loss = loss + early_exit_aux_loss

    return loss, routing_aux_loss, early_exit_aux_loss


def _backward_loss(
    loss: torch.Tensor,
    *,
    optimizer: Any,
    grad_clip_norm: float,
    model_params: Tuple[torch.Tensor, ...],
) -> float:
    """Run backward and optional grad clipping, leaving stepping to the caller."""

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip_norm > 0.0:
        grad_norm = clip_grad_norm(model_params, grad_clip_norm).item()
    else:
        grad_norm = 0.0
    return float(grad_norm)


def _optimizer_step(optimizer: Any) -> None:
    """Run the optimizer step as its own boundary for profiling and replacement."""

    optimizer.step()


@dataclass
class _TrainingLoopState:
    """Bundle of training loop state variables passed to post-training helpers."""

    __slots__ = (
        "initial_loss",
        "final_loss",
        "min_loss",
        "total_tokens",
        "total_time_ms",
        "step_count",
        "step_time_sum_ms",
        "grad_norm_sum",
        "grad_norm_sq_sum",
        "grad_norm_max",
        "grad_norm_count",
        "training_curve",
        "collect_curve",
        "seq_len",
        "seed",
        "entropy_gate_trajectory",
        "routing_aux_loss_sum",
        "routing_aux_loss_count",
    )

    initial_loss: float | None
    final_loss: float | None
    min_loss: float
    total_tokens: int
    total_time_ms: float
    step_count: int
    step_time_sum_ms: float
    grad_norm_sum: float
    grad_norm_sq_sum: float
    grad_norm_max: float
    grad_norm_count: int
    training_curve: list
    collect_curve: bool
    seq_len: int
    seed: int
    entropy_gate_trajectory: list
    routing_aux_loss_sum: float
    routing_aux_loss_count: int

    def native_summary(self) -> dict[str, Any]:
        return _native_training_summary(self)


@dataclass
class _MicroTrainLoopProgress:
    """Mutable loop progress for micro-train bookkeeping.

    This keeps Python-side orchestration state compact while carving out a
    boundary that can later move to Rust/C++ without changing `_micro_train`.
    """

    initial_loss: float | None = None
    final_loss: float | None = None
    min_loss: float = float("inf")
    total_tokens: int = 0
    step_count: int = 0
    step_time_sum_ms: float = 0.0
    grad_norm_sum: float = 0.0
    grad_norm_sq_sum: float = 0.0
    grad_norm_max: float = 0.0
    grad_norm_count: int = 0
    training_curve: list[dict[str, float]] = field(default_factory=list)
    entropy_gate_trajectory: list[float] = field(default_factory=list)
    routing_aux_loss_sum: float = 0.0
    routing_aux_loss_count: int = 0
    loss_at_250: float | None = None
    loss_at_500: float | None = None

    def record_cuda_graph_step(self, *, token_count: int, step_time_ms: float) -> None:
        self.step_count += 1
        self.step_time_sum_ms += step_time_ms
        self.total_tokens += token_count

    def record_loss_snapshot(self, *, loss_val: float) -> None:
        if self.initial_loss is None:
            self.initial_loss = loss_val
        self.final_loss = loss_val
        self.min_loss = min(self.min_loss, loss_val)

    def record_routing_aux_loss(self, routing_aux_loss: float | None) -> None:
        if routing_aux_loss is None:
            return
        self.routing_aux_loss_sum += routing_aux_loss
        self.routing_aux_loss_count += 1

    def commit_eager_step(
        self,
        *,
        step: int,
        loss_val: float,
        grad_norm: float,
        step_time_ms: float,
        token_count: int,
        collect_curve: bool,
    ) -> None:
        self.record_loss_snapshot(loss_val=loss_val)
        self.total_tokens += token_count
        self.step_count += 1
        self.step_time_sum_ms += step_time_ms
        self.grad_norm_sum += grad_norm
        self.grad_norm_sq_sum += grad_norm * grad_norm
        self.grad_norm_max = max(self.grad_norm_max, grad_norm)
        self.grad_norm_count += 1
        if collect_curve:
            self.training_curve.append(
                {
                    "step": step,
                    "loss": loss_val,
                    "grad_norm": grad_norm,
                    "step_time_ms": step_time_ms,
                }
            )

    def to_loop_state(
        self,
        *,
        total_time_ms: float,
        collect_curve: bool,
        seq_len: int,
        seed: int,
    ) -> _TrainingLoopState:
        return _TrainingLoopState(
            initial_loss=self.initial_loss,
            final_loss=self.final_loss,
            min_loss=self.min_loss,
            total_tokens=self.total_tokens,
            total_time_ms=total_time_ms,
            step_count=self.step_count,
            step_time_sum_ms=self.step_time_sum_ms,
            grad_norm_sum=self.grad_norm_sum,
            grad_norm_sq_sum=self.grad_norm_sq_sum,
            grad_norm_max=self.grad_norm_max,
            grad_norm_count=self.grad_norm_count,
            training_curve=self.training_curve,
            collect_curve=collect_curve,
            seq_len=seq_len,
            seed=seed,
            entropy_gate_trajectory=self.entropy_gate_trajectory,
            routing_aux_loss_sum=self.routing_aux_loss_sum,
            routing_aux_loss_count=self.routing_aux_loss_count,
        )


def _maybe_extend_training_budget(
    progress: _MicroTrainLoopProgress,
    result: dict[str, Any],
    *,
    step: int,
    loss_val: float,
    total_steps: int,
) -> int:
    """Extend the micro-train budget when the step-500 trajectory is improving."""

    if step == 250:
        progress.loss_at_250 = loss_val
        return total_steps

    if step != 500:
        return total_steps

    progress.loss_at_500 = loss_val
    if progress.loss_at_250 is None:
        return total_steps

    improvement_rate = (progress.loss_at_250 - progress.loss_at_500) / max(
        progress.loss_at_250,
        1e-6,
    )
    if improvement_rate > 0 and total_steps < 1000:
        result["adaptive_budget_extension"] = True
        return 1000
    return total_steps


def _training_step_error(
    *,
    step: int,
    loss_val: float,
    grad_norm: float,
) -> dict[str, Any] | None:
    """Return terminal step failures in a shape `_micro_train` can merge."""

    if not math.isfinite(loss_val):
        return {"error": f"NaN/Inf loss at step {step}", "n_train_steps": step}

    if step == 0 and (not math.isfinite(grad_norm) or grad_norm <= 1e-10):
        return {
            "error": "zero_grad_precheck_failed",
            "n_train_steps": 0,
            "max_grad_norm": grad_norm,
            "mean_grad_norm": grad_norm,
            "grad_norm_std": 0.0,
        }

    return None


def _build_training_step_event(
    live_context: dict[str, Any] | None,
    *,
    step: int,
    total_steps: int,
    loss_val: float,
    grad_norm: float,
    routing_aux_loss: float | None,
) -> dict[str, Any] | None:
    """Shape dashboard step telemetry outside the main training loop."""

    if not live_context or step % 10 != 0:
        return None

    step_event: dict[str, Any] = {
        "experiment_id": live_context.get("exp_id", ""),
        "step": step,
        "loss": round(loss_val, 6),
        "total_steps": total_steps,
        "phase": live_context.get("phase", ""),
    }
    if routing_aux_loss is not None:
        step_event["routing_aux_loss"] = round(routing_aux_loss, 6)
    if grad_norm > 0:
        step_event["grad_norm"] = round(grad_norm, 4)
    return step_event
