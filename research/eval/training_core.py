"""Shared optimization loops for eval-time micro-training."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Iterable, Optional, Sequence

import torch

from ._runner_native import load_runner_native
from .utils import clip_grad_norm, make_adamw
from ..scientist.shared_utils import coerce_finite_float as _safe_float


@dataclass(slots=True)
class TrainLoopResult:
    final_loss: float
    steps_completed: int
    diverged: bool
    telemetry: Optional[dict[str, Any]] = None


_EMPTY_GRAD_STATS: dict[str, Any] = {
    "total_norm": 0.0,
    "layer_norms": {},
    "max_layer": None,
    "max_layer_norm": 0.0,
    "has_nonfinite": False,
    "num_grads": 0,
}


def _grad_stats(
    parameters: Sequence[torch.Tensor],
    parameter_names: Optional[Sequence[str]],
) -> dict[str, Any]:
    grads: list[torch.Tensor] = []
    names: list[str] = []
    for idx, param in enumerate(parameters):
        if param.grad is None:
            continue
        grads.append(param.grad)
        names.append(
            parameter_names[idx] if parameter_names is not None else f"param_{idx}"
        )
    if not grads:
        return dict(_EMPTY_GRAD_STATS)
    return dict(load_runner_native().grad_stats_fused(grads, names))


def _append_step_telemetry(
    train_telemetry: dict[str, Any],
    *,
    step: int,
    loss: float | None,
    lr_expected: list[float],
    lr_actual_before_step: list[float],
    lr_actual_after_scheduler: list[float],
    pre_clip: dict[str, Any],
    post_clip: dict[str, Any],
    clipped: bool,
) -> None:
    steps = train_telemetry.setdefault("steps", [])
    steps.append(
        {
            "step": int(step),
            "loss": loss,
            "lr_expected": lr_expected,
            "lr_actual_before_step": lr_actual_before_step,
            "lr_actual_after_scheduler": lr_actual_after_scheduler,
            "pre_clip_total_grad_norm": pre_clip["total_norm"],
            "post_clip_total_grad_norm": post_clip["total_norm"],
            "pre_clip_layer_norms": pre_clip["layer_norms"],
            "post_clip_layer_norms": post_clip["layer_norms"],
            "pre_clip_max_layer": pre_clip["max_layer"],
            "post_clip_max_layer": post_clip["max_layer"],
            "pre_clip_max_layer_norm": pre_clip["max_layer_norm"],
            "post_clip_max_layer_norm": post_clip["max_layer_norm"],
            "clipped": bool(clipped),
            "has_nonfinite_grad": bool(
                pre_clip["has_nonfinite"] or post_clip["has_nonfinite"]
            ),
        }
    )


def _finalize_telemetry(
    train_telemetry: dict[str, Any],
    *,
    diverged: bool,
    steps_completed: int,
) -> None:
    steps = train_telemetry.get("steps", [])
    if not steps:
        train_telemetry["summary"] = {
            "steps_completed": int(steps_completed),
            "diverged": bool(diverged),
            "max_pre_clip_grad_norm": None,
            "max_post_clip_grad_norm": None,
            "max_lr_delta": 0.0,
            "nonfinite_grad_steps": 0,
        }
        return
    max_pre = max(step["pre_clip_total_grad_norm"] for step in steps)
    max_post = max(step["post_clip_total_grad_norm"] for step in steps)
    max_lr_delta = 0.0
    nonfinite_grad_steps = 0
    for step in steps:
        deltas = [
            abs(expected - actual)
            for expected, actual in zip(
                step["lr_expected"], step["lr_actual_before_step"], strict=False
            )
        ]
        if deltas:
            max_lr_delta = max(max_lr_delta, max(deltas))
        if step["has_nonfinite_grad"]:
            nonfinite_grad_steps += 1
    train_telemetry["summary"] = {
        "steps_completed": int(steps_completed),
        "diverged": bool(diverged),
        "max_pre_clip_grad_norm": max_pre,
        "max_post_clip_grad_norm": max_post,
        "max_lr_delta": max_lr_delta,
        "nonfinite_grad_steps": nonfinite_grad_steps,
    }


class _NativeOptimizerBase:
    def __init__(self, parameters: Sequence[torch.Tensor], *, lr: float):
        self.params = list(parameters)
        self.param_groups = [{"params": self.params, "lr": lr}]

    def zero_grad(self, set_to_none: bool = True):
        for param in self.params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()


class _NativeSGDOptimizer(_NativeOptimizerBase):
    def __init__(
        self,
        parameters: Sequence[torch.Tensor],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
    ):
        super().__init__(parameters, lr=lr)
        self._native = load_runner_native()
        self._momentum = float(momentum)
        self._weight_decay = float(weight_decay)
        self._state = {
            id(param): torch.zeros_like(param)
            for param in self.params
            if momentum != 0.0
        }

    @torch.no_grad()
    def step(self):
        lr = float(self.param_groups[0]["lr"])
        for param in self.params:
            grad = param.grad
            if grad is None:
                continue
            buf = self._state.get(id(param))
            if buf is None:
                buf = torch.zeros_like(param)
            self._native.sgd_step_inplace(
                param,
                grad,
                buf,
                lr,
                self._momentum,
                self._weight_decay,
                bool(self._momentum != 0.0),
            )
            if self._momentum != 0.0:
                self._state[id(param)] = buf


class _NativeAdamWOptimizer(_NativeOptimizerBase):
    def __init__(
        self,
        parameters: Sequence[torch.Tensor],
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        eps: float = 1e-8,
    ):
        super().__init__(parameters, lr=lr)
        self._native = load_runner_native()
        self._beta1 = float(betas[0])
        self._beta2 = float(betas[1])
        self._weight_decay = float(weight_decay)
        self._eps = float(eps)
        self._step = 0
        self._state = {
            id(param): (torch.zeros_like(param), torch.zeros_like(param))
            for param in self.params
        }

    @torch.no_grad()
    def step(self):
        self._step += 1
        lr = float(self.param_groups[0]["lr"])
        for param in self.params:
            grad = param.grad
            if grad is None:
                continue
            exp_avg, exp_avg_sq = self._state[id(param)]
            self._native.adamw_step_inplace(
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                lr,
                self._beta1,
                self._beta2,
                self._eps,
                self._weight_decay,
                int(self._step),
            )


def make_optimizer(
    parameters: Sequence[torch.Tensor],
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float = 0.01,
    momentum: float = 0.0,
    betas: Optional[tuple[float, float]] = None,
):
    opt = (optimizer_name or "adamw").lower()
    enable_paramwise_native = os.getenv(
        "ARIA_ENABLE_EVAL_PARAMWISE_NATIVE_OPTIMIZER", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if enable_paramwise_native:
        try:
            # Probe: fail fast if the native extension can't be built/loaded.
            load_runner_native()
            if opt == "sgd":
                return _NativeSGDOptimizer(
                    parameters,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )
            adamw_betas = betas if betas is not None else (0.9, 0.999)
            return _NativeAdamWOptimizer(
                parameters,
                lr=lr,
                betas=adamw_betas,
                weight_decay=weight_decay,
            )
        except Exception:
            pass
    if opt == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=bool(momentum != 0.0),
        )
    adamw_betas = betas if betas is not None else (0.9, 0.999)
    return make_adamw(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
        betas=adamw_betas,
    )


def run_training_loop(
    parameters: Iterable[torch.Tensor],
    compute_loss: Callable[[int], torch.Tensor],
    *,
    n_steps: int,
    optimizer=None,
    optimizer_name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    momentum: float = 0.0,
    betas: Optional[tuple[float, float]] = None,
    clip_grad: float = 1.0,
    warmup_steps: int = 0,
    loss_trajectory: Optional[dict] = None,
    scheduler_step: Optional[Callable[[], None]] = None,
    train_telemetry: Optional[dict[str, Any]] = None,
    parameter_names: Optional[Sequence[str]] = None,
) -> TrainLoopResult:
    param_values = list(parameters)
    if optimizer is None:
        optimizer = make_optimizer(
            param_values,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            betas=betas,
        )
    if parameter_names is not None and len(parameter_names) != len(param_values):
        raise ValueError("parameter_names length must match parameters")

    base_group_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if train_telemetry is not None:
        train_telemetry.clear()
        train_telemetry["base_group_lrs"] = list(base_group_lrs)
        train_telemetry["optimizer_name"] = (optimizer_name or "adamw").lower()
        train_telemetry["warmup_steps"] = int(warmup_steps)
        train_telemetry["clip_grad"] = float(clip_grad)

    if train_telemetry is None and parameter_names is None:
        final_loss = float("inf")
        steps_completed = 0
        diverged = False
        for step in range(n_steps):
            if warmup_steps > 0 and step < warmup_steps:
                warmup_factor = (step + 1) / warmup_steps
                for group_idx, group in enumerate(optimizer.param_groups):
                    group["lr"] = base_group_lrs[group_idx] * warmup_factor

            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(step)
            if not torch.isfinite(loss):
                diverged = True
                break

            loss.backward()
            if clip_grad > 0:
                clip_grad_norm(param_values, clip_grad)
            optimizer.step()
            if scheduler_step is not None:
                scheduler_step()

            final_loss = float(loss.item())
            steps_completed = step + 1
            if loss_trajectory is not None:
                loss_trajectory[steps_completed] = final_loss

        return TrainLoopResult(
            final_loss=final_loss,
            steps_completed=steps_completed,
            diverged=diverged,
            telemetry=None,
        )

    final_loss = float("inf")
    steps_completed = 0
    diverged = False

    for step in range(n_steps):
        current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if warmup_steps > 0 and step < warmup_steps:
            warmup_factor = (step + 1) / warmup_steps
            expected_lrs = []
            for group_idx, group in enumerate(optimizer.param_groups):
                target_lr = base_group_lrs[group_idx] * warmup_factor
                group["lr"] = target_lr
                expected_lrs.append(float(target_lr))
        else:
            expected_lrs = current_lrs

        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(step)
        if not torch.isfinite(loss):
            diverged = True
            break

        loss.backward()
        pre_clip = _grad_stats(param_values, parameter_names)
        if pre_clip["has_nonfinite"]:
            diverged = True
            if train_telemetry is not None:
                _append_step_telemetry(
                    train_telemetry,
                    step=step + 1,
                    loss=_safe_float(loss.item()),
                    lr_expected=list(expected_lrs),
                    lr_actual_before_step=[
                        float(group["lr"]) for group in optimizer.param_groups
                    ],
                    lr_actual_after_scheduler=[
                        float(group["lr"]) for group in optimizer.param_groups
                    ],
                    pre_clip=pre_clip,
                    post_clip=pre_clip,
                    clipped=False,
                )
            break

        clipped = False
        if clip_grad > 0:
            clip_grad_norm(param_values, clip_grad)
            clipped = pre_clip["total_norm"] > float(clip_grad)
        post_clip = _grad_stats(param_values, parameter_names)
        actual_lrs_before_step = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        if post_clip["has_nonfinite"]:
            diverged = True
            if train_telemetry is not None:
                _append_step_telemetry(
                    train_telemetry,
                    step=step + 1,
                    loss=_safe_float(loss.item()),
                    lr_expected=list(expected_lrs),
                    lr_actual_before_step=actual_lrs_before_step,
                    lr_actual_after_scheduler=actual_lrs_before_step,
                    pre_clip=pre_clip,
                    post_clip=post_clip,
                    clipped=clipped,
                )
            break
        optimizer.step()
        if scheduler_step is not None:
            scheduler_step()
        actual_lrs_after_scheduler = [
            float(group["lr"]) for group in optimizer.param_groups
        ]

        final_loss = float(loss.item())
        steps_completed = step + 1
        if loss_trajectory is not None:
            loss_trajectory[steps_completed] = final_loss
        if train_telemetry is not None:
            _append_step_telemetry(
                train_telemetry,
                step=steps_completed,
                loss=final_loss,
                lr_expected=list(expected_lrs),
                lr_actual_before_step=actual_lrs_before_step,
                lr_actual_after_scheduler=actual_lrs_after_scheduler,
                pre_clip=pre_clip,
                post_clip=post_clip,
                clipped=clipped,
            )

    if train_telemetry is not None:
        _finalize_telemetry(
            train_telemetry,
            diverged=diverged,
            steps_completed=steps_completed,
        )
    return TrainLoopResult(
        final_loss=final_loss,
        steps_completed=steps_completed,
        diverged=diverged,
        telemetry=train_telemetry,
    )
