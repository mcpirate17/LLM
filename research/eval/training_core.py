"""Shared optimization loops for eval-time micro-training."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Iterable, Optional, Sequence

import torch

from ._runner_native import load_runner_native
from .utils import make_adamw


@dataclass(slots=True)
class TrainLoopResult:
    final_loss: float
    steps_completed: int
    diverged: bool


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
            native = load_runner_native()
            del native
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

    final_loss = float("inf")
    steps_completed = 0
    diverged = False

    for step in range(n_steps):
        if warmup_steps > 0 and step < warmup_steps:
            warmup_factor = (step + 1) / warmup_steps
            for group in optimizer.param_groups:
                group["lr"] = lr * warmup_factor

        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(step)
        if not torch.isfinite(loss):
            diverged = True
            break

        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(param_values, clip_grad)
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
    )
