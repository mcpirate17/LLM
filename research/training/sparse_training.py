"""
Dynamic Sparse Training — RigL-Style Mask Updates

Implementation of RigL (Rigging the Lottery) sparse-from-scratch training.
The model maintains a fixed parameter budget but periodically updates
*which* parameters are active based on gradient magnitude: grow connections
where gradients are large, prune where magnitudes are small.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import torch

from ._loss_native import load_loss_native


@dataclass(slots=True)
class _SparseParamState:
    param: torch.Tensor
    mask: torch.Tensor


class RigLScheduler:
    """
    Manages the RigL sparse topology update schedule and mask enforcement.
    Operates on parameterized tensors passed to the optimizer.
    """

    __slots__ = (
        "optimizer",
        "dense_allocation",
        "T_end",
        "delta",
        "alpha",
        "step_count",
        "_sparse_params",
        "_hook_handles",
        "_native",
    )

    def __init__(
        self,
        params,
        optimizer: torch.optim.Optimizer,
        dense_allocation: float = 0.2,
        T_end: int = 1000,
        delta: int = 100,
        alpha: float = 0.3,
    ):
        self.optimizer = optimizer
        self.dense_allocation = dense_allocation
        self.T_end = T_end
        self.delta = delta
        self.alpha = alpha  # initial proportion of weights to update

        self.step_count = 0
        self._sparse_params: list[_SparseParamState] = []
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._native = load_loss_native()

        for param in _flatten_params(params):
            if (
                isinstance(param, torch.Tensor)
                and param.requires_grad
                and param.dim() >= 2
            ):
                self._sparse_params.append(
                    _SparseParamState(
                        param=param,
                        mask=torch.zeros_like(param, dtype=torch.bool),
                    )
                )

        if self._sparse_params:
            self.init_masks()
            self.apply_masks()
            self._register_hooks()

    def init_masks(self):
        """Randomly initialize sparsity masks according to the dense_allocation."""
        for state in self._sparse_params:
            param = state.param
            k = max(1, int(self.dense_allocation * param.numel()))
            perm = torch.randperm(param.numel(), device=param.device)
            mask = torch.zeros_like(param, dtype=torch.bool)
            mask.view(-1).index_fill_(0, perm[:k], True)
            state.mask = mask

    def apply_masks(self):
        """Ensure the disabled weights are exactly zero."""
        with torch.no_grad():
            for state in self._sparse_params:
                state.param.data.mul_(state.mask)

    def _register_hooks(self):
        """Register backward hooks to zero out gradients for pruned weights."""
        for state in self._sparse_params:
            if not state.param.requires_grad:
                continue
            # Bind the state via default arg so each hook captures its own
            # mask reference without an extra closure layer.
            self._hook_handles.append(
                state.param.register_hook(lambda grad, s=state: grad * s.mask)
            )

    def cosine_annealing(self) -> float:
        """Compute the fraction of active weights to drop/grow at the current step."""
        if self.step_count >= self.T_end:
            return 0.0
        return self.alpha / 2 * (1 + math.cos(math.pi * self.step_count / self.T_end))

    def update_topology(self):
        """The core RigL update: prune lowest magnitude, grow highest gradient."""
        drop_fraction = self.cosine_annealing()
        if drop_fraction <= 0.0:
            return

        with torch.no_grad():
            for state in self._sparse_params:
                param = state.param
                old_mask = state.mask
                num_active = int(old_mask.sum().item())
                num_to_update = int(num_active * drop_fraction)
                if num_to_update == 0:
                    continue

                grad = param.grad if param.grad is not None else torch.zeros_like(param)
                new_mask = self._native.rigl_compute_new_mask(
                    param, grad, old_mask, int(num_to_update)
                )
                state.mask = new_mask

                param.data.mul_(new_mask)

                # Reset optimizer momentum for newly grown weights only.
                # Pruned-weight momentum is harmless (their grad is zeroed by
                # the backward hook) and writing to it wastes memory traffic.
                grown = new_mask & ~old_mask
                opt_state = self.optimizer.state[param]
                if "exp_avg" in opt_state:
                    opt_state["exp_avg"][grown] = 0.0
                if "exp_avg_sq" in opt_state:
                    opt_state["exp_avg_sq"][grown] = 0.0

    def step(self):
        """Called every training step. Periodically updates topology."""
        self.step_count += 1
        if self.step_count % self.delta == 0 and self.step_count < self.T_end:
            self.update_topology()


class RigLOptimizer(torch.optim.Optimizer):
    """
    Wrapper optimizer for RigL algorithm. Use as a transparent replacement for AdamW.
    """

    def __init__(
        self,
        params,
        base_optimizer_cls=torch.optim.AdamW,
        dense_allocation=0.2,
        T_end=1000,
        delta=100,
        **kwargs,
    ):
        if isinstance(params, torch.Tensor):
            params = [params]
        params_list = list(params)

        self.base_optimizer = base_optimizer_cls(params_list, **kwargs)
        self.defaults = self.base_optimizer.defaults
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

        self.scheduler = RigLScheduler(
            params=params_list,
            optimizer=self.base_optimizer,
            dense_allocation=dense_allocation,
            T_end=T_end,
            delta=delta,
        )

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.scheduler.step()
        return loss

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


def _flatten_params(params) -> list[torch.Tensor]:
    if isinstance(params, dict):
        return list(params.get("params", []))
    if isinstance(params, list) and params and isinstance(params[0], dict):
        return [param for group in params for param in group["params"]]
    return list(params)
