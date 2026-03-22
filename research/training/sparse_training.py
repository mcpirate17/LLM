"""
Dynamic Sparse Training — RigL-Style Mask Updates

Implementation of RigL (Rigging the Lottery) sparse-from-scratch training.
The model maintains a fixed parameter budget but periodically updates
*which* parameters are active based on gradient magnitude: grow connections
where gradients are large, prune where magnitudes are small.
"""

from __future__ import annotations
import math
import torch
from typing import Dict


class RigLScheduler:
    """
    Manages the RigL sparse topology update schedule and mask enforcement.
    Operates on parameterized tensors passed to the optimizer.
    """

    def __init__(
        self,
        params,
        optimizer: torch.optim.Optimizer,
        dense_allocation: float = 0.2,
        T_end: int = 1000,
        delta: int = 100,
        alpha: float = 0.3,
        grad_accumulation_n: int = 1,
    ):
        self.optimizer = optimizer
        self.dense_allocation = dense_allocation
        self.T_end = T_end
        self.delta = delta
        self.alpha = alpha  # initial proportion of weights to update
        self.grad_accumulation_n = grad_accumulation_n

        self.step_count = 0
        self.masks: Dict[int, torch.Tensor] = {}

        # Identify params to sparsify (typically just 2D+ weights like Linear/Conv)
        self.target_modules = {}
        idx = 0

        # Unpack param groups if necessary
        if isinstance(params, dict):
            p_list = params.get("params", [])
        elif (
            isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict)
        ):
            p_list = [p for g in params for p in g["params"]]
        else:
            p_list = list(params)

        for param in p_list:
            # We only sparsify weights that are 2D or more (ignore bias and LayerNorm)
            # and ignore embeddings usually handled sparsely by default, but checking dimension is easiest heuristics
            if (
                isinstance(param, torch.Tensor)
                and param.requires_grad
                and len(param.shape) >= 2
            ):
                self.target_modules[idx] = param
                idx += 1

        if len(self.target_modules) > 0:
            self.init_masks()
            self.apply_masks()
            self._hook_handles = []
            self._register_hooks()

    def init_masks(self):
        """Randomly initialize sparsity masks according to the dense_allocation."""
        for name, param in self.target_modules.items():
            k = int(self.dense_allocation * param.numel())
            # Ensure at least 1 parameter is active
            k = max(1, k)
            # Random permutation to select active connections
            perm = torch.randperm(param.numel(), device=param.device)
            active_indices = perm[:k]
            mask = torch.zeros_like(param, dtype=torch.bool)
            mask.view(-1)[active_indices] = True
            self.masks[name] = mask

    def apply_masks(self):
        """Ensure the disabled weights are exactly zero."""
        with torch.no_grad():
            for name, param in self.target_modules.items():
                if name in self.masks:
                    param.data.mul_(self.masks[name])

    def _register_hooks(self):
        """Register backward hooks to zero out gradients for pruned weights."""
        for name, param in self.target_modules.items():
            if param.requires_grad:

                def get_hook(mask):
                    def hook(grad):
                        return grad * mask

                    return hook

                handle = param.register_hook(get_hook(self.masks[name]))
                self._hook_handles.append(handle)

    def _clear_hooks(self):
        for handle in getattr(self, "_hook_handles", []):
            handle.remove()
        self._hook_handles = []

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
            for name, param in self.target_modules.items():
                mask = self.masks[name]
                num_active = mask.sum().item()
                num_to_update = int(num_active * drop_fraction)

                if num_to_update == 0:
                    continue

                # 1. Prune
                # Mask out inactive weights
                w_mag = param.abs()
                w_mag[~mask] = -1.0  # inactive weights are ignored

                keep_k = int(num_active) - num_to_update
                if keep_k > 0:
                    _, keep_indices = torch.topk(w_mag.view(-1), keep_k)
                    new_mask = torch.zeros_like(mask).view(-1)
                    new_mask[keep_indices] = True
                else:
                    new_mask = torch.zeros_like(mask).view(-1)

                # 2. Grow
                grad_mag = (
                    param.grad.abs()
                    if param.grad is not None
                    else torch.zeros_like(param)
                )
                # Don't grow where we already kept weights
                grad_mag.view(-1)[new_mask] = -1.0

                if num_to_update > 0:
                    _, grow_indices = torch.topk(grad_mag.view(-1), num_to_update)
                    new_mask[grow_indices] = True

                new_mask = new_mask.view(mask.shape)
                self.masks[name] = new_mask

                # Apply new mask
                param.data.mul_(new_mask)

                # Zero out optimizer momentum for the newly grown weights
                state = self.optimizer.state[param]
                if "exp_avg" in state:
                    state["exp_avg"][~new_mask] = 0.0
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"][~new_mask] = 0.0

        # Replace hooks with new masks
        self._clear_hooks()
        self._register_hooks()

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
        # We must support normal optimizer initialization so extract param groups safely
        if isinstance(params, torch.Tensor):
            params = [params]
        params_list = list(params)

        self.base_optimizer = base_optimizer_cls(params_list, **kwargs)
        # Expose defaults from base optimizer so wrap works properly
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
        if hasattr(self, "scheduler"):
            self.scheduler.step()
        return loss

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
