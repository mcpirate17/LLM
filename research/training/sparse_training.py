"""
Dynamic Sparse Training — RigL-Style Mask Updates

Implements sparse-from-scratch training where the model maintains a
fixed parameter budget but periodically updates *which* parameters are
active based on gradient magnitude. This is the RigL (Rigging the Lottery)
approach: grow connections where gradients are large, prune connections
where magnitudes are small.

Key idea: total non-zero parameters stays constant throughout training,
but the sparsity pattern evolves to find a better sparse topology.

Usage:
    scheduler = RigLScheduler(model, sparsity=0.8, update_freq=100)
    for step in range(n_steps):
        loss = model(x)
        loss.backward()
        optimizer.step()
        scheduler.step()  # periodically updates masks
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class RigLScheduler:
    """RigL-style dynamic sparse training scheduler.

    Maintains binary masks on weight matrices and periodically updates them:
    1. DROP: Remove fraction of smallest-magnitude active weights
    2. GROW: Activate same number of currently-masked weights with largest gradients

    The drop fraction follows a cosine decay schedule so updates become
    more conservative as training progresses.

    Args:
        model: The nn.Module to sparsify
        sparsity: Target sparsity ratio (0.0 = dense, 1.0 = all zeros)
        update_freq: Steps between mask updates
        total_steps: Total training steps (for cosine decay of drop fraction)
        initial_drop_fraction: Fraction of active weights to drop per update
        min_param_size: Minimum parameter numel to sparsify (skip small params)
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.8,
        update_freq: int = 100,
        total_steps: int = 1000,
        initial_drop_fraction: float = 0.3,
        min_param_size: int = 64,
    ):
        self.sparsity = max(0.0, min(0.95, sparsity))
        self.update_freq = max(1, update_freq)
        self.total_steps = max(1, total_steps)
        self.initial_drop_fraction = max(0.01, min(0.5, initial_drop_fraction))
        self.min_param_size = min_param_size
        self.step_count = 0
        self.n_updates = 0

        # Initialize masks for eligible parameters
        self.masks: Dict[str, torch.Tensor] = {}
        self.param_refs: Dict[str, nn.Parameter] = {}
        for name, param in model.named_parameters():
            if param.dim() >= 2 and param.numel() >= min_param_size:
                mask = self._init_mask(param)
                self.masks[name] = mask
                self.param_refs[name] = param
                # Apply initial mask
                param.data.mul_(mask)

    def _init_mask(self, param: torch.Tensor) -> torch.Tensor:
        """Initialize a random sparsity mask at the target sparsity level."""
        mask = torch.ones_like(param)
        n_total = param.numel()
        n_zeros = int(n_total * self.sparsity)
        if n_zeros <= 0 or n_zeros >= n_total:
            return mask
        # Random initial topology
        flat = mask.flatten()
        perm = torch.randperm(n_total, device=param.device)
        flat[perm[:n_zeros]] = 0.0
        return flat.view_as(param)

    def _cosine_drop_fraction(self) -> float:
        """Cosine-decay the drop fraction toward zero."""
        progress = min(1.0, self.step_count / self.total_steps)
        return self.initial_drop_fraction * (1 + math.cos(math.pi * progress)) / 2

    def step(self) -> Optional[Dict[str, float]]:
        """Call after optimizer.step(). Returns update stats if mask was updated."""
        self.step_count += 1

        if self.step_count % self.update_freq != 0:
            # Just enforce masks (zero out pruned weights that optimizer may have updated)
            self._apply_masks()
            return None

        # Perform mask update
        drop_fraction = self._cosine_drop_fraction()
        total_grown = 0
        total_dropped = 0
        total_active = 0
        total_params = 0

        for name, param in self.param_refs.items():
            mask = self.masks[name]
            grad = param.grad

            if grad is None:
                continue

            active = mask.bool()
            n_active = int(active.sum().item())
            n_to_drop = max(1, int(n_active * drop_fraction))

            # DROP: remove smallest-magnitude active weights
            active_magnitudes = param.data.abs() * mask
            # Set inactive positions to inf so they're never selected for drop
            active_magnitudes[~active] = float('inf')
            flat_mag = active_magnitudes.flatten()
            _, drop_indices = flat_mag.topk(n_to_drop, largest=False)
            flat_mask = mask.flatten()
            flat_mask[drop_indices] = 0.0

            # GROW: activate masked positions with largest gradient magnitude
            inactive = ~flat_mask.bool()
            flat_grad = grad.abs().flatten()
            # Only consider currently-inactive positions
            grow_scores = flat_grad.clone()
            grow_scores[~inactive] = -1.0  # ignore active positions
            _, grow_indices = grow_scores.topk(n_to_drop, largest=True)
            flat_mask[grow_indices] = 1.0

            self.masks[name] = flat_mask.view_as(param)

            # Re-initialize newly grown weights (small random values)
            grown_mask = torch.zeros_like(param).flatten()
            grown_mask[grow_indices] = 1.0
            grown_mask = grown_mask.view_as(param)
            param.data[grown_mask.bool()] = (
                torch.randn_like(param.data[grown_mask.bool()])
                * (1.0 / math.sqrt(param.shape[-1]))
                * 0.1
            )

            # Apply updated mask
            param.data.mul_(self.masks[name])

            total_grown += n_to_drop
            total_dropped += n_to_drop
            total_active += int(self.masks[name].sum().item())
            total_params += param.numel()

        self.n_updates += 1

        return {
            "update_step": self.step_count,
            "n_updates": self.n_updates,
            "drop_fraction": round(drop_fraction, 4),
            "total_grown": total_grown,
            "total_dropped": total_dropped,
            "active_params": total_active,
            "total_params": total_params,
            "effective_density": round(total_active / max(total_params, 1), 4),
        }

    def _apply_masks(self) -> None:
        """Zero out pruned weights (gradient updates may have revived them)."""
        for name, param in self.param_refs.items():
            param.data.mul_(self.masks[name])

    def get_density(self) -> float:
        """Current effective density across all masked parameters."""
        total_active = sum(int(m.sum().item()) for m in self.masks.values())
        total_params = sum(m.numel() for m in self.masks.values())
        return total_active / max(total_params, 1)

    def get_telemetry(self) -> Dict:
        """Return telemetry summary for logging."""
        return {
            "sparsity_target": self.sparsity,
            "effective_density": round(self.get_density(), 4),
            "n_mask_updates": self.n_updates,
            "step_count": self.step_count,
            "n_masked_params": len(self.masks),
        }


class RigLOptimizer(torch.optim.Optimizer):
    """AdamW optimizer with integrated RigL dynamic sparse mask updates.

    Wraps standard AdamW with a RigL scheduler that periodically updates
    the sparsity topology. The model trains sparse-from-scratch with a
    fixed parameter budget.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        sparsity: float = 0.8,
        update_freq: int = 100,
        total_steps: int = 1000,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._sparsity = sparsity
        self._update_freq = update_freq
        self._total_steps = total_steps
        self._rigl: Optional[RigLScheduler] = None
        self._inner_step = 0

    def _ensure_rigl(self) -> None:
        """Lazily initialize RigL scheduler on first step."""
        if self._rigl is not None:
            return
        # Build a temporary module to collect all parameters
        all_params = []
        for group in self.param_groups:
            all_params.extend(group["params"])
        # Create a wrapper module holding refs
        wrapper = nn.Module()
        for i, p in enumerate(all_params):
            wrapper.register_parameter(f"p{i}", p)
        self._rigl = RigLScheduler(
            wrapper,
            sparsity=self._sparsity,
            update_freq=self._update_freq,
            total_steps=self._total_steps,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Standard AdamW update
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                state["step"] += 1
                m, v = state["m"], state["v"]
                beta1, beta2 = 0.9, 0.999
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step = state["step"]
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.addcdiv_(m_hat, v_hat.sqrt() + 1e-8, value=-lr)

        # RigL mask update
        self._ensure_rigl()
        self._rigl.step()
        return loss

    def get_rigl_telemetry(self) -> Dict:
        """Return RigL scheduler telemetry if available."""
        if self._rigl is not None:
            return self._rigl.get_telemetry()
        return {}
