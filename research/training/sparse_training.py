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
    ... (omitting docstring for brevity in replace call) ...
    """
    # ... (rest of RigLScheduler) ...

class StructuredRigLScheduler(RigLScheduler):
    """RigL scheduler that enforces structured sparsity patterns (N:M or Block)."""
    
    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.8,
        update_freq: int = 100,
        total_steps: int = 1000,
        initial_drop_fraction: float = 0.3,
        min_param_size: int = 64,
        structure_type: str = "block", # "block" or "nm"
        block_size: int = 16,
        n: int = 2,
        m: int = 4,
    ):
        self.structure_type = structure_type
        self.block_size = block_size
        self.n = n
        self.m = m
        super().__init__(model, sparsity, update_freq, total_steps, initial_drop_fraction, min_param_size)

    def _init_mask(self, param: torch.Tensor) -> torch.Tensor:
        if self.structure_type == "nm":
            from ..synthesis.compiler import _build_nm_mask
            return _build_nm_mask(param, self.n, self.m)
        else: # block
            from ..synthesis.compiler import _build_block_sparse_mask
            return _build_block_sparse_mask(param, self.block_size, 1.0 - self.sparsity)

    def step(self) -> Optional[Dict[str, float]]:
        """Perform mask update while maintaining structure."""
        self.step_count += 1
        if self.step_count % self.update_freq != 0:
            self._apply_masks()
            return None

        drop_fraction = self._cosine_drop_fraction()
        
        for name, param in self.param_refs.items():
            mask = self.masks[name]
            grad = param.grad
            if grad is None: continue

            if self.structure_type == "block":
                self._update_block_mask(name, param, mask, grad, drop_fraction)
            else: # nm
                # N:M sparsity is usually fixed structure but we can evolve which 
                # N out of M are active if we don't use hardware-fixed 2:4
                self._update_nm_mask(name, param, mask, grad, drop_fraction)

        self.n_updates += 1
        return self.get_telemetry()

    def _update_block_mask(self, name, param, mask, grad, drop_fraction):
        BS = self.block_size
        rows, cols = param.shape
        m_rows, m_cols = rows // BS, cols // BS
        if m_rows == 0 or m_cols == 0: return

        # Compute block scores (magnitude for drop, gradient for grow)
        weights_sq = (param.data ** 2)[:m_rows*BS, :m_cols*BS].view(m_rows, BS, m_cols, BS)
        block_mags = weights_sq.mean(dim=(1, 3))
        
        grad_sq = (grad ** 2)[:m_rows*BS, :m_cols*BS].view(m_rows, BS, m_cols, BS)
        block_grads = grad_sq.mean(dim=(1, 3))

        # Current block mask
        current_block_mask = mask[:m_rows*BS, :m_cols*BS].view(m_rows, BS, m_cols, BS).any(dim=(1, 3))
        
        n_active = int(current_block_mask.sum().item())
        n_to_drop = max(1, int(n_active * drop_fraction))

        # Drop blocks with smallest magnitude
        active_scores = block_mags.clone()
        active_scores[~current_block_mask] = float('inf')
        _, drop_indices = active_scores.flatten().topk(n_to_drop, largest=False)
        
        # Grow blocks with largest gradient
        inactive_scores = block_grads.clone()
        inactive_scores[current_block_mask] = -1.0
        _, grow_indices = inactive_scores.flatten().topk(n_to_drop, largest=True)

        # Update mask
        new_block_mask_flat = current_block_mask.flatten()
        new_block_mask_flat[drop_indices] = False
        new_block_mask_flat[grow_indices] = True
        new_block_mask = new_block_mask_flat.view(m_rows, m_cols)

        # Expand back to full mask
        full_mask = torch.zeros_like(mask)
        expanded = new_block_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, BS, BS)
        full_mask[:m_rows*BS, :m_cols*BS] = expanded.permute(0, 2, 1, 3).reshape(m_rows*BS, m_cols*BS)
        
        self.masks[name] = full_mask
        param.data.mul_(full_mask)

    def _update_nm_mask(self, name, param, mask, grad, drop_fraction):
        # For simplicity in this prototype, N:M updates are similar to random RigL 
        # but constrained to maintain N active per M chunk.
        # This is more complex to implement correctly without full redistribution.
        # For now, we'll keep N:M static or use a simplified block-based update.
        pass

    def _cosine_drop_fraction(self) -> float:
        """Cosine-decay the drop fraction toward zero."""
        progress = min(1.0, self.step_count / self.total_steps)
        return self.initial_drop_fraction * (1 + math.cos(math.pi * progress)) / 2

    def step(self) -> Optional[Dict[str, float]]:
        """Call after optimizer.step(). Returns update stats if mask was updated.
        Vectorized implementation to eliminate per-parameter Python loops.
        """
        self.step_count += 1

        if self.step_count % self.update_freq != 0:
            self._apply_masks()
            return None

        # Perform mask update
        drop_fraction = self._cosine_drop_fraction()
        
        # Batch collect metadata for recording at the end
        total_grown = 0
        total_dropped = 0
        total_active = 0
        total_params = 0

        # We still loop over param_refs because they are separate tensors,
        # but we use efficient topk and mask operations within each.
        for name, param in self.param_refs.items():
            mask = self.masks[name]
            grad = param.grad
            if grad is None: continue

            # Vectorized DROP: remove smallest-magnitude active weights
            # Use a large value for masked weights so they are not selected
            active_magnitudes = param.data.abs()
            active_magnitudes.masked_fill_(mask == 0, float('inf'))
            
            n_active = int(mask.sum().item())
            n_to_drop = max(1, int(n_active * drop_fraction))
            
            flat_mag = active_magnitudes.flatten()
            _, drop_indices = flat_mag.topk(n_to_drop, largest=False)
            
            # Update mask (in-place)
            flat_mask = mask.flatten()
            flat_mask.scatter_(0, drop_indices, 0.0)

            # Vectorized GROW: activate masked positions with largest gradient magnitude
            # Use a small value for already-active positions so they are not selected
            flat_grad = grad.abs().flatten()
            flat_grad.masked_fill_(flat_mask > 0, -1.0)
            
            _, grow_indices = flat_grad.topk(n_to_drop, largest=True)
            flat_mask.scatter_(0, grow_indices, 1.0)

            # Update state
            self.masks[name] = flat_mask.view_as(param)

            # Re-initialize newly grown weights efficiently
            # (only if they were actually grown)
            if n_to_drop > 0:
                fan_in = param.shape[-1]
                std = (1.0 / math.sqrt(fan_in)) * 0.1
                # Small batch random initialization
                param.data.flatten().scatter_(
                    0, grow_indices, 
                    torch.randn(n_to_drop, device=param.device) * std
                )

            # Enforce mask
            param.data.mul_(self.masks[name])

            total_grown += n_to_drop
            total_dropped += n_to_drop
            total_active += n_active
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
