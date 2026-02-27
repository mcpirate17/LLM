"""Scaling ceiling predictor via NTK condition number analysis.

Evaluates whether an architecture's training dynamics will remain stable
as width/depth increase, using the empirical Neural Tangent Kernel and
muP transferability checks.

Reference: ARIA_NEXT_GEN_ARCHITECTURE.md §2
"""

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ScalingPredictor:
    """Predicts scaling ceilings from micro-training diagnostics.

    Uses empirical NTK condition number growth between two model sizes
    to estimate whether an architecture will scale gracefully.

    Reference: ARIA_NEXT_GEN_ARCHITECTURE.md §2
    """

    def __init__(self, model_fn: Callable, dataloader):
        """
        Args:
            model_fn: Callable(dim, depth) -> nn.Module
            dataloader: Iterable yielding (input, target) batches
        """
        self.model_fn = model_fn
        self.dataloader = dataloader

    def compute_ntk_condition_number(
        self, model: nn.Module, x: torch.Tensor, max_outputs: int = 32
    ) -> float:
        """Compute condition number of the empirical NTK.

        The NTK K = J @ J^T where J is the Jacobian dy/d(theta).
        A well-conditioned NTK implies stable, scale-invariant learning.

        Args:
            model: The model to analyse.
            x: Input tensor (first dim is batch).
            max_outputs: Cap on output dimensions to keep computation tractable.

        Returns:
            Condition number (ratio of largest to smallest eigenvalue).
        """
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return float('inf')

        y = model(x)
        # Flatten output to 1-D per sample
        y_flat = y.reshape(y.shape[0], -1)
        n_out = min(y_flat.shape[1], max_outputs)
        y_flat = y_flat[:, :n_out]

        # Build Jacobian row by row
        jacobian_rows = []
        for i in range(y_flat.shape[0]):
            for j in range(n_out):
                model.zero_grad()
                y_flat[i, j].backward(retain_graph=True)
                grads = torch.cat(
                    [p.grad.flatten() for p in params if p.grad is not None]
                )
                jacobian_rows.append(grads.detach())
                # Reset grads for next backward
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()

        J = torch.stack(jacobian_rows)  # (n_samples * n_out, n_params)

        # NTK = J @ J^T
        ntk = J @ J.T
        eigenvalues = torch.linalg.eigvalsh(ntk)
        pos = eigenvalues[eigenvalues > 0]
        if len(pos) < 2:
            return float('inf')
        cond = (pos[-1] / pos[0]).item()
        return cond

    def check_mup_transferability(
        self, dim_small: int = 64, dim_large: int = 256, lr: float = 1e-3, steps: int = 20
    ) -> float:
        """Check if optimal learning rate transfers across widths (muP test).

        Trains at two scales and compares loss trajectories. If they diverge
        significantly, the architecture doesn't obey muP scaling.

        Returns:
            Ratio of loss trajectories (closer to 1.0 = better transferability).
        """
        model_s = self.model_fn(dim=dim_small, depth=2)
        model_l = self.model_fn(dim=dim_large, depth=2)

        losses = {}
        for tag, model in [('small', model_s), ('large', model_l)]:
            opt = torch.optim.SGD(model.parameters(), lr=lr)
            model.train()
            trajectory = []
            for step, (xb, yb) in enumerate(self.dataloader):
                if step >= steps:
                    break
                opt.zero_grad()
                out = model(xb)
                loss = nn.functional.mse_loss(out.flatten(), yb.flatten()[:out.numel()])
                loss.backward()
                opt.step()
                trajectory.append(loss.item())
            losses[tag] = trajectory

        if not losses.get('small') or not losses.get('large'):
            return 0.0

        # Compare final losses
        final_s = losses['small'][-1]
        final_l = losses['large'][-1]
        if final_s < 1e-12:
            return 1.0
        return min(final_l / (final_s + 1e-12), final_s / (final_l + 1e-12))

    def evaluate_scaling_ceiling(self) -> float:
        """Evaluate the architecture at two micro-scales to predict macro-scaling.

        Returns:
            Score in [0, 1]: 1.0 = excellent scaling, 0.0 = catastrophic.
        """
        model_small = self.model_fn(dim=64, depth=2)
        model_med = self.model_fn(dim=128, depth=4)

        x, _ = next(iter(self.dataloader))

        cond_small = self.compute_ntk_condition_number(model_small, x[:4])
        cond_med = self.compute_ntk_condition_number(model_med, x[:4])

        logger.info("NTK cond: small=%.2f, med=%.2f", cond_small, cond_med)

        if cond_small < 1e-8:
            return 0.0

        growth_rate = cond_med / (cond_small + 1e-8)
        # Score: 1.0 is perfect scaling, 0.0 is catastrophic
        scaling_score = max(0.0, 1.0 - 0.1 * growth_rate)
        return scaling_score
