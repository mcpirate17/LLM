"""Muon optimizer: Newton-Schulz orthogonalized momentum SGD for 2D+ parameters.

Applies NS iteration to approximate the matrix square root of the
second moment, giving natural-gradient-like updates without Adam's
memory cost. 1D parameters (biases, norms) fall back to AdamW.
"""

from __future__ import annotations

import torch


class _MuonGroup:
    """Newton-Schulz orthogonalized momentum SGD for 2D+ parameters."""

    __slots__ = ("params", "lr", "momentum", "wd", "ns_steps", "buffers")

    def __init__(self, params, lr, momentum, wd, ns_steps=5):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.wd = wd
        self.ns_steps = ns_steps
        self.buffers = [torch.zeros_like(p) for p in params]

    @torch.no_grad()
    def step(self):
        for p, buf in zip(self.params, self.buffers):
            if p.grad is None:
                continue
            g = p.grad

            if g.ndim >= 2:
                shape = g.shape
                g2d = g.reshape(shape[0], -1) if g.ndim > 2 else g
                g2d = g2d.float()
                g2d = g2d / (g2d.norm() + 1e-8)

                rows, cols = g2d.shape
                X = g2d
                if rows <= cols:
                    for _ in range(self.ns_steps):
                        A = X @ X.T
                        X = 1.5 * X - 0.5 * A @ X
                else:
                    for _ in range(self.ns_steps):
                        A = X.T @ X
                        X = 1.5 * X - 0.5 * X @ A

                g = X.reshape(shape).to(p.dtype)

            buf.mul_(self.momentum).add_(g)
            p.mul_(1 - self.lr * self.wd)
            p.add_(buf, alpha=-self.lr)


class CombinedOptimizer:
    """Muon (2D+ params) + AdamW (1D params) combined optimizer."""

    __slots__ = ("muon", "adamw", "_all_muon_params")

    def __init__(self, muon, adamw):
        self.muon = muon
        self.adamw = adamw
        self._all_muon_params = muon.params

    def step(self):
        self.muon.step()
        self.adamw.step()

    def zero_grad(self):
        for p in self._all_muon_params:
            p.grad = None
        self.adamw.zero_grad(set_to_none=True)


def get_muon_optimizer(model, lr=0.02, momentum=0.95, wd=0.01):
    """Create Muon+AdamW combined optimizer for the given model."""
    params_2d = []
    params_other = []

    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            params_2d.append(p)
        else:
            params_other.append(p)

    muon = _MuonGroup(params_2d, lr=lr, momentum=momentum, wd=wd)
    adamw = torch.optim.AdamW(params_other, lr=lr * 0.1, weight_decay=wd)
    return CombinedOptimizer(muon, adamw)
