from __future__ import annotations

import torch

from ._muon_native import load_muon_native


def _orthogonalize_update(
    matrix: torch.Tensor,
    n_steps: int,
    native_ext,
) -> torch.Tensor:
    if matrix.ndim != 2:
        return matrix
    return native_ext.orthogonalize_update(matrix, n_steps)


class MuonOptimizer(torch.optim.Optimizer):
    """Momentum optimizer with Newton-Schulz orthogonalized 2D updates."""

    __slots__ = ("_native_ext",)

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
        self._native_ext = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(grad)
                update = grad + momentum * buffer if nesterov else buffer

                if param.ndim >= 2:
                    if self._native_ext is None:
                        self._native_ext = load_muon_native()
                    update = _orthogonalize_update(
                        update.view(param.shape[0], -1),
                        n_steps=ns_steps,
                        native_ext=self._native_ext,
                    ).view_as(param)

                if weight_decay > 0:
                    param.data.mul_(1 - lr * weight_decay)
                param.data.add_(update, alpha=-lr)

        return loss
