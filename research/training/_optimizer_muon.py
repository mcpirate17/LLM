from __future__ import annotations

import torch

_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315
_NS_EPS = 1e-30


def _orthogonalize_update(matrix: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Newton-Schulz orthogonalization — device-side only, no host syncs."""
    if matrix.ndim != 2:
        return matrix

    rows, cols = matrix.shape
    transposed = rows < cols
    working = matrix.transpose(0, 1) if transposed else matrix
    # clamp_min keeps near-zero matrices from producing NaNs while preserving
    # the original behavior for any gradient with non-trivial norm (the clamp
    # is a no-op for norms > 1e-30, i.e. essentially always).
    norm = working.norm().clamp_min(_NS_EPS)
    x = working / norm
    for _ in range(n_steps):
        gram = x.transpose(0, 1).matmul(x)
        xg = x.matmul(gram)
        # Fused polynomial step: x ← a·x + b·xg + c·(xg @ gram), in place.
        # Eliminates two temporaries per iteration vs. the functional form.
        x.mul_(_NS_A).add_(xg, alpha=_NS_B).addmm_(xg, gram, alpha=_NS_C)

    return x.transpose(0, 1) if transposed else x


class MuonOptimizer(torch.optim.Optimizer):
    """Momentum optimizer with Newton-Schulz orthogonalized 2D updates.

    Batched with torch._foreach_* ops so the Python overhead is amortized
    across every parameter in a group; orthogonalization is still per-matrix
    because each matrix has a different shape.
    """

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

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])

            params: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            buffers: list[torch.Tensor] = []
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                buffer = state.get("momentum_buffer")
                if buffer is None:
                    buffer = torch.zeros_like(grad)
                    state["momentum_buffer"] = buffer
                params.append(param)
                grads.append(grad)
                buffers.append(buffer)

            if not params:
                continue

            # Batched momentum: buffer = momentum * buffer + grad
            torch._foreach_mul_(buffers, momentum)
            torch._foreach_add_(buffers, grads)

            if nesterov:
                updates = list(torch._foreach_add(grads, buffers, alpha=momentum))
            else:
                updates = list(buffers)

            # Orthogonalize each 2D+ update (shapes differ — per-matrix call).
            for i, param in enumerate(params):
                if param.ndim >= 2:
                    reshaped = updates[i].view(param.shape[0], -1)
                    updates[i] = _orthogonalize_update(reshaped, ns_steps).view_as(
                        param
                    )

            if weight_decay > 0.0:
                torch._foreach_mul_(params, 1.0 - lr * weight_decay)
            torch._foreach_add_(params, updates, alpha=-lr)

        return loss
