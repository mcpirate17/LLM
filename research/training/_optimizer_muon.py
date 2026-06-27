from __future__ import annotations

import torch

_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315
_NS_EPS = 1e-30


def _orthogonalize_batched(stack: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Newton-Schulz orthogonalization on a [N, R, C] stack — device-side only.

    One bmm kernel sequence covers every matrix in the stack, so same-shape
    matrices cost one launch set instead of N.
    """
    rows, cols = stack.shape[1], stack.shape[2]
    transposed = rows < cols
    working = stack.transpose(1, 2) if transposed else stack
    # clamp_min keeps near-zero matrices from producing NaNs while preserving
    # the original behavior for any gradient with non-trivial norm (the clamp
    # is a no-op for norms > 1e-30, i.e. essentially always).
    norm = working.flatten(1).norm(dim=1).clamp_min(_NS_EPS).view(-1, 1, 1)
    x = working / norm
    for _ in range(n_steps):
        gram = x.transpose(1, 2).bmm(x)
        xg = x.bmm(gram)
        # Fused polynomial step: x ← a·x + b·xg + c·(xg @ gram), in place.
        x.mul_(_NS_A).add_(xg, alpha=_NS_B).baddbmm_(xg, gram, alpha=_NS_C)

    return x.transpose(1, 2) if transposed else x


def _orthogonalize_update(matrix: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Newton-Schulz orthogonalization of a single 2D matrix."""
    if matrix.ndim != 2:
        return matrix
    return _orthogonalize_batched(matrix.unsqueeze(0), n_steps).squeeze(0)


class MuonOptimizer(torch.optim.Optimizer):
    """Momentum optimizer with Newton-Schulz orthogonalized 2D updates.

    Batched with torch._foreach_* ops so the Python overhead is amortized
    across every parameter in a group; orthogonalization buckets same-shape
    matrices into one stacked Newton-Schulz (bmm) so a model with many
    identically shaped layers pays one kernel sequence per shape, not per
    matrix.
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

            # Bucket 2D+ updates by reshaped 2D shape (and dtype/device, so
            # mixed-precision groups never stack together), then run one
            # batched Newton-Schulz per bucket.
            buckets: dict[tuple, list[int]] = {}
            for i, param in enumerate(params):
                if param.ndim >= 2:
                    key = (
                        param.shape[0],
                        param[0].numel(),
                        updates[i].dtype,
                        updates[i].device,
                    )
                    buckets.setdefault(key, []).append(i)

            for (n_rows, n_cols, _, _), idxs in buckets.items():
                if len(idxs) == 1:
                    i = idxs[0]
                    reshaped = updates[i].view(n_rows, n_cols)
                    updates[i] = _orthogonalize_update(reshaped, ns_steps).view_as(
                        params[i]
                    )
                    continue
                stack = torch.stack([updates[i].view(n_rows, n_cols) for i in idxs])
                ortho = _orthogonalize_batched(stack, ns_steps)
                for j, i in enumerate(idxs):
                    updates[i] = ortho[j].view_as(params[i])

            if weight_decay > 0.0:
                torch._foreach_mul_(params, 1.0 - lr * weight_decay)
            torch._foreach_add_(params, updates, alpha=-lr)

        return loss
