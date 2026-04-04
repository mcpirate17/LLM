"""Python fallback kernel for gated_linear block."""

import torch
import torch.nn.functional as F
from aria_designer.components._weight_cache import cached_randn
from aria_designer.runtime.dispatch import KernelDispatcher


class ComponentHandler:
    """Fused gated linear: (x @ W) * sigmoid(x @ W_gate)."""

    __slots__ = ()
    _dispatcher = KernelDispatcher(use_native=True)

    def validate_config(self, config):
        errors = []
        out_dim = config.get("out_dim", 128)
        if not isinstance(out_dim, int) or out_dim < 1:
            errors.append("out_dim must be int >= 1")
        return errors

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        D = x.shape[-1]
        out_dim = config.get("out_dim", D)

        W = cached_randn(
            out_dim,
            D,
            seed=D * 65537 + out_dim,
            device=x.device,
            dtype=x.dtype,
            scale=D**-0.5,
        )
        W_gate = cached_randn(
            out_dim,
            D,
            seed=D * 131071 + out_dim,
            device=x.device,
            dtype=x.dtype,
            scale=D**-0.5,
        )
        try:
            return {"y": self._dispatcher.gated_linear(x, W, None, W_gate, None)}
        except Exception:
            return {"y": F.linear(x, W) * torch.sigmoid(F.linear(x, W_gate))}
