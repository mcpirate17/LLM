"""Python fallback kernel for gated_linear block."""

import torch
import torch.nn.functional as F
from components._weight_cache import cached_randn
from components.base import _try_native


class ComponentHandler:
    """Fused gated linear: (x @ W) * sigmoid(x @ W_gate)."""

    __slots__ = ()

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

        # Try native kernel first
        result = _try_native("gated_linear", x, D, out_dim)
        if result is not None:
            return {"y": result}

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
        return {"y": F.linear(x, W) * torch.sigmoid(F.linear(x, W_gate))}
