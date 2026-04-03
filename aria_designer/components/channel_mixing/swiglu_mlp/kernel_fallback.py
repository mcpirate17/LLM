"""Python fallback kernel for swiglu_mlp."""

import torch.nn.functional as F
from components._weight_cache import cached_randn


class ComponentHandler:
    """Fallback handler for swiglu_mlp: SwiGLU feed-forward with cached projections."""

    __slots__ = ()

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        if x.dim() != 3:
            raise ValueError("swiglu_mlp expects x with shape [B, S, D]")
        D = x.shape[-1]
        hidden = max(1, int(D * float(config.get("mlp_ratio", 3.0))))
        # Proper SwiGLU: separate gate and value projections
        W_gate = cached_randn(
            hidden,
            D,
            seed=D * 65537 + hidden,
            device=x.device,
            dtype=x.dtype,
            scale=D**-0.5,
        )
        W_up = cached_randn(
            hidden,
            D,
            seed=D * 131071 + hidden,
            device=x.device,
            dtype=x.dtype,
            scale=D**-0.5,
        )
        W_down = cached_randn(
            D,
            hidden,
            seed=hidden * 65537 + D,
            device=x.device,
            dtype=x.dtype,
            scale=hidden**-0.5,
        )
        gate = F.silu(F.linear(x, W_gate))
        up = F.linear(x, W_up)
        return {"y": F.linear(gate * up, W_down)}
