"""Python fallback kernel for swiglu_mlp."""

import torch.nn.functional as F


class ComponentHandler:
    """Fallback handler for swiglu_mlp."""

    __slots__ = ()

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        if x.dim() != 3:
            raise ValueError("swiglu_mlp expects x with shape [B, S, D]")
        hidden = max(1, int(x.shape[-1] * float(config.get("mlp_ratio", 3.0))))
        gate = F.silu(x.mean(dim=-1, keepdim=True)).expand(*x.shape[:-1], hidden)
        up = x.mean(dim=-1, keepdim=True).expand(*x.shape[:-1], hidden)
        mixed = gate * up
        return {"y": mixed[..., : x.shape[-1]]}
