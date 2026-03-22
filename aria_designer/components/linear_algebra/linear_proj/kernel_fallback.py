"""Python fallback kernel for linear_proj."""

import torch.nn.functional as F


class ComponentHandler:
    """Fallback handler for linear_proj: F.linear(x, W, b)."""

    def validate_config(self, config):
        return []

    def build(self, config):
        out_dim = config.get("out_dim")
        if out_dim:
            return None  # dims not known until forward
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        out_dim = config.get("out_dim") or x.shape[-1]
        # Lazy projection: create weight on the fly for preview/shape inference.
        # Real training uses the research pipeline compiler, not this fallback.
        w = _lazy_linear(x.shape[-1], out_dim, x.device, x.dtype)
        return {"y": F.linear(x, w)}


def _lazy_linear(in_dim, out_dim, device, dtype):
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(in_dim * 65537 + out_dim)
    w = torch.randn(out_dim, in_dim, generator=gen, dtype=dtype).to(device)
    w *= in_dim**-0.5
    return w
