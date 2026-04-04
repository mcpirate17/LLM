"""Kernel handler for rope_rotate with an honest Torch fallback."""

import torch
from aria_designer.components.base import NativeComponentHandler
from research.defaults import ROPE_THETA_BASE


class ComponentHandler(NativeComponentHandler):
    native_op_name = "rope_rotate"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        theta_base = float(config.get("theta_base", ROPE_THETA_BASE))
        return (x, theta_base)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        theta_base = float(config.get("theta_base", ROPE_THETA_BASE))
        _, seq_len, dim = x.shape
        pos = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        freqs = 1.0 / (
            theta_base
            ** (torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) / dim)
        )
        angles = pos * freqs.unsqueeze(0)
        cos_a = torch.cos(angles).unsqueeze(0)
        sin_a = torch.sin(angles).unsqueeze(0)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        y = torch.zeros_like(x)
        y[..., 0::2] = x1 * cos_a - x2 * sin_a
        y[..., 1::2] = x1 * sin_a + x2 * cos_a
        return {"y": y}
