"""Kernel handler for hyperbolic_norm — dispatches to aria_core.hyperbolic_norm_f32."""

import torch
from components.base import NativeComponentHandler


class ComponentHandler(NativeComponentHandler):
    native_op_name = "hyperbolic_norm"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        c = config.get("curvature", 1.0)
        return (x, c)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=1e-8)
        return {"y": x / norm}
