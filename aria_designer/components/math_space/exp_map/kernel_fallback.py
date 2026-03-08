"""Kernel handler for exp_map — dispatches to aria_core.exp_map_f32."""
import torch
import torch.nn as nn
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "exp_map"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        v = inputs.get("v", x).detach().contiguous().float()
        c = config.get("curvature", 1.0)
        return (x, v, c)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        v = inputs.get("v", x)
        c = config.get("curvature", 1.0)
        sqrt_c = c ** 0.5
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-8)
        return {"y": x + torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)}
