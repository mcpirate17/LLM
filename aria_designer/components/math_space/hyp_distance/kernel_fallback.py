"""Kernel handler for hyp_distance — dispatches to aria_core.hyp_distance_f32."""
import torch
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "hyp_distance"

    def _get_native_args(self, inputs, config):
        x = inputs.get("x", inputs.get("a")).detach().contiguous().float()
        y = inputs.get("y", inputs.get("b", x)).detach().contiguous().float()
        return (x, y)

    def _fallback(self, inputs, config):
        x = inputs.get("x", inputs.get("a"))
        y = inputs.get("y", inputs.get("b", x))
        return {"y": torch.norm(x - y, dim=-1, keepdim=True)}
