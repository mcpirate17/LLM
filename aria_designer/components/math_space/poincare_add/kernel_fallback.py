"""Kernel handler for poincare_add — dispatches to aria_core.poincare_add_f32."""
import torch
import torch.nn as nn
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "poincare_add"

    def _get_native_args(self, inputs, config):
        x = inputs.get("x", inputs.get("a")).detach().contiguous().float()
        y = inputs.get("y", inputs.get("b", x)).detach().contiguous().float()
        c = config.get("curvature", 1.0)
        return (x, y, c)

    def _fallback(self, inputs, config):
        x = inputs.get("x", inputs.get("a"))
        y = inputs.get("y", inputs.get("b", x))
        return {"y": x + y}
