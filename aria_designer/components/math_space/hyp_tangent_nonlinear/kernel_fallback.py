"""Kernel handler for hyp_tangent_nonlinear — dispatches to aria_core.hyp_tangent_nonlinear_f32."""
import torch
import torch.nn as nn
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "hyp_tangent_nonlinear"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        c = config.get("curvature", 1.0)
        return (x, c)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        return {"y": torch.tanh(x)}
