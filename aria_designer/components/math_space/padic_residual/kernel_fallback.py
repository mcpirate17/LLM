"""Kernel handler for padic_residual — dispatches to aria_core.padic_residual_f32."""
import torch
import torch.nn as nn
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "padic_residual"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        residual = inputs.get("residual", x).detach().contiguous().float()
        p = config.get("p", 2)
        n_digits = config.get("n_digits", 4)
        return (x, residual, p, n_digits)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        residual = inputs.get("residual", x)
        return {"y": x + residual}
