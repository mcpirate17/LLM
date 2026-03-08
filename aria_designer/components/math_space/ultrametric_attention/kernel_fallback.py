"""Kernel handler for ultrametric_attention — dispatches to aria_core.ultrametric_attention_f32."""
import torch
import torch.nn as nn
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "ultrametric_attention"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        p = config.get("p", 2)
        n_digits = config.get("n_digits", 4)
        return (x, p, n_digits)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        return {"y": x}
