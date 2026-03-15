"""Kernel handler for clifford_attention — dispatches to aria_core.clifford_attention_f32."""
import torch
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "clifford_attention"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        n_grades = config.get("n_grades", 4)
        return (x, n_grades)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        return {"y": x}
