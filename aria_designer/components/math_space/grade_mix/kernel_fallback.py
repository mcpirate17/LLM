"""Kernel handler for grade_mix — dispatches to aria_core.grade_mix_f32."""
import torch
import torch.nn as nn
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "grade_mix"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        weights = torch.ones(x.shape[-1])
        return (x, weights)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        return {"y": x}
