"""Kernel handler for grade_select — dispatches to aria_core.grade_select_f32."""

import torch
from components.base import NativeComponentHandler


class ComponentHandler(NativeComponentHandler):
    native_op_name = "grade_select"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        grade = config.get("grade", 0)
        return (x, grade)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        grade = config.get("grade", 0)
        D = x.shape[-1]
        out = torch.zeros_like(x)
        if grade == 0:
            out[..., :1] = x[..., :1]
        elif grade == 1 and D >= 4:
            out[..., 1:4] = x[..., 1:4]
        return {"y": out}
