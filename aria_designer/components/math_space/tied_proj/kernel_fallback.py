"""Kernel handler for tied_proj — dispatches to aria_core.linear_tied_f32."""
import torch
from components.base import NativeComponentHandler, _make_weight

class ComponentHandler(NativeComponentHandler):
    native_op_name = "linear_tied"

    def _ensure_weights(self, x, config):
        D = x.shape[-1]
        self._weights["w"] = _make_weight((D, D), fan_in=D)

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        return (x, self._weights["w"])

    def _fallback(self, inputs, config):
        x = inputs["x"]
        return {"y": x}
