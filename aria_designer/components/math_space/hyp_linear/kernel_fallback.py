"""Kernel handler for hyp_linear — dispatches to aria_core.hyp_linear_f32."""
import torch
import torch.nn as nn
from components.base import NativeComponentHandler, _make_weight

class ComponentHandler(NativeComponentHandler):
    native_op_name = "hyp_linear"

    def _ensure_weights(self, x, config):
        D = x.shape[-1]
        out_dim = config.get("out_dim", D)
        self._weights["w"] = _make_weight((out_dim, D), fan_in=D)

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        c = config.get("curvature", 1.0)
        return (x, self._weights["w"], None, c)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        return {"y": torch.nn.functional.linear(x, self._weights["w"])}
