"""Kernel handler for bottleneck_proj — dispatches to aria_core.linear_bottleneck_f32."""
import torch
from components.base import NativeComponentHandler, _make_weight

class ComponentHandler(NativeComponentHandler):
    native_op_name = "linear_bottleneck"

    def _ensure_weights(self, x, config):
        D = x.shape[-1]
        ratio = config.get("ratio", 4)
        bottleneck = max(1, D // ratio)
        self._weights["down"] = _make_weight((bottleneck, D), fan_in=D)
        self._weights["up"] = _make_weight((D, bottleneck), fan_in=bottleneck)

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        return (x, self._weights["down"], self._weights["up"], None)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        h = torch.relu(x @ self._weights["down"].T)
        return {"y": h @ self._weights["up"].T}
