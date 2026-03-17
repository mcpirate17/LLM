"""Kernel handler for rwkv_time_mixing — dispatches to aria_core.rwkv_time_mixing_f32."""

import torch
from components.base import NativeComponentHandler, _make_weight


class ComponentHandler(NativeComponentHandler):
    native_op_name = "rwkv_time_mixing"

    def _ensure_weights(self, x, config):
        D = x.shape[-1]
        self._weights["decay"] = torch.randn(D) * 0.1
        self._weights["bonus"] = torch.randn(D) * 0.1
        self._weights["wk"] = _make_weight((D, D), fan_in=D)
        self._weights["wv"] = _make_weight((D, D), fan_in=D)
        self._weights["wr"] = _make_weight((D, D), fan_in=D)

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        return (
            x,
            self._weights["decay"],
            self._weights["bonus"],
            self._weights["wk"],
            self._weights["wv"],
            self._weights["wr"],
        )

    def _fallback(self, inputs, config):
        x = inputs["x"]
        return {"y": x}
