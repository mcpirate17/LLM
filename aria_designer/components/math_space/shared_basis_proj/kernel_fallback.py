"""Kernel handler for shared_basis_proj — dispatches to aria_core.linear_shared_basis_f32."""
import torch
import torch.nn as nn
from components.base import NativeComponentHandler, _make_weight

class ComponentHandler(NativeComponentHandler):
    native_op_name = "linear_shared_basis"

    def _ensure_weights(self, x, config):
        D = x.shape[-1]
        n_basis = config.get("n_basis", 8)
        self._weights["basis"] = _make_weight((n_basis, D), fan_in=D)
        self._weights["coeffs"] = _make_weight((D, n_basis), fan_in=n_basis)

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        return (x, self._weights["basis"], self._weights["coeffs"])

    def _fallback(self, inputs, config):
        x = inputs["x"]
        return {"y": x}
