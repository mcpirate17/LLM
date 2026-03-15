"""Kernel handler for rotor_transform — dispatches to aria_core.rotor_transform_f32."""
import torch
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "rotor_transform"

    def _ensure_weights(self, x, config):
        D = x.shape[-1]
        rotor = torch.randn(*x.shape) * 0.1
        rotor[..., 0] = 1.0
        self._weights["rotor"] = rotor

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        rotor = self._weights["rotor"]
        if rotor.shape != x.shape:
            rotor = torch.randn_like(x) * 0.1
            rotor[..., 0] = 1.0
            self._weights["rotor"] = rotor
        return (x, rotor)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        return {"y": x}
