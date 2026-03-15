"""Kernel handler for spike_rate_code — dispatches to aria_core.spike_rate_code_f32."""
import torch
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "spike_rate_code"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        n_steps = config.get("n_steps", 10)
        return (x, n_steps)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        n_steps = config.get("n_steps", 10)
        probs = torch.sigmoid(x)
        return {"y": (probs * n_steps) / n_steps}
