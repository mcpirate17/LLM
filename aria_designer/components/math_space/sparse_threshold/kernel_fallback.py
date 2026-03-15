"""Kernel handler for sparse_threshold — dispatches to aria_core.sparse_threshold_f32."""
import torch
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "sparse_threshold"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        threshold = config.get("threshold", 0.01)
        return (x, threshold)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        threshold = config.get("threshold", 0.01)
        return {"y": torch.where(x.abs() > threshold, x, torch.zeros_like(x))}
