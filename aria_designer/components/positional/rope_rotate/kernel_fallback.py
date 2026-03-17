"""Kernel handler for rope_rotate — dispatches to aria_core.rope_rotate_f32."""

from components.base import NativeComponentHandler
from research.defaults import ROPE_THETA_BASE


class ComponentHandler(NativeComponentHandler):
    native_op_name = "rope_rotate"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        theta_base = config.get("theta_base", ROPE_THETA_BASE)
        return (x, theta_base)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        return {"y": x}
