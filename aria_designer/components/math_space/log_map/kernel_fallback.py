"""Kernel handler for log_map — dispatches to aria_core.log_map_f32."""

from components.base import NativeComponentHandler


class ComponentHandler(NativeComponentHandler):
    native_op_name = "log_map"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        y = inputs.get("y", x).detach().contiguous().float()
        c = config.get("curvature", 1.0)
        return (x, y, c)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        y = inputs.get("y", x)
        return {"y": y - x}
