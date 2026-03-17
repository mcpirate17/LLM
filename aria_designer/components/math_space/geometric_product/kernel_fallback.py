"""Kernel handler for geometric_product — dispatches to aria_core.clifford_geometric_product_cl30_f32."""

from components.base import NativeComponentHandler


class ComponentHandler(NativeComponentHandler):
    native_op_name = "clifford_geometric_product_cl30"

    def _get_native_args(self, inputs, config):
        x = inputs.get("x", inputs.get("a")).detach().contiguous().float()
        y = inputs.get("y", inputs.get("b", x)).detach().contiguous().float()
        return (x, y)

    def _fallback(self, inputs, config):
        x = inputs.get("x", inputs.get("a"))
        y = inputs.get("y", inputs.get("b", x))
        return {"y": x * y}
