"""Kernel handler for grouped_linear — dispatches to aria_core.linear_grouped_f32."""

from components.base import NativeComponentHandler, _make_weight


class ComponentHandler(NativeComponentHandler):
    native_op_name = "linear_grouped"

    def _ensure_weights(self, x, config):
        D = x.shape[-1]
        self._weights["w"] = _make_weight((D, D), fan_in=D)

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        n_groups = config.get("n_groups", 4)
        return (x, self._weights["w"], n_groups, None)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        return {"y": x}
