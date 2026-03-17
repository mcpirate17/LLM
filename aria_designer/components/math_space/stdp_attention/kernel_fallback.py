"""Kernel handler for stdp_attention — dispatches to aria_core.stdp_attention_f32."""

from components.base import NativeComponentHandler


class ComponentHandler(NativeComponentHandler):
    native_op_name = "stdp_attention"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        tau_plus = config.get("tau_plus", 20.0)
        tau_minus = config.get("tau_minus", 20.0)
        return (x, tau_plus, tau_minus)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        return {"y": x}
