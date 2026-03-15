"""Kernel handler for padic_gate — dispatches to aria_core.padic_gate_f32."""
import torch
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "padic_gate"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        p = config.get("p", 2.0)
        return (x, p)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        p = config.get("p", 2.0)
        abs_x = torch.clamp(x.abs(), min=1e-10)
        valuation = -(torch.log(abs_x) / torch.log(torch.tensor(p)))
        valuation = torch.clamp(valuation, -10.0, 10.0)
        gate = torch.sigmoid(valuation)
        return {"y": x * gate}
