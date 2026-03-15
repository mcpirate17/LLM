"""Kernel handler for lif_neuron — dispatches to aria_core.lif_neuron_f32."""
import torch
from components.base import NativeComponentHandler

class ComponentHandler(NativeComponentHandler):
    native_op_name = "lif_neuron"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        tau = config.get("tau", 20.0)
        threshold = config.get("threshold", 1.0)
        return (x, tau, threshold)

    def _fallback(self, inputs, config):
        x = inputs["x"]
        tau = config.get("tau", 20.0)
        threshold = config.get("threshold", 1.0)
        decay = torch.exp(torch.tensor(-1.0 / tau))
        out = torch.zeros_like(x)
        membrane = torch.zeros(*x.shape[:-2], x.shape[-1], device=x.device)
        for t in range(x.shape[-2]):
            membrane = decay * membrane + x[..., t, :]
            spike = (membrane >= threshold).float()
            out[..., t, :] = spike
            membrane = membrane * (1 - spike)
        return {"y": out}
