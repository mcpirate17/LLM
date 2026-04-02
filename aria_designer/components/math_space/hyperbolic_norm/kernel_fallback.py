"""Kernel handler for hyperbolic_norm — delegates to research.mathspaces.hyperbolic."""

import torch
import torch.nn as nn
from components.base import NativeComponentHandler


class ComponentHandler(NativeComponentHandler):
    native_op_name = "hyperbolic_norm"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        return (x,)

    def _fallback(self, inputs, config):
        from research.mathspaces.hyperbolic import execute_hyperbolic_norm

        x = inputs["x"]
        D = x.shape[-1]
        stub = nn.Module()
        stub.weight = nn.Parameter(torch.ones(D, device=x.device, dtype=x.dtype))
        stub.bias = nn.Parameter(torch.zeros(D, device=x.device, dtype=x.dtype))
        return {"y": execute_hyperbolic_norm(stub, x)}
