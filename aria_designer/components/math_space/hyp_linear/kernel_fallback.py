"""Kernel handler for hyp_linear — delegates to research.mathspaces.hyperbolic."""

import math
import torch
import torch.nn as nn
from components.base import NativeComponentHandler


class ComponentHandler(NativeComponentHandler):
    native_op_name = "hyp_linear"

    def _get_native_args(self, inputs, config):
        x = inputs["x"].detach().contiguous().float()
        return (x,)

    def _fallback(self, inputs, config):
        from research.mathspaces.hyperbolic import execute_hyp_linear

        x = inputs["x"]
        D = x.shape[-1]
        stub = nn.Module()
        stub.weight = nn.Parameter(
            torch.randn(D, D, device=x.device, dtype=x.dtype) / math.sqrt(D)
        )
        return {"y": execute_hyp_linear(stub, x)}
