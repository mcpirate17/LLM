"""Kernel handler for gather_topk — dispatches to aria_core.gather_topk_f32."""
import torch
import torch.nn as nn
from components.base import BaseComponentHandler, _try_native

class ComponentHandler(BaseComponentHandler):
    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        x = inputs["x"]
        k = config.get("k", 8)
        _, idx = x.topk(k, dim=-1)
        result = _try_native("gather_topk", x, idx.int())
        if result is not None:
            return {"y": result}
        return {"y": torch.gather(x, -1, idx)}
