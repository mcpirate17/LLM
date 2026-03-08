"""Kernel handler for conditional_gather — Phase A weighted average."""
import torch
import torch.nn as nn


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return nn.Identity()

    def forward(self, inputs, config):
        a = inputs["a"]
        b = inputs["b"]
        # Phase A: equal-weight average
        return {"y": (a + b) / 2.0}
