"""Shared utilities for evaluation probes.

These helpers run *between* the trained-model forward and the probe's own
training/eval phase. They exist because probes commonly ``copy.deepcopy`` the
trained model, and upstream evaluation paths leave inference-mode tensors
behind that break autograd or corrupt tensor metadata on subsequent CUDA ops.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _materialize_non_inference_(module: nn.Module) -> None:
    """In-place: replace inference-mode params/buffers with autograd-safe storage.

    Probes deepcopy the trained investigation model, then train the copy.
    Upstream evaluation paths run forward under ``torch.inference_mode()``
    which marks any cached buffers (RoPE rotations, attention biases, KV
    caches) as inference tensors. Inference tensors propagate through
    ``deepcopy``, and trying to use them in autograd-tracked computation
    raises ``RuntimeError: Inference tensors cannot be saved for backward``.
    Cloning the storage outside ``inference_mode`` mints a normal tensor;
    swapping ``.data`` rebinds the wrapping Parameter/buffer to it.
    """
    with torch.no_grad():
        for p in module.parameters(recurse=True):
            if p.is_inference():
                fresh = torch.empty_like(p.data)
                fresh.copy_(p.data)
                p.data = fresh
        for b in module.buffers(recurse=True):
            if b.is_inference():
                fresh = torch.empty_like(b.data)
                fresh.copy_(b.data)
                b.data = fresh
