"""Shared utilities for evaluation probes.

These helpers run *between* the trained-model forward and the probe's own
training/eval phase. They exist because probes commonly ``copy.deepcopy`` the
trained model, and upstream evaluation paths leave inference-mode tensors
behind that break autograd or corrupt tensor metadata on subsequent CUDA ops.
"""

from __future__ import annotations

import copy

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


_MODULE_INTERNAL_DICTS = frozenset(
    {
        "_parameters",
        "_buffers",
        "_modules",
        "_non_persistent_buffers_set",
        "_backward_pre_hooks",
        "_backward_hooks",
        "_forward_hooks",
        "_forward_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    }
)


def _detach_non_leaf_attrs_(module: nn.Module) -> None:
    """In-place: detach any non-leaf tensors cached on modules.

    Patterns like ``nn.utils.weight_norm`` and synthesis-graph op caches store
    a *computed* tensor (with ``grad_fn``, i.e. non-leaf) directly on the
    module as a Python attribute. ``copy.deepcopy`` then raises
    ``RuntimeError: Only Tensors created explicitly by the user (graph leaves)
    support the deepcopy protocol``. Detaching drops ``grad_fn`` so the copy
    succeeds; the next forward pass will recompute the cache.
    """
    for mod in module.modules():
        for name, buf in list(mod._buffers.items()):
            if buf is not None and not buf.is_leaf:
                mod._buffers[name] = buf.detach()
        for name, val in list(mod.__dict__.items()):
            if name in _MODULE_INTERNAL_DICTS:
                continue
            if isinstance(val, torch.Tensor) and not val.is_leaf:
                setattr(mod, name, val.detach())


def safe_deepcopy_module(module: nn.Module) -> nn.Module:
    """Materialize, detach, then deepcopy — survives the two known fail modes.

    ``copy.deepcopy`` on a trained nn.Module fails in two distinct ways:

    1. **Inference-mode tensors**: an upstream ``torch.inference_mode()`` eval
       pass left cached buffers (RoPE tables, attention masks) marked as
       inference tensors. ``_materialize_non_inference_`` clones their storage
       outside inference_mode to remint them as normal tensors.
    2. **Non-leaf cached tensors**: synthesis-graph ops or ``weight_norm``-style
       wrappers cache a computed tensor (with ``grad_fn``) as a raw module
       attribute. ``_detach_non_leaf_attrs_`` strips ``grad_fn`` so deepcopy
       can walk the tensor.

    Both passes are idempotent: once a module is cleaned, future calls are
    no-ops. We also clean the copy as belt-and-braces against any new bad
    tensors that landed via the deepcopy of internal caches.
    """
    _materialize_non_inference_(module)
    _detach_non_leaf_attrs_(module)
    copied = copy.deepcopy(module)
    _materialize_non_inference_(copied)
    _detach_non_leaf_attrs_(copied)
    return copied
