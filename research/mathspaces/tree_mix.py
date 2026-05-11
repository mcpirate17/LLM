"""
Binary-Tree Feature Mixing — atomic mixer node

Per external_research_2026-05-10.md §2.1 — "leafed layers / fractal /
tree-structured connectivity". The grammar can build linear chains and
parallel branches today; what it cannot build is **balanced-tree depth
structure**.

`tree_mix(x, y)` is the atomic binary mixer node:

    z = sigmoid(W) ⊙ x + (1 - sigmoid(W)) ⊙ y

with a learned per-feature gate W ∈ R^D. Both children get gradient
proportional to the gate (sigmoid keeps W strictly in (0, 1)), so no
hard routing.

A balanced binary tree of depth K is built at the *template* level by
composing 2^K - 1 tree_mix nodes — exactly the research spec's
``tree_mix(left, right, depth)`` semantics, with depth chosen by the
template. The atomic op is binary; the structure is grammar-emergent.

All operations are torch primitives — sigmoid, mul, add — so the hot
path stays in the native C++/CUDA dispatch. No custom kernel.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def execute_tree_mix(
    module: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Binary gated mixer: z = sigmoid(W) ⊙ x + (1 - sigmoid(W)) ⊙ y.

    Defensive fallbacks:
    - If ``module.gate`` is missing, return ``x`` (identity on first arg
      — matches the rest of the mathspaces ops' fail-soft convention).
    - If ``y`` has a different shape than ``x``, broadcast where possible
      and fall back to ``x`` otherwise.
    """
    if not hasattr(module, "gate"):
        return x

    if x.shape != y.shape:
        # Allow broadcastable y (e.g. (D,) vs (B,S,D)); otherwise identity.
        try:
            y = y.expand_as(x)
        except RuntimeError:
            return x

    gate = torch.sigmoid(module.gate.to(x.dtype))  # (D,) — broadcasts over batch + seq
    return gate * x + (1.0 - gate) * y
