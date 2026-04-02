"""Shared utilities for mathspace modules."""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=8)
def causal_mask(S: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular bool mask for causal attention (True = masked)."""
    return torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
