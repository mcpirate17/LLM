from __future__ import annotations

from typing import Dict, Tuple

import torch

from ._optimizer_muon import MuonOptimizer


def build_optimizer(
    params,
    optimizer_type: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
    momentum: float = 0.95,
    fused: bool = False,
    foreach: bool = False,
) -> torch.optim.Optimizer:
    """Construct the supported runtime optimizer set."""
    name = optimizer_type.lower().strip()

    if name == "muon":
        return MuonOptimizer(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

    if name == "adamw":
        kwargs: Dict[str, object] = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
        # fused/foreach are explicit caller requests: if the build can't
        # honor them, fail loud rather than silently training differently.
        if fused:
            kwargs["fused"] = True
        elif foreach:
            kwargs["foreach"] = True
        return torch.optim.AdamW(params, **kwargs)  # type: ignore[arg-type]

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
        )

    raise ValueError(
        f"Unknown optimizer_type {name!r}. Valid options: muon, adamw, sgd"
    )
