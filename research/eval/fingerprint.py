"""Behavioral fingerprint public surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch.nn as nn

from research.defaults import VOCAB_SIZE

from .fingerprint_runtime import compute_fingerprint, compute_lightning_fingerprint
from .fingerprint_types import BehavioralFingerprint

if TYPE_CHECKING:
    from research.synthesis.graph import ComputationGraph


def compute_gated_fingerprint(
    model: nn.Module,
    *,
    seq_len: int = 64,
    model_dim: int = 256,
    vocab_size: int = VOCAB_SIZE,
    device: str = "cpu",
    full_gate_enabled: bool = True,
    _lightning_novelty_threshold: float = 0.15,
    force_lightning_only: bool = False,
    graph: Optional["ComputationGraph"] = None,
    structural_floor: float = 0.10,
) -> Tuple[BehavioralFingerprint, bool]:
    del _lightning_novelty_threshold
    if not full_gate_enabled:
        return (
            compute_fingerprint(
                model,
                seq_len=seq_len,
                model_dim=model_dim,
                vocab_size=vocab_size,
                device=device,
                include_cka=False,
                include_behavioral_probes=False,
            ),
            True,
        )

    lightning_fp = compute_lightning_fingerprint(
        model,
        seq_len=seq_len,
        model_dim=model_dim,
        device=device,
        graph=graph,
        structural_floor=structural_floor,
    )
    if force_lightning_only or float(lightning_fp.novelty_score or 0.0) < float(
        structural_floor
    ):
        return lightning_fp, False

    return (
        compute_fingerprint(
            model,
            seq_len=seq_len,
            model_dim=model_dim,
            vocab_size=vocab_size,
            device=device,
            include_cka=False,
            include_behavioral_probes=False,
        ),
        True,
    )
