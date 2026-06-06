"""Intrinsic Dimensionality Collapse Rate.

Tracks how the participation ratio (PR) of hidden-state covariance eigenvalues
changes between two points during training. PR is a smooth, differentiable
estimator of intrinsic dimensionality:

    PR(C) = (Σ λ_i)² / Σ λ_i²

with λ_i the eigenvalues of the activation covariance C. PR ∈ [1, d] where d is
the hidden width.

A model that learns to compress concepts into a low-dimensional manifold shows
**rapid decrease** in PR between early and late training. A model that just
memorizes token co-occurrences keeps high PR (or even grows it). The collapse
rate is the slope:

    collapse_rate = (PR_late − PR_early) / (step_late − step_early)

Negative = compression (good). Zero / positive = memorization or no learning.

This module exposes two callables:

    capture_hidden_state_snapshot(model, probe_ids) -> SnapshotResult
        Runs one no-grad forward pass with the model in eval mode, captures
        the pre-LM-head hidden state, computes its participation ratio.

    compute_id_collapse_rate(early, late) -> ID_CollapseResult
        Combines two snapshots into a rate.

The smoke driver / pipeline integration is responsible for invoking
``capture_hidden_state_snapshot`` at the desired training step (e.g. 150 and
750). Decoupling capture from rate-computation keeps this metric reusable for
any training loop that wants to observe collapse at arbitrary checkpoints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from ._probe_runtime import disable_native_probe_dispatch

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HiddenStateSnapshot:
    step: int
    participation_ratio: float
    intrinsic_dim_normalized: float  # PR / hidden_dim, in [0, 1]
    n_eigenvalues: int
    elapsed_ms: float
    status: str = "ok"


@dataclass(slots=True)
class IDCollapseResult:
    pr_early: Optional[float] = None
    pr_late: Optional[float] = None
    intrinsic_dim_normalized_early: Optional[float] = None
    intrinsic_dim_normalized_late: Optional[float] = None
    step_early: Optional[int] = None
    step_late: Optional[int] = None
    collapse_rate: Optional[float] = None
    collapse_rate_normalized: Optional[float] = None  # rate / hidden_dim
    status: str = "init"
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, float | int | str | None]:
        return {
            "fp_id_pr_early": self.pr_early,
            "fp_id_pr_late": self.pr_late,
            "fp_id_norm_early": self.intrinsic_dim_normalized_early,
            "fp_id_norm_late": self.intrinsic_dim_normalized_late,
            "fp_id_step_early": self.step_early,
            "fp_id_step_late": self.step_late,
            "fp_id_collapse_rate": self.collapse_rate,
            "fp_id_collapse_rate_normalized": self.collapse_rate_normalized,
            "fp_id_collapse_status": self.status,
            "fp_id_collapse_elapsed_ms": self.elapsed_ms,
        }


def _participation_ratio(eigenvalues: torch.Tensor) -> float:
    """Participation ratio: (sum λ)² / sum(λ²). Stable to noise."""
    eigenvalues = eigenvalues.clamp_min(0.0)
    s1 = eigenvalues.sum()
    s2 = (eigenvalues * eigenvalues).sum().clamp_min(1e-20)
    return float((s1 * s1 / s2).item())


def capture_hidden_state_snapshot(
    model: nn.Module,
    probe_ids: torch.Tensor,
    *,
    step: int,
    device: str | torch.device,
) -> HiddenStateSnapshot:
    """Forward-pass on probe_ids, capture pre-LM-head hidden state, compute PR.

    The hidden state is taken from ``_fingerprint_pre_logits_from_embed``
    when available — that's the canonical pre-LM-head representation
    (model body output → norm → ready-for-LM-head). Falls back to
    running the body via ``_fingerprint_forward_from_embed`` and applying
    the model's ``norm`` if present.
    """
    t0 = time.perf_counter()
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    device_str = str(dev)

    try:
        model.eval()
        with disable_native_probe_dispatch(model, device=device_str), torch.no_grad():
            embed = model.embed(probe_ids)
            if hasattr(model, "_fingerprint_pre_logits_from_embed"):
                hidden = model._fingerprint_pre_logits_from_embed(embed)
            elif hasattr(model, "_fingerprint_forward_from_embed"):
                hidden = model._fingerprint_forward_from_embed(embed)
                if hasattr(model, "norm") and model.norm is not None:
                    hidden = model.norm(hidden)
            else:
                hidden = embed

            # Flatten (B, S, D) -> (B*S, D) and center.
            flat = hidden.reshape(-1, hidden.shape[-1]).float()
            flat = flat - flat.mean(dim=0, keepdim=True)

            # Use the smaller-dim Gram for cheaper eigvalsh.
            n_samples = flat.shape[0]
            n_feats = flat.shape[1]
            if n_samples >= n_feats:
                gram = flat.transpose(0, 1) @ flat  # (D, D)
            else:
                gram = flat @ flat.transpose(0, 1)  # (N, N)

            eigenvalues = torch.linalg.eigvalsh(gram)
            pr = _participation_ratio(eigenvalues)
            normalized = pr / max(n_feats, 1)

        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        return HiddenStateSnapshot(
            step=step,
            participation_ratio=pr,
            intrinsic_dim_normalized=normalized,
            n_eigenvalues=eigenvalues.numel(),
            elapsed_ms=elapsed,
            status="ok",
        )
    except RuntimeError as exc:
        logger.warning("ID snapshot capture failed at step %d: %s", step, exc)
        return HiddenStateSnapshot(
            step=step,
            participation_ratio=float("nan"),
            intrinsic_dim_normalized=float("nan"),
            n_eigenvalues=0,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status=f"failed: {exc.__class__.__name__}",
        )


def compute_id_collapse_rate(
    early: HiddenStateSnapshot,
    late: HiddenStateSnapshot,
) -> IDCollapseResult:
    """Combine two snapshots into a collapse-rate metric.

    Args:
        early: snapshot at lower training step (e.g. step 150).
        late: snapshot at higher training step (e.g. step 750).

    Returns a result with positive collapse_rate meaning growth (memorization)
    and negative meaning compression (generalization).
    """
    t0 = time.perf_counter()
    result = IDCollapseResult(
        pr_early=early.participation_ratio,
        pr_late=late.participation_ratio,
        intrinsic_dim_normalized_early=early.intrinsic_dim_normalized,
        intrinsic_dim_normalized_late=late.intrinsic_dim_normalized,
        step_early=early.step,
        step_late=late.step,
        status="ok",
    )

    if early.status != "ok" or late.status != "ok":
        result.status = f"snapshot_failed: early={early.status}, late={late.status}"
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    step_diff = late.step - early.step
    if step_diff <= 0:
        result.status = "invalid_steps"
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    pr_diff = late.participation_ratio - early.participation_ratio
    norm_diff = late.intrinsic_dim_normalized - early.intrinsic_dim_normalized
    result.collapse_rate = pr_diff / step_diff
    result.collapse_rate_normalized = norm_diff / step_diff
    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result
