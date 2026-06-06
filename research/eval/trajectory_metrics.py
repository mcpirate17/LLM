"""Unified trajectory metric entry point.

One function — ``compute_trajectory_metrics(model, ...)`` — runs every
Gemini metric we have. Sharing the entry point ensures all callers get
the same suite, the same defaults, and one place to add the next metric.

The four metrics:

* ``fp_jacobian_spectral_norm`` (existing) — operator norm of
  ``J(output, embed)``. Already in the fingerprint pipeline; we just
  re-expose it here so the smoke driver writes everything from one
  call.
* ``fp_jacobian_erf_*`` — density / variance / decay slope of
  ``J(last_token_output, embed)`` across input positions.
* ``fp_id_collapse_*`` — ratio of intrinsic-dimension snapshots taken
  at two training steps; lives separately from the static metrics
  because it requires a *during-training* hook.
* ``fp_icld_*`` — slope of per-position loss on synthetic Dyck.
* ``fp_logit_margin_*`` — slope of logit-margin trajectory while
  briefly training the model on synthetic transitive triples.

Three of the four metrics are static probes — they take the model in
its current state and return a value. ``fp_id_collapse_*`` is the
exception: it needs *two* hidden-state snapshots taken at different
training steps, and the surrounding pipeline is responsible for
calling ``capture_hidden_state_snapshot`` at the right moments and
combining them with ``compute_id_collapse_rate``. The smoke driver
demonstrates the full pattern.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .fingerprint_sensitivity import analyze_sensitivity
from .icld_velocity import ICLDResult, compute_icld_velocity
from .intrinsic_dim_collapse import (
    HiddenStateSnapshot,
    IDCollapseResult,
    capture_hidden_state_snapshot,
    compute_id_collapse_rate,
)
from .jacobian_erf import JacobianERFResult, compute_jacobian_erf
from .transitive_logit_margin import (
    LogitMarginResult,
    compute_transitive_logit_margin,
)

logger = logging.getLogger(__name__)


def _logit_margin_core_is_finite(result: LogitMarginResult) -> bool:
    for value in (
        result.velocity,
        result.initial_margin,
        result.final_margin,
        result.delta_margin,
    ):
        if value is None or not math.isfinite(float(value)):
            return False
    return True


@dataclass(slots=True)
class TrajectoryMetricsResult:
    spec_norm: Optional[float] = None
    spec_norm_eff_rank: Optional[float] = None
    spec_norm_uniformity: Optional[float] = None
    spec_norm_status: str = "init"

    jacobian_erf: JacobianERFResult = field(default_factory=JacobianERFResult)
    icld: ICLDResult = field(default_factory=ICLDResult)
    logit_margin: LogitMarginResult = field(default_factory=LogitMarginResult)
    id_collapse: Optional[IDCollapseResult] = (
        None  # only set when caller provides snapshots
    )

    metric_phase: str = (
        "unknown"  # init | screening_750 | investigation_full | validation_full
    )
    elapsed_ms_total: float = 0.0

    def to_column_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "fp_jacobian_spectral_norm": self.spec_norm,
            "fp_jacobian_effective_rank": self.spec_norm_eff_rank,
            "fp_sensitivity_uniformity": self.spec_norm_uniformity,
            "fp_spec_norm_status": self.spec_norm_status,
            "fp_metric_phase": self.metric_phase,
        }
        out.update(self.jacobian_erf.to_dict())
        out.update(self.icld.to_dict())
        out.update(self.logit_margin.to_dict())
        if self.id_collapse is not None:
            out.update(self.id_collapse.to_dict())
        return out


def compute_trajectory_metrics(
    model: nn.Module,
    *,
    metric_phase: str,
    device: str | torch.device = "cuda",
    seq_len: int = 64,
    erf_n_samples: int = 4,
    icld_seq_len: int = 64,
    icld_batch_size: int = 32,
    logit_margin_n_train_steps: int = 60,
    logit_margin_batch_size: int = 32,
    logit_margin_lr: float = 1e-3,
    logit_margin_retry_lr: float | None = 3e-4,
    id_collapse_early: HiddenStateSnapshot | None = None,
    id_collapse_late: HiddenStateSnapshot | None = None,
    spec_norm_vocab_size: int = 32000,
) -> TrajectoryMetricsResult:
    """Run all static trajectory metrics on ``model``.

    Args:
        model: ``SynthesizedModel``-compatible module.
        metric_phase: label for ``fp_metric_phase`` so ML training can
            condition on which lifecycle stage produced these values.
            Use ``"init"`` if model is freshly built without training,
            ``"screening_750"`` if measured at end of screening tier,
            ``"investigation_full"`` for end of investigation training,
            ``"validation_full"`` for end of validation training.
        id_collapse_early / id_collapse_late: optional hidden-state
            snapshots taken at two training steps. When both are
            provided, the result's ``id_collapse`` field is populated;
            otherwise it stays ``None`` and the caller is expected to
            run the snapshot/collapse machinery separately.
    """
    t0 = time.perf_counter()
    out = TrajectoryMetricsResult(metric_phase=metric_phase)

    dev = torch.device(device) if not isinstance(device, torch.device) else device

    # 1) Spec norm — existing pipeline metric. Run it via analyze_sensitivity
    # directly so we don't have to rebuild the fingerprint container.
    sensitivity = analyze_sensitivity(
        model, dev, seq_len=seq_len, vocab_size=spec_norm_vocab_size
    )
    if sensitivity.get("_succeeded"):
        out.spec_norm = float(sensitivity["spectral_norm"])
        out.spec_norm_eff_rank = float(sensitivity["effective_rank"])
        out.spec_norm_uniformity = float(sensitivity["uniformity"])
        out.spec_norm_status = "ok"
    else:
        out.spec_norm_status = "failed"

    # 2) Jacobian ERF — same autograd plumbing as spec_norm.
    out.jacobian_erf = compute_jacobian_erf(
        model,
        seq_len=seq_len,
        vocab_size=spec_norm_vocab_size,
        device=dev,
        n_samples=erf_n_samples,
    )

    # 3) ICLD — synthetic Dyck, no training, just per-position loss slope.
    out.icld = compute_icld_velocity(
        model,
        seq_len=icld_seq_len,
        batch_size=icld_batch_size,
        device=dev,
    )

    # 4) Logit margin — short on-the-fly training of a deepcopy.
    out.logit_margin = compute_transitive_logit_margin(
        model,
        n_train_steps=logit_margin_n_train_steps,
        batch_size=logit_margin_batch_size,
        lr=logit_margin_lr,
        device=dev,
    )
    if logit_margin_retry_lr is not None and not _logit_margin_core_is_finite(
        out.logit_margin
    ):
        retry = compute_transitive_logit_margin(
            model,
            n_train_steps=logit_margin_n_train_steps,
            batch_size=logit_margin_batch_size,
            lr=logit_margin_retry_lr,
            device=dev,
        )
        if _logit_margin_core_is_finite(retry):
            retry.status = f"{retry.status}_lr{logit_margin_retry_lr:g}_fallback"
            out.logit_margin = retry

    # 5) ID collapse — only if caller supplied snapshots.
    if id_collapse_early is not None and id_collapse_late is not None:
        out.id_collapse = compute_id_collapse_rate(id_collapse_early, id_collapse_late)

    out.elapsed_ms_total = round((time.perf_counter() - t0) * 1000, 1)
    return out


__all__ = [
    "TrajectoryMetricsResult",
    "compute_trajectory_metrics",
    "capture_hidden_state_snapshot",  # re-export so callers don't import
    # from intrinsic_dim_collapse directly
]
