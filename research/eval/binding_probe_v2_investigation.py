"""Investigation-tier binding probe (v2, 2026-04-18).

Drop-in addition that lives alongside the production screening-tier binding
probe (`research.eval.binding_curriculum.curriculum_binding_range_profile`).

Key differences vs screening-tier:
  * Longer training budget (2400 steps vs 400/800) so slow-converging
    architectures (e.g. 2-layer attention at lr=3e-4) reach their true
    capability ceiling. The PROBE_CALIBRATION_2026-04-17.md sweep showed
    attn_2l stuck at 0.026 while attn_4l reached 0.976 at 1600 steps —
    that looked like a convergence issue, not an architectural limit.
  * Extended distance set {4, 8, 16, 32, 64} (screening uses {4,8,16,32}).
    The distance=64 eval requires seq_len ≥ 128 which all investigation-
    tier candidates satisfy.
  * Dedicated protocol-versioned columns so scoring can swap v1→v2 when
    both are present.

This module does NOT modify the screening probe. Integration steps match
induction_probe_v2_investigation.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch.nn as nn

from ._probe_runtime import disable_native_probe_dispatch
from .binding_curriculum import curriculum_binding_range_profile

logger = logging.getLogger(__name__)

BINDING_V2_PROTOCOL_VERSION = "binding_investigation_extended_v1"
BINDING_V2_DISTANCES: Tuple[int, ...] = (4, 8, 16, 32, 64)
BINDING_V2_TRAIN_STEPS = 2400
BINDING_V2_EVAL_EXAMPLES = 200
BINDING_V2_TRAIN_SEQ_LEN = 128
BINDING_V2_EVAL_SEQ_LEN = 128
BINDING_V2_LR = 3e-4
BINDING_V2_TIMEOUT_S = 240.0
BINDING_V2_SEEDS: Tuple[int, ...] = (11, 23, 47)


@dataclass(slots=True)
class BindingV2Result:
    """Result from the v2 investigation-tier binding probe."""

    auc: float = 0.0
    max_distance_acc: float = 0.0
    distance_accuracies: Dict[int, float] | None = None
    train_steps: int = 0
    status: str = "ok"
    elapsed_ms: float = 0.0
    protocol_version: str = BINDING_V2_PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "binding_v2_investigation_auc": self.auc,
            "binding_v2_investigation_max_distance_acc": self.max_distance_acc,
            "binding_v2_investigation_distance_accuracies": self.distance_accuracies,
            "binding_v2_investigation_train_steps": self.train_steps,
            "binding_v2_investigation_status": self.status,
            "binding_v2_investigation_elapsed_ms": self.elapsed_ms,
            "binding_v2_investigation_protocol_version": self.protocol_version,
        }


def _run_binding_v2_single_seed(
    model: nn.Module,
    *,
    distances: Tuple[int, ...] = BINDING_V2_DISTANCES,
    n_train_steps: int = BINDING_V2_TRAIN_STEPS,
    n_eval: int = BINDING_V2_EVAL_EXAMPLES,
    train_seq_len: int = BINDING_V2_TRAIN_SEQ_LEN,
    eval_seq_len: int = BINDING_V2_EVAL_SEQ_LEN,
    lr: float = BINDING_V2_LR,
    device: str = "cuda",
    seed: int | None = None,
) -> BindingV2Result:
    """Single-seed binding v2 probe. Prefer
    :func:`run_binding_v2_investigation` which takes the median across seeds
    to avoid occasional optimizer-state-death (attn_2l@2400 seeds: 4 of 5
    reached 0.995 but one seed flatlined at 0.005; see
    `tasks/probe_calibration_results/variance_summary.md`, 2026-04-18).
    """
    t0 = time.perf_counter()

    with disable_native_probe_dispatch(model, device=device):
        raw = curriculum_binding_range_profile(
            model,
            distances=distances,
            n_train_steps=n_train_steps,
            n_eval=n_eval,
            train_seq_len=train_seq_len,
            eval_seq_len=eval_seq_len,
            lr=lr,
            device=device,
            seed=seed,
        )

    dist_accs = dict(raw.distance_accuracies or {})
    vals = list(dist_accs.values())
    auc = round(sum(vals) / len(vals), 4) if vals else 0.0
    peak = round(max(vals), 4) if vals else 0.0

    return BindingV2Result(
        auc=auc,
        max_distance_acc=peak,
        distance_accuracies=dist_accs,
        train_steps=int(raw.train_steps),
        status=str(raw.status),
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


def run_binding_v2_investigation(
    model: nn.Module,
    *,
    seeds: Tuple[int, ...] = BINDING_V2_SEEDS,
    distances: Tuple[int, ...] = BINDING_V2_DISTANCES,
    n_train_steps: int = BINDING_V2_TRAIN_STEPS,
    n_eval: int = BINDING_V2_EVAL_EXAMPLES,
    train_seq_len: int = BINDING_V2_TRAIN_SEQ_LEN,
    eval_seq_len: int = BINDING_V2_EVAL_SEQ_LEN,
    lr: float = BINDING_V2_LR,
    device: str = "cuda",
) -> BindingV2Result:
    """Median-of-N-seeds binding v2 probe (public API).

    Runs :func:`_run_binding_v2_single_seed` once per seed and returns the
    result from the seed whose AUC is the median. Protects against the
    occasional dead-optimizer seed at the capability frontier (see
    `tasks/probe_calibration_results/variance_summary.md`, 2026-04-18).
    """
    t0 = time.perf_counter()
    runs: list[BindingV2Result] = []
    for seed in seeds:
        r = _run_binding_v2_single_seed(
            model,
            distances=distances,
            n_train_steps=n_train_steps,
            n_eval=n_eval,
            train_seq_len=train_seq_len,
            eval_seq_len=eval_seq_len,
            lr=lr,
            device=device,
            seed=int(seed),
        )
        runs.append(r)

    runs.sort(key=lambda r: r.auc)
    median = runs[len(runs) // 2]
    median.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return median
