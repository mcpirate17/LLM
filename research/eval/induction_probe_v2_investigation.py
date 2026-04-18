"""Investigation-tier induction probe (v2, 2026-04-17).

Drop-in addition that lives *alongside* the production screening-tier probe
(`research.eval.induction_probe`, `tasks.induction_native_probe.fast_induction_probe`).

Key difference from the screening-tier probe:
  * Training cycles through the eval gap set instead of fixing gap=8. The
    production regime causes well-architected models to over-specialize
    (spike at the trained gap, random elsewhere) — see
    `PROBE_CALIBRATION_2026-04-17.md` for empirical measurements. Mixed-gap
    training lets the model learn the abstract induction-head mechanism
    rather than a per-position shortcut.
  * Only runs at investigation tier (post-screening). ~hundreds of runs per
    day vs screening's 10K+, so we can afford the extra signal without
    backfilling history.
  * Writes to a dedicated column (`induction_v2_investigation_auc`) so it
    can be scored independently of the screening-tier column.

This module does NOT modify the production probe. Integration:
  1. Add `induction_v2_investigation_auc REAL` to the program_results schema.
  2. In _eval_registry.py, wire a new EvalSpec that calls
     :func:`run_induction_v2_investigation` at investigation tier.
  3. In leaderboard_scoring.py, add the new kwarg and a small subscore
     (suggested: 15 pts S-curve centered on 0.7).
  4. No change to screening-tier scoring; pre-fix rows get `None` and skip
     the new subscore gracefully.
"""

from __future__ import annotations

import copy
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._probe_runtime import disable_native_probe_dispatch
from .induction_probe import _generate_induction_batch  # reuse the batch gen
from .utils import make_adamw

logger = logging.getLogger(__name__)

# Protocol constants — bump the version string when you change any of these.
INDUCTION_V2_PROTOCOL_VERSION = "induction_investigation_mixed_v1"
INDUCTION_V2_GAPS: Tuple[int, ...] = (4, 8, 16, 32, 64)
INDUCTION_V2_TRAIN_STEPS = 500
INDUCTION_V2_EVAL_EXAMPLES = 200
INDUCTION_V2_BATCH_SIZE = 32
INDUCTION_V2_LR = 1e-3
INDUCTION_V2_TIMEOUT_S = 120.0
INDUCTION_V2_SEEDS: Tuple[int, ...] = (11, 23, 47)
_RESTRICTED_VOCAB = 256


def _amp_context(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@dataclass(slots=True)
class InductionV2Result:
    """Result from the v2 investigation-tier induction probe."""

    auc: float = 0.0
    max_gap_acc: float = 0.0  # peak per-gap accuracy — useful to distinguish
    # "learned one gap perfectly" from "learned all gaps partially"
    gap_accuracies: Dict[int, float] | None = None
    steps_trained: int = 0
    status: str = "ok"
    elapsed_ms: float = 0.0
    protocol_version: str = INDUCTION_V2_PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "induction_v2_investigation_auc": self.auc,
            "induction_v2_investigation_max_gap_acc": self.max_gap_acc,
            "induction_v2_investigation_gap_accuracies": self.gap_accuracies,
            "induction_v2_investigation_steps_trained": self.steps_trained,
            "induction_v2_investigation_status": self.status,
            "induction_v2_investigation_elapsed_ms": self.elapsed_ms,
            "induction_v2_investigation_protocol_version": self.protocol_version,
        }


def _run_induction_v2_single_seed(
    model: nn.Module,
    *,
    gaps: Tuple[int, ...] = INDUCTION_V2_GAPS,
    n_train_steps: int = INDUCTION_V2_TRAIN_STEPS,
    n_eval: int = INDUCTION_V2_EVAL_EXAMPLES,
    batch_size: int = INDUCTION_V2_BATCH_SIZE,
    lr: float = INDUCTION_V2_LR,
    device: str = "cuda",
    timeout_s: float = INDUCTION_V2_TIMEOUT_S,
    seed: int | None = None,
) -> InductionV2Result:
    """Single-seed induction v2 probe. Prefer
    :func:`run_induction_v2_investigation` which takes the median across
    seeds to avoid single-seed convergence failures at the mechanism-forming
    threshold (shallow attention models are seed-sensitive — see
    `tasks/probe_calibration_results/variance_summary.md`, 2026-04-18).
    """
    t0 = time.perf_counter()
    result = InductionV2Result(gap_accuracies={})
    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    try:
        probe_model = copy.deepcopy(model).to(device)
        probe_model.train()
    except Exception as exc:
        result.status = f"copy_failed: {exc}"
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    opt = make_adamw(probe_model.parameters(), lr=lr)

    try:
        with disable_native_probe_dispatch(probe_model, device=device):
            for step in range(1, n_train_steps + 1):
                if time.perf_counter() - t0 > timeout_s:
                    result.status = "timeout"
                    break
                # The critical difference vs production: cycle gaps rather
                # than fix gap=8. This makes the model learn the abstract
                # "find previous occurrence of same token" pattern instead
                # of a position-8-specific shortcut.
                g = gaps[step % len(gaps)]
                input_ids, targets = _generate_induction_batch(
                    batch_size, g, device, generator=generator
                )
                opt.zero_grad(set_to_none=True)
                with _amp_context(device):
                    logits = probe_model(input_ids)
                    pred_logits = logits[:, input_ids.shape[1] - 1, :_RESTRICTED_VOCAB]
                    loss = F.cross_entropy(pred_logits.float(), targets)
                if not torch.isfinite(loss):
                    result.status = "diverged"
                    break
                loss.backward()
                nn.utils.clip_grad_norm_(probe_model.parameters(), 1.0)
                opt.step()
                result.steps_trained = step

            probe_model.eval()
            with torch.inference_mode():
                for gap in sorted(gaps):
                    if time.perf_counter() - t0 > timeout_s:
                        result.gap_accuracies[gap] = 0.0
                        continue
                    correct = 0
                    total = 0
                    remaining = n_eval
                    while remaining > 0:
                        bs = min(batch_size, remaining)
                        inp, tgt = _generate_induction_batch(
                            bs, gap, device, generator=generator
                        )
                        out = probe_model(inp)
                        preds = out[:, inp.shape[1] - 1, :_RESTRICTED_VOCAB].argmax(-1)
                        correct += (preds == tgt).sum().item()
                        total += tgt.numel()
                        remaining -= bs
                    result.gap_accuracies[gap] = round(correct / max(total, 1), 4)
    except Exception as exc:
        result.status = f"train_failed: {exc}"
    finally:
        del probe_model
        if device == "cuda":
            torch.cuda.empty_cache()

    if result.gap_accuracies:
        vals = list(result.gap_accuracies.values())
        result.auc = round(sum(vals) / len(vals), 4)
        result.max_gap_acc = round(max(vals), 4)

    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result


def run_induction_v2_investigation(
    model: nn.Module,
    *,
    seeds: Tuple[int, ...] = INDUCTION_V2_SEEDS,
    gaps: Tuple[int, ...] = INDUCTION_V2_GAPS,
    n_train_steps: int = INDUCTION_V2_TRAIN_STEPS,
    n_eval: int = INDUCTION_V2_EVAL_EXAMPLES,
    batch_size: int = INDUCTION_V2_BATCH_SIZE,
    lr: float = INDUCTION_V2_LR,
    device: str = "cuda",
    timeout_s: float = INDUCTION_V2_TIMEOUT_S,
) -> InductionV2Result:
    """Median-of-N-seeds induction v2 probe (public API).

    Runs :func:`_run_induction_v2_single_seed` once per seed and returns the
    result from the seed whose AUC is the median. Protects against
    occasional convergence failures at the capability frontier (e.g.
    1-layer attention at seed=11 → 0.044 while other seeds ≥ 0.82 in the
    Phase 1 variance sweep). The median pick keeps gap-level breakdowns
    consistent with the reported AUC (vs averaging, which would mix
    incompatible gap-accuracy dictionaries).
    """
    t0 = time.perf_counter()
    runs: list[InductionV2Result] = []
    for seed in seeds:
        r = _run_induction_v2_single_seed(
            model,
            gaps=gaps,
            n_train_steps=n_train_steps,
            n_eval=n_eval,
            batch_size=batch_size,
            lr=lr,
            device=device,
            timeout_s=timeout_s,
            seed=int(seed),
        )
        runs.append(r)

    runs.sort(key=lambda r: r.auc)
    median = runs[len(runs) // 2]
    median.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return median
