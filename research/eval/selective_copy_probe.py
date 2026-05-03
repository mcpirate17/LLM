"""Selective copy — measures ability to select the most recent value after a marker.

Tests Mamba-style state selectivity: the model must ignore irrelevant filler and
copy the value that followed the LAST occurrence of MARKER, even when multiple
MARKER occurrences appear earlier in the sequence.

Sequence format:
    [filler...] [MARKER][v1] [filler...] [MARKER][v2] [filler...]
    [MARKER][v3] [filler...] [QUERY_MARKER]
Target: v3 (the value that followed the *most recent* MARKER before QUERY).

Distractors (earlier MARKER occurrences with different values) test selectivity:
attention with positional bias and SSMs with selective state should both succeed;
pure linear attention typically averages across markers and gets it wrong.

Output column: ``selective_copy_score``  (mean accuracy across 3 distractor counts).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .retrieval_eval_utils import (
    run_retrieval_probe_config,
)

logger = logging.getLogger(__name__)

_TIMEOUT_S = 90.0
_MARKER = 4
_QUERY_MARKER = 5
_FILL_LO = 100
_FILL_HI = 356  # 256 possible values
_FILL_N = _FILL_HI - _FILL_LO
_CHANCE = 1.0 / _FILL_N
_PASS_THRESHOLD = 3 * _CHANCE
_DEFAULT_DISTRACTORS = (1, 2, 3)
_DEFAULT_SEQ_LEN = 256
_DEFAULT_BATCH = 16
_DEFAULT_TRAIN_STEPS = 200
_DEFAULT_EVAL_BATCHES = 8
_DEFAULT_LR = 1e-3


@dataclass(slots=True)
class SelectiveCopyResult:
    score: float = 0.0
    per_distractor_count: Dict[int, float] = field(default_factory=dict)
    distractor_counts_passed: int = 0
    distractor_counts_tested: int = 0
    elapsed_ms: float = 0.0
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selective_copy_score": self.score,
            "selective_copy_per_distractor": self.per_distractor_count,
            "selective_copy_passed": self.distractor_counts_passed,
            "selective_copy_tested": self.distractor_counts_tested,
            "selective_copy_elapsed_ms": self.elapsed_ms,
            "selective_copy_status": self.status,
        }


def _generate_batch(
    batch_size: int,
    seq_len: int,
    n_distractors: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Place ``n_distractors+1`` (MARKER, value) pairs at random positions; the
    LAST pair's value is the target. QUERY_MARKER is at seq_len-1.
    """
    if 2 * (n_distractors + 1) + 2 > seq_len:
        raise ValueError(
            f"seq_len={seq_len} too short for n_distractors={n_distractors}"
        )

    input_ids = torch.randint(
        _FILL_LO, _FILL_HI, (batch_size, seq_len), dtype=torch.long, device=device
    )

    # For each example, pick n_distractors+1 marker positions in increasing order.
    n_pairs = n_distractors + 1
    targets = torch.empty(batch_size, dtype=torch.long, device=device)
    last_marker_pos = seq_len - 4  # leave room for value + filler + query
    for b in range(batch_size):
        # Sample n_pairs positions in [1, last_marker_pos], ascending, with gaps≥2
        slots = torch.randperm(last_marker_pos - 2, device="cpu")[:n_pairs] + 1
        slots, _ = torch.sort(slots)
        # Force minimum spacing of 2 (so MARKER and value don't overlap)
        for i in range(1, n_pairs):
            if int(slots[i]) - int(slots[i - 1]) < 2:
                slots[i] = slots[i - 1] + 2
        slots = slots.clamp(max=last_marker_pos)
        for i in range(n_pairs):
            pos = int(slots[i])
            v = int(torch.randint(_FILL_LO, _FILL_HI, (1,)).item())
            input_ids[b, pos] = _MARKER
            input_ids[b, pos + 1] = v
            if i == n_pairs - 1:
                targets[b] = v
    input_ids[:, -1] = _QUERY_MARKER
    return input_ids, targets


def selective_copy_score(
    model: nn.Module,
    *,
    distractor_counts: Tuple[int, ...] = _DEFAULT_DISTRACTORS,
    seq_len: int = _DEFAULT_SEQ_LEN,
    n_train_steps: int = _DEFAULT_TRAIN_STEPS,
    n_eval_batches: int = _DEFAULT_EVAL_BATCHES,
    batch_size: int = _DEFAULT_BATCH,
    lr: float = _DEFAULT_LR,
    device: str = "cuda",
    timeout_s: float = _TIMEOUT_S,
) -> SelectiveCopyResult:
    """Train and evaluate selective-copy at varying distractor counts.

    Score = fraction of distractor-counts where accuracy > 3× chance baseline.
    """
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    per: Dict[int, float] = {}
    passed = 0

    for n_dist in distractor_counts:
        if time.perf_counter() > deadline:
            break

        def make_train_batch(bs: int, dev: str) -> Tuple[torch.Tensor, torch.Tensor]:
            return _generate_batch(bs, seq_len, n_dist, dev)

        eval_ids, eval_targets = _generate_batch(
            n_eval_batches * batch_size, seq_len, n_dist, device
        )

        try:
            acc, _timed = run_retrieval_probe_config(
                model,
                n_train_steps=n_train_steps,
                eval_ids=eval_ids,
                eval_targets=eval_targets,
                batch_size=batch_size,
                lr=lr,
                device=device,
                deadline=deadline,
                make_train_batch=make_train_batch,
                query_pos=seq_len - 1,
                vocab_lo=_FILL_LO,
                vocab_hi=_FILL_HI,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "selective_copy n_dist=%d failed: %s", n_dist, exc, exc_info=False
            )
            per[n_dist] = 0.0
            continue

        per[n_dist] = round(float(acc), 4)
        if acc > _PASS_THRESHOLD:
            passed += 1

    n_tested = len(per)
    score = passed / max(n_tested, 1) if n_tested > 0 else 0.0
    return SelectiveCopyResult(
        score=round(float(score), 4),
        per_distractor_count=per,
        distractor_counts_passed=passed,
        distractor_counts_tested=n_tested,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        status="ok",
    )
