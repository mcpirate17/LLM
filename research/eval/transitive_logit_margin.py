"""Continuous Logit Margin on Transitive Closure.

Trains a deep-copied probe model briefly on synthetic transitive-relation
triples ``A is X . X is Y . A is Y`` and tracks the logit margin —

    M(step) = logit(target_token) − mean(logit over candidate non-target tokens))

— at every training step. The slope of M across steps is the metric. A
reasoning-capable architecture begins a monotonic upward trend almost
immediately; a bottlenecked architecture keeps M flat or noisy.

Per Gemini's spec, even when the target is not yet the argmax (binary
accuracy = 0 %), the *trend* in M discriminates architectures cheaply at
small budgets. This is therefore the inverse of the existing v2 binding
AUC — both measure relational composition, but binding requires the model
to actually win the argmax while logit margin tracks early progress.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._probe_runtime import disable_native_probe_dispatch
from ._trajectory_datasets import (
    TRANSITIVE_ENTITY_RANGE,
    TRANSITIVE_TARGET_INDEX,
    transitive_triples,
)

logger = logging.getLogger(__name__)


@dataclass
class LogitMarginResult:
    velocity: Optional[float] = (
        None  # slope of margin vs training step (positive is good)
    )
    initial_margin: Optional[float] = None  # margin at step 0
    final_margin: Optional[float] = None  # margin at last step
    delta_margin: Optional[float] = None  # final - initial
    n_steps: Optional[int] = None
    margin_trajectory: List[float] = field(default_factory=list)  # length n_steps+1
    status: str = "init"
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, float | int | str | None]:
        # Keep the per-step trajectory out of the column dict; it's
        # auxiliary diagnostic data. Only the slope and bounds go to DB.
        return {
            "fp_logit_margin_velocity": self.velocity,
            "fp_logit_margin_initial": self.initial_margin,
            "fp_logit_margin_final": self.final_margin,
            "fp_logit_margin_delta": self.delta_margin,
            "fp_logit_margin_n_steps": self.n_steps,
            "fp_logit_margin_status": self.status,
            "fp_logit_margin_elapsed_ms": self.elapsed_ms,
        }


def _measure_margin_at_target(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> float:
    """Mean over batch of logit(target) − mean(logit over non-target candidates).

    Candidates are all entity tokens in the transitive vocabulary. The
    target is the correct Y entity for each row. ``inputs`` carries the
    full sequence including the target token at TRANSITIVE_TARGET_INDEX
    (so it's a teacher-forcing setup); the prediction we score is at
    position ``TRANSITIVE_TARGET_INDEX − 1``.
    """
    with torch.no_grad():
        logits = model(inputs)  # (B, S, V)
    # Position to score is the one *before* the target token, since the
    # model is asked to predict the next token from the prefix.
    pred_pos = TRANSITIVE_TARGET_INDEX - 1
    logits_at_pred = logits[:, pred_pos, :]  # (B, V)

    lo, hi = TRANSITIVE_ENTITY_RANGE
    candidate_logits = logits_at_pred[:, lo:hi]  # (B, n_entities)
    target_in_pool = targets - lo  # (B,) target index inside candidate pool

    target_logit = candidate_logits.gather(1, target_in_pool.unsqueeze(1)).squeeze(
        1
    )  # (B,)

    # Mean over non-target candidates: subtract the target contribution.
    candidate_sum = candidate_logits.sum(dim=1)
    n_candidates = candidate_logits.shape[1]
    mean_non_target = (candidate_sum - target_logit) / max(n_candidates - 1, 1)

    return float((target_logit - mean_non_target).mean().item())


def _train_and_track_margin(
    probe: nn.Module,
    *,
    eval_inputs: torch.Tensor,
    eval_targets: torch.Tensor,
    gen: torch.Generator,
    n_train_steps: int,
    batch_size: int,
    lr: float,
    dev: torch.device,
) -> tuple[List[float], str]:
    """Run probe training and collect per-step margin trajectory.

    Returns ``(trajectory, status)`` — ``status`` is ``"ok"`` for a clean
    run or ``"diverged_at_step_N"`` for a NaN/inf loss.
    """
    initial_margin = _measure_margin_at_target(probe, eval_inputs, eval_targets)
    trajectory: List[float] = [initial_margin]

    probe.train()
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=lr,
        foreach=False,
        fused=False,
    )

    status = "ok"
    for step in range(n_train_steps):
        inputs, targets = transitive_triples(
            batch_size=batch_size, device=dev, generator=gen
        )
        logits = probe(inputs)
        pred_pos = TRANSITIVE_TARGET_INDEX - 1
        loss = F.cross_entropy(logits[:, pred_pos, :], targets)
        if not torch.isfinite(loss):
            status = f"diverged_at_step_{step}"
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            margin = _measure_margin_at_target(probe, eval_inputs, eval_targets)
        trajectory.append(margin)
        probe.train()

    return trajectory, status


def _fit_margin_slope(trajectory: List[float], dev: torch.device) -> float:
    """Linear-regression slope of margin trajectory vs step index."""
    xs = torch.arange(len(trajectory), dtype=torch.float32, device=dev)
    ys = torch.tensor(trajectory, dtype=torch.float32, device=dev)
    mean_x = xs.mean()
    mean_y = ys.mean()
    cov = ((xs - mean_x) * (ys - mean_y)).sum()
    var_x = ((xs - mean_x) ** 2).sum().clamp_min(1e-12)
    return float((cov / var_x).item())


def compute_transitive_logit_margin(
    model: nn.Module,
    *,
    n_train_steps: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | torch.device = "cuda",
    seed: int = 4321,
) -> LogitMarginResult:
    """Train a deep-copied probe on transitive triples and return margin slope.

    The supplied ``model`` is not modified — we deepcopy it so the
    fingerprint pipeline can safely call this at any point in the
    model's lifecycle. ``n_train_steps`` controls how long we observe
    the margin trajectory; default 60 is enough to see the early-trend
    signal Gemini described while keeping per-architecture cost ≤ ~3 s
    on CUDA.

    Args:
        model: ``SynthesizedModel``-compatible module that emits
            next-token logits.
        n_train_steps: number of optimizer steps to run.
        batch_size: triples per step.
        lr: optimizer learning rate. Kept high so margin moves visibly
            at small step counts.
        device: cuda or cpu.
        seed: rng seed for reproducible triple sampling.
    """
    result = LogitMarginResult(n_steps=n_train_steps, status="failed")
    t0 = time.perf_counter()

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    device_str = str(dev)

    if n_train_steps < 4:
        result.status = "n_train_steps_too_small"
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    try:
        probe = copy.deepcopy(model).to(dev)
    except RuntimeError as exc:
        logger.warning("Logit-margin deepcopy failed: %s", exc)
        result.status = f"deepcopy_failed: {exc.__class__.__name__}"
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    try:
        gen = torch.Generator(device=dev).manual_seed(int(seed))

        # Pre-generate eval batch once so margin is measured against a
        # stable distribution across steps. Training batches are fresh
        # each step.
        eval_inputs, eval_targets = transitive_triples(
            batch_size=128, device=dev, generator=gen
        )

        probe.eval()
        with disable_native_probe_dispatch(probe, device=device_str):
            trajectory, status = _train_and_track_margin(
                probe,
                eval_inputs=eval_inputs,
                eval_targets=eval_targets,
                gen=gen,
                n_train_steps=n_train_steps,
                batch_size=batch_size,
                lr=lr,
                dev=dev,
            )
            result.status = status

        if len(trajectory) >= 4:
            result.velocity = _fit_margin_slope(trajectory, dev)
            result.initial_margin = trajectory[0]
            result.final_margin = trajectory[-1]
            result.delta_margin = trajectory[-1] - trajectory[0]
            result.margin_trajectory = trajectory
            result.n_steps = len(trajectory) - 1
        elif result.status == "ok":
            result.status = "trajectory_too_short"
    finally:
        del probe
        if device_str.startswith("cuda"):
            torch.cuda.empty_cache()
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return result
