"""Investigation-tier binding probe.

Drop-in addition that lives alongside the production screening-tier binding
probe (`research.eval.binding_curriculum.curriculum_binding_range_profile`).

Differences from screening-tier:
  * Longer training budget (2400 steps vs 400/800) so slow-converging
    architectures reach their capability ceiling (attn_2l at 1600 steps
    was stuck at 0.026 while attn_4l hit 0.976).
  * Extended distance set {4, 8, 16, 32, 64} (screening uses {4,8,16,32}).
  * Dedicated protocol-versioned columns so scoring can swap v1→v2 when
    both are present.
  * Median-of-3 seeds — single-seed fails ~1-in-5 at the capability
    frontier (dead-optimizer seeds).

Performance:
  * Pre-generate all 2400 training batches up-front in a single
    vectorized randint — eliminates per-step dispatch overhead.
  * Single deepcopy per fingerprint + state_dict reload for seeds 2/3
    (cuts deepcopy cost by ~65%).
  * Seq_len is fixed at 128 for both train and eval regardless of
    distance, so the forward graph is shape-invariant. torch.compile
    works cleanly here (gated on ``ARIA_PROBE_COMPILE``) — but remains
    opt-in because the IR executor has Python-side bookkeeping that
    still triggers recompiles on model families that use the
    rich-telemetry path.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._probe_runtime import disable_native_probe_dispatch
from .utils import clip_grad_norm

logger = logging.getLogger(__name__)

BINDING_V2_PROTOCOL_VERSION = "binding_investigation_extended_v2"
BINDING_V2_DISTANCES: Tuple[int, ...] = (4, 8, 16, 32, 64)
BINDING_V2_TRAIN_STEPS = 2400
BINDING_V2_EVAL_EXAMPLES = 200
BINDING_V2_TRAIN_SEQ_LEN = 128
BINDING_V2_EVAL_SEQ_LEN = 128
BINDING_V2_TRAIN_BATCH_SIZE = 16
BINDING_V2_EVAL_BATCH_SIZE = 32
BINDING_V2_LR = 3e-4
BINDING_V2_TIMEOUT_S = 240.0
BINDING_V2_SEEDS: Tuple[int, ...] = (11, 23, 47)


def _maybe_compile(model: nn.Module) -> nn.Module:
    """Optionally wrap with ``torch.compile``.

    Shape-invariant here (seq_len=128 always), so compile *could* amortize
    across the 2400 training steps. Gated behind ``ARIA_PROBE_COMPILE``
    because some IR-executor graphs still trigger Python-side side-effect
    recompiles that erase the gain.
    """
    import os as _os

    if _os.environ.get("ARIA_PROBE_COMPILE", "") != "1":
        return model
    if not torch.cuda.is_available():
        return model
    try:
        return torch.compile(model, mode="default", dynamic=False, fullgraph=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch.compile unavailable for binding probe model: %s", exc)
        return model


def _generate_copy_batches_bulk(
    n_batches: int,
    batch_size: int,
    seq_len: int,
    distance: int,
    vocab_size: int,
    device: str,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Vectorized bulk generator for copy-at-distance sequences.

    Shape: (n_batches, batch_size, seq_len). Each row is a repeated seed
    of length ``distance`` tiled to fill ``seq_len`` — same as the v1
    probe but built in one kernel dispatch for the whole training budget.
    """
    # Seed tokens, one per example
    seeds = torch.randint(
        1,
        vocab_size,
        (n_batches, batch_size, distance),
        device=device,
        generator=generator,
    )
    n_rep = (seq_len + distance - 1) // distance
    return seeds.repeat(1, 1, n_rep)[:, :, :seq_len].contiguous()


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
            "binding_intermediate_auc": self.auc,
            "binding_intermediate_max_distance_acc": self.max_distance_acc,
            "binding_intermediate_distance_accuracies": self.distance_accuracies,
            "binding_intermediate_train_steps": self.train_steps,
            "binding_intermediate_status": self.status,
            "binding_intermediate_elapsed_ms": self.elapsed_ms,
            "binding_intermediate_protocol_version": self.protocol_version,
        }


@torch.inference_mode()
def _eval_distances_bulk(
    model: nn.Module,
    *,
    distances: Tuple[int, ...],
    n_eval: int,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    device: str,
    generator: torch.Generator | None,
) -> Dict[int, float]:
    accs: Dict[int, float] = {}
    for distance in distances:
        if distance <= 0 or distance + 1 >= seq_len:
            accs[int(distance)] = 0.0
            continue
        n_batches = (n_eval + batch_size - 1) // batch_size
        batches = _generate_copy_batches_bulk(
            n_batches, batch_size, seq_len, distance, vocab_size, device, generator
        )
        correct = 0
        total = 0
        seen = 0
        for b in range(n_batches):
            if seen >= n_eval:
                break
            take = min(batch_size, n_eval - seen)
            batch = batches[b, :take]
            logits = model(batch)
            preds = logits[:, distance - 1 : seq_len - 1, :vocab_size].argmax(dim=-1)
            targets = batch[:, distance:seq_len]
            correct += preds.eq(targets).sum().item()
            total += targets.numel()
            seen += take
        accs[int(distance)] = round(correct / max(total, 1), 4)
    return accs


def _run_binding_intermediate_on(
    probe_model: nn.Module,
    *,
    distances: Tuple[int, ...],
    n_train_steps: int,
    n_eval: int,
    train_seq_len: int,
    eval_seq_len: int,
    train_batch_size: int,
    eval_batch_size: int,
    lr: float,
    device: str,
    timeout_s: float,
    generator: torch.Generator | None,
) -> BindingV2Result:
    """Run the probe training+eval on an already-prepared ``probe_model``."""
    t0 = time.perf_counter()
    result = BindingV2Result(distance_accuracies={})
    valid_distances = _valid_binding_distances(distances, train_seq_len)
    if not valid_distances:
        result.status = "no_valid_distances"
        result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result
    n_dists = len(valid_distances)

    vocab_size = int(getattr(probe_model, "vocab_size", 256) or 256)
    probe_model.train()
    compiled = _maybe_compile(probe_model)

    # Pre-generate all training batches per distance up-front.
    pre_train = _pre_generate_binding_train_batches(
        valid_distances,
        n_train_steps,
        train_batch_size,
        train_seq_len,
        vocab_size,
        device,
        generator,
    )
    cursor = {d: 0 for d in valid_distances}

    optimizer = torch.optim.AdamW(
        probe_model.parameters(),
        lr=lr,
        foreach=False,
        fused=False,
    )

    try:
        with disable_native_probe_dispatch(probe_model, device=device):
            for step in range(n_train_steps):
                if time.perf_counter() - t0 > timeout_s:
                    result.status = "timeout"
                    break
                distance = valid_distances[step % n_dists]
                batch = pre_train[distance][cursor[distance]]
                cursor[distance] += 1

                loss = _binding_train_loss(
                    compiled,
                    batch,
                    distance=distance,
                    train_seq_len=train_seq_len,
                    vocab_size=vocab_size,
                )
                if not torch.isfinite(loss):
                    result.status = "diverged"
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm(probe_model.parameters(), 1.0)
                optimizer.step()
                result.train_steps = step + 1

            probe_model.eval()
            result.distance_accuracies = _eval_distances_bulk(
                compiled,
                distances=valid_distances,
                n_eval=n_eval,
                seq_len=eval_seq_len,
                batch_size=eval_batch_size,
                vocab_size=vocab_size,
                device=device,
                generator=generator,
            )
            vals = list(result.distance_accuracies.values())
            result.auc = round(sum(vals) / len(vals), 4) if vals else 0.0
            result.max_distance_acc = round(max(vals), 4) if vals else 0.0
    except Exception as exc:
        result.status = f"train_failed: {exc}"

    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result


def _valid_binding_distances(
    distances: Tuple[int, ...],
    train_seq_len: int,
) -> Tuple[int, ...]:
    return tuple(int(d) for d in distances if int(d) > 0 and int(d) + 1 < train_seq_len)


def _pre_generate_binding_train_batches(
    valid_distances: Tuple[int, ...],
    n_train_steps: int,
    train_batch_size: int,
    train_seq_len: int,
    vocab_size: int,
    device: str,
    generator: torch.Generator | None,
) -> Dict[int, torch.Tensor]:
    steps_per_dist: Dict[int, int] = {d: 0 for d in valid_distances}
    n_dists = len(valid_distances)
    for step in range(n_train_steps):
        steps_per_dist[valid_distances[step % n_dists]] += 1

    pre_train: Dict[int, torch.Tensor] = {}
    for distance, count in steps_per_dist.items():
        if count > 0:
            pre_train[distance] = _generate_copy_batches_bulk(
                count,
                train_batch_size,
                train_seq_len,
                distance,
                vocab_size,
                device,
                generator,
            )
    return pre_train


def _binding_train_loss(
    compiled_model: nn.Module,
    batch: torch.Tensor,
    *,
    distance: int,
    train_seq_len: int,
    vocab_size: int,
) -> torch.Tensor:
    logits = compiled_model(batch)
    pred_logits = logits[:, distance - 1 : train_seq_len - 1, :vocab_size]
    targets = batch[:, distance:train_seq_len]
    return F.cross_entropy(
        pred_logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )


def _run_binding_intermediate_single_seed(
    model: nn.Module,
    *,
    distances: Tuple[int, ...] = BINDING_V2_DISTANCES,
    n_train_steps: int = BINDING_V2_TRAIN_STEPS,
    n_eval: int = BINDING_V2_EVAL_EXAMPLES,
    train_seq_len: int = BINDING_V2_TRAIN_SEQ_LEN,
    eval_seq_len: int = BINDING_V2_EVAL_SEQ_LEN,
    train_batch_size: int = BINDING_V2_TRAIN_BATCH_SIZE,
    eval_batch_size: int = BINDING_V2_EVAL_BATCH_SIZE,
    lr: float = BINDING_V2_LR,
    device: str = "cuda",
    timeout_s: float = BINDING_V2_TIMEOUT_S,
    seed: int | None = None,
) -> BindingV2Result:
    """Single-seed binding v2 probe. Prefer
    :func:`run_binding_intermediate` which takes the median across
    seeds.
    """
    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    try:
        probe_model = copy.deepcopy(model).to(device)
    except Exception as exc:
        return BindingV2Result(status=f"copy_failed: {exc}", distance_accuracies={})
    try:
        return _run_binding_intermediate_on(
            probe_model,
            distances=distances,
            n_train_steps=n_train_steps,
            n_eval=n_eval,
            train_seq_len=train_seq_len,
            eval_seq_len=eval_seq_len,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            lr=lr,
            device=device,
            timeout_s=timeout_s,
            generator=generator,
        )
    finally:
        del probe_model
        if device == "cuda":
            torch.cuda.empty_cache()


def run_binding_intermediate(
    model: nn.Module,
    *,
    seeds: Tuple[int, ...] = BINDING_V2_SEEDS,
    distances: Tuple[int, ...] = BINDING_V2_DISTANCES,
    n_train_steps: int = BINDING_V2_TRAIN_STEPS,
    n_eval: int = BINDING_V2_EVAL_EXAMPLES,
    train_seq_len: int = BINDING_V2_TRAIN_SEQ_LEN,
    eval_seq_len: int = BINDING_V2_EVAL_SEQ_LEN,
    train_batch_size: int = BINDING_V2_TRAIN_BATCH_SIZE,
    eval_batch_size: int = BINDING_V2_EVAL_BATCH_SIZE,
    lr: float = BINDING_V2_LR,
    device: str = "cuda",
    timeout_s: float = BINDING_V2_TIMEOUT_S,
) -> BindingV2Result:
    """Median-of-N-seeds binding v2 probe (public API).

    One deepcopy, then state_dict reload between seeds.
    """
    t0 = time.perf_counter()
    try:
        probe_model = copy.deepcopy(model).to(device)
    except Exception as exc:
        return BindingV2Result(
            status=f"copy_failed: {exc}",
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            distance_accuracies={},
        )
    init_state = {k: v.detach().clone() for k, v in probe_model.state_dict().items()}

    runs: List[BindingV2Result] = []
    try:
        for idx, seed in enumerate(seeds):
            if idx > 0:
                probe_model.load_state_dict(init_state, strict=False)
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            r = _run_binding_intermediate_on(
                probe_model,
                distances=distances,
                n_train_steps=n_train_steps,
                n_eval=n_eval,
                train_seq_len=train_seq_len,
                eval_seq_len=eval_seq_len,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                lr=lr,
                device=device,
                timeout_s=timeout_s,
                generator=generator,
            )
            runs.append(r)
    finally:
        del probe_model, init_state
        if device == "cuda":
            torch.cuda.empty_cache()

    runs.sort(key=lambda r: r.auc)
    median = runs[len(runs) // 2]
    median.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return median
