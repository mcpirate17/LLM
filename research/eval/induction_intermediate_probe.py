"""Investigation-tier induction probe.

Drop-in addition that lives alongside the production screening-tier probe
(`research.eval.induction_probe`, `tasks.induction_native_probe.fast_induction_probe`).

Differences from the screening-tier probe:
  * Training cycles through the eval gap set instead of fixing gap=8. The
    production regime causes well-architected models to over-specialize
    (spike at the trained gap, random elsewhere).
  * Only runs at investigation tier (post-screening).
  * Writes to dedicated columns (`induction_intermediate_*`).

Performance:
  * Sync-free batch generation: no ``.any()`` collision check — noise is
    sampled from vocab-1 and shift-past-A deterministically, giving a
    uniform-over-non-A distribution without any CPU↔GPU sync.
  * Bulk pre-generation: all training batches are generated in one
    vectorized call per gap before the train loop starts — eliminates
    per-step kernel dispatch overhead (~28% of prior runtime).
  * torch.compile(mode="reduce-overhead") caches the forward graph.
  * Multi-seed probe uses a single deepcopy plus state_dict reload for
    seeds 2/3, cutting per-fingerprint deepcopy cost by ~65%.
"""

from __future__ import annotations

import copy
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._probe_runtime import disable_native_probe_dispatch
from ._probe_utils import _materialize_non_inference_
from .utils import clip_grad_norm, make_adamw

logger = logging.getLogger(__name__)

# Protocol constants — bump the version string when you change any of these.
INDUCTION_V2_PROTOCOL_VERSION = "induction_investigation_mixed_v2"
INDUCTION_V2_GAPS: Tuple[int, ...] = (4, 8, 16, 32, 64)
INDUCTION_V2_TRAIN_STEPS = 500
INDUCTION_V2_EVAL_EXAMPLES = 200
INDUCTION_V2_BATCH_SIZE = 32
INDUCTION_V2_LR = 1e-3
INDUCTION_V2_TIMEOUT_S = 120.0
INDUCTION_V2_SEEDS: Tuple[int, ...] = (11, 23, 47)
_RESTRICTED_VOCAB = 256


def _snapshot_module_tensors(
    module: nn.Module,
) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Capture parameters and buffers for cheap in-place restoration."""
    refs = [*module.parameters(), *module.buffers()]
    with torch.no_grad():
        snapshot = [tensor.detach().clone() for tensor in refs]
    return refs, snapshot


def _restore_module_tensors(
    refs: List[torch.Tensor], snapshot: List[torch.Tensor]
) -> None:
    """Restore parameters and buffers without rebuilding a state_dict."""
    with torch.no_grad():
        for ref, original in zip(refs, snapshot):
            ref.copy_(original)


def _amp_context(device: str):
    return nullcontext()


def _generate_induction_batch_bulk(
    n_batches: int,
    batch_size: int,
    gap: int,
    device: str,
    generator: torch.Generator | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized bulk batch generator for the induction task.

    Produces ``(inputs, targets)`` where:
      inputs[i] shape (batch_size, gap+3): ``[A B n1 n2 ... nG A]``
      targets[i] shape (batch_size,): the B token.

    Sync-free — noise is sampled from ``[1, _RESTRICTED_VOCAB-1)`` and
    shift-past-A so values ≥ A become A+1, yielding a uniform distribution
    over ``_RESTRICTED_VOCAB \\ {A}``. No ``.any()`` sync, no masked
    scatter, no conditional branch that depends on a CUDA tensor.
    """
    seq_len = gap + 3
    # Sample A, B once per (batch, example)
    A = torch.randint(
        1,
        _RESTRICTED_VOCAB,
        (n_batches, batch_size),
        device=device,
        generator=generator,
    )
    B = torch.randint(
        1,
        _RESTRICTED_VOCAB,
        (n_batches, batch_size),
        device=device,
        generator=generator,
    )
    # Noise sampled from the reduced range [1, _RESTRICTED_VOCAB-1). Any
    # token equal-or-greater than A is shifted up by 1, producing a
    # collision-free uniform draw from {1..vocab-1} \ {A}.
    noise_raw = torch.randint(
        1,
        _RESTRICTED_VOCAB - 1,
        (n_batches, batch_size, gap),
        device=device,
        generator=generator,
    )
    # Shift past A: since A ∈ [1, vocab), any raw value ≥ A maps to raw+1.
    noise = noise_raw + (noise_raw >= A.unsqueeze(-1)).to(torch.int64)

    # Assemble [A, B, noise, A]
    inputs = torch.empty(
        (n_batches, batch_size, seq_len), dtype=torch.int64, device=device
    )
    inputs[:, :, 0] = A
    inputs[:, :, 1] = B
    inputs[:, :, 2 : gap + 2] = noise
    inputs[:, :, gap + 2] = A
    return inputs, B


@dataclass(slots=True)
class InductionV2Result:
    """Result from the v2 investigation-tier induction probe."""

    auc: float = 0.0
    max_gap_acc: float = 0.0
    gap_accuracies: Dict[int, float] | None = None
    steps_trained: int = 0
    status: str = "ok"
    elapsed_ms: float = 0.0
    protocol_version: str = INDUCTION_V2_PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "induction_intermediate_auc": self.auc,
            "induction_intermediate_max_gap_acc": self.max_gap_acc,
            "induction_intermediate_gap_accuracies": self.gap_accuracies,
            "induction_intermediate_steps_trained": self.steps_trained,
            "induction_intermediate_status": self.status,
            "induction_intermediate_elapsed_ms": self.elapsed_ms,
            "induction_intermediate_protocol_version": self.protocol_version,
        }


def _maybe_compile(model: nn.Module) -> nn.Module:
    """Optionally wrap with ``torch.compile``.

    Disabled by default: probe training uses 5 distinct seq_lens (one per
    gap: 7/11/19/35/67), and ``torch.compile`` pays a per-shape compile
    cost that does NOT amortize at 500-step training budgets. Measured
    ~23s of compile overhead that the in-loop savings (~0.5ms per step
    kernel-launch reduction × 500 steps = 0.25s) do not recover.

    Can be re-enabled via ``ARIA_PROBE_COMPILE=1`` once we either:
      (a) add a prewarm phase that amortizes compile across a batch of
          fingerprints, or
      (b) pad all inputs to max seq_len so there's a single compiled
          graph.
    """
    import os as _os

    if _os.environ.get("ARIA_PROBE_COMPILE", "") != "1":
        return model
    if not torch.cuda.is_available():
        return model
    try:
        return torch.compile(model, mode="default", dynamic=True, fullgraph=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch.compile unavailable for probe model: %s", exc)
        return model


def _steps_per_gap(gaps: Tuple[int, ...], n_train_steps: int) -> Dict[int, int]:
    n_gaps = len(gaps)
    steps = {gap: 0 for gap in gaps}
    for step in range(n_train_steps):
        steps[gaps[step % n_gaps]] += 1
    return steps


def _pregenerate_training_batches(
    gaps: Tuple[int, ...],
    steps_per_gap: Dict[int, int],
    batch_size: int,
    device: str,
    generator: torch.Generator | None,
) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    inputs: Dict[int, torch.Tensor] = {}
    targets: Dict[int, torch.Tensor] = {}
    for gap in gaps:
        count = steps_per_gap[gap]
        if count <= 0:
            continue
        inp, tgt = _generate_induction_batch_bulk(
            count, batch_size, gap, device, generator
        )
        inputs[gap] = inp
        targets[gap] = tgt
    return inputs, targets


def _evaluate_gap_accuracy(
    compiled: Any,
    *,
    gap: int,
    n_eval: int,
    batch_size: int,
    device: str,
    generator: torch.Generator | None,
) -> float:
    n_batches = (n_eval + batch_size - 1) // batch_size
    eval_inp, eval_tgt = _generate_induction_batch_bulk(
        n_batches, batch_size, gap, device, generator
    )
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    seen = 0
    for batch_idx in range(n_batches):
        if seen >= n_eval:
            break
        take = min(batch_size, n_eval - seen)
        inp = eval_inp[batch_idx, :take]
        tgt = eval_tgt[batch_idx, :take]
        out = compiled(inp)
        preds = out[:, inp.shape[1] - 1, :_RESTRICTED_VOCAB].argmax(-1)
        correct += (preds == tgt).sum()
        total += tgt.numel()
        seen += take
    return round(int(correct.item()) / max(total, 1), 4)


def _run_induction_intermediate_on(
    probe_model: nn.Module,
    *,
    gaps: Tuple[int, ...],
    n_train_steps: int,
    n_eval: int,
    batch_size: int,
    lr: float,
    device: str,
    timeout_s: float,
    generator: torch.Generator | None,
) -> InductionV2Result:
    """Run the probe training+eval on an already-prepared ``probe_model``.

    The caller owns model lifetime (deepcopy / state-reload) so we can do
    one deepcopy per fingerprint and reuse for all seeds.
    """
    t0 = time.perf_counter()
    result = InductionV2Result(gap_accuracies={})
    probe_model.train()
    # Fused AdamW has produced graph-dependent CUDA grouped-tensor failures
    # here, which should not become false zero-capability evidence.
    opt = make_adamw(
        probe_model.parameters(),
        lr=lr,
        fused_if_available=False,
        foreach=False,
    )
    compiled = _maybe_compile(probe_model)

    n_gaps = len(gaps)
    pre_inputs, pre_targets = _pregenerate_training_batches(
        gaps,
        _steps_per_gap(gaps, n_train_steps),
        batch_size,
        device,
        generator,
    )

    # Per-gap cursor to index into pre-generated buffers
    cursor = {g: 0 for g in gaps}

    try:
        with disable_native_probe_dispatch(probe_model, device=device):
            for step in range(1, n_train_steps + 1):
                if time.perf_counter() - t0 > timeout_s:
                    result.status = "timeout"
                    break
                g = gaps[(step - 1) % n_gaps]
                i = cursor[g]
                cursor[g] = i + 1
                input_ids = pre_inputs[g][i]
                targets = pre_targets[g][i]

                opt.zero_grad(set_to_none=True)
                with _amp_context(device):
                    logits = compiled(input_ids)
                    pred_logits = logits[:, input_ids.shape[1] - 1, :_RESTRICTED_VOCAB]
                    loss = F.cross_entropy(pred_logits.float(), targets)
                if not torch.isfinite(loss):
                    result.status = "diverged"
                    break
                loss.backward()
                clip_grad_norm(probe_model.parameters(), 1.0)
                opt.step()
                result.steps_trained = step

            # Eval phase — bulk-generate eval batches per gap.
            probe_model.eval()
            with torch.inference_mode():
                for gap in sorted(gaps):
                    if time.perf_counter() - t0 > timeout_s:
                        result.gap_accuracies[gap] = 0.0
                        continue
                    result.gap_accuracies[gap] = _evaluate_gap_accuracy(
                        compiled,
                        gap=gap,
                        n_eval=n_eval,
                        batch_size=batch_size,
                        device=device,
                        generator=generator,
                    )
    except Exception as exc:
        result.status = f"train_failed: {exc}"

    if result.gap_accuracies:
        vals = list(result.gap_accuracies.values())
        result.auc = round(sum(vals) / len(vals), 4)
        result.max_gap_acc = round(max(vals), 4)

    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result


def _run_induction_intermediate_single_seed(
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
    """Single-seed induction v2 probe.

    Kept for compatibility and targeted debugging. Production callers
    should use :func:`run_induction_intermediate` which takes the
    median across seeds — shallow attention models are seed-sensitive at
    the mechanism-forming threshold (see
    `tasks/probe_calibration_results/variance_summary.md`, 2026-04-18).
    """
    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    try:
        probe_model = copy.deepcopy(model).to(device)
        _materialize_non_inference_(probe_model)
    except Exception as exc:
        return InductionV2Result(
            status=f"copy_failed: {exc}",
            elapsed_ms=0.0,
            gap_accuracies={},
        )
    try:
        return _run_induction_intermediate_on(
            probe_model,
            gaps=gaps,
            n_train_steps=n_train_steps,
            n_eval=n_eval,
            batch_size=batch_size,
            lr=lr,
            device=device,
            timeout_s=timeout_s,
            generator=generator,
        )
    finally:
        del probe_model
        if device == "cuda":
            torch.cuda.empty_cache()


def run_induction_intermediate(
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

    Performance note: rather than deepcopy the model N times, we deepcopy
    once, save the initial state_dict, then for each seed we optimize the
    probe_model in place and restore the state_dict between seeds. This
    is ~10× cheaper than deepcopy for the 10-100M-param probe models.
    """
    t0 = time.perf_counter()
    try:
        probe_model = copy.deepcopy(model).to(device)
        _materialize_non_inference_(probe_model)
    except Exception as exc:
        return InductionV2Result(
            status=f"copy_failed: {exc}",
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            gap_accuracies={},
        )
    # Snapshot initial params/buffers once; restore between seeds in-place
    # instead of rebuilding/loading a full state_dict every time.
    state_refs, init_state = _snapshot_module_tensors(probe_model)

    runs: List[InductionV2Result] = []
    try:
        for idx, seed in enumerate(seeds):
            if idx > 0:
                _restore_module_tensors(state_refs, init_state)
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            r = _run_induction_intermediate_on(
                probe_model,
                gaps=gaps,
                n_train_steps=n_train_steps,
                n_eval=n_eval,
                batch_size=batch_size,
                lr=lr,
                device=device,
                timeout_s=timeout_s,
                generator=generator,
            )
            runs.append(r)
    finally:
        del probe_model, state_refs, init_state
        if device == "cuda":
            torch.cuda.empty_cache()

    runs.sort(key=lambda r: r.auc)
    median = runs[len(runs) // 2]
    median.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return median
