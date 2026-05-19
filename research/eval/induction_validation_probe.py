"""Champion-tier induction probe.

A harder, champion-only associative retrieval probe.  It trains the
candidate on multi-binding retrieval examples paired with contradicted
counterfactual contexts, then scores each gap by requiring both the original and
counterfactual contexts to be answered correctly.  The calibrated 2K default is
short enough that untrained SSM/RWKV controls do not solve the task from
scratch, while trained GPT-2/Retrieval controls separate cleanly.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._probe_runtime import disable_native_probe_dispatch
from ._probe_utils import _materialize_non_inference_
from .induction_intermediate_probe import (
    INDUCTION_V2_EVAL_EXAMPLES,
    INDUCTION_V2_SEEDS,
    _restore_module_tensors,
    _snapshot_module_tensors,
)
from .utils import clip_grad_norm, make_adamw

INDUCTION_VALIDATION_2K_STEPS = 2_000
INDUCTION_VALIDATION_5K_STEPS = 5_000
INDUCTION_VALIDATION_10K_STEPS = 10_000
INDUCTION_VALIDATION_DEFAULT_STEPS = INDUCTION_VALIDATION_2K_STEPS
INDUCTION_VALIDATION_PROTOCOL_VERSION_2K = "induction_validation_full_counterfactual_2k"
INDUCTION_VALIDATION_PROTOCOL_VERSION_5K = "induction_validation_full_counterfactual_5k"
INDUCTION_VALIDATION_PROTOCOL_VERSION_10K = (
    "induction_validation_full_counterfactual_10k"
)
INDUCTION_VALIDATION_TIMEOUT_MIN_S = 300.0
INDUCTION_VALIDATION_TIMEOUT_S_PER_STEP = 0.08
INDUCTION_VALIDATION_PAIRS_PER_EXAMPLE = 4
INDUCTION_VALIDATION_GAPS: Tuple[int, ...] = (4, 8, 16)
INDUCTION_VALIDATION_BATCH_SIZE = 16
INDUCTION_VALIDATION_LR = 3e-4
_KEY_POOL = 127
_VALUE_POOL = 128
_VALUE_OFFSET = 128
_RESTRICTED_VOCAB = _VALUE_OFFSET + _VALUE_POOL


def induction_validation_protocol_version_for_steps(n_train_steps: int) -> str:
    steps = int(n_train_steps)
    if steps == INDUCTION_VALIDATION_2K_STEPS:
        return INDUCTION_VALIDATION_PROTOCOL_VERSION_2K
    if steps == INDUCTION_VALIDATION_5K_STEPS:
        return INDUCTION_VALIDATION_PROTOCOL_VERSION_5K
    if steps == INDUCTION_VALIDATION_10K_STEPS:
        return INDUCTION_VALIDATION_PROTOCOL_VERSION_10K
    raise ValueError(
        "induction_validation only supports explicit champion budgets "
        f"{INDUCTION_VALIDATION_2K_STEPS}, {INDUCTION_VALIDATION_5K_STEPS}, "
        f"and {INDUCTION_VALIDATION_10K_STEPS}; got {steps}"
    )


def select_induction_validation_budget(
    *, extended_budget: bool = False, n_train_steps: int | None = None
) -> tuple[int, str]:
    steps = (
        int(n_train_steps)
        if n_train_steps is not None
        else INDUCTION_VALIDATION_5K_STEPS
        if extended_budget
        else INDUCTION_VALIDATION_DEFAULT_STEPS
    )
    return steps, induction_validation_protocol_version_for_steps(steps)


def _gap_accuracy_cv(gap_accuracies: Dict[int, float] | None) -> float:
    vals = [float(v) for v in (gap_accuracies or {}).values()]
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    if mean == 0.0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    return round(math.sqrt(variance) / abs(mean), 4)


def _freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _infer_model_dim(model: nn.Module) -> int:
    dim = getattr(model, "model_dim", None)
    if dim is not None:
        return int(dim)
    norm = getattr(model, "norm", None)
    shape = getattr(getattr(norm, "weight", None), "shape", None)
    if shape:
        return int(shape[0])
    raise ValueError("model does not expose model_dim or norm.weight")


def _pre_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "_fingerprint_representations"):
        _logits, reps = model._fingerprint_representations(input_ids)
        return reps
    output = model(input_ids)
    if output.ndim != 3:
        raise ValueError("model output must be rank-3 logits or representations")
    return output


def _sample_unique_tokens(
    n_batches: int,
    batch_size: int,
    n_tokens: int,
    pool_size: int,
    *,
    offset: int,
    device: str,
    generator: torch.Generator | None,
) -> torch.Tensor:
    scores = torch.rand(
        (n_batches, batch_size, pool_size),
        device=device,
        generator=generator,
    )
    return scores.argsort(dim=-1)[..., :n_tokens].to(torch.int64) + int(offset)


def _generate_counterfactual_binding_batches(
    n_batches: int,
    batch_size: int,
    gap: int,
    *,
    pairs_per_example: int,
    device: str,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate normal and contradicted multi-binding examples.

    Normal examples look like ``k1 v1 k2 v2 ... filler ... kq -> vq``.
    Counterfactual examples replace the value paired with ``kq`` and require the
    prediction to change.  Keys and values come from disjoint token ranges.
    """

    n_pairs = int(pairs_per_example)
    if n_pairs <= 0 or n_pairs > min(_KEY_POOL, _VALUE_POOL):
        raise ValueError("pairs_per_example must fit within the key/value pools")
    gap_i = max(0, int(gap))
    seq_len = n_pairs * 2 + gap_i + 1
    keys = _sample_unique_tokens(
        n_batches,
        batch_size,
        n_pairs,
        _KEY_POOL,
        offset=1,
        device=device,
        generator=generator,
    )
    values = _sample_unique_tokens(
        n_batches,
        batch_size,
        n_pairs,
        _VALUE_POOL,
        offset=_VALUE_OFFSET,
        device=device,
        generator=generator,
    )
    query_idx = torch.randint(
        0,
        n_pairs,
        (n_batches, batch_size),
        device=device,
        generator=generator,
    )
    batch_idx = torch.arange(n_batches, device=device).view(n_batches, 1)
    row_idx = torch.arange(batch_size, device=device).view(1, batch_size)
    targets = values[batch_idx, row_idx, query_idx]

    inputs = torch.empty(
        (n_batches, batch_size, seq_len),
        dtype=torch.int64,
        device=device,
    )
    inputs[:, :, 0 : n_pairs * 2 : 2] = keys
    inputs[:, :, 1 : n_pairs * 2 : 2] = values
    if gap_i:
        fillers = torch.randint(
            1,
            _RESTRICTED_VOCAB,
            (n_batches, batch_size, gap_i),
            device=device,
            generator=generator,
        )
        inputs[:, :, n_pairs * 2 : n_pairs * 2 + gap_i] = fillers
    inputs[:, :, -1] = keys[batch_idx, row_idx, query_idx]

    cf_inputs = inputs.clone()
    cf_raw = ((targets - _VALUE_OFFSET + 1) % _VALUE_POOL) + _VALUE_OFFSET
    target_value_pos = query_idx * 2 + 1
    cf_inputs[batch_idx, row_idx, target_value_pos] = cf_raw
    return inputs, targets, cf_inputs, cf_raw


def _run_induction_validation_on(
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
    pairs_per_example: int = INDUCTION_VALIDATION_PAIRS_PER_EXAMPLE,
) -> "InductionValidationResult":
    t0 = time.perf_counter()
    result = InductionValidationResult(gap_accuracies={})
    opt = make_adamw(
        probe_model.parameters(),
        lr=lr,
        fused_if_available=False,
        foreach=False,
    )
    n_gaps = len(gaps)
    steps_per_gap: Dict[int, int] = {g: 0 for g in gaps}
    for step in range(n_train_steps):
        steps_per_gap[gaps[step % n_gaps]] += 1

    pre_inputs: Dict[int, torch.Tensor] = {}
    pre_targets: Dict[int, torch.Tensor] = {}
    pre_cf_inputs: Dict[int, torch.Tensor] = {}
    pre_cf_targets: Dict[int, torch.Tensor] = {}
    for gap in gaps:
        cnt = steps_per_gap[gap]
        if cnt <= 0:
            continue
        inp, tgt, cf_inp, cf_tgt = _generate_counterfactual_binding_batches(
            cnt,
            batch_size,
            gap,
            pairs_per_example=pairs_per_example,
            device=device,
            generator=generator,
        )
        pre_inputs[gap] = inp
        pre_targets[gap] = tgt
        pre_cf_inputs[gap] = cf_inp
        pre_cf_targets[gap] = cf_tgt
    cursor = {g: 0 for g in gaps}

    try:
        with disable_native_probe_dispatch(probe_model, device=device):
            probe_model.train()
            for step in range(1, n_train_steps + 1):
                if time.perf_counter() - t0 > timeout_s:
                    result.status = "timeout"
                    break
                gap = gaps[(step - 1) % n_gaps]
                idx = cursor[gap]
                cursor[gap] = idx + 1
                input_ids = pre_inputs[gap][idx]
                targets = pre_targets[gap][idx]
                cf_input_ids = pre_cf_inputs[gap][idx]
                cf_targets = pre_cf_targets[gap][idx]
                opt.zero_grad(set_to_none=True)
                pred_logits = probe_model(input_ids)[
                    :, input_ids.shape[1] - 1, :_RESTRICTED_VOCAB
                ]
                cf_logits = probe_model(cf_input_ids)[
                    :, cf_input_ids.shape[1] - 1, :_RESTRICTED_VOCAB
                ]
                loss = (
                    F.cross_entropy(pred_logits.float(), targets)
                    + F.cross_entropy(cf_logits.float(), cf_targets)
                ) * 0.5
                if not torch.isfinite(loss):
                    result.status = "diverged"
                    break
                loss.backward()
                clip_grad_norm(probe_model.parameters(), 1.0)
                opt.step()
                result.steps_trained = step

            probe_model.eval()
            with torch.inference_mode():
                for gap in sorted(gaps):
                    if time.perf_counter() - t0 > timeout_s:
                        result.gap_accuracies[gap] = 0.0
                        continue
                    n_batches = (n_eval + batch_size - 1) // batch_size
                    eval_inp, eval_tgt, cf_inp, cf_tgt = (
                        _generate_counterfactual_binding_batches(
                            n_batches,
                            batch_size,
                            gap,
                            pairs_per_example=pairs_per_example,
                            device=device,
                            generator=generator,
                        )
                    )
                    normal_correct = torch.zeros((), dtype=torch.long, device=device)
                    cf_correct = torch.zeros((), dtype=torch.long, device=device)
                    total = 0
                    seen = 0
                    for batch in range(n_batches):
                        if seen >= n_eval:
                            break
                        take = min(batch_size, n_eval - seen)
                        inp = eval_inp[batch, :take]
                        tgt = eval_tgt[batch, :take]
                        preds = probe_model(inp)[
                            :, inp.shape[1] - 1, :_RESTRICTED_VOCAB
                        ].argmax(-1)
                        normal_correct += (preds == tgt).sum()
                        c_inp = cf_inp[batch, :take]
                        c_tgt = cf_tgt[batch, :take]
                        c_preds = probe_model(c_inp)[
                            :, c_inp.shape[1] - 1, :_RESTRICTED_VOCAB
                        ].argmax(-1)
                        cf_correct += (c_preds == c_tgt).sum()
                        total += tgt.numel()
                        seen += take
                    normal_n, cf_n = torch.stack([normal_correct, cf_correct]).tolist()
                    normal_acc = normal_n / max(total, 1)
                    cf_acc = cf_n / max(total, 1)
                    result.gap_accuracies[gap] = round(min(normal_acc, cf_acc), 4)
    except Exception as exc:  # noqa: BLE001
        result.status = f"train_failed: {exc}"

    if result.gap_accuracies:
        vals = list(result.gap_accuracies.values())
        result.auc = round(sum(vals) / len(vals), 4)
        result.max_gap_acc = round(max(vals), 4)
    result.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return result


@dataclass(slots=True)
class InductionValidationResult:
    """Result from the champion-tier induction v3 probe."""

    auc: float = 0.0
    max_gap_acc: float = 0.0
    gap_accuracies: Dict[int, float] | None = None
    gap_accuracy_cv: float = 0.0
    steps_trained: int = 0
    status: str = "ok"
    elapsed_ms: float = 0.0
    protocol_version: str = INDUCTION_VALIDATION_PROTOCOL_VERSION_2K

    def to_dict(self) -> Dict[str, Any]:
        return {
            "induction_validation_auc": self.auc,
            "induction_validation_max_gap_acc": self.max_gap_acc,
            "induction_validation_gap_accuracy_cv": self.gap_accuracy_cv,
            "induction_validation_gap_accuracies": self.gap_accuracies,
            "induction_validation_steps_trained": self.steps_trained,
            "induction_validation_status": self.status,
            "induction_validation_elapsed_ms": self.elapsed_ms,
            "induction_validation_protocol_version": self.protocol_version,
        }


def _run_induction_validation_median(
    model: nn.Module,
    *,
    seeds: Tuple[int, ...],
    gaps: Tuple[int, ...],
    n_train_steps: int,
    n_eval: int,
    batch_size: int,
    lr: float,
    device: str,
    timeout_s: float,
) -> InductionValidationResult:
    t0 = time.perf_counter()
    try:
        probe_model = copy.deepcopy(model).to(device)
        _materialize_non_inference_(probe_model)
    except Exception as exc:
        return InductionValidationResult(
            status=f"copy_failed: {exc}",
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            gap_accuracies={},
        )

    state_refs, init_state = _snapshot_module_tensors(probe_model)
    runs = []
    try:
        for idx, seed in enumerate(seeds):
            if idx > 0:
                _restore_module_tensors(state_refs, init_state)
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            runs.append(
                _run_induction_validation_on(
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
            )
    finally:
        del probe_model, state_refs, init_state
        if device == "cuda":
            torch.cuda.empty_cache()

    runs.sort(key=lambda r: r.auc)
    median = runs[len(runs) // 2]
    gap_accuracies = dict(median.gap_accuracies or {})
    return InductionValidationResult(
        auc=median.auc,
        max_gap_acc=median.max_gap_acc,
        gap_accuracies=gap_accuracies,
        gap_accuracy_cv=_gap_accuracy_cv(gap_accuracies),
        steps_trained=median.steps_trained,
        status=median.status,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


def run_induction_validation_champion(
    model: nn.Module,
    *,
    extended_budget: bool = False,
    n_train_steps: int | None = None,
    seeds: Tuple[int, ...] = INDUCTION_V2_SEEDS,
    gaps: Tuple[int, ...] = INDUCTION_VALIDATION_GAPS,
    n_eval: int = INDUCTION_V2_EVAL_EXAMPLES,
    batch_size: int = INDUCTION_VALIDATION_BATCH_SIZE,
    lr: float = INDUCTION_VALIDATION_LR,
    device: str = "cuda",
    timeout_s: float | None = None,
) -> InductionValidationResult:
    """Run the champion-tier v3 induction protocol.

    The calibrated default budget is 2K probe-training steps. Set
    ``extended_budget`` for the explicit 5K diagnostic protocol, or pass
    ``n_train_steps=10000`` for the explicit 10K diagnostic protocol.
    """
    steps, protocol_version = select_induction_validation_budget(
        extended_budget=extended_budget,
        n_train_steps=n_train_steps,
    )
    timeout = (
        float(timeout_s)
        if timeout_s is not None
        else max(
            INDUCTION_VALIDATION_TIMEOUT_MIN_S,
            INDUCTION_VALIDATION_TIMEOUT_S_PER_STEP * steps,
        )
    )
    result = _run_induction_validation_median(
        model,
        seeds=seeds,
        gaps=gaps,
        n_train_steps=steps,
        n_eval=n_eval,
        batch_size=batch_size,
        lr=lr,
        device=device,
        timeout_s=timeout,
    )
    result.protocol_version = protocol_version
    return result
