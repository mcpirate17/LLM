"""Shared utilities for evaluation probes.

These helpers run *between* the trained-model forward and the probe's own
training/eval phase. They exist because probes commonly ``copy.deepcopy`` the
trained model, and upstream evaluation paths leave inference-mode tensors
behind that break autograd or corrupt tensor metadata on subsequent CUDA ops.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Callable, Protocol, Sequence, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clip_grad_norm

logger = logging.getLogger(__name__)


class _HasAuc(Protocol):
    auc: float


_ProbeRun = TypeVar("_ProbeRun", bound=_HasAuc)


class ProbeCopyError(RuntimeError):
    """Raised when the probe's safe-deepcopy of the host model fails."""


def mean_auc(values: Sequence[float]) -> float:
    """Mean accuracy rounded to 4 places — the shared probe AUC reduction."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def probe_steps_to_threshold(
    learning_curve: Sequence[dict[str, float | int]],
    *,
    metric_key: str,
    threshold: float,
) -> int | None:
    for row in learning_curve:
        if float(row.get(metric_key) or 0.0) >= float(threshold):
            return int(row["step"])
    return None


def probe_curve_summary(
    learning_curve: Sequence[dict[str, float | int]],
    *,
    metric_key: str,
    final_step: int,
) -> tuple[float, float, float, float, float]:
    if not learning_curve:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    early = float(learning_curve[0].get(metric_key) or 0.0)
    final = float(learning_curve[-1].get(metric_key) or 0.0)
    best = max(float(row.get(metric_key) or 0.0) for row in learning_curve)
    auc = sum(float(row.get(metric_key) or 0.0) for row in learning_curve) / float(
        len(learning_curve)
    )
    first_step = int(learning_curve[0].get("step") or 0)
    span = max(1, int(final_step) - first_step)
    return early, final, best, auc, (final - early) * 100.0 / float(span)


def next_token_train_step(
    model: nn.Module,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    pad_id: int,
    grad_clip: float = 1.0,
) -> bool:
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    targets = batch[:, 1:].contiguous()
    pred = logits[:, :-1, :].contiguous()
    mask = targets != int(pad_id)
    if not bool(mask.any()):
        return True
    loss = F.cross_entropy(pred[mask].float(), targets[mask])
    if not torch.isfinite(loss):
        return False
    loss.backward()
    clip_grad_norm(model.parameters(), float(grad_clip))
    optimizer.step()
    return True


def maybe_compile_probe_model(model: nn.Module, *, dynamic: bool) -> nn.Module:
    """Optionally wrap a probe copy with ``torch.compile``.

    Gated behind ``ARIA_PROBE_COMPILE=1`` (off by default): per-shape compile
    cost does not amortize at typical probe budgets, and some IR-executor
    graphs trigger Python-side side-effect recompiles that erase the gain.
    ``dynamic`` should be True when the probe trains across multiple sequence
    lengths (e.g. one per gap) and False for shape-invariant probes.
    """
    if os.environ.get("ARIA_PROBE_COMPILE", "") != "1":
        return model
    if not torch.cuda.is_available():
        return model
    try:
        return torch.compile(model, mode="default", dynamic=dynamic, fullgraph=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch.compile unavailable for probe model: %s", exc)
        return model


def run_probe_seeds(
    model: nn.Module,
    *,
    seeds: Sequence[int],
    device: str,
    run_single_seed: Callable[[nn.Module, torch.Generator], _ProbeRun],
) -> list[_ProbeRun]:
    """Shared multi-seed probe harness: one deepcopy, restore between seeds.

    Deepcopies the host model once, snapshots its initial tensors, then runs
    ``run_single_seed(probe_model, generator)`` per seed with the initial
    weights restored in between (~10× cheaper than deepcopying per seed for
    10-100M-param probe models). Returns the runs sorted by ``.auc`` so the
    caller can take ``runs[len(runs) // 2]`` as the median.

    Raises :class:`ProbeCopyError` if the deepcopy fails; run errors
    propagate unchanged.
    """
    try:
        probe_model = safe_deepcopy_module(model).to(device)
    except Exception as exc:
        raise ProbeCopyError(str(exc)) from exc
    state_refs, init_state = snapshot_module_tensors(probe_model)

    runs: list[_ProbeRun] = []
    try:
        for idx, seed in enumerate(seeds):
            if idx > 0:
                restore_module_tensors(state_refs, init_state)
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            runs.append(run_single_seed(probe_model, generator))
    finally:
        del probe_model, state_refs, init_state
        if device == "cuda":
            torch.cuda.empty_cache()

    runs.sort(key=lambda r: r.auc)
    return runs


def _materialize_non_inference_(module: nn.Module) -> None:
    """In-place: replace inference-mode params/buffers with autograd-safe storage.

    Probes deepcopy the trained investigation model, then train the copy.
    Upstream evaluation paths run forward under ``torch.inference_mode()``
    which marks any cached buffers (RoPE rotations, attention biases, KV
    caches) as inference tensors. Inference tensors propagate through
    ``deepcopy``, and trying to use them in autograd-tracked computation
    raises ``RuntimeError: Inference tensors cannot be saved for backward``.
    Cloning the storage outside ``inference_mode`` mints a normal tensor;
    swapping ``.data`` rebinds the wrapping Parameter/buffer to it.
    """
    with torch.no_grad():
        for p in module.parameters(recurse=True):
            if p.is_inference():
                fresh = torch.empty_like(p.data)
                fresh.copy_(p.data)
                p.data = fresh
        for b in module.buffers(recurse=True):
            if b.is_inference():
                fresh = torch.empty_like(b.data)
                fresh.copy_(b.data)
                b.data = fresh


_MODULE_INTERNAL_DICTS = frozenset(
    {
        "_parameters",
        "_buffers",
        "_modules",
        "_non_persistent_buffers_set",
        "_backward_pre_hooks",
        "_backward_hooks",
        "_forward_hooks",
        "_forward_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    }
)


def _detach_non_leaf_attrs_(module: nn.Module) -> None:
    """In-place: detach any non-leaf tensors cached on modules.

    Patterns like ``nn.utils.weight_norm`` and synthesis-graph op caches store
    a *computed* tensor (with ``grad_fn``, i.e. non-leaf) directly on the
    module as a Python attribute. ``copy.deepcopy`` then raises
    ``RuntimeError: Only Tensors created explicitly by the user (graph leaves)
    support the deepcopy protocol``. Detaching drops ``grad_fn`` so the copy
    succeeds; the next forward pass will recompute the cache.
    """
    for mod in module.modules():
        for name, buf in list(mod._buffers.items()):
            if buf is not None and not buf.is_leaf:
                mod._buffers[name] = buf.detach()
        for name, val in list(mod.__dict__.items()):
            if name in _MODULE_INTERNAL_DICTS:
                continue
            if isinstance(val, torch.Tensor) and not val.is_leaf:
                setattr(mod, name, val.detach())


def snapshot_module_tensors(
    module: nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Capture parameters and buffers for cheap in-place restoration.

    Pair with :func:`restore_module_tensors` to reuse one deepcopy across
    multiple probe runs (seeds, sequence lengths) instead of re-deepcopying.
    """
    refs = [*module.parameters(), *module.buffers()]
    with torch.no_grad():
        snapshot = [tensor.detach().clone() for tensor in refs]
    return refs, snapshot


def restore_module_tensors(
    refs: list[torch.Tensor], snapshot: list[torch.Tensor]
) -> None:
    """Restore parameters and buffers without rebuilding a state_dict."""
    with torch.no_grad():
        for ref, original in zip(refs, snapshot):
            ref.copy_(original)


def safe_deepcopy_module(module: nn.Module) -> nn.Module:
    """Materialize, detach, then deepcopy — survives the two known fail modes.

    ``copy.deepcopy`` on a trained nn.Module fails in two distinct ways:

    1. **Inference-mode tensors**: an upstream ``torch.inference_mode()`` eval
       pass left cached buffers (RoPE tables, attention masks) marked as
       inference tensors. ``_materialize_non_inference_`` clones their storage
       outside inference_mode to remint them as normal tensors.
    2. **Non-leaf cached tensors**: synthesis-graph ops or ``weight_norm``-style
       wrappers cache a computed tensor (with ``grad_fn``) as a raw module
       attribute. ``_detach_non_leaf_attrs_`` strips ``grad_fn`` so deepcopy
       can walk the tensor.

    Both passes are idempotent: once a module is cleaned, future calls are
    no-ops. We also clean the copy as belt-and-braces against any new bad
    tensors that landed via the deepcopy of internal caches.
    """
    _materialize_non_inference_(module)
    _detach_non_leaf_attrs_(module)
    copied = copy.deepcopy(module)
    _materialize_non_inference_(copied)
    _detach_non_leaf_attrs_(copied)
    return copied
