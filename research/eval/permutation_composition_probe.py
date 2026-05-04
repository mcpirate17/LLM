"""Permutation composition probe.

Trains a model on tokenized transposition chains:

    start, a1, b1, a2, b2, ..., QUERY -> final

where each ``(a, b)`` pair swaps two symbols and the target is the result of
applying the full swap chain to ``start``. The held-out evaluation uses both
the trained chain length and a longer chain length, so the score rewards
cross-token composition instead of local association.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clip_grad_norm, make_adamw

PERMUTATION_COMPOSITION_METRIC_VERSION = "permutation_composition_v1"

_PAD = 0
_QUERY = 1
_SYMBOL_LO = 4
_MIN_ITEMS = 4
_DEFAULT_N_ITEMS = 8
_DEFAULT_TRAIN_CHAIN_LEN = 2
_DEFAULT_EVAL_CHAIN_LEN = 4
_DEFAULT_TRAIN_STEPS = 80
_DEFAULT_EVAL_BATCHES = 8
_DEFAULT_BATCH = 64
_DEFAULT_LR = 2e-3
_TIMEOUT_S = 45.0


@dataclass(frozen=True, slots=True)
class PermutationLayout:
    n_items: int
    symbol_lo: int = _SYMBOL_LO

    @property
    def symbol_hi(self) -> int:
        return self.symbol_lo + self.n_items

    @property
    def vocab_size(self) -> int:
        return self.symbol_hi

    @property
    def chance(self) -> float:
        return 1.0 / float(self.n_items)


@dataclass(slots=True)
class PermutationCompositionResult:
    score: float = 0.0
    train_chain_accuracy: float = 0.0
    extrapolation_accuracy: float = 0.0
    n_items: int = 0
    train_chain_len: int = 0
    eval_chain_len: int = 0
    n_train_steps: int = 0
    chance: float = 0.0
    elapsed_ms: float = 0.0
    status: str = "ok"
    metric_version: str = PERMUTATION_COMPOSITION_METRIC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "permutation_composition_score": self.score,
            "permutation_composition_train_chain_acc": self.train_chain_accuracy,
            "permutation_composition_extrapolation_acc": self.extrapolation_accuracy,
            "permutation_composition_n_items": self.n_items,
            "permutation_composition_train_chain_len": self.train_chain_len,
            "permutation_composition_eval_chain_len": self.eval_chain_len,
            "permutation_composition_train_steps": self.n_train_steps,
            "permutation_composition_chance": self.chance,
            "permutation_composition_elapsed_ms": self.elapsed_ms,
            "permutation_composition_status": self.status,
            "permutation_composition_metric_version": self.metric_version,
        }


def _make_layout(n_items: int) -> PermutationLayout:
    return PermutationLayout(n_items=max(_MIN_ITEMS, int(n_items)))


def _apply_transpositions(
    start_symbols: torch.Tensor,
    left_symbols: torch.Tensor,
    right_symbols: torch.Tensor,
) -> torch.Tensor:
    current = start_symbols.clone()
    for idx in range(left_symbols.shape[1]):
        left = left_symbols[:, idx]
        right = right_symbols[:, idx]
        current = torch.where(
            current == left,
            right,
            torch.where(current == right, left, current),
        )
    return current


def _make_batch(
    layout: PermutationLayout,
    batch_size: int,
    chain_len: int,
    device: str,
    rng: torch.Generator | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    chain = max(1, int(chain_len))
    start = torch.randint(
        layout.symbol_lo,
        layout.symbol_hi,
        (batch_size,),
        device=device,
        generator=rng,
    )
    left = torch.randint(
        layout.symbol_lo,
        layout.symbol_hi,
        (batch_size, chain),
        device=device,
        generator=rng,
    )
    offset = torch.randint(
        1,
        layout.n_items,
        (batch_size, chain),
        device=device,
        generator=rng,
    )
    right = layout.symbol_lo + ((left - layout.symbol_lo + offset) % layout.n_items)
    targets = _apply_transpositions(start, left, right)

    input_ids = torch.full(
        (batch_size, 2 * chain + 2),
        _PAD,
        dtype=torch.long,
        device=device,
    )
    input_ids[:, 0] = start
    for idx in range(chain):
        input_ids[:, 1 + 2 * idx] = left[:, idx]
        input_ids[:, 2 + 2 * idx] = right[:, idx]
    input_ids[:, -1] = _QUERY
    return input_ids, targets


def _drop_state_keys_added_after_snapshot(
    model: nn.Module, saved_keys: set[str] | frozenset[str]
) -> None:
    current_keys = set(model.state_dict().keys())
    new_keys = sorted(
        current_keys - saved_keys,
        key=lambda item: item.count("."),
        reverse=True,
    )
    for key in new_keys:
        module_path, _, leaf = key.rpartition(".")
        parent: object = model
        if module_path:
            for part in module_path.split("."):
                if not isinstance(parent, nn.Module):
                    parent = None
                    break
                parent = getattr(parent, part, None)
        if not isinstance(parent, nn.Module):
            continue
        if leaf in parent._parameters:
            del parent._parameters[leaf]
        elif leaf in parent._buffers:
            del parent._buffers[leaf]


def _eval_accuracy(
    model: nn.Module,
    layout: PermutationLayout,
    *,
    chain_len: int,
    n_batches: int,
    batch_size: int,
    device: str,
    seed: int,
) -> float:
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(max(1, int(n_batches))):
            input_ids, targets = _make_batch(layout, batch_size, chain_len, device, rng)
            logits = model(input_ids)
            query_logits = logits[:, -1, layout.symbol_lo : layout.symbol_hi]
            pred = query_logits.argmax(dim=-1) + layout.symbol_lo
            correct += int((pred == targets).sum().item())
            total += int(targets.numel())
    return correct / max(total, 1)


def permutation_composition_score(
    model: nn.Module,
    *,
    n_items: int = _DEFAULT_N_ITEMS,
    train_chain_len: int = _DEFAULT_TRAIN_CHAIN_LEN,
    eval_chain_len: int = _DEFAULT_EVAL_CHAIN_LEN,
    n_train_steps: int = _DEFAULT_TRAIN_STEPS,
    n_eval_batches: int = _DEFAULT_EVAL_BATCHES,
    batch_size: int = _DEFAULT_BATCH,
    lr: float = _DEFAULT_LR,
    device: str = "cuda",
    seed: int = 42,
    timeout_s: float = _TIMEOUT_S,
) -> PermutationCompositionResult:
    """Train on short transposition chains and evaluate composition accuracy."""
    t0 = time.perf_counter()
    layout = _make_layout(n_items)
    train_chain = max(1, int(train_chain_len))
    eval_chain = max(train_chain, int(eval_chain_len))
    if layout.symbol_hi > int(getattr(model, "vocab_size", layout.symbol_hi)):
        return PermutationCompositionResult(
            n_items=layout.n_items,
            train_chain_len=train_chain,
            eval_chain_len=eval_chain,
            chance=layout.chance,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status="model_vocab_too_small",
        )

    saved_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    saved_keys = frozenset(saved_state)
    was_training = model.training
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    deadline = t0 + float(timeout_s)
    steps = 0
    status = "ok"

    try:
        model.train()
        opt = make_adamw(model.parameters(), lr=lr)
        for step in range(1, int(n_train_steps) + 1):
            if time.perf_counter() > deadline:
                status = "timeout"
                break
            input_ids, targets = _make_batch(
                layout,
                batch_size,
                train_chain,
                device,
                rng,
            )
            opt.zero_grad(set_to_none=True)
            logits = model(input_ids)
            pred_logits = logits[:, -1, layout.symbol_lo : layout.symbol_hi]
            loss = F.cross_entropy(pred_logits, targets - layout.symbol_lo)
            if not torch.isfinite(loss):
                status = "non_finite_loss"
                break
            loss.backward()
            clip_grad_norm(model.parameters(), 1.0)
            opt.step()
            steps = step

        if status not in ("ok", "timeout"):
            return PermutationCompositionResult(
                n_items=layout.n_items,
                train_chain_len=train_chain,
                eval_chain_len=eval_chain,
                n_train_steps=steps,
                chance=layout.chance,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                status=status,
            )

        train_acc = _eval_accuracy(
            model,
            layout,
            chain_len=train_chain,
            n_batches=n_eval_batches,
            batch_size=batch_size,
            device=device,
            seed=seed + 1,
        )
        extrap_acc = _eval_accuracy(
            model,
            layout,
            chain_len=eval_chain,
            n_batches=n_eval_batches,
            batch_size=batch_size,
            device=device,
            seed=seed + 2,
        )
        raw = 0.5 * (train_acc + extrap_acc)
        normalized = max(0.0, (raw - layout.chance) / max(1.0 - layout.chance, 1e-6))
        return PermutationCompositionResult(
            score=round(float(normalized), 4),
            train_chain_accuracy=round(float(train_acc), 4),
            extrapolation_accuracy=round(float(extrap_acc), 4),
            n_items=layout.n_items,
            train_chain_len=train_chain,
            eval_chain_len=eval_chain,
            n_train_steps=steps,
            chance=round(float(layout.chance), 4),
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status=status,
        )
    finally:
        _drop_state_keys_added_after_snapshot(model, saved_keys)
        model.load_state_dict(saved_state)
        model.train(was_training)
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


__all__ = [
    "PERMUTATION_COMPOSITION_METRIC_VERSION",
    "PermutationCompositionResult",
    "PermutationLayout",
    "permutation_composition_score",
]
