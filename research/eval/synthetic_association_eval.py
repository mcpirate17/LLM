"""Tiny controlled association probe.

This is a deliberately small, HellaSwag-shaped replacement candidate for
nano-scale architectures.  It trains a model on a fixed low-vocabulary mapping:

    noun -> associated verb
    noun -> associated adjective

Then it evaluates forced-choice accuracy among four same-type candidates.  If an
architecture cannot learn this probe, real HellaSwag is too broad to be a useful
ranking signal for that architecture scale.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clip_grad_norm, make_adamw

logger = logging.getLogger(__name__)

SYNTHETIC_ASSOCIATION_METRIC_VERSION = "synthetic_association_v1"

_PAD = 0
_VERB_QUERY = 1
_ADJ_QUERY = 2
_MIN_ACTIVE_VOCAB = 20
_DEFAULT_ACTIVE_VOCAB = 40
_DEFAULT_TRAIN_STEPS = 20
_DEFAULT_EVAL_REPEATS = 8
_DEFAULT_BATCH = 32
_DEFAULT_LR = 2e-3
_TIMEOUT_S = 45.0


@dataclass(frozen=True, slots=True)
class AssociationLayout:
    active_vocab_size: int
    n_per_type: int
    noun_lo: int
    noun_hi: int
    verb_lo: int
    verb_hi: int
    adjective_lo: int
    adjective_hi: int

    @property
    def answer_lo(self) -> int:
        return self.verb_lo

    @property
    def answer_hi(self) -> int:
        return self.adjective_hi

    @property
    def chance(self) -> float:
        return 0.25


@dataclass(slots=True)
class SyntheticAssociationResult:
    score: float = 0.0
    verb_accuracy: float = 0.0
    adjective_accuracy: float = 0.0
    n_words: int = 0
    n_pairs: int = 0
    n_train_steps: int = 0
    active_vocab_size: int = 0
    chance: float = 0.25
    elapsed_ms: float = 0.0
    status: str = "ok"
    metric_version: str = SYNTHETIC_ASSOCIATION_METRIC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "synthetic_association_score": self.score,
            "synthetic_association_verb_acc": self.verb_accuracy,
            "synthetic_association_adjective_acc": self.adjective_accuracy,
            "synthetic_association_n_words": self.n_words,
            "synthetic_association_n_pairs": self.n_pairs,
            "synthetic_association_train_steps": self.n_train_steps,
            "synthetic_association_active_vocab_size": self.active_vocab_size,
            "synthetic_association_chance": self.chance,
            "synthetic_association_elapsed_ms": self.elapsed_ms,
            "synthetic_association_status": self.status,
            "synthetic_association_metric_version": self.metric_version,
        }


def _make_layout(active_vocab_size: int) -> AssociationLayout:
    active = max(_MIN_ACTIVE_VOCAB, int(active_vocab_size))
    n_per_type = (active - 4) // 3
    if n_per_type < 4:
        raise ValueError(
            f"active_vocab_size={active_vocab_size} leaves fewer than 4 words per class"
        )
    noun_lo = 4
    noun_hi = noun_lo + n_per_type
    verb_lo = noun_hi
    verb_hi = verb_lo + n_per_type
    adjective_lo = verb_hi
    adjective_hi = adjective_lo + n_per_type
    return AssociationLayout(
        active_vocab_size=active,
        n_per_type=n_per_type,
        noun_lo=noun_lo,
        noun_hi=noun_hi,
        verb_lo=verb_lo,
        verb_hi=verb_hi,
        adjective_lo=adjective_lo,
        adjective_hi=adjective_hi,
    )


def _association_targets(
    noun_ids: torch.Tensor,
    relation_ids: torch.Tensor,
    layout: AssociationLayout,
) -> torch.Tensor:
    noun_idx = noun_ids - layout.noun_lo
    verb_targets = layout.verb_lo + ((noun_idx + 1) % layout.n_per_type)
    adjective_targets = layout.adjective_lo + ((noun_idx + 2) % layout.n_per_type)
    return torch.where(relation_ids == _VERB_QUERY, verb_targets, adjective_targets)


def _association_target_int(
    noun: int,
    relation: int,
    layout: AssociationLayout,
) -> int:
    noun_idx = int(noun) - layout.noun_lo
    if int(relation) == _VERB_QUERY:
        return layout.verb_lo + ((noun_idx + 1) % layout.n_per_type)
    return layout.adjective_lo + ((noun_idx + 2) % layout.n_per_type)


def _make_train_batch(
    layout: AssociationLayout,
    batch_size: int,
    device: str,
    rng: torch.Generator | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    nouns = torch.randint(
        layout.noun_lo,
        layout.noun_hi,
        (batch_size,),
        device=device,
        generator=rng,
    )
    relation_flag = torch.randint(0, 2, (batch_size,), device=device, generator=rng)
    relations = torch.where(
        relation_flag == 0,
        torch.full_like(relation_flag, _VERB_QUERY),
        torch.full_like(relation_flag, _ADJ_QUERY),
    )
    targets = _association_targets(nouns, relations, layout)
    input_ids = torch.empty(batch_size, 3, dtype=torch.long, device=device)
    input_ids[:, 0] = nouns
    input_ids[:, 1] = relations
    input_ids[:, 2] = _PAD
    return input_ids, targets


def _candidate_ids(
    target: int,
    lo: int,
    hi: int,
    true_slot: int,
) -> list[int]:
    pool_size = hi - lo
    candidates = [target]
    offset = 1
    while len(candidates) < 4:
        candidate = lo + ((target - lo + offset) % pool_size)
        if candidate != target:
            candidates.append(candidate)
        offset += 1
    ordered = candidates[1:]
    ordered.insert(true_slot, target)
    return ordered


def _eval_forced_choice_accuracy(
    model: nn.Module,
    layout: AssociationLayout,
    *,
    relation_id: int,
    eval_repeats: int,
    batch_size: int,
    device: str,
) -> float:
    model.eval()
    seqs: list[list[int]] = []
    targets: list[int] = []
    candidates: list[list[int]] = []
    for repeat in range(max(1, int(eval_repeats))):
        for noun in range(layout.noun_lo, layout.noun_hi):
            relation = int(relation_id)
            target = _association_target_int(noun, relation, layout)
            lo, hi = (
                (layout.verb_lo, layout.verb_hi)
                if relation == _VERB_QUERY
                else (layout.adjective_lo, layout.adjective_hi)
            )
            true_slot = (noun + repeat + relation) % 4
            seqs.append([noun, relation, _PAD])
            targets.append(true_slot)
            candidates.append(_candidate_ids(target, lo, hi, true_slot))

    input_ids = torch.tensor(seqs, dtype=torch.long, device=device)
    target_slots = torch.tensor(targets, dtype=torch.long, device=device)
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
    correct = 0
    total = int(input_ids.shape[0])
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            logits = model(input_ids[start:end])
            query_logits = logits[:, 1, :]
            scores = query_logits.gather(1, candidate_tensor[start:end])
            pred_slots = scores.argmax(dim=-1)
            correct += int((pred_slots == target_slots[start:end]).sum().item())
    return correct / max(total, 1)


def synthetic_association_score(
    model: nn.Module,
    *,
    active_vocab_size: int = _DEFAULT_ACTIVE_VOCAB,
    n_train_steps: int = _DEFAULT_TRAIN_STEPS,
    eval_repeats: int = _DEFAULT_EVAL_REPEATS,
    batch_size: int = _DEFAULT_BATCH,
    lr: float = _DEFAULT_LR,
    device: str = "cuda",
    seed: int = 42,
    timeout_s: float = _TIMEOUT_S,
) -> SyntheticAssociationResult:
    """Train noun-association mappings, then evaluate four-way choices."""
    t0 = time.perf_counter()
    dev = torch.device(device)
    layout = _make_layout(active_vocab_size)
    if layout.adjective_hi > int(getattr(model, "vocab_size", layout.adjective_hi)):
        return SyntheticAssociationResult(
            active_vocab_size=layout.active_vocab_size,
            n_words=layout.n_per_type * 3,
            n_pairs=layout.n_per_type * 2,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status="model_vocab_too_small",
        )

    rng = torch.Generator(device=dev.type)
    rng.manual_seed(int(seed))
    saved_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    deadline = t0 + float(timeout_s)
    steps_completed = 0
    status = "ok"

    try:
        model.train()
        opt = make_adamw(model.parameters(), lr=lr)
        for step in range(1, int(n_train_steps) + 1):
            if time.perf_counter() > deadline:
                status = "timeout"
                break
            input_ids, targets = _make_train_batch(layout, batch_size, device, rng)
            opt.zero_grad(set_to_none=True)
            logits = model(input_ids)
            pred_logits = logits[:, 1, layout.answer_lo : layout.answer_hi]
            loss = F.cross_entropy(pred_logits, targets - layout.answer_lo)
            if not torch.isfinite(loss):
                status = "diverged"
                break
            loss.backward()
            clip_grad_norm(model.parameters(), 1.0)
            opt.step()
            steps_completed = step

        verb_acc = _eval_forced_choice_accuracy(
            model,
            layout,
            relation_id=_VERB_QUERY,
            eval_repeats=eval_repeats,
            batch_size=batch_size,
            device=device,
        )
        adjective_acc = _eval_forced_choice_accuracy(
            model,
            layout,
            relation_id=_ADJ_QUERY,
            eval_repeats=eval_repeats,
            batch_size=batch_size,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("synthetic association probe failed: %s", exc, exc_info=False)
        return SyntheticAssociationResult(
            active_vocab_size=layout.active_vocab_size,
            n_words=layout.n_per_type * 3,
            n_pairs=layout.n_per_type * 2,
            n_train_steps=steps_completed,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status=f"failed:{type(exc).__name__}",
        )
    finally:
        model.load_state_dict(saved_state)
        model.train(was_training)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    raw = 0.5 * (verb_acc + adjective_acc)
    chance = layout.chance
    normalized = max(0.0, (raw - chance) / max(1.0 - chance, 1e-6))
    return SyntheticAssociationResult(
        score=round(float(normalized), 4),
        verb_accuracy=round(float(verb_acc), 4),
        adjective_accuracy=round(float(adjective_acc), 4),
        n_words=layout.n_per_type * 3,
        n_pairs=layout.n_per_type * 2,
        n_train_steps=steps_completed,
        active_vocab_size=layout.active_vocab_size,
        chance=chance,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        status=status,
    )


__all__ = [
    "AssociationLayout",
    "SyntheticAssociationResult",
    "SYNTHETIC_ASSOCIATION_METRIC_VERSION",
    "synthetic_association_score",
]
