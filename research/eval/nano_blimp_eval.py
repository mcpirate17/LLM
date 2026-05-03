"""Nano-BLiMP: minimal-pair grammaticality on the controlled-language vocab.

Sister probe to ``synthetic_association_eval`` (codex). Both share the
``AssociationLayout`` and training step machinery, but evaluate
complementary capabilities:

  * ``synthetic_association_score`` (codex)  — HellaSwag-shaped 4-way forced
    choice: "given a noun + query, pick the associated word from 4
    same-class candidates"
  * ``nano_blimp_score`` (this file)         — BLiMP-shaped log-likelihood
    minimal pairs:
      - ``class_coherence``: model prefers `[noun][verb_query][verb]` over
        `[noun][verb_query][noun]` (right-class continuation)
      - ``binding_fidelity``: model prefers `[noun_A][verb_query][verb_A]`
        over `[noun_A][verb_query][verb_B]` where verb_B is associated with
        a different noun (binding-correctness)
      - ``order_grammaticality``: model prefers
        `[noun][verb_query][verb]` over `[verb_query][noun][verb]`
        (well-formed order)

Score uses log-prob of full sequences (sum log p(t_i | t_<i)). Fraction of
pairs where the well-formed sequence has higher log-prob → accuracy.
Random baseline = 0.5 (binary good/bad). Three sub-accuracies averaged
into ``nano_blimp_score``.

Output is intentionally not yet wired to leaderboard scoring — this is an
experimental probe. Wire only after cohort calibration shows real
architecture-to-architecture spread.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .synthetic_association_eval import (
    AssociationLayout,
    _ADJ_QUERY,
    _VERB_QUERY,
    _association_target_int,
    _make_layout,
    _make_train_batch,
)
from .utils import clip_grad_norm, make_adamw

logger = logging.getLogger(__name__)

NANO_BLIMP_METRIC_VERSION = "nano_blimp_v1"

_DEFAULT_ACTIVE_VOCAB = 32
_DEFAULT_TRAIN_STEPS = 300
_DEFAULT_BATCH = 32
_DEFAULT_LR = 1e-3
_TIMEOUT_S = 60.0


@dataclass(slots=True)
class NanoBLiMPResult:
    score: float = 0.0
    class_coherence_acc: float = 0.0
    binding_fidelity_acc: float = 0.0
    order_grammaticality_acc: float = 0.0
    n_pairs_per_test: int = 0
    n_train_steps: int = 0
    active_vocab_size: int = 0
    chance: float = 0.5
    elapsed_ms: float = 0.0
    status: str = "ok"
    metric_version: str = NANO_BLIMP_METRIC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nano_blimp_score": self.score,
            "nano_blimp_class_coherence_acc": self.class_coherence_acc,
            "nano_blimp_binding_fidelity_acc": self.binding_fidelity_acc,
            "nano_blimp_order_grammaticality_acc": self.order_grammaticality_acc,
            "nano_blimp_n_pairs_per_test": self.n_pairs_per_test,
            "nano_blimp_train_steps": self.n_train_steps,
            "nano_blimp_active_vocab_size": self.active_vocab_size,
            "nano_blimp_chance": self.chance,
            "nano_blimp_elapsed_ms": self.elapsed_ms,
            "nano_blimp_status": self.status,
            "nano_blimp_metric_version": self.metric_version,
        }


def _seq_logprob(model: nn.Module, seqs: torch.Tensor) -> torch.Tensor:
    """Return sum-log-prob of next-token over each sequence (B,S) -> (B,)."""
    with torch.no_grad():
        logits = model(seqs)  # (B, S, V)
        log_probs = F.log_softmax(logits, dim=-1)
        # next-token log-prob: log p(t_i | t_<i) for i in [1, S-1]
        # = log_probs[:, :-1].gather(2, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
        gathered = log_probs[:, :-1].gather(2, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
        return gathered.sum(dim=-1)


def _build_class_coherence_pairs(
    layout: AssociationLayout, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """For every (noun, query) combo, build (good, bad) seq pair where
    good ends in the right class and bad ends in the wrong class."""
    good = []
    bad = []
    for noun in range(layout.noun_lo, layout.noun_hi):
        for query, right_lo, right_hi, wrong_lo, wrong_hi in (
            (
                _VERB_QUERY,
                layout.verb_lo,
                layout.verb_hi,
                layout.noun_lo,
                layout.noun_hi,
            ),
            (
                _ADJ_QUERY,
                layout.adjective_lo,
                layout.adjective_hi,
                layout.noun_lo,
                layout.noun_hi,
            ),
        ):
            target = _association_target_int(noun, query, layout)
            # Bad: continuation from a different (noun) class — uses the
            # noun position adjacent to `target` so token IDs differ by
            # class membership but stay in the active vocab.
            wrong_token = wrong_lo + ((target - right_lo) % (wrong_hi - wrong_lo))
            good.append([noun, query, target])
            bad.append([noun, query, wrong_token])
    return (
        torch.tensor(good, dtype=torch.long, device=device),
        torch.tensor(bad, dtype=torch.long, device=device),
    )


def _build_binding_fidelity_pairs(
    layout: AssociationLayout, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Good pair: (noun_A, query, associated_word_A).
    Bad pair: (noun_A, query, associated_word_B) where B≠A is another
    noun's association in the same class. Tests whether the model bound
    the right pair, not just the right class."""
    good = []
    bad = []
    nouns = list(range(layout.noun_lo, layout.noun_hi))
    n = len(nouns)
    for i, noun_a in enumerate(nouns):
        # Use the cyclic neighbor's noun as the source of the wrong
        # association — guaranteed to differ from noun_a's target.
        noun_b = nouns[(i + 1) % n]
        for query in (_VERB_QUERY, _ADJ_QUERY):
            target_a = _association_target_int(noun_a, query, layout)
            target_b = _association_target_int(noun_b, query, layout)
            if target_a == target_b:
                # Same target by coincidence — skip (rare with the
                # codex offset constants 5,1 and 7,2).
                continue
            good.append([noun_a, query, target_a])
            bad.append([noun_a, query, target_b])
    return (
        torch.tensor(good, dtype=torch.long, device=device),
        torch.tensor(bad, dtype=torch.long, device=device),
    )


def _build_order_pairs(
    layout: AssociationLayout, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Good: [noun, query, target]. Bad: [query, noun, target] — same
    tokens, swapped order. Tests whether the model learned the canonical
    `noun-then-query` ordering observed in training, not just bag-of-words."""
    good = []
    bad = []
    for noun in range(layout.noun_lo, layout.noun_hi):
        for query in (_VERB_QUERY, _ADJ_QUERY):
            target = _association_target_int(noun, query, layout)
            good.append([noun, query, target])
            bad.append([query, noun, target])
    return (
        torch.tensor(good, dtype=torch.long, device=device),
        torch.tensor(bad, dtype=torch.long, device=device),
    )


def _pair_accuracy(model: nn.Module, good: torch.Tensor, bad: torch.Tensor) -> float:
    """Fraction of pairs where good has higher sum-log-prob than bad."""
    g = _seq_logprob(model, good)
    b = _seq_logprob(model, bad)
    correct = (g > b).sum().item()
    total = good.shape[0]
    return float(correct) / max(total, 1)


def _train_probe(
    model: nn.Module,
    layout: AssociationLayout,
    *,
    n_train_steps: int,
    batch_size: int,
    lr: float,
    device: str,
    deadline: float,
    seed: int,
) -> tuple[int, str]:
    """Train the model in place; caller must snapshot+restore state_dict.
    Avoids ``copy.deepcopy`` which fails on models with ``weight_norm``
    parametrizations (silently scoring 0.0/0.0). state_dict snapshot
    survives weight_norm by serializing only leaf parameters."""
    model.train()
    opt = make_adamw(model.parameters(), lr=lr)
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    steps = 0
    status = "ok"
    for step in range(int(n_train_steps)):
        if time.perf_counter() > deadline:
            status = "timeout"
            break
        input_ids, targets = _make_train_batch(layout, batch_size, device, rng)
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids)
        pred_logits = logits[:, 1, layout.answer_lo : layout.answer_hi]
        loss = F.cross_entropy(pred_logits, targets - layout.answer_lo)
        if not torch.isfinite(loss):
            status = "non_finite_loss"
            break
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step()
        steps = step + 1
    return steps, status


def nano_blimp_score(
    model: nn.Module,
    *,
    active_vocab_size: int = _DEFAULT_ACTIVE_VOCAB,
    n_train_steps: int = _DEFAULT_TRAIN_STEPS,
    batch_size: int = _DEFAULT_BATCH,
    lr: float = _DEFAULT_LR,
    device: str = "cuda",
    seed: int = 42,
    timeout_s: float = _TIMEOUT_S,
) -> NanoBLiMPResult:
    """Train on association mappings, evaluate three minimal-pair tests."""
    t0 = time.perf_counter()
    deadline = t0 + float(timeout_s)
    layout = _make_layout(active_vocab_size)

    if layout.adjective_hi > int(getattr(model, "vocab_size", layout.adjective_hi)):
        return NanoBLiMPResult(
            active_vocab_size=layout.active_vocab_size,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status="model_vocab_too_small",
        )

    saved_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    try:
        steps, status = _train_probe(
            model,
            layout,
            n_train_steps=n_train_steps,
            batch_size=batch_size,
            lr=lr,
            device=device,
            deadline=deadline,
            seed=seed,
        )
        if status not in ("ok", "timeout"):
            return NanoBLiMPResult(
                n_train_steps=steps,
                active_vocab_size=layout.active_vocab_size,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                status=status,
            )

        model.eval()
        good_c, bad_c = _build_class_coherence_pairs(layout, device)
        good_b, bad_b = _build_binding_fidelity_pairs(layout, device)
        good_o, bad_o = _build_order_pairs(layout, device)
        class_acc = _pair_accuracy(model, good_c, bad_c)
        binding_acc = _pair_accuracy(model, good_b, bad_b)
        order_acc = _pair_accuracy(model, good_o, bad_o)
        score = (class_acc + binding_acc + order_acc) / 3.0
        n_pairs = good_c.shape[0]

        return NanoBLiMPResult(
            score=round(float(score), 4),
            class_coherence_acc=round(float(class_acc), 4),
            binding_fidelity_acc=round(float(binding_acc), 4),
            order_grammaticality_acc=round(float(order_acc), 4),
            n_pairs_per_test=int(n_pairs),
            n_train_steps=int(steps),
            active_vocab_size=layout.active_vocab_size,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status=status,
        )
    finally:
        # Always restore caller's model state — the probe trained in place.
        model.load_state_dict(saved_state)
        model.train(was_training)
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def nano_blimp_eval_only(
    trained_model: nn.Module,
    layout: AssociationLayout,
    *,
    device: str = "cuda",
) -> NanoBLiMPResult:
    """Evaluate nano-BLiMP on an *already-trained* model.

    Use this when sharing the training pass between
    ``synthetic_association_score`` and ``nano_blimp_score`` — train once
    on the association data, then run both eval functions back-to-back.
    The model is assumed to be in a state that has seen the (noun, query,
    target) training distribution from ``_make_train_batch``.

    Does NOT train, does NOT mutate model state.
    """
    t0 = time.perf_counter()
    was_training = trained_model.training
    trained_model.eval()
    try:
        good_c, bad_c = _build_class_coherence_pairs(layout, device)
        good_b, bad_b = _build_binding_fidelity_pairs(layout, device)
        good_o, bad_o = _build_order_pairs(layout, device)
        class_acc = _pair_accuracy(trained_model, good_c, bad_c)
        binding_acc = _pair_accuracy(trained_model, good_b, bad_b)
        order_acc = _pair_accuracy(trained_model, good_o, bad_o)
        score = (class_acc + binding_acc + order_acc) / 3.0
        n_pairs = good_c.shape[0]
        return NanoBLiMPResult(
            score=round(float(score), 4),
            class_coherence_acc=round(float(class_acc), 4),
            binding_fidelity_acc=round(float(binding_acc), 4),
            order_grammaticality_acc=round(float(order_acc), 4),
            n_pairs_per_test=int(n_pairs),
            n_train_steps=0,  # eval-only; caller did the training
            active_vocab_size=layout.active_vocab_size,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status="ok",
        )
    finally:
        trained_model.train(was_training)
