"""Nano-BLiMP: minimal-pair grammaticality on the language-control vocab.

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

v2 adds a held-out compositional split. ``class_coherence`` and
``binding_fidelity`` saturated for every architecture in v1 because the
training loop sampled every noun. The probe now trains on an
``in-distribution`` subset of nouns and reports class/binding accuracies
separately on training-noun pairs (``*_in_dist_acc``) and on held-out
nouns (``*_held_out_acc``). ``order_grammaticality_acc`` stays as the
primary diagnostic — its discrimination is structural (recurrent
state-tracking vs attention) and does not benefit from a held-out split.

Score uses log-prob of full sequences (sum log p(t_i | t_<i)). Fraction of
pairs where the well-formed sequence has higher log-prob → accuracy.
Random baseline = 0.5 (binary good/bad).

Output is intentionally not yet wired to leaderboard scoring — this is an
experimental probe. Wire only after cohort calibration shows real
architecture-to-architecture spread.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .synthetic_association_eval import (
    AssociationLayout,
    _ADJ_QUERY,
    _PAD,
    _VERB_QUERY,
    _association_target_int,
    _association_targets,
    _make_layout,
)
from .utils import _get_tiktoken_encoder, clip_grad_norm, make_adamw

logger = logging.getLogger(__name__)

NANO_BLIMP_METRIC_VERSION = "nano_blimp_v2"

_DEFAULT_ACTIVE_VOCAB = 32
_DEFAULT_TRAIN_STEPS = 300
_DEFAULT_BATCH = 32
_DEFAULT_LR = 1e-3
_DEFAULT_HELD_OUT_COUNT = 2
_TIMEOUT_S = 60.0


@dataclass(slots=True)
class NanoBLiMPResult:
    score: float = 0.0
    class_coherence_acc: float = 0.0
    binding_fidelity_acc: float = 0.0
    order_grammaticality_acc: float = 0.0
    # v2: held-out compositional split
    class_coherence_in_dist_acc: float = 0.0
    class_coherence_held_out_acc: float = 0.0
    binding_fidelity_in_dist_acc: float = 0.0
    binding_fidelity_held_out_acc: float = 0.0
    held_out_score: float = 0.0
    held_out_count: int = 0
    held_out_noun_ids: tuple[int, ...] = field(default_factory=tuple)
    n_pairs_per_test: int = 0
    n_in_dist_pairs: int = 0
    n_held_out_pairs: int = 0
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
            "nano_blimp_class_coherence_in_dist_acc": self.class_coherence_in_dist_acc,
            "nano_blimp_class_coherence_held_out_acc": self.class_coherence_held_out_acc,
            "nano_blimp_binding_fidelity_in_dist_acc": self.binding_fidelity_in_dist_acc,
            "nano_blimp_binding_fidelity_held_out_acc": self.binding_fidelity_held_out_acc,
            "nano_blimp_held_out_score": self.held_out_score,
            "nano_blimp_held_out_count": self.held_out_count,
            "nano_blimp_held_out_noun_ids": list(self.held_out_noun_ids),
            "nano_blimp_n_pairs_per_test": self.n_pairs_per_test,
            "nano_blimp_n_in_dist_pairs": self.n_in_dist_pairs,
            "nano_blimp_n_held_out_pairs": self.n_held_out_pairs,
            "nano_blimp_train_steps": self.n_train_steps,
            "nano_blimp_active_vocab_size": self.active_vocab_size,
            "nano_blimp_chance": self.chance,
            "nano_blimp_elapsed_ms": self.elapsed_ms,
            "nano_blimp_status": self.status,
            "nano_blimp_metric_version": self.metric_version,
        }


def _select_held_out_nouns(
    layout: AssociationLayout,
    held_out_count: int,
    seed: int,
) -> tuple[int, ...]:
    """Deterministically pick ``held_out_count`` noun IDs to hold out.

    Picked from the *end* of the noun range so the in-distribution pool is
    a contiguous prefix — keeps held-out noun indices stable across
    different vocab sizes while still varying with seed via a small offset.
    """
    n = layout.n_per_type
    held_out_count = max(0, int(held_out_count))
    if held_out_count == 0:
        return ()
    if held_out_count >= n - 1:
        # Need at least 2 in-dist nouns for the binding eval (cyclic neighbor).
        held_out_count = max(0, n - 2)
    if held_out_count == 0:
        return ()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed) ^ 0x1F2E3D)
    perm = torch.randperm(n, generator=rng).tolist()
    picks = sorted(perm[:held_out_count])
    return tuple(layout.noun_lo + idx for idx in picks)


def _seq_logprob(model: nn.Module, seqs: torch.Tensor) -> torch.Tensor:
    """Return sum-log-prob of next-token over each sequence (B,S) -> (B,)."""
    with torch.no_grad():
        logits = model(seqs)  # (B, S, V)
        log_probs = F.log_softmax(logits, dim=-1)
        # next-token log-prob: log p(t_i | t_<i) for i in [1, S-1]
        gathered = log_probs[:, :-1].gather(2, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
        return gathered.sum(dim=-1)


def _build_class_coherence_pairs(
    layout: AssociationLayout,
    device: str,
    nouns: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """For every (noun, query) in ``nouns``, build (good, bad) seq pairs:
    good ends in the right class; bad ends in the wrong class."""
    good = []
    bad = []
    for noun in nouns:
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
            wrong_token = wrong_lo + ((target - right_lo) % (wrong_hi - wrong_lo))
            good.append([noun, query, target])
            bad.append([noun, query, wrong_token])
    return (
        torch.tensor(good, dtype=torch.long, device=device),
        torch.tensor(bad, dtype=torch.long, device=device),
    )


def _build_binding_fidelity_pairs(
    layout: AssociationLayout,
    device: str,
    nouns: Sequence[int],
    *,
    distractor_pool: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Good: (noun_A, query, associated_word_A).
    Bad: (noun_A, query, associated_word_B), where B is drawn from
    ``distractor_pool`` (must contain ≥1 noun ≠ A).

    For the held-out split we want the distractor mapping to come from a
    noun the model *did* see during training, so the test isolates whether
    the model can generalize the binding rule to a new prefix — not whether
    it can compare two unseen mappings."""
    good: list[list[int]] = []
    bad: list[list[int]] = []
    pool = [n for n in distractor_pool]
    if not pool:
        return (
            torch.empty((0, 3), dtype=torch.long, device=device),
            torch.empty((0, 3), dtype=torch.long, device=device),
        )
    for i, noun_a in enumerate(nouns):
        # Pick a deterministic distractor from the pool that differs from A.
        offset = 1
        noun_b = pool[(i + offset) % len(pool)]
        guard = 0
        while noun_b == noun_a and guard < len(pool):
            offset += 1
            noun_b = pool[(i + offset) % len(pool)]
            guard += 1
        if noun_b == noun_a:
            continue
        for query in (_VERB_QUERY, _ADJ_QUERY):
            target_a = _association_target_int(noun_a, query, layout)
            target_b = _association_target_int(noun_b, query, layout)
            if target_a == target_b:
                continue
            good.append([noun_a, query, target_a])
            bad.append([noun_a, query, target_b])
    if not good:
        return (
            torch.empty((0, 3), dtype=torch.long, device=device),
            torch.empty((0, 3), dtype=torch.long, device=device),
        )
    return (
        torch.tensor(good, dtype=torch.long, device=device),
        torch.tensor(bad, dtype=torch.long, device=device),
    )


def _build_order_pairs(
    layout: AssociationLayout,
    device: str,
    nouns: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Good: [noun, query, target]. Bad: [query, noun, target] — same
    tokens, swapped order."""
    good = []
    bad = []
    for noun in nouns:
        for query in (_VERB_QUERY, _ADJ_QUERY):
            target = _association_target_int(noun, query, layout)
            good.append([noun, query, target])
            bad.append([query, noun, target])
    if not good:
        return (
            torch.empty((0, 3), dtype=torch.long, device=device),
            torch.empty((0, 3), dtype=torch.long, device=device),
        )
    return (
        torch.tensor(good, dtype=torch.long, device=device),
        torch.tensor(bad, dtype=torch.long, device=device),
    )


def _pair_accuracy(model: nn.Module, good: torch.Tensor, bad: torch.Tensor) -> float:
    """Fraction of pairs where good has higher sum-log-prob than bad."""
    if good.shape[0] == 0:
        return 0.0
    g = _seq_logprob(model, good)
    b = _seq_logprob(model, bad)
    correct = (g > b).sum().item()
    total = good.shape[0]
    return float(correct) / max(total, 1)


def _make_in_dist_train_batch(
    layout: AssociationLayout,
    in_dist_pool: torch.Tensor,
    batch_size: int,
    device: str,
    rng: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (noun, query, PAD) batches restricted to ``in_dist_pool``.

    Mirrors ``synthetic_association_eval._make_train_batch`` but draws
    nouns from a caller-supplied pool tensor — so the held-out nouns
    never appear in the training distribution."""
    pool_idx = torch.randint(
        0, in_dist_pool.shape[0], (int(batch_size),), device=device, generator=rng
    )
    nouns = in_dist_pool.index_select(0, pool_idx)
    relation_flag = torch.randint(0, 2, (batch_size,), device=device, generator=rng)
    relations = torch.where(
        relation_flag == 0,
        torch.full_like(relation_flag, _VERB_QUERY),
        torch.full_like(relation_flag, _ADJ_QUERY),
    )
    targets = _association_targets(nouns, relations, layout)
    input_ids = torch.empty(int(batch_size), 3, dtype=torch.long, device=device)
    input_ids[:, 0] = nouns
    input_ids[:, 1] = relations
    input_ids[:, 2] = _PAD
    return input_ids, targets


def _train_probe(
    model: nn.Module,
    layout: AssociationLayout,
    in_dist_nouns: tuple[int, ...],
    *,
    n_train_steps: int,
    batch_size: int,
    lr: float,
    device: str,
    deadline: float,
    seed: int,
) -> tuple[int, str]:
    """Train the model in place on the in-distribution noun subset.

    Caller must snapshot+restore state_dict (avoids ``copy.deepcopy``
    which fails on models with ``weight_norm`` parametrizations)."""
    if not in_dist_nouns:
        return 0, "no_in_dist_nouns"
    model.train()
    opt = make_adamw(model.parameters(), lr=lr)
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    pool = torch.tensor(in_dist_nouns, dtype=torch.long, device=device)
    steps = 0
    status = "ok"
    for step in range(int(n_train_steps)):
        if time.perf_counter() > deadline:
            status = "timeout"
            break
        input_ids, targets = _make_in_dist_train_batch(
            layout, pool, batch_size, device, rng
        )
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


def _weighted_split_accuracy(
    in_dist_acc: float,
    in_dist_count: int,
    held_out_acc: float,
    held_out_count: int,
) -> float:
    total = in_dist_count + held_out_count
    if total <= 0:
        return 0.0
    return (in_dist_acc * in_dist_count + held_out_acc * held_out_count) / total


def _evaluate_pair_split_metrics(
    model: nn.Module,
    *,
    class_in: tuple[torch.Tensor, torch.Tensor],
    class_held_out: tuple[torch.Tensor, torch.Tensor],
    binding_in: tuple[torch.Tensor, torch.Tensor],
    binding_held_out: tuple[torch.Tensor, torch.Tensor],
    order: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    good_c_in, bad_c_in = class_in
    good_c_ho, bad_c_ho = class_held_out
    good_b_in, bad_b_in = binding_in
    good_b_ho, bad_b_ho = binding_held_out
    good_o, bad_o = order

    cls_in = _pair_accuracy(model, good_c_in, bad_c_in)
    cls_ho = _pair_accuracy(model, good_c_ho, bad_c_ho)
    bind_in = _pair_accuracy(model, good_b_in, bad_b_in)
    bind_ho = _pair_accuracy(model, good_b_ho, bad_b_ho)
    order_acc = _pair_accuracy(model, good_o, bad_o)

    n_c_in = int(good_c_in.shape[0])
    n_c_ho = int(good_c_ho.shape[0])
    n_b_in = int(good_b_in.shape[0])
    n_b_ho = int(good_b_ho.shape[0])

    return {
        "class_coherence_acc": _weighted_split_accuracy(cls_in, n_c_in, cls_ho, n_c_ho),
        "class_coherence_in_dist_acc": cls_in,
        "class_coherence_held_out_acc": cls_ho,
        "binding_fidelity_acc": _weighted_split_accuracy(
            bind_in, n_b_in, bind_ho, n_b_ho
        ),
        "binding_fidelity_in_dist_acc": bind_in,
        "binding_fidelity_held_out_acc": bind_ho,
        "order_grammaticality_acc": order_acc,
        "n_in_dist_pairs": n_c_in,
        "n_held_out_pairs": n_c_ho,
        "n_pairs_per_test": int(good_o.shape[0]),
    }


def _evaluate_splits(
    model: nn.Module,
    layout: AssociationLayout,
    in_dist_nouns: tuple[int, ...],
    held_out_nouns: tuple[int, ...],
    *,
    device: str,
) -> dict[str, Any]:
    """Compute class/binding/order accuracies on the in-dist and held-out
    noun splits. Returns a dict with raw fields populated."""
    all_nouns = tuple(sorted(in_dist_nouns + held_out_nouns))

    good_c_in, bad_c_in = _build_class_coherence_pairs(layout, device, in_dist_nouns)
    good_c_ho, bad_c_ho = _build_class_coherence_pairs(layout, device, held_out_nouns)

    good_b_in, bad_b_in = _build_binding_fidelity_pairs(
        layout, device, in_dist_nouns, distractor_pool=in_dist_nouns
    )
    # Held-out binding: distractor pool = in-distribution nouns. The
    # comparison answer is built from a noun the model actually saw, so
    # this isolates "did the model learn the noun→association rule well
    # enough to apply it to a new noun?"
    good_b_ho, bad_b_ho = _build_binding_fidelity_pairs(
        layout, device, held_out_nouns, distractor_pool=in_dist_nouns
    )

    good_o, bad_o = _build_order_pairs(layout, device, all_nouns)

    return _evaluate_pair_split_metrics(
        model,
        class_in=(good_c_in, bad_c_in),
        class_held_out=(good_c_ho, bad_c_ho),
        binding_in=(good_b_in, bad_b_in),
        binding_held_out=(good_b_ho, bad_b_ho),
        order=(good_o, bad_o),
    )


def _build_result(
    *,
    splits: dict[str, Any],
    layout: AssociationLayout,
    held_out_nouns: tuple[int, ...],
    n_train_steps: int,
    elapsed_ms: float,
    status: str,
) -> NanoBLiMPResult:
    cls_in = float(splits["class_coherence_in_dist_acc"])
    cls_ho = float(splits["class_coherence_held_out_acc"])
    bind_in = float(splits["binding_fidelity_in_dist_acc"])
    bind_ho = float(splits["binding_fidelity_held_out_acc"])
    cls_all = float(splits["class_coherence_acc"])
    bind_all = float(splits["binding_fidelity_acc"])
    order_acc = float(splits["order_grammaticality_acc"])
    score = (cls_all + bind_all + order_acc) / 3.0
    if held_out_nouns:
        held_out_score = (cls_ho + bind_ho + order_acc) / 3.0
    else:
        held_out_score = 0.0
    return NanoBLiMPResult(
        score=round(score, 4),
        class_coherence_acc=round(cls_all, 4),
        binding_fidelity_acc=round(bind_all, 4),
        order_grammaticality_acc=round(order_acc, 4),
        class_coherence_in_dist_acc=round(cls_in, 4),
        class_coherence_held_out_acc=round(cls_ho, 4),
        binding_fidelity_in_dist_acc=round(bind_in, 4),
        binding_fidelity_held_out_acc=round(bind_ho, 4),
        held_out_score=round(held_out_score, 4),
        held_out_count=len(held_out_nouns),
        held_out_noun_ids=tuple(int(n) for n in held_out_nouns),
        n_pairs_per_test=int(splits["n_pairs_per_test"]),
        n_in_dist_pairs=int(splits["n_in_dist_pairs"]),
        n_held_out_pairs=int(splits["n_held_out_pairs"]),
        n_train_steps=int(n_train_steps),
        active_vocab_size=layout.active_vocab_size,
        elapsed_ms=round(elapsed_ms, 1),
        status=status,
    )


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
    held_out_count: int = _DEFAULT_HELD_OUT_COUNT,
) -> NanoBLiMPResult:
    """Train on association mappings (in-dist noun subset), then evaluate
    minimal-pair tests on both training-noun and held-out-noun splits."""
    t0 = time.perf_counter()
    deadline = t0 + float(timeout_s)
    layout = _make_layout(active_vocab_size)

    if layout.adjective_hi > int(getattr(model, "vocab_size", layout.adjective_hi)):
        return NanoBLiMPResult(
            active_vocab_size=layout.active_vocab_size,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status="model_vocab_too_small",
        )

    held_out_nouns = _select_held_out_nouns(layout, held_out_count, seed)
    all_nouns = tuple(range(layout.noun_lo, layout.noun_hi))
    in_dist_nouns = tuple(n for n in all_nouns if n not in set(held_out_nouns))

    saved_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    try:
        steps, status = _train_probe(
            model,
            layout,
            in_dist_nouns,
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
                held_out_count=len(held_out_nouns),
                held_out_noun_ids=held_out_nouns,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                status=status,
            )

        model.eval()
        splits = _evaluate_splits(
            model, layout, in_dist_nouns, held_out_nouns, device=device
        )
        return _build_result(
            splits=splits,
            layout=layout,
            held_out_nouns=held_out_nouns,
            n_train_steps=steps,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            status=status,
        )
    finally:
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
    held_out_nouns: Sequence[int] = (),
) -> NanoBLiMPResult:
    """Evaluate nano-BLiMP on an *already-trained* model.

    Use this when sharing the training pass between
    ``synthetic_association_score`` and ``nano_blimp_score`` — train once
    on the association data, then run both eval functions back-to-back.

    ``held_out_nouns`` should be the noun IDs the caller intentionally
    withheld from training. When empty (the v1 default), held-out fields
    are all 0 and the v1-shaped overall accuracies are returned. When
    provided, accuracies are reported separately for in-dist and held-out
    splits.

    Does NOT train, does NOT mutate model state.
    """
    t0 = time.perf_counter()
    was_training = trained_model.training
    trained_model.eval()
    try:
        held_set = set(int(n) for n in held_out_nouns)
        all_nouns = tuple(range(layout.noun_lo, layout.noun_hi))
        in_dist_nouns = tuple(n for n in all_nouns if n not in held_set)
        held_out_tuple = tuple(sorted(held_set))
        splits = _evaluate_splits(
            trained_model,
            layout,
            in_dist_nouns,
            held_out_tuple,
            device=device,
        )
        return _build_result(
            splits=splits,
            layout=layout,
            held_out_nouns=held_out_tuple,
            n_train_steps=0,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            status="ok",
        )
    finally:
        trained_model.train(was_training)


# ────────────────────────────────────────────────────────────────────────
# v3 — real-word held-out compositional probe
# ────────────────────────────────────────────────────────────────────────
#
# v2 audit on the 5-arch champion/frontier cohort produced inverted
# rankings: the genuine low-ppl real-text arch scored lowest, no-signal
# SSM hybrids scored highest. Diagnosis: ``order_grammaticality_acc``
# dominates ``held_out_score``, and on synthetic IDs 4–30 it actually
# measures "how fast does the embedding adapt to a 30-ID vocab",
# which is anti-correlated with having well-tuned wikitext embeddings.
#
# v3 keeps the held-out methodology from v2 but swaps the synthetic IDs
# for real words (codex's curated noun/verb lists, single-token under
# tiktoken-gpt2). Same compositional offset rule applied to real-word
# lexical anchors; same in-dist / held-out split; same eval shape.
#
# Sequence shape: ``[the, noun, verb]`` — 3 tokens, all single-token
# under tiktoken-gpt2 with leading space.
# Compositional rule: ``verb_idx = (noun_idx + 1) mod n_per_type``.

NANO_BLIMP_V3_METRIC_VERSION = "nano_blimp_v3_real_word"

_V3_DEFAULT_N_PER_TYPE = 24
_V3_DEFAULT_TRAIN_STEPS = 200
_V3_DEFAULT_BATCH = 32
_V3_DEFAULT_LR = 1e-3
_V3_DEFAULT_HELD_OUT_COUNT = 4
_V3_TIMEOUT_S = 90.0
_V3_TIKTOKEN_ENCODING = "gpt2"


@dataclass(frozen=True, slots=True)
class RealWordLayout:
    nouns: tuple[str, ...]
    verbs: tuple[str, ...]
    noun_ids: tuple[int, ...]
    verb_ids: tuple[int, ...]
    determiner: str
    determiner_id: int
    n_per_type: int
    encoding: str

    @property
    def max_token_id(self) -> int:
        return max(max(self.noun_ids), max(self.verb_ids), self.determiner_id)


@dataclass(slots=True)
class NanoBLiMPV3Result:
    score: float = 0.0
    class_coherence_acc: float = 0.0
    binding_fidelity_acc: float = 0.0
    order_grammaticality_acc: float = 0.0
    class_coherence_in_dist_acc: float = 0.0
    class_coherence_held_out_acc: float = 0.0
    binding_fidelity_in_dist_acc: float = 0.0
    binding_fidelity_held_out_acc: float = 0.0
    held_out_score: float = 0.0
    held_out_count: int = 0
    held_out_noun_indices: tuple[int, ...] = field(default_factory=tuple)
    held_out_noun_words: tuple[str, ...] = field(default_factory=tuple)
    n_pairs_per_test: int = 0
    n_in_dist_pairs: int = 0
    n_held_out_pairs: int = 0
    n_train_steps: int = 0
    n_per_type: int = 0
    elapsed_ms: float = 0.0
    status: str = "ok"
    metric_version: str = NANO_BLIMP_V3_METRIC_VERSION
    encoding: str = _V3_TIKTOKEN_ENCODING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nano_blimp_v3_score": self.score,
            "nano_blimp_v3_class_coherence_acc": self.class_coherence_acc,
            "nano_blimp_v3_binding_fidelity_acc": self.binding_fidelity_acc,
            "nano_blimp_v3_order_grammaticality_acc": self.order_grammaticality_acc,
            "nano_blimp_v3_class_coherence_in_dist_acc": self.class_coherence_in_dist_acc,
            "nano_blimp_v3_class_coherence_held_out_acc": self.class_coherence_held_out_acc,
            "nano_blimp_v3_binding_fidelity_in_dist_acc": self.binding_fidelity_in_dist_acc,
            "nano_blimp_v3_binding_fidelity_held_out_acc": self.binding_fidelity_held_out_acc,
            "nano_blimp_v3_held_out_score": self.held_out_score,
            "nano_blimp_v3_held_out_count": self.held_out_count,
            "nano_blimp_v3_held_out_noun_indices": list(self.held_out_noun_indices),
            "nano_blimp_v3_held_out_noun_words": list(self.held_out_noun_words),
            "nano_blimp_v3_n_pairs_per_test": self.n_pairs_per_test,
            "nano_blimp_v3_n_in_dist_pairs": self.n_in_dist_pairs,
            "nano_blimp_v3_n_held_out_pairs": self.n_held_out_pairs,
            "nano_blimp_v3_train_steps": self.n_train_steps,
            "nano_blimp_v3_n_per_type": self.n_per_type,
            "nano_blimp_v3_elapsed_ms": self.elapsed_ms,
            "nano_blimp_v3_status": self.status,
            "nano_blimp_v3_metric_version": self.metric_version,
            "nano_blimp_v3_encoding": self.encoding,
        }


def _filter_single_token_words(
    words: Sequence[str], enc: Any, *, leading_space: bool = True
) -> list[tuple[str, int]]:
    """Keep only words that tokenize to exactly one token (with leading
    space when ``leading_space`` is true). Preserves input order, deduped."""
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    for w in words:
        text = " " + w if leading_space else w
        ids = enc.encode(text, allowed_special=set())
        if len(ids) == 1 and w not in seen:
            out.append((w, int(ids[0])))
            seen.add(w)
    return out


@lru_cache(maxsize=4)
def build_real_word_layout(
    n_per_type: int = _V3_DEFAULT_N_PER_TYPE,
    *,
    encoding: str = _V3_TIKTOKEN_ENCODING,
    determiner: str = "the",
) -> RealWordLayout:
    """Construct a real-word layout from the shared real-word vocab,
    filtered to single-token words. Imported lazily to avoid pulling in
    tiktoken at module load. Append-only ordering invariant on
    ``REAL_WORD_NOUNS`` / ``REAL_WORD_VERBS`` keeps the first-24 single-
    token slice byte-identical to the original cohort audit."""
    from ._real_word_vocab import REAL_WORD_NOUNS, REAL_WORD_VERBS

    enc = _get_tiktoken_encoder(encoding)
    nouns_pairs = _filter_single_token_words(REAL_WORD_NOUNS, enc)
    verbs_pairs = _filter_single_token_words(REAL_WORD_VERBS, enc)
    if len(nouns_pairs) < n_per_type or len(verbs_pairs) < n_per_type:
        raise ValueError(
            f"only {len(nouns_pairs)} single-token nouns and "
            f"{len(verbs_pairs)} single-token verbs available; need {n_per_type}"
        )
    nouns_pairs = nouns_pairs[:n_per_type]
    verbs_pairs = verbs_pairs[:n_per_type]
    det_ids = enc.encode(" " + determiner, allowed_special=set())
    if len(det_ids) != 1:
        raise ValueError(f"determiner {determiner!r} did not encode to a single token")
    return RealWordLayout(
        nouns=tuple(w for w, _ in nouns_pairs),
        verbs=tuple(w for w, _ in verbs_pairs),
        noun_ids=tuple(i for _, i in nouns_pairs),
        verb_ids=tuple(i for _, i in verbs_pairs),
        determiner=determiner,
        determiner_id=int(det_ids[0]),
        n_per_type=n_per_type,
        encoding=encoding,
    )


def _associated_verb_idx(noun_idx: int, n_per_type: int) -> int:
    """Compositional offset rule, mirroring the synthetic v2 mapping."""
    return (int(noun_idx) + 1) % int(n_per_type)


def _select_held_out_v3(
    layout: RealWordLayout, held_out_count: int, seed: int
) -> tuple[int, ...]:
    n = layout.n_per_type
    held_out_count = max(0, int(held_out_count))
    if held_out_count == 0:
        return ()
    if held_out_count >= n - 1:
        held_out_count = max(0, n - 2)
    if held_out_count == 0:
        return ()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed) ^ 0x3D2C1B)
    perm = torch.randperm(n, generator=rng).tolist()
    return tuple(sorted(perm[:held_out_count]))


def _v3_class_pairs(
    layout: RealWordLayout, device: str, noun_indices: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Good: ``[the, noun, associated_verb]``. Bad: ``[the, noun, other_noun]``
    — wrong-class continuation."""
    good: list[list[int]] = []
    bad: list[list[int]] = []
    for ni in noun_indices:
        v_idx = _associated_verb_idx(ni, layout.n_per_type)
        wrong_noun_idx = (ni + 1) % layout.n_per_type
        good.append([layout.determiner_id, layout.noun_ids[ni], layout.verb_ids[v_idx]])
        bad.append(
            [layout.determiner_id, layout.noun_ids[ni], layout.noun_ids[wrong_noun_idx]]
        )
    if not good:
        empty = torch.empty((0, 3), dtype=torch.long, device=device)
        return empty, empty
    return (
        torch.tensor(good, dtype=torch.long, device=device),
        torch.tensor(bad, dtype=torch.long, device=device),
    )


def _v3_binding_pairs(
    layout: RealWordLayout,
    device: str,
    target_noun_indices: Sequence[int],
    distractor_pool: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Good: ``[the, noun_A, verb_A]``. Bad: ``[the, noun_A, verb_B]`` where
    verb_B is the verb associated with some in-distribution noun ≠ A.
    Tests whether the model bound the right (noun, verb) pair, not just
    "verb after noun"."""
    good: list[list[int]] = []
    bad: list[list[int]] = []
    pool = [int(i) for i in distractor_pool]
    if not pool:
        empty = torch.empty((0, 3), dtype=torch.long, device=device)
        return empty, empty
    for k, ni in enumerate(target_noun_indices):
        my_v = layout.verb_ids[_associated_verb_idx(ni, layout.n_per_type)]
        chosen_verb: int | None = None
        for off in range(1, len(pool) + 1):
            cand = pool[(k + off) % len(pool)]
            if cand == ni:
                continue
            cand_v = layout.verb_ids[_associated_verb_idx(cand, layout.n_per_type)]
            if cand_v != my_v:
                chosen_verb = cand_v
                break
        if chosen_verb is None:
            continue
        good.append([layout.determiner_id, layout.noun_ids[ni], my_v])
        bad.append([layout.determiner_id, layout.noun_ids[ni], chosen_verb])
    if not good:
        empty = torch.empty((0, 3), dtype=torch.long, device=device)
        return empty, empty
    return (
        torch.tensor(good, dtype=torch.long, device=device),
        torch.tensor(bad, dtype=torch.long, device=device),
    )


def _v3_order_pairs(
    layout: RealWordLayout, device: str, noun_indices: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Good: ``[the, noun, verb]``. Bad: ``[the, verb, noun]`` — same
    content tokens, swapped order."""
    good: list[list[int]] = []
    bad: list[list[int]] = []
    for ni in noun_indices:
        v_id = layout.verb_ids[_associated_verb_idx(ni, layout.n_per_type)]
        good.append([layout.determiner_id, layout.noun_ids[ni], v_id])
        bad.append([layout.determiner_id, v_id, layout.noun_ids[ni]])
    if not good:
        empty = torch.empty((0, 3), dtype=torch.long, device=device)
        return empty, empty
    return (
        torch.tensor(good, dtype=torch.long, device=device),
        torch.tensor(bad, dtype=torch.long, device=device),
    )


def _v3_make_train_batch(
    layout: RealWordLayout,
    in_dist_idx_pool: torch.Tensor,
    batch_size: int,
    device: str,
    rng: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample noun *indices* from ``in_dist_idx_pool``, build
    ``[the, noun_id, PAD]`` sequences, return target verb-pool indices."""
    pool_idx = torch.randint(
        0,
        in_dist_idx_pool.shape[0],
        (int(batch_size),),
        device=device,
        generator=rng,
    )
    noun_indices = in_dist_idx_pool.index_select(0, pool_idx)
    noun_ids = torch.tensor(
        [layout.noun_ids[int(ni)] for ni in noun_indices.tolist()],
        dtype=torch.long,
        device=device,
    )
    target_verb_pool_idx = torch.tensor(
        [
            _associated_verb_idx(int(ni), layout.n_per_type)
            for ni in noun_indices.tolist()
        ],
        dtype=torch.long,
        device=device,
    )
    input_ids = torch.empty(int(batch_size), 3, dtype=torch.long, device=device)
    input_ids[:, 0] = layout.determiner_id
    input_ids[:, 1] = noun_ids
    input_ids[:, 2] = _PAD
    return input_ids, target_verb_pool_idx


def _v3_train_probe(
    model: nn.Module,
    layout: RealWordLayout,
    in_dist_indices: tuple[int, ...],
    *,
    n_train_steps: int,
    batch_size: int,
    lr: float,
    device: str,
    deadline: float,
    seed: int,
) -> tuple[int, str]:
    """Train the model in place to predict the associated verb after
    ``[the, noun]``. Restricts training to the in-dist noun subset.
    Cross-entropy is taken over the n_per_type verb-IDs (not the full
    vocab) so the loss is comparable across model vocab sizes."""
    if not in_dist_indices:
        return 0, "no_in_dist_nouns"
    model.train()
    opt = make_adamw(model.parameters(), lr=lr)
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    in_dist_pool = torch.tensor(in_dist_indices, dtype=torch.long, device=device)
    verb_id_tensor = torch.tensor(layout.verb_ids, dtype=torch.long, device=device)
    steps = 0
    status = "ok"
    for step in range(int(n_train_steps)):
        if time.perf_counter() > deadline:
            status = "timeout"
            break
        input_ids, target_verb_pool_idx = _v3_make_train_batch(
            layout, in_dist_pool, batch_size, device, rng
        )
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids)
        verb_logits = logits[:, 1, :].index_select(1, verb_id_tensor)
        loss = F.cross_entropy(verb_logits, target_verb_pool_idx)
        if not torch.isfinite(loss):
            status = "non_finite_loss"
            break
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step()
        steps = step + 1
    return steps, status


def _v3_evaluate_splits(
    model: nn.Module,
    layout: RealWordLayout,
    in_dist_indices: tuple[int, ...],
    held_out_indices: tuple[int, ...],
    *,
    device: str,
) -> dict[str, Any]:
    all_indices = tuple(sorted(set(in_dist_indices) | set(held_out_indices)))

    good_c_in, bad_c_in = _v3_class_pairs(layout, device, in_dist_indices)
    good_c_ho, bad_c_ho = _v3_class_pairs(layout, device, held_out_indices)
    good_b_in, bad_b_in = _v3_binding_pairs(
        layout, device, in_dist_indices, distractor_pool=in_dist_indices
    )
    # Held-out binding distractor pool is in-dist only — same rationale as
    # v2: isolate "applies the rule to a new noun" from "comparison of two
    # unseen mappings."
    good_b_ho, bad_b_ho = _v3_binding_pairs(
        layout, device, held_out_indices, distractor_pool=in_dist_indices
    )
    good_o, bad_o = _v3_order_pairs(layout, device, all_indices)

    return _evaluate_pair_split_metrics(
        model,
        class_in=(good_c_in, bad_c_in),
        class_held_out=(good_c_ho, bad_c_ho),
        binding_in=(good_b_in, bad_b_in),
        binding_held_out=(good_b_ho, bad_b_ho),
        order=(good_o, bad_o),
    )


def _v3_build_result(
    *,
    splits: dict[str, Any],
    layout: RealWordLayout,
    held_out_indices: tuple[int, ...],
    n_train_steps: int,
    elapsed_ms: float,
    status: str,
) -> NanoBLiMPV3Result:
    cls_in = float(splits["class_coherence_in_dist_acc"])
    cls_ho = float(splits["class_coherence_held_out_acc"])
    bind_in = float(splits["binding_fidelity_in_dist_acc"])
    bind_ho = float(splits["binding_fidelity_held_out_acc"])
    order_acc = float(splits["order_grammaticality_acc"])
    cls_all = float(splits["class_coherence_acc"])
    bind_all = float(splits["binding_fidelity_acc"])
    score = (cls_all + bind_all + order_acc) / 3.0
    held_out_score = (cls_ho + bind_ho + order_acc) / 3.0 if held_out_indices else 0.0
    return NanoBLiMPV3Result(
        score=round(score, 4),
        class_coherence_acc=round(cls_all, 4),
        binding_fidelity_acc=round(bind_all, 4),
        order_grammaticality_acc=round(order_acc, 4),
        class_coherence_in_dist_acc=round(cls_in, 4),
        class_coherence_held_out_acc=round(cls_ho, 4),
        binding_fidelity_in_dist_acc=round(bind_in, 4),
        binding_fidelity_held_out_acc=round(bind_ho, 4),
        held_out_score=round(held_out_score, 4),
        held_out_count=len(held_out_indices),
        held_out_noun_indices=tuple(int(i) for i in held_out_indices),
        held_out_noun_words=tuple(layout.nouns[i] for i in held_out_indices),
        n_pairs_per_test=int(splits["n_pairs_per_test"]),
        n_in_dist_pairs=int(splits["n_in_dist_pairs"]),
        n_held_out_pairs=int(splits["n_held_out_pairs"]),
        n_train_steps=int(n_train_steps),
        n_per_type=layout.n_per_type,
        elapsed_ms=round(elapsed_ms, 1),
        status=status,
        encoding=layout.encoding,
    )


def nano_blimp_v3_score(
    model: nn.Module,
    *,
    n_per_type: int = _V3_DEFAULT_N_PER_TYPE,
    n_train_steps: int = _V3_DEFAULT_TRAIN_STEPS,
    batch_size: int = _V3_DEFAULT_BATCH,
    lr: float = _V3_DEFAULT_LR,
    device: str = "cuda",
    seed: int = 42,
    timeout_s: float = _V3_TIMEOUT_S,
    held_out_count: int = _V3_DEFAULT_HELD_OUT_COUNT,
    encoding: str = _V3_TIKTOKEN_ENCODING,
) -> NanoBLiMPV3Result:
    """Real-word held-out compositional probe.

    Trains the model to predict the offset-rule-associated verb after
    ``[the, noun]`` for an in-distribution noun subset, then evaluates
    class coherence, binding fidelity, and order grammaticality on
    minimal pairs. Class and binding are split into in-dist and held-out
    accuracies; order uses all nouns.

    Caller's model state is preserved (state_dict snapshot/restore;
    survives weight_norm parametrize).
    """
    t0 = time.perf_counter()
    deadline = t0 + float(timeout_s)
    try:
        layout = build_real_word_layout(n_per_type=n_per_type, encoding=encoding)
    except ValueError as exc:
        logger.debug("v3 layout build failed: %s", exc)
        return NanoBLiMPV3Result(
            n_per_type=n_per_type,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status="layout_unavailable",
        )

    model_vocab = int(getattr(model, "vocab_size", 0) or 0)
    if model_vocab <= 0 or layout.max_token_id >= model_vocab:
        return NanoBLiMPV3Result(
            n_per_type=layout.n_per_type,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            status="model_vocab_too_small",
        )

    held_out_indices = _select_held_out_v3(layout, held_out_count, seed)
    held_set = set(held_out_indices)
    in_dist_indices = tuple(i for i in range(layout.n_per_type) if i not in held_set)

    saved_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    try:
        steps, status = _v3_train_probe(
            model,
            layout,
            in_dist_indices,
            n_train_steps=n_train_steps,
            batch_size=batch_size,
            lr=lr,
            device=device,
            deadline=deadline,
            seed=seed,
        )
        if status not in ("ok", "timeout"):
            return NanoBLiMPV3Result(
                n_train_steps=steps,
                n_per_type=layout.n_per_type,
                held_out_count=len(held_out_indices),
                held_out_noun_indices=held_out_indices,
                held_out_noun_words=tuple(layout.nouns[i] for i in held_out_indices),
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                status=status,
                encoding=layout.encoding,
            )

        model.eval()
        splits = _v3_evaluate_splits(
            model, layout, in_dist_indices, held_out_indices, device=device
        )
        return _v3_build_result(
            splits=splits,
            layout=layout,
            held_out_indices=held_out_indices,
            n_train_steps=steps,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            status=status,
        )
    finally:
        model.load_state_dict(saved_state)
        model.train(was_training)
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
