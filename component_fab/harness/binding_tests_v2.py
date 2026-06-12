"""Binding/induction tests v2 — five nano-scale capability probes.

Adds discrimination axes that fab's current gate stack misses:
- multi-token induction (Anthropic-style induction head circuit)
- selective copy (Mamba-paper task; isolates selective state)
- dyck-2 (hierarchical state; correlates with BLiMP island/filler-gap)
- variable-delay repeat (information preservation across arbitrary gap)
- NPI-shaped synthetic (flag-propagation across context)

Each test:
1. Generates synthetic data (deterministic from seed).
2. Trains a TinyLM(lane) for a small number of steps.
3. Measures task-specific accuracy on held-out batch.

Returns ``BindingTestsV2Result`` summarizing all 5 tests per spec.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .tiny_lm import TinyLM
from .training_probe import build_tiny_lm, gather_logits_at

logger = logging.getLogger(__name__)


def _train_and_score(
    model: nn.Module,
    train_batches: list,
    eval_batches: list,
    n_train_steps: int,
    *,
    probe: str,
) -> float:
    """Train then score a probe. A failed probe scores 0.0 (so ranking can
    proceed), but the failure is logged loudly rather than silently swallowed."""
    try:
        _train_steps(model, train_batches, n_train_steps)
        return _eval_accuracy(model, eval_batches)
    except Exception:  # noqa: BLE001 - one bad lane must not abort the cohort
        logger.warning("binding probe %r failed; scoring 0.0", probe, exc_info=True)
        return 0.0


@dataclass(frozen=True, slots=True)
class BindingTestsV2Result:
    label: str
    multi_token_induction_acc: float
    selective_copy_acc: float
    variable_delay_acc_dict: dict[str, float]
    variable_delay_acc_mean: float
    npi_synthetic_acc: float
    overall_score: float
    elapsed_s: float


def _build_lm(
    lane_factory: Callable[[int], nn.Module],
    *,
    vocab_size: int,
    dim: int,
    n_blocks: int,
    max_seq_len: int,
) -> TinyLM:
    return build_tiny_lm(
        lane_factory,
        vocab_size=vocab_size,
        dim=dim,
        n_blocks=n_blocks,
        max_seq_len=max_seq_len,
        use_position_embedding=True,
        stable_init=True,
    )


def _model_dim(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    dim = getattr(cfg, "dim", None)
    if not dim:
        raise ValueError(
            f"cannot determine model dim for {type(model).__name__}: "
            "expected a `.config.dim` attribute"
        )
    return int(dim)


def _train_steps(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    n_steps: int,
    lr: float | None = None,
    clip_grad: float = 1.0,
) -> None:
    """Train ``model`` on ``(input_ids, target_positions, target_ids)`` batches."""
    dim = _model_dim(model)
    actual_lr = float(lr) if lr is not None else 3e-3 * math.sqrt(64.0 / max(1, dim))
    optim = torch.optim.Adam(model.parameters(), lr=actual_lr)
    model.train()
    for step in range(n_steps):
        ids, pos, tgt = batches[step % len(batches)]
        gathered = gather_logits_at(model(ids), pos)
        loss = nn.functional.cross_entropy(
            gathered.reshape(-1, gathered.shape[-1]), tgt.reshape(-1)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite binding loss at step {step}")
        optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        if torch.is_tensor(grad_norm) and not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite binding grad at step {step}")
        optim.step()


def _eval_accuracy(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ids, pos, tgt in batches:
            preds = gather_logits_at(model(ids), pos).argmax(dim=-1)
            correct += int((preds == tgt).sum().item())
            total += int(tgt.numel())
    return correct / max(1, total)


# ---------- Test 1: multi-token (3-gram) induction ----------


def _gen_induction_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    n_distractors: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bigram induction: [a, b, distractor*N, a, ?] → predict b.

    Canonical single-step Anthropic induction head: at the second
    occurrence of `a`, model must look back to find the token that
    followed the first `a` (= `b`) and emit it. Earlier 3-gram variant
    was too hard at dim=32, 120 steps. Bigram is the standard
    induction-head probe.
    """
    ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    pos = torch.zeros(batch_size, 1, dtype=torch.long)
    tgt = torch.zeros(batch_size, 1, dtype=torch.long)
    for sample in range(batch_size):
        # Pick a, b — distinct.
        pair = torch.randperm(vocab_size, generator=rng)[:2].tolist()
        a, b_tok = pair
        # Plant (a, b) at start.
        ids[sample, 0] = a
        ids[sample, 1] = b_tok
        cursor = 2
        # Insert distractor tokens (not a, not b).
        for _ in range(n_distractors):
            if cursor >= seq_len - 2:
                break
            d = int(torch.randint(0, vocab_size, (1,), generator=rng).item())
            if d in pair:
                d = (d + 2) % vocab_size
                if d in pair:
                    d = (d + 1) % vocab_size
            ids[sample, cursor] = d
            cursor += 1
        # Query: at position of second `a`, predict b_tok (the next token).
        if cursor + 1 < seq_len:
            ids[sample, cursor] = a
            pos[sample, 0] = cursor  # predict from logits at this position
            tgt[sample, 0] = b_tok
    return ids, pos, tgt


def test_multi_token_induction(
    lane_factory: Callable[[int], nn.Module],
    *,
    dim: int = 32,
    n_blocks: int = 2,
    vocab_size: int = 16,
    seq_len: int = 32,
    batch_size: int = 32,
    n_distractors: int = 8,
    n_train_steps: int = 80,
    n_eval_batches: int = 4,
    seed: int = 0,
) -> float:
    """Train + eval; returns held-out accuracy."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = _build_lm(
        lane_factory,
        vocab_size=vocab_size,
        dim=dim,
        n_blocks=n_blocks,
        max_seq_len=seq_len,
    )
    train_batches = [
        _gen_induction_batch(batch_size, seq_len, vocab_size, n_distractors, rng)
        for _ in range(16)
    ]
    eval_batches = [
        _gen_induction_batch(batch_size, seq_len, vocab_size, n_distractors, rng)
        for _ in range(n_eval_batches)
    ]
    return _train_and_score(
        model, train_batches, eval_batches, n_train_steps, probe="multi_token_induction"
    )


# ---------- Test 2: selective copy ----------


def _gen_selective_copy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    n_content: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Input: content tokens scattered among NOISE; mask region at end.

    Layout: [t1, NOISE*r1, t2, NOISE*r2, ..., t_N, NOISE*rN, MASK*N]
    Targets: at the N mask positions, output t1, t2, ..., tN in order.
    """
    NOISE = vocab_size - 2
    MASK = vocab_size - 1
    ids = torch.full((batch_size, seq_len), NOISE, dtype=torch.long)
    pos = torch.zeros(batch_size, n_content, dtype=torch.long)
    tgt = torch.zeros(batch_size, n_content, dtype=torch.long)
    for b in range(batch_size):
        contents = [
            int(torch.randint(0, vocab_size - 2, (1,), generator=rng).item())
            for _ in range(n_content)
        ]
        # Reserve last n_content positions for masks.
        mask_start = seq_len - n_content
        # Randomly place contents in [0, mask_start) maintaining order.
        possible_slots = torch.randperm(mask_start, generator=rng)[:n_content].tolist()
        possible_slots.sort()
        for i, slot in enumerate(possible_slots):
            ids[b, slot] = contents[i]
        # Place mask region.
        for i in range(n_content):
            ids[b, mask_start + i] = MASK
            pos[b, i] = mask_start + i
            tgt[b, i] = contents[i]
    return ids, pos, tgt


def test_selective_copy(
    lane_factory: Callable[[int], nn.Module],
    *,
    dim: int = 32,
    n_blocks: int = 2,
    vocab_size: int = 16,
    seq_len: int = 32,
    batch_size: int = 32,
    n_content: int = 4,
    n_train_steps: int = 80,
    n_eval_batches: int = 4,
    seed: int = 0,
) -> float:
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = _build_lm(
        lane_factory,
        vocab_size=vocab_size,
        dim=dim,
        n_blocks=n_blocks,
        max_seq_len=seq_len,
    )
    train_batches = [
        _gen_selective_copy_batch(batch_size, seq_len, vocab_size, n_content, rng)
        for _ in range(16)
    ]
    eval_batches = [
        _gen_selective_copy_batch(batch_size, seq_len, vocab_size, n_content, rng)
        for _ in range(n_eval_batches)
    ]
    return _train_and_score(
        model, train_batches, eval_batches, n_train_steps, probe="selective_copy"
    )


# ---------- Test 3: dyck-2 (next valid bracket) ----------


def _gen_dyck2_v3_batch(
    batch_size: int,
    seq_len: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vocab: 0=(, 1=[, 2=), 3=], 4=PAD.

    Hard dyck-2: place 4 OPENS at the start, then noise, then ask which
    close-type matches the FIRST open. Recency heuristics fail because
    the relevant open is far from the query.

    Layout: [open_1, open_2, open_3, open_4, noise*K, ?] → predict
    CLOSE(open_1) — the type that would close the OUTERMOST (first) open.

    For the query at the end of a balanced-in-noise context, the model
    must remember open_1's TYPE despite three intervening opens and many
    noise tokens. Recency = wrong.
    """
    NOISE_OPEN_PAIRS = (
        (0, 2),  # ()
        (1, 3),  # []
    )
    OPEN = (0, 1)
    CLOSE = {0: 2, 1: 3}
    PAD = 4
    ids = torch.full((batch_size, seq_len), PAD, dtype=torch.long)
    pos = torch.zeros(batch_size, 1, dtype=torch.long)
    tgt = torch.zeros(batch_size, 1, dtype=torch.long)
    for b in range(batch_size):
        # 4 outermost opens, all distinct positions, alternating types occasionally.
        outer_opens = [
            int(torch.randint(0, 2, (1,), generator=rng).item()) for _ in range(4)
        ]
        outer_open_tokens = [OPEN[o] for o in outer_opens]
        seq: list[int] = list(outer_open_tokens)
        # Add balanced noise pairs (full open-close cycles) until close to end.
        while len(seq) + 2 < seq_len - 1:
            pair_idx = int(torch.randint(0, 2, (1,), generator=rng).item())
            o, c = NOISE_OPEN_PAIRS[pair_idx]
            seq.append(o)
            seq.append(c)
        # Truncate.
        seq = seq[: seq_len - 1]
        for i, s in enumerate(seq):
            ids[b, i] = s
        # Query: at the last filled position, predict the close that matches outer_opens[3]
        # (the FOURTH outermost, which is the innermost of the original outer 4 = most recent
        # outer, but separated by noise from query). Tests state-tracking across noise.
        # NOTE: noise pairs are balanced — they don't change the stack of outer opens.
        # So the next valid close is CLOSE[outer_opens[3]] = matching the innermost outer.
        pos[b, 0] = len(seq) - 1
        tgt[b, 0] = CLOSE[outer_opens[3]]
    return ids, pos, tgt


def test_dyck2_v3(
    lane_factory: Callable[[int], nn.Module],
    *,
    dim: int = 64,
    n_blocks: int = 2,
    seq_len: int = 48,
    batch_size: int = 32,
    n_train_steps: int = 300,
    n_eval_batches: int = 4,
    seed: int = 0,
) -> float:
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    VOCAB = 5
    model = _build_lm(
        lane_factory,
        vocab_size=VOCAB,
        dim=dim,
        n_blocks=n_blocks,
        max_seq_len=seq_len,
    )
    train_batches = [_gen_dyck2_v3_batch(batch_size, seq_len, rng) for _ in range(16)]
    eval_batches = [
        _gen_dyck2_v3_batch(batch_size, seq_len, rng) for _ in range(n_eval_batches)
    ]
    return _train_and_score(
        model, train_batches, eval_batches, n_train_steps, probe="dyck2_v3"
    )


# ---------- Test 4: variable-delay repeat ----------


def _gen_repeat_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    delay: int,
    n_repeat: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Input: [x1, x2, x3, NOISE*delay, REPEAT, ?, ?, ?] -> targets x1, x2, x3."""
    NOISE = vocab_size - 2
    REPEAT = vocab_size - 1
    ids = torch.full((batch_size, seq_len), NOISE, dtype=torch.long)
    pos = torch.zeros(batch_size, n_repeat, dtype=torch.long)
    tgt = torch.zeros(batch_size, n_repeat, dtype=torch.long)
    for b in range(batch_size):
        contents = [
            int(torch.randint(0, vocab_size - 2, (1,), generator=rng).item())
            for _ in range(n_repeat)
        ]
        for i, c in enumerate(contents):
            if i < seq_len:
                ids[b, i] = c
        repeat_marker_pos = n_repeat + delay
        if repeat_marker_pos < seq_len:
            ids[b, repeat_marker_pos] = REPEAT
        for i in range(n_repeat):
            qpos = repeat_marker_pos + 1 + i
            if qpos >= seq_len:
                break
            pos[b, i] = qpos
            tgt[b, i] = contents[i]
    return ids, pos, tgt


def test_variable_delay_repeat(
    lane_factory: Callable[[int], nn.Module],
    *,
    dim: int = 32,
    n_blocks: int = 2,
    vocab_size: int = 16,
    n_repeat: int = 3,
    delays: tuple[int, ...] = (1, 4, 16),
    seq_len_per_delay: dict[int, int] | None = None,
    batch_size: int = 32,
    n_train_steps: int = 80,
    n_eval_batches: int = 4,
    seed: int = 0,
) -> tuple[dict[str, float], float]:
    """Returns (per-delay acc dict, mean across delays)."""
    seq_len_per_delay = seq_len_per_delay or {
        d: max(24, n_repeat * 2 + d + 4) for d in delays
    }
    per_delay: dict[str, float] = {}
    for delay in delays:
        torch.manual_seed(seed + delay)
        rng = torch.Generator().manual_seed(seed + delay)
        sl = seq_len_per_delay[delay]
        model = _build_lm(
            lane_factory,
            vocab_size=vocab_size,
            dim=dim,
            n_blocks=n_blocks,
            max_seq_len=sl,
        )
        train_batches = [
            _gen_repeat_batch(batch_size, sl, vocab_size, delay, n_repeat, rng)
            for _ in range(16)
        ]
        eval_batches = [
            _gen_repeat_batch(batch_size, sl, vocab_size, delay, n_repeat, rng)
            for _ in range(n_eval_batches)
        ]
        try:
            _train_steps(model, train_batches, n_train_steps)
            per_delay[f"delay_{delay}"] = _eval_accuracy(model, eval_batches)
        except Exception:  # noqa: BLE001 - one bad delay must not abort the probe
            logger.warning(
                "variable-delay repeat probe failed at delay=%d; scoring 0.0",
                delay,
                exc_info=True,
            )
            per_delay[f"delay_{delay}"] = 0.0
    mean = sum(per_delay.values()) / max(1, len(per_delay))
    return per_delay, mean


# ---------- Test 5: NPI-shaped synthetic ----------


def _gen_npi_v2_batch(
    batch_size: int,
    seq_len: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NPI v2: licensor is at position 0, then K noise clauses, then NPI query.

    Vocab: 0=NEG, 1=POS, 2=NOUN, 3=VERB, 4=ANY (NPI), 5=OTHER, 6=PAD, 7=CONJ.

    Layout: [LICENSOR, NOUN, VERB, CONJ, NOUN, VERB, CONJ, ..., NOUN, VERB, ?]
    The polarity item position is at the END, separated from LICENSOR by K
    intervening NOUN-VERB-CONJ triples. Recency-based prediction fails because
    nothing between LICENSOR and ? indicates polarity.
    """
    NEG, POS, NOUN, VERB, ANY, OTHER, PAD, CONJ = range(8)
    ids = torch.full((batch_size, seq_len), PAD, dtype=torch.long)
    pos = torch.zeros(batch_size, 1, dtype=torch.long)
    tgt = torch.zeros(batch_size, 1, dtype=torch.long)
    for b in range(batch_size):
        is_neg = torch.rand((), generator=rng).item() < 0.5
        licensor = NEG if is_neg else POS
        ids[b, 0] = licensor
        cursor = 1
        # Fill with NOUN-VERB-CONJ triples until 4 tokens before end.
        while cursor + 5 < seq_len:
            ids[b, cursor] = NOUN
            ids[b, cursor + 1] = VERB
            ids[b, cursor + 2] = CONJ
            cursor += 3
        # Query: position seq_len-1 — predict ANY (well-formed) or OTHER (ill-formed)
        if cursor + 2 < seq_len:
            ids[b, cursor] = NOUN
            ids[b, cursor + 1] = VERB
            ids[b, cursor + 2] = PAD  # query slot
            pos[b, 0] = cursor + 1
            tgt[b, 0] = ANY if is_neg else OTHER
    return ids, pos, tgt


def test_npi_synthetic_v2(
    lane_factory: Callable[[int], nn.Module],
    *,
    dim: int = 64,
    n_blocks: int = 2,
    seq_len: int = 48,
    batch_size: int = 32,
    n_train_steps: int = 300,
    n_eval_batches: int = 4,
    seed: int = 0,
) -> float:
    """NPI v2 with long-distance licensor → polarity-item separation.

    Defeats simple bag-of-tokens prediction by inserting many NOUN-VERB-CONJ
    triples between the licensor and the query. Model must carry the polarity
    flag across 10-15 intervening tokens.
    """
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    VOCAB = 8
    model = _build_lm(
        lane_factory,
        vocab_size=VOCAB,
        dim=dim,
        n_blocks=n_blocks,
        max_seq_len=seq_len,
    )
    train_batches = [_gen_npi_v2_batch(batch_size, seq_len, rng) for _ in range(16)]
    eval_batches = [
        _gen_npi_v2_batch(batch_size, seq_len, rng) for _ in range(n_eval_batches)
    ]
    return _train_and_score(
        model, train_batches, eval_batches, n_train_steps, probe="npi_synthetic_v2"
    )


# ---------- Combined runner ----------


def run_all_binding_tests_v2(
    lane_factory: Callable[[int], nn.Module],
    label: str,
    *,
    dim: int = 32,
    n_blocks: int = 2,
    n_train_steps: int = 80,
    seed: int = 0,
) -> BindingTestsV2Result:
    """Run all 5 tests on a single lane factory. Returns a result dataclass."""
    t0 = time.monotonic()
    mti = test_multi_token_induction(
        lane_factory, dim=dim, n_blocks=n_blocks, n_train_steps=n_train_steps, seed=seed
    )
    sc = test_selective_copy(
        lane_factory,
        dim=dim,
        n_blocks=n_blocks,
        n_train_steps=n_train_steps,
        seed=seed + 1,
    )
    # dyck-2 excluded from composite: validation showed every architecture
    # scored 1.0 (deterministic target from balanced-string structure).
    # The saturated v1/v2 variants were deleted; ``test_dyck2_v3`` is the
    # surviving hard variant (used by research/tools/scaling_blimp_study).
    rep_dict, rep_mean = test_variable_delay_repeat(
        lane_factory,
        dim=dim,
        n_blocks=n_blocks,
        n_train_steps=n_train_steps,
        seed=seed + 3,
    )
    # NPI v2 (long-distance licensor separation) replaced the recency-soluble
    # v1 variant in the composite; same seed offset, same step budget.
    npi = test_npi_synthetic_v2(
        lane_factory,
        dim=dim,
        n_blocks=n_blocks,
        n_train_steps=n_train_steps,
        seed=seed + 4,
    )
    elapsed = time.monotonic() - t0
    # 4-test composite (dyck-2 dropped).
    overall = (mti + sc + rep_mean + npi) / 4.0
    return BindingTestsV2Result(
        label=label,
        multi_token_induction_acc=mti,
        selective_copy_acc=sc,
        variable_delay_acc_dict=rep_dict,
        variable_delay_acc_mean=rep_mean,
        npi_synthetic_acc=npi,
        overall_score=overall,
        elapsed_s=elapsed,
    )
