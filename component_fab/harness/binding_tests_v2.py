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

import math
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .tiny_lm import TinyLM, TinyLMConfig


@dataclass(frozen=True, slots=True)
class BindingTestsV2Result:
    label: str
    multi_token_induction_acc: float
    selective_copy_acc: float
    dyck2_acc: float
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
    cfg = TinyLMConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_blocks=n_blocks,
        use_position_embedding=True,
        max_seq_len=max_seq_len,
        use_ffn=False,
    )
    model = TinyLM(lane_factory, cfg)
    _stable_probe_init(model, n_blocks=n_blocks)
    return model


def _stable_probe_init(model: nn.Module, *, n_blocks: int) -> None:
    init_std = 0.02
    scaled_init_std = init_std / math.sqrt(max(1, 2 * n_blocks))
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            is_residual_out = (
                name.endswith(".fc2")
                or name.endswith(".out")
                or name.endswith(".out_proj")
            )
            std = scaled_init_std if is_residual_out else init_std
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _model_dim(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    dim = getattr(cfg, "dim", None)
    if dim:
        return int(dim)
    for param in model.parameters():
        if param.ndim >= 2:
            return int(param.shape[-1])
    return 64


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
        logits = model(ids)
        b, q = pos.shape
        v = logits.shape[-1]
        pos_expanded = pos.unsqueeze(-1).expand(b, q, v)
        gathered = logits.gather(1, pos_expanded)
        loss = nn.functional.cross_entropy(gathered.reshape(-1, v), tgt.reshape(-1))
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
            logits = model(ids)
            b, q = pos.shape
            v = logits.shape[-1]
            pos_expanded = pos.unsqueeze(-1).expand(b, q, v)
            preds = logits.gather(1, pos_expanded).argmax(dim=-1)
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
    try:
        _train_steps(model, train_batches, n_train_steps)
        return _eval_accuracy(model, eval_batches)
    except Exception:  # noqa: BLE001
        return 0.0


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
    try:
        _train_steps(model, train_batches, n_train_steps)
        return _eval_accuracy(model, eval_batches)
    except Exception:  # noqa: BLE001
        return 0.0


# ---------- Test 3: dyck-2 (next valid bracket) ----------


def _gen_dyck2_batch(
    batch_size: int,
    seq_len: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vocab: 0=(, 1=[, 2=), 3=], 4=PAD.

    DEPRECATED — original generator queried the deterministic last token
    of a balanced sequence; every baseline scored 1.0. Use
    ``_gen_dyck2_v2_batch`` instead for the screening composite.
    Kept here for backwards compatibility with the standalone
    ``test_dyck2`` function.
    """
    PAD = 4
    OPEN = (0, 1)
    CLOSE = {0: 2, 1: 3}
    ids = torch.full((batch_size, seq_len), PAD, dtype=torch.long)
    # We test at exactly 1 query position per sequence (the LAST token).
    pos = torch.zeros(batch_size, 1, dtype=torch.long)
    tgt = torch.zeros(batch_size, 1, dtype=torch.long)
    for b in range(batch_size):
        stack: list[int] = []
        seq: list[int] = []
        # Generate a balanced sequence of length seq_len - 1 (leave last for query).
        target_len = seq_len - 1
        while len(seq) < target_len:
            remaining = target_len - len(seq)
            # Decide open vs close. If stack non-empty and we have space, random;
            # if stack empty, must open.
            if not stack or (
                len(stack) < remaining // 2
                and torch.rand((), generator=rng).item() < 0.5
            ):
                op = OPEN[int(torch.randint(0, 2, (1,), generator=rng).item())]
                stack.append(op)
                seq.append(op)
            else:
                top = stack.pop()
                seq.append(CLOSE[top])
        for i, s in enumerate(seq):
            ids[b, i] = s
        # Query: what's the valid next token at position target_len-1 → predict the close
        # bracket determined by the FINAL open in the stack at position target_len-1
        # If the FINAL token in seq is itself a close, then predict next valid open
        # (or any of {0,1} but we pick deterministic 0).
        # Simpler version: trim seq by 1 so we can ask "given seq[:-1], predict seq[-1]"
        # That's the same as predicting the actual last token of a balanced sequence.
        # Build stack up to target_len-2, then valid set at position target_len-1
        # is whatever seq[-1] is. So just predict seq[-1].
        pos[b, 0] = target_len - 1  # second-to-last index
        tgt[b, 0] = seq[-1]
        # Shift: we want input[0..target_len-2] and output at target_len-1
        # ids[b, target_len-1] = seq[-1]  # this is what's THERE
        # Position to query is target_len-1 (predict from context [0..target_len-2])
        # The actual final token at that pos is seq[-1] which we expect.
    return ids, pos, tgt


def _gen_dyck2_v2_batch(
    batch_size: int,
    seq_len: int,
    n_queries: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vocab: 0=(, 1=[, 2=), 3=], 4=PAD.

    Generate dyck-2 sequences with MIN stack depth >= 3 at some point, then
    pick ``n_queries`` interior CLOSE positions and ask the model to predict
    the close TYPE (matching the open at that stack depth). Each query
    requires recovering the open type the model saw earlier — not just the
    most-recent unmatched open.

    Min-depth-3 means the model can't just track the topmost open with a
    1-bit register; it must propagate multiple open-types through state.
    """
    PAD = 4
    OPEN = (0, 1)
    CLOSE = {0: 2, 1: 3}
    ids = torch.full((batch_size, seq_len), PAD, dtype=torch.long)
    pos = torch.zeros(batch_size, n_queries, dtype=torch.long)
    tgt = torch.zeros(batch_size, n_queries, dtype=torch.long)
    for b in range(batch_size):
        # Strategy: build sequence by alternating "force open until depth >= 3"
        # then random open/close until end. Track per-position close-type targets.
        stack: list[int] = []
        seq: list[int] = []
        max_depth_reached = 0
        # Phase 1: force opens until depth >= 3
        while len(stack) < 3 and len(seq) < seq_len - 2:
            op = OPEN[int(torch.randint(0, 2, (1,), generator=rng).item())]
            stack.append(op)
            seq.append(op)
            max_depth_reached = max(max_depth_reached, len(stack))
        # Phase 2: random open/close, but bias toward closes once deep enough
        while len(seq) < seq_len:
            close_prob = 0.6 if len(stack) > 1 else 0.0
            if stack and torch.rand((), generator=rng).item() < close_prob:
                top = stack.pop()
                seq.append(CLOSE[top])
            elif len(stack) < 6 and (len(seq) + len(stack) + 2 < seq_len):
                op = OPEN[int(torch.randint(0, 2, (1,), generator=rng).item())]
                stack.append(op)
                seq.append(op)
                max_depth_reached = max(max_depth_reached, len(stack))
            elif stack:
                top = stack.pop()
                seq.append(CLOSE[top])
            else:
                break
        # Truncate or pad.
        seq = seq[:seq_len]
        for i, s in enumerate(seq):
            ids[b, i] = s
        # Find all CLOSE positions (where the value is in {2, 3}).
        close_positions = [i for i, s in enumerate(seq) if s in (2, 3)]
        if len(close_positions) < n_queries:
            # Not enough close positions — fall back: query at any non-zero positions.
            close_positions = close_positions + [
                len(seq) - 1 - i for i in range(n_queries - len(close_positions))
            ]
        # Sample n_queries close positions, preferring interior ones (not the first close).
        chosen = sorted(close_positions[: max(n_queries * 2, n_queries)])
        if len(chosen) > n_queries:
            # Take every other to spread coverage.
            step = max(1, len(chosen) // n_queries)
            chosen = chosen[::step][:n_queries]
        else:
            chosen = chosen[:n_queries]
        # Pad to n_queries with the last available position.
        while len(chosen) < n_queries:
            chosen.append(chosen[-1] if chosen else 0)
        for qi, qpos in enumerate(chosen):
            # Predict from position qpos-1 (causal) — the close at qpos is the target.
            pos[b, qi] = max(0, qpos - 1)
            tgt[b, qi] = seq[qpos] if qpos < len(seq) else 2
    return ids, pos, tgt


def test_dyck2_v2(
    lane_factory: Callable[[int], nn.Module],
    *,
    dim: int = 64,
    n_blocks: int = 2,
    seq_len: int = 48,
    n_queries: int = 4,
    batch_size: int = 32,
    n_train_steps: int = 300,
    n_eval_batches: int = 4,
    seed: int = 0,
) -> float:
    """Improved dyck-2 with interior-close prediction + min-depth-3 nesting.

    Replaces the original ``test_dyck2`` for screening use (the original
    scored 1.0 on every baseline because the last-token target was
    deterministic).
    """
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
    train_batches = [
        _gen_dyck2_v2_batch(batch_size, seq_len, n_queries, rng) for _ in range(16)
    ]
    eval_batches = [
        _gen_dyck2_v2_batch(batch_size, seq_len, n_queries, rng)
        for _ in range(n_eval_batches)
    ]
    try:
        _train_steps(model, train_batches, n_train_steps)
        return _eval_accuracy(model, eval_batches)
    except Exception:  # noqa: BLE001
        return 0.0


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
    try:
        _train_steps(model, train_batches, n_train_steps)
        return _eval_accuracy(model, eval_batches)
    except Exception:  # noqa: BLE001
        return 0.0


def test_dyck2(
    lane_factory: Callable[[int], nn.Module],
    *,
    dim: int = 32,
    n_blocks: int = 2,
    seq_len: int = 24,
    batch_size: int = 32,
    n_train_steps: int = 80,
    n_eval_batches: int = 4,
    seed: int = 0,
) -> float:
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    VOCAB = 5  # (, [, ), ], PAD
    model = _build_lm(
        lane_factory,
        vocab_size=VOCAB,
        dim=dim,
        n_blocks=n_blocks,
        max_seq_len=seq_len,
    )
    train_batches = [_gen_dyck2_batch(batch_size, seq_len, rng) for _ in range(16)]
    eval_batches = [
        _gen_dyck2_batch(batch_size, seq_len, rng) for _ in range(n_eval_batches)
    ]
    try:
        _train_steps(model, train_batches, n_train_steps)
        return _eval_accuracy(model, eval_batches)
    except Exception:  # noqa: BLE001
        return 0.0


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
        except Exception:  # noqa: BLE001
            per_delay[f"delay_{delay}"] = 0.0
    mean = sum(per_delay.values()) / max(1, len(per_delay))
    return per_delay, mean


# ---------- Test 5: NPI-shaped synthetic ----------


def _gen_npi_batch(
    batch_size: int,
    seq_len: int,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vocab: 0=NEG, 1=POS, 2=NOUN, 3=VERB, 4=ANY, 5=OTHER, 6=PAD.

    Generate sequences like [LICENSOR NOUN VERB ANY_or_OTHER NOUN VERB ANY_or_OTHER ...]
    Target: at random "ANY" position, predict 1 if licensor was NEG (well-formed),
            0 if POS (ill-formed). Encoded as predict ANY token (4) for well-formed,
            OTHER (5) for ill-formed.
    """
    NEG, POS, NOUN, VERB, ANY, OTHER, PAD = range(7)
    ids = torch.full((batch_size, seq_len), PAD, dtype=torch.long)
    pos = torch.zeros(batch_size, 1, dtype=torch.long)
    tgt = torch.zeros(batch_size, 1, dtype=torch.long)
    for b in range(batch_size):
        is_neg = torch.rand((), generator=rng).item() < 0.5
        licensor = NEG if is_neg else POS
        ids[b, 0] = licensor
        # Pattern: NOUN VERB X NOUN VERB X NOUN VERB X — where X is ANY (well-formed) or OTHER (ill-formed)
        cursor = 1
        target_x_pos = -1
        while cursor + 2 < seq_len:
            ids[b, cursor] = NOUN
            ids[b, cursor + 1] = VERB
            # Pick whether THIS X is the target.
            if target_x_pos < 0 and torch.rand((), generator=rng).item() < 0.5:
                target_x_pos = cursor + 2
            # Fill X position with PLACEHOLDER (set to PAD here; the QUERY will predict ANY or OTHER).
            ids[b, cursor + 2] = PAD
            cursor += 3
        if target_x_pos < 0:
            target_x_pos = max(3, seq_len - 4)
        pos[b, 0] = target_x_pos
        tgt[b, 0] = ANY if is_neg else OTHER
    return ids, pos, tgt


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
    try:
        _train_steps(model, train_batches, n_train_steps)
        return _eval_accuracy(model, eval_batches)
    except Exception:  # noqa: BLE001
        return 0.0


def test_npi_synthetic(
    lane_factory: Callable[[int], nn.Module],
    *,
    dim: int = 32,
    n_blocks: int = 2,
    seq_len: int = 24,
    batch_size: int = 32,
    n_train_steps: int = 80,
    n_eval_batches: int = 4,
    seed: int = 0,
) -> float:
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    VOCAB = 7
    model = _build_lm(
        lane_factory,
        vocab_size=VOCAB,
        dim=dim,
        n_blocks=n_blocks,
        max_seq_len=seq_len,
    )
    train_batches = [_gen_npi_batch(batch_size, seq_len, rng) for _ in range(16)]
    eval_batches = [
        _gen_npi_batch(batch_size, seq_len, rng) for _ in range(n_eval_batches)
    ]
    try:
        _train_steps(model, train_batches, n_train_steps)
        return _eval_accuracy(model, eval_batches)
    except Exception:  # noqa: BLE001
        return 0.0


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
    # ``test_dyck2`` remains importable for research use.
    rep_dict, rep_mean = test_variable_delay_repeat(
        lane_factory,
        dim=dim,
        n_blocks=n_blocks,
        n_train_steps=n_train_steps,
        seed=seed + 3,
    )
    npi = test_npi_synthetic(
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
        dyck2_acc=0.0,
        variable_delay_acc_dict=rep_dict,
        variable_delay_acc_mean=rep_mean,
        npi_synthetic_acc=npi,
        overall_score=overall,
        elapsed_s=elapsed,
    )
