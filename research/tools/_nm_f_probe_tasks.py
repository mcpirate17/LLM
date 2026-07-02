"""NM-F probe tasks C-F: generators, split evals, and runners (F2/F3/F5/F6).

Companion module of ``research/tools/nm_f_capability_probes.py`` (split out
2026-07-02 to keep the harness under the god-file limit). The harness owns the
model plumbing (``ProbeLM``, ``build_mixer``, ``train_model``,
``eval_accuracy``) and the CLI; this module owns the token-layout constants,
the four newer task generators, their split-metric evals, and their runners:

  * **overwrite (NM-F2)** — k->v1 ... k->v2 rebinding; split accuracy
    overwritten vs once-written control keys (exact replacement vs additive
    blend).
  * **anagram (NM-F3)** — same/permuted window discrimination; a commutative
    mixer is at chance by construction.
  * **modcounter (NM-F5)** — marked-token running count mod 2 / mod 4, trained
    at body length 128, evaled to 4096 (32x extrapolation).
  * **induction (NM-F6)** — classic induction with the A...B copy distance
    swept far past the train range.

Probe rationale, JSON schema, and usage live in the harness module docstring.
This module never imports the harness at module level: the harness imports the
runners from here, so the back-reference is resolved lazily at call time
(``_harness()``), which also keeps script (``__main__``) execution sound.
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from research.tools.nm_f_capability_probes import ProbeLM

QUERY_TOK = 1
JUDGE_TOK = 2  # F3 anagram: same/permuted marker position
MARK_TOK = 3  # F5 modcounter: the token being counted
MOD2_QUERY_TOK = 4  # F5 modcounter: trailing query position for count % 2
MOD4_QUERY_TOK = 5  # F5 modcounter: trailing query position for count % 4
KEYS = (8, 72)  # 64 keys
ALT_VOCAB = (72, 128)  # disjoint alphabet: F3 window tokens, F6 induction A/B
VALUES = (128, 192)  # 64 values
MOD2_LABELS = (192, 193)  # F5: count % 2 in {0, 1}
MOD4_LABELS = (194, 195, 196, 197)  # F5: count % 4 in {0, 1, 2, 3}
MARK_PROB = 0.3  # F5: Bernoulli-per-position mark rate (unpredictable count)
FILLER = (200, 250)
SAME_TOK = 250  # F3 anagram label: probe window == original
PERM_TOK = 251  # F3 anagram label: probe window is a genuine permutation


def _harness():
    """The parent harness module, imported lazily (call time, both modules
    fully initialized) — a module-level back-import would be circular."""
    from research.tools import nm_f_capability_probes

    return nm_f_capability_probes


# ── task generators (positions randomized per sequence — no recency shortcut) ──


def _place_write_block(
    x: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    seg_start: int,
    seg_len: int,
    n: int,
    device: torch.device,
    gen: torch.Generator,
) -> None:
    """In-place: scatter ``n`` (key, value) pairs at a random contiguous offset
    within ``x[:, seg_start:seg_start+seg_len]`` — the write-block placement
    shared by both segments of the F2 overwrite probe."""
    rows = torch.arange(x.shape[0], device=device).unsqueeze(1)
    offset = torch.randint(0, seg_len - 2 * n + 1, (x.shape[0], 1), generator=gen).to(
        device
    )
    pos = seg_start + offset + 2 * torch.arange(n, device=device).unsqueeze(0)
    x[rows, pos] = keys
    x[rows, pos + 1] = values


def make_overwrite_batch(
    batch: int,
    n_pairs: int,
    n_overwrite: int,
    seg_len: int,
    device: torch.device,
    gen: torch.Generator,
    n_keys: int = 64,
    n_values: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """F2 overwrite probe: write block 1 binds ``n_pairs`` keys to v1 (RANDOM
    offset within the first ``seg_len``-token segment); write block 2 REBINDS a
    random ``n_overwrite``-key subset to v2 != v1 (random offset within the
    second segment); a single dense query lists all ``n_pairs`` keys in random
    order. Labels are each key's FINAL value (v2 for overwritten keys, v1 for
    once-written control keys) — an additive-blend memory that never fully
    erases v1 fails the overwritten split while an exact-overwrite memory does
    not. Returns ``(x, labels, is_overwritten)``: ``is_overwritten`` marks each
    query position's split for accuracy reporting."""
    if 2 * n_pairs > seg_len or n_overwrite > n_pairs or n_overwrite < 1:
        raise ValueError(
            f"seg_len={seg_len} too short, or bad n_overwrite={n_overwrite}"
            f" for n_pairs={n_pairs}"
        )
    if not (
        n_pairs <= n_keys <= KEYS[1] - KEYS[0]
        and 2 <= n_values <= VALUES[1] - VALUES[0]
    ):
        raise ValueError(f"bad difficulty n_keys={n_keys}, n_values={n_values}")
    total_len = 2 * seg_len + 1 + n_pairs
    x = torch.randint(FILLER[0], FILLER[1], (batch, total_len), generator=gen).to(
        device
    )
    keys = (
        torch.stack(
            [torch.randperm(n_keys, generator=gen)[:n_pairs] for _ in range(batch)]
        ).to(device)
        + KEYS[0]
    )
    values1 = (
        torch.randint(0, n_values, (batch, n_pairs), generator=gen).to(device)
        + VALUES[0]
    )
    # v2 sampled to differ from v1 at each overwritten key (a real overwrite).
    v1_off = values1[:, :n_overwrite] - VALUES[0]
    v2_off = torch.randint(0, n_values - 1, (batch, n_overwrite), generator=gen).to(
        device
    )
    v2_off = v2_off + (v2_off >= v1_off).long()
    values2 = v2_off + VALUES[0]

    _place_write_block(x, keys, values1, 0, seg_len, n_pairs, device, gen)
    _place_write_block(
        x, keys[:, :n_overwrite], values2, seg_len, seg_len, n_overwrite, device, gen
    )

    qpos = 2 * seg_len
    x[:, qpos] = QUERY_TOK
    order = torch.stack(
        [torch.randperm(n_pairs, generator=gen) for _ in range(batch)]
    ).to(device)
    x[:, qpos + 1 :] = torch.gather(keys, 1, order)

    final_values = values1.clone()
    final_values[:, :n_overwrite] = values2
    is_overwritten_key = torch.zeros(batch, n_pairs, dtype=torch.bool, device=device)
    is_overwritten_key[:, :n_overwrite] = True

    labels = torch.full((batch, total_len), -100, dtype=torch.long, device=device)
    labels[:, qpos + 1 :] = torch.gather(final_values, 1, order)
    overwritten_mask = torch.zeros(batch, total_len, dtype=torch.bool, device=device)
    overwritten_mask[:, qpos + 1 :] = torch.gather(is_overwritten_key, 1, order)
    return x, labels, overwritten_mask


def _nonidentity_permutation(n: int, gen: torch.Generator) -> torch.Tensor:
    """A ``torch.randperm`` draw rejection-resampled to exclude the identity —
    every "permuted" example in the anagram probe must be genuinely reordered."""
    identity = torch.arange(n)
    perm = torch.randperm(n, generator=gen)
    while torch.equal(perm, identity):
        perm = torch.randperm(n, generator=gen)
    return perm


def make_anagram_batch(
    batch: int, window: int, device: torch.device, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """F3 order-discrimination probe: a ``window``-length span drawn from the
    ``ALT_VOCAB`` alphabet (disjoint from filler, so a permutation can't be
    faked by repeated filler tokens), then — after a random gap — either an
    EXACT repeat of the span (label ``SAME_TOK``) or a genuine random
    PERMUTATION of the same multiset (label ``PERM_TOK``; identity permutation
    is rejected). A commutative/sum-pooling mixer sees an identical
    bag-of-tokens in both cases and sits at chance (0.5); ``NilpotentLieScan``'s
    Chen-identity level-2 term is order-sensitive by construction."""
    if window < 2:
        raise ValueError(f"window must be >= 2 to permute, got {window}")
    if window > ALT_VOCAB[1] - ALT_VOCAB[0]:
        raise ValueError(f"window={window} exceeds the ALT_VOCAB alphabet size")
    prefix = int(torch.randint(1, 9, (1,), generator=gen))
    gap = int(torch.randint(1, 9, (1,), generator=gen))
    seq_len = prefix + window + gap + window + 1
    x = torch.randint(FILLER[0], FILLER[1], (batch, seq_len), generator=gen).to(device)
    window_tok = (
        torch.stack(
            [
                torch.randperm(ALT_VOCAB[1] - ALT_VOCAB[0], generator=gen)[:window]
                for _ in range(batch)
            ]
        ).to(device)
        + ALT_VOCAB[0]
    )
    x[:, prefix : prefix + window] = window_tok
    is_same = torch.rand(batch, generator=gen) < 0.5
    probe_idx = torch.stack(
        [
            torch.arange(window)
            if is_same[b]
            else _nonidentity_permutation(window, gen)
            for b in range(batch)
        ]
    ).to(device)
    probe = torch.gather(window_tok, 1, probe_idx)
    probe_start = prefix + window + gap
    x[:, probe_start : probe_start + window] = probe
    x[:, -1] = JUDGE_TOK
    labels = torch.full((batch, seq_len), -100, dtype=torch.long, device=device)
    labels[:, -1] = torch.where(
        is_same.to(device),
        torch.full((batch,), SAME_TOK, device=device),
        torch.full((batch,), PERM_TOK, device=device),
    )
    return x, labels


def make_modcounter_batch(
    batch: int, body_len: int, device: torch.device, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """F5 length-extrapolated mod-counter: ``MARK_TOK`` appears at each body
    position with probability ``MARK_PROB`` (Bernoulli count, not fixed — the
    parity/mod4 total can't be read off from position alone); two trailing
    query positions (``MOD2_QUERY_TOK``, ``MOD4_QUERY_TOK``) predict the
    running mark count mod 2 and mod 4 (dense: both positions supervised). No
    key/value binding — this isolates state-tracking arithmetic, the axis
    ``PortHamiltonianMixer``'s certified energy-norm contraction targets at
    length extrapolation."""
    if body_len < 1:
        raise ValueError(f"body_len must be >= 1, got {body_len}")
    seq_len = body_len + 2
    x = torch.randint(FILLER[0], FILLER[1], (batch, seq_len), generator=gen).to(device)
    is_mark = torch.rand(batch, body_len, generator=gen).to(device) < MARK_PROB
    x[:, :body_len] = torch.where(
        is_mark, torch.full_like(x[:, :body_len], MARK_TOK), x[:, :body_len]
    )
    count = is_mark.long().sum(dim=1)
    x[:, body_len] = MOD2_QUERY_TOK
    x[:, body_len + 1] = MOD4_QUERY_TOK
    labels = torch.full((batch, seq_len), -100, dtype=torch.long, device=device)
    labels[:, body_len] = MOD2_LABELS[0] + (count % 2)
    labels[:, body_len + 1] = MOD4_LABELS[0] + (count % 4)
    return x, labels


def make_induction_batch(
    batch: int, gap: int, device: torch.device, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """F6 induction distance-generalization probe: ``[... A B ... (gap fillers)
    ... A] -> predict B`` at the position right after the repeated A (classic
    induction, ``A``/``B`` drawn from ``ALT_VOCAB``, no separate query marker —
    the repeated token itself is the trigger). Trained with ``gap`` sampled
    from ``{8,32,64}``; eval sweeps gap buckets far past train range (up to
    2048, 32x) to test whether ``ScaleEquivariantWaveletStack``'s dyadic
    dilation set (receptive field ``2**n_scales`` while params grow only
    linearly) tracks an A...B pair whose distance exceeds every literal filter
    tap."""
    if gap < 1:
        raise ValueError(f"gap must be >= 1, got {gap}")
    prefix = int(torch.randint(1, 9, (1,), generator=gen))
    seq_len = prefix + 2 + gap + 1
    x = torch.randint(FILLER[0], FILLER[1], (batch, seq_len), generator=gen).to(device)
    a_tok = (
        torch.randint(0, ALT_VOCAB[1] - ALT_VOCAB[0], (batch,), generator=gen).to(
            device
        )
        + ALT_VOCAB[0]
    )
    b_tok = (
        torch.randint(0, ALT_VOCAB[1] - ALT_VOCAB[0], (batch,), generator=gen).to(
            device
        )
        + ALT_VOCAB[0]
    )
    x[:, prefix] = a_tok
    x[:, prefix + 1] = b_tok
    x[:, -1] = a_tok
    labels = torch.full((batch, seq_len), -100, dtype=torch.long, device=device)
    labels[:, -1] = b_tok
    return x, labels


# ── split-metric evals ──


@torch.no_grad()
def eval_overwrite_accuracy(
    model: "ProbeLM", sample_batch, n_seq: int, batch: int, gen
) -> dict[str, float]:
    """F2 split accuracy: overwritten-key queries (must return v2) vs
    once-written control-key queries (must return v1)."""
    model.eval()
    correct = {"overwritten": 0, "control": 0}
    total = {"overwritten": 0, "control": 0}
    for _ in range(math.ceil(n_seq / batch)):
        x, y, is_overwritten = sample_batch(batch, gen)
        hit = model(x).argmax(dim=-1) == y
        qmask = y != -100
        for split, split_mask in (
            ("overwritten", is_overwritten),
            ("control", ~is_overwritten),
        ):
            sel = qmask & split_mask
            correct[split] += int((hit & sel).sum())
            total[split] += int(sel.sum())
    return {k: correct[k] / total[k] for k in correct}


@torch.no_grad()
def eval_modcounter_accuracy(
    model: "ProbeLM", sample_batch, n_seq: int, batch: int, gen
) -> dict[str, float]:
    """F5 split accuracy for the two trailing query positions: running count
    mod 2 (second-to-last) and mod 4 (last)."""
    model.eval()
    correct = {"mod2": 0, "mod4": 0}
    total = 0
    for _ in range(math.ceil(n_seq / batch)):
        x, y = sample_batch(batch, gen)
        preds = model(x).argmax(dim=-1)
        correct["mod2"] += int((preds[:, -2] == y[:, -2]).sum())
        correct["mod4"] += int((preds[:, -1] == y[:, -1]).sum())
        total += x.shape[0]
    return {k: correct[k] / total for k in correct}


# ── probe runners (same loop shape as the harness's binding/retention) ──


def run_overwrite_probe(args, device: torch.device) -> dict:
    """Probe C: exact-overwrite vs additive-blend leakage. Accuracy split into
    OVERWRITTEN keys (re-bound k->v2, must return v2) and CONTROL keys (bound
    once, must return v1) — an additive fast-weight memory returns a v1/v2
    mixture on the overwritten split and fails there while passing control;
    ``IdempotentObliqueMemory``'s projector update targets exact replacement
    on both."""
    h = _harness()
    train_pairs, eval_pairs, seg_len = (4, 8), (4, 8, 16), args.body_len
    mixers = args.mixers.split(",") if args.mixers else ["oblique", "attn"]
    nk, nv = args.n_keys, args.n_values
    results: dict = {
        "task": "overwrite",
        "train_pairs": train_pairs,
        "seg_len": seg_len,
        "n_keys": nk,
        "n_values": nv,
        "saturation_control": "none",
    }
    for name in mixers:
        per_seed: dict[int, dict] = {}
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            model = h.ProbeLM(name).to(device)

            def full_sample(
                b: int, g, seg_len=seg_len, nk=nk, nv=nv
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                n = train_pairs[
                    int(torch.randint(0, len(train_pairs), (1,), generator=g))
                ]
                n_ow = max(1, n // 2)
                return make_overwrite_batch(b, n, n_ow, seg_len, device, g, nk, nv)

            def train_sample(b: int, g) -> tuple[torch.Tensor, torch.Tensor]:
                x, y, _is_overwritten = full_sample(b, g)
                return x, y

            losses = h.train_model(
                model, train_sample, args.steps, args.batch, device, gen, args.lr
            )
            accs = {
                n: eval_overwrite_accuracy(
                    model,
                    lambda b, g, n=n: make_overwrite_batch(
                        b, n, max(1, n // 2), seg_len, device, g, nk, nv
                    ),
                    args.eval_seqs,
                    args.batch,
                    torch.Generator().manual_seed(10_000 + seed),
                )
                for n in eval_pairs
            }
            per_seed[seed] = {
                "acc_by_pairs": accs,
                "final_loss": losses[-1],
                "first_loss": losses[0],
            }
            print(f"[overwrite] {name} seed={seed} acc={accs}", flush=True)
        results[name] = {
            "per_seed": per_seed,
            "median_acc_by_pairs": {
                n: {
                    split: statistics.median(
                        per_seed[s]["acc_by_pairs"][n][split] for s in per_seed
                    )
                    for split in ("overwritten", "control")
                }
                for n in eval_pairs
            },
            "non_embedding_params": h.ProbeLM(name).non_embedding_params(),
        }
    return results


def run_anagram_probe(args, device: torch.device) -> dict:
    """Probe D: same/permuted binary accuracy vs window size. A commutative
    (order-insensitive/sum-pooling) mixer is at CHANCE (0.5) by construction —
    ``NilpotentLieScan``'s zero-param Chen-identity level-2 term is the
    mechanism that should clear this task."""
    h = _harness()
    train_windows, eval_windows = (4, 8), (4, 8, 16, 32)
    mixers = args.mixers.split(",") if args.mixers else ["lie", "attn"]
    results: dict = {
        "task": "anagram",
        "train_windows": train_windows,
        "eval_windows": eval_windows,
        "chance_accuracy": 0.5,
        "saturation_control": "none",
    }
    for name in mixers:
        per_seed: dict[int, dict] = {}
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            model = h.ProbeLM(name).to(device)

            def sample(b: int, g) -> tuple[torch.Tensor, torch.Tensor]:
                w = train_windows[
                    int(torch.randint(0, len(train_windows), (1,), generator=g))
                ]
                return make_anagram_batch(b, w, device, g)

            losses = h.train_model(
                model, sample, args.steps, args.batch, device, gen, args.lr
            )
            accs = {
                w: h.eval_accuracy(
                    model,
                    lambda b, g, w=w: make_anagram_batch(b, w, device, g),
                    args.eval_seqs,
                    args.batch,
                    torch.Generator().manual_seed(10_000 + seed),
                )
                for w in eval_windows
            }
            per_seed[seed] = {
                "acc_by_window": accs,
                "final_loss": losses[-1],
                "first_loss": losses[0],
            }
            print(f"[anagram] {name} seed={seed} acc={accs}", flush=True)
        results[name] = {
            "per_seed": per_seed,
            "median_acc_by_window": {
                w: statistics.median(per_seed[s]["acc_by_window"][w] for s in per_seed)
                for w in eval_windows
            },
            "non_embedding_params": h.ProbeLM(name).non_embedding_params(),
        }
    return results


def run_modcounter_probe(args, device: torch.device) -> dict:
    """Probe E: mod2/mod4 running-count accuracy vs length (128->4096, 32x
    extrapolation) per mixer. ``PortHamiltonianMixer``'s certified energy-norm
    contraction should keep state bounded and trackable far past train
    length; the attention control is reported honestly at every length — the
    lane a non-QKV mixer should beat it outright."""
    h = _harness()
    train_body_len, eval_body_lens = 128, (128, 512, 1024, 4096)
    mixers = args.mixers.split(",") if args.mixers else ["phmix", "attn"]
    results: dict = {
        "task": "modcounter",
        "train_body_len": train_body_len,
        "eval_body_lens": eval_body_lens,
        "saturation_control": "none",
    }
    for name in mixers:
        per_seed: dict[int, dict] = {}
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            model = h.ProbeLM(name).to(device)

            def sample(b: int, g) -> tuple[torch.Tensor, torch.Tensor]:
                return make_modcounter_batch(b, train_body_len, device, g)

            losses = h.train_model(
                model, sample, args.steps, args.batch, device, gen, args.lr
            )
            accs = {
                length: eval_modcounter_accuracy(
                    model,
                    lambda b, g, length=length: make_modcounter_batch(
                        b, length, device, g
                    ),
                    args.eval_seqs,
                    args.batch,
                    torch.Generator().manual_seed(10_000 + seed),
                )
                for length in eval_body_lens
            }
            per_seed[seed] = {
                "acc_by_length": accs,
                "final_loss": losses[-1],
                "first_loss": losses[0],
            }
            print(f"[modcounter] {name} seed={seed} acc={accs}", flush=True)
        results[name] = {
            "per_seed": per_seed,
            "median_acc_by_length": {
                length: {
                    split: statistics.median(
                        per_seed[s]["acc_by_length"][length][split] for s in per_seed
                    )
                    for split in ("mod2", "mod4")
                }
                for length in eval_body_lens
            },
            "non_embedding_params": h.ProbeLM(name).non_embedding_params(),
        }
    return results


def run_induction_probe(args, device: torch.device) -> dict:
    """Probe F: induction accuracy vs A-B distance, incl. 32x extrapolation
    beyond the train range. ``ScaleEquivariantWaveletStack``'s shared mother
    filter reused at dyadic dilations gives receptive field ``2**n_scales``
    while params grow only linearly — it should track the A...B gap far past
    anything a fixed local conv spans."""
    h = _harness()
    train_gaps, eval_gaps = (8, 32, 64), (8, 32, 64, 256, 1024, 2048)
    mixers = args.mixers.split(",") if args.mixers else ["wavelet", "attn"]
    results: dict = {
        "task": "induction_distance",
        "train_gaps": train_gaps,
        "eval_gaps": eval_gaps,
        "in_distribution_gaps": (8, 64),
        "saturation_control": "none",
    }
    for name in mixers:
        per_seed: dict[int, dict] = {}
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed)
            model = h.ProbeLM(name).to(device)

            def sample(b: int, g) -> tuple[torch.Tensor, torch.Tensor]:
                gap = train_gaps[
                    int(torch.randint(0, len(train_gaps), (1,), generator=g))
                ]
                return make_induction_batch(b, gap, device, g)

            losses = h.train_model(
                model, sample, args.steps, args.batch, device, gen, args.lr
            )
            accs = {
                gap: h.eval_accuracy(
                    model,
                    lambda b, g, gap=gap: make_induction_batch(b, gap, device, g),
                    args.eval_seqs,
                    args.batch,
                    torch.Generator().manual_seed(10_000 + seed),
                )
                for gap in eval_gaps
            }
            per_seed[seed] = {
                "acc_by_gap": accs,
                "final_loss": losses[-1],
                "first_loss": losses[0],
            }
            print(f"[induction] {name} seed={seed} acc={accs}", flush=True)
        results[name] = {
            "per_seed": per_seed,
            "median_acc_by_gap": {
                gap: statistics.median(per_seed[s]["acc_by_gap"][gap] for s in per_seed)
                for gap in eval_gaps
            },
            "non_embedding_params": h.ProbeLM(name).non_embedding_params(),
        }
    return results
