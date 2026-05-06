"""NanoBind — S0 hard rejection gate for compositional binding.

Tests whether an architecture (synthesized from a graph spec, freshly
initialised) can learn the noun→adjective binding rule on a small
controlled corpus.  Architectures that score *exactly 0.00* across a
short repeat sweep are flagged as frequency-mode-collapse degenerates
and rejected at S0 BEFORE consuming tier-1 evaluation budget.

Production framing (per codex-claude joint matrix, 2026-05-03):

* This is a **no-go gate only**.  Persistent zero ⇒ reject.
* It is **not** a positive endorsement.  Passing means "eligible to
  continue evaluation," nothing more.  Higher scores do not mean
  "better model" — see the Workstream E strict-1 follow-up for why
  raw score is unreliable as a positive ranking signal.
* Bilateral validation: independent 144-cell matrices (claude
  ``nano_corpus_v0`` + codex ``nano_corpus_codex``) on the same
  4 architectures both showed 0/36 for the two token_merge_block
  failers and 36/36 for ec7025d7.  ref_gpt2 was never zero.
* The defensible decision rule is therefore: reject iff slot-ending
  accuracy is **exactly 0.00 at every checkpoint of the sweep**.

Default sweep: corpus=280 sentences (80 A + 120 B + 80 C, strict
n_adj_per_noun=3, vocab=10), 5 held-out nouns, train fresh-init from
``graph_json`` directly on the corpus (no random-token warmup),
checkpoints at 1000 and 2000 steps.

Cost: ~5–8 s per architecture on a 5090.  Tested at 17K-arch scale
in a single overnight run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch

from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_from_json
from research.tools.nano_corpus_v0 import (
    ADJECTIVES,
    DEFAULT_N_ADJECTIVES,
    _bucket_a,
    _bucket_b,
    _bucket_c,
    _test_prompts,
)

logger = logging.getLogger(__name__)

NANO_BIND_METRIC_VERSION = "nano_bind_v0"

DEFAULT_HELD_OUT: tuple[str, ...] = ("cat", "book", "lamp", "child", "ship")
DEFAULT_TEST_NOUNS: tuple[str, ...] = (
    "cat",
    "book",
    "lamp",
    "child",
    "ship",  # held-out
    "dog",
    "bird",
    "man",
    "boy",
    "woman",  # in-distribution
)
DEFAULT_CHECKPOINTS: tuple[int, ...] = (1000, 2000)
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 32
DEFAULT_N_ADJ_PER_NOUN = 3
DEFAULT_TIMEOUT_S = 60.0
TIKTOKEN_ENCODING = "cl100k_base"  # matches compile_model vocab=100277
PAD_ID = 0


@dataclass(slots=True)
class NanoBindResult:
    """Result of a single NanoBind evaluation.

    `is_no_go` is True iff slot-ending accuracy is **exactly 0.00** at
    every checkpoint of the sweep — the only condition for S0 rejection.
    Other fields are stored for audit so rejected archs can be reviewed.
    Passing this test does NOT imply the architecture is good.
    """

    metric_version: str = NANO_BIND_METRIC_VERSION
    is_no_go: bool = False  # persistent-exact-zero across full sweep
    scores: tuple[float, ...] = ()
    held_acc: tuple[float, ...] = ()
    n_unique: tuple[int, ...] = ()
    # Per-checkpoint, per-prompt top-5 token IDs and decoded strings.
    # Outer index = checkpoint, inner index = prompt.
    top5_token_ids: tuple[tuple[tuple[int, ...], ...], ...] = ()
    top5_tokens: tuple[tuple[tuple[str, ...], ...], ...] = ()
    prompt_sentences: tuple[str, ...] = ()
    sweep_metadata: dict[str, Any] | None = None
    elapsed_ms: float = 0.0
    status: str = "ok"  # 'ok' | 'compile_failed' | 'timeout' | 'error'
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nano_bind_metric_version": self.metric_version,
            "nano_bind_is_no_go": self.is_no_go,
            "nano_bind_scores": list(self.scores),
            "nano_bind_held_acc": list(self.held_acc),
            "nano_bind_n_unique": list(self.n_unique),
            "nano_bind_top5_token_ids": [
                [list(p) for p in cp] for cp in self.top5_token_ids
            ],
            "nano_bind_top5_tokens": [[list(p) for p in cp] for cp in self.top5_tokens],
            "nano_bind_prompt_sentences": list(self.prompt_sentences),
            "nano_bind_sweep_metadata": self.sweep_metadata,
            "nano_bind_elapsed_ms": self.elapsed_ms,
            "nano_bind_status": self.status,
            "nano_bind_error": self.error,
        }


def _get_encoder():
    from research.eval.utils import _get_tiktoken_encoder

    return _get_tiktoken_encoder(TIKTOKEN_ENCODING)


_ADJ_TOKEN_ID_CACHE: dict[tuple[str, int], frozenset[int]] = {}


def _get_adj_token_ids(enc, n_adjectives: int) -> frozenset[int]:
    """Cached single-token IDs for the first ``n_adjectives`` adjectives.

    Replaces the inline set comprehension at the start of ``nano_bind()`` so
    a sweep across many archs only pays the encode cost once per
    (encoding, n_adjectives) pair.
    """
    cache_key = (TIKTOKEN_ENCODING, int(n_adjectives))
    cached = _ADJ_TOKEN_ID_CACHE.get(cache_key)
    if cached is not None:
        return cached
    ids = frozenset(
        int(enc.encode(" " + a, allowed_special=set())[0])
        for a in ADJECTIVES[: int(n_adjectives)]
    )
    _ADJ_TOKEN_ID_CACHE[cache_key] = ids
    return ids


def _build_corpus_tensors(
    *,
    enc,
    device: torch.device,
    held_out: tuple[str, ...],
    test_nouns: tuple[str, ...],
    n_a: int,
    n_b: int,
    n_c: int,
    n_adj_per_noun: int,
    n_adjectives: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str]]:
    import random

    from research.eval.utils import pack_token_rows, tokenize_words_serial

    rng = random.Random(int(seed))
    held = frozenset(held_out)
    a = _bucket_a(
        rng,
        n_a,
        exclude_nouns=held,
        strict_selection=True,
        n_adj_per_noun=n_adj_per_noun,
        n_adjectives=n_adjectives,
    )
    b = _bucket_b(rng, n_b)
    c = _bucket_c(rng, n_c)
    train_sentences = a + b + c
    prompt_sentences = list(_test_prompts(test_nouns))

    def _tok(words: list[str]) -> list[int]:
        return tokenize_words_serial(enc, words, encoding_name=TIKTOKEN_ENCODING)

    train_rows = [_tok(s.split()) for s in train_sentences]
    train_ids = pack_token_rows(train_rows, device, pad_id=PAD_ID)
    prompt_rows = [_tok(s.split()) for s in prompt_sentences]
    prompt_ids = pack_token_rows(prompt_rows, device, pad_id=PAD_ID)
    last_pos = [len(r) - 1 for r in prompt_rows]
    return train_ids, prompt_ids, last_pos, prompt_sentences


def _train_one_step(
    model: torch.nn.Module,
    train_ids: torch.Tensor,
    opt: torch.optim.Optimizer,
    rng: torch.Generator,
    batch_size: int,
) -> bool:
    """One AdamW step.  Returns False on non-finite loss."""
    import torch.nn.functional as F
    from research.eval.utils import clip_grad_norm

    n = train_ids.shape[0]
    idx = torch.randint(
        0, n, (int(batch_size),), generator=rng, device=train_ids.device
    )
    batch = train_ids.index_select(0, idx)
    opt.zero_grad(set_to_none=True)
    logits = model(batch)
    targets = batch[:, 1:].contiguous()
    pred = logits[:, :-1, :].contiguous()
    mask = targets != PAD_ID
    if not bool(mask.any()):
        return True
    loss = F.cross_entropy(pred[mask].float(), targets[mask])
    if not torch.isfinite(loss):
        return False
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    opt.step()
    return True


@torch.no_grad()
def _eval_checkpoint(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    last_pos: list[int],
    prompt_sentences: list[str],
    *,
    held_out: frozenset[str],
    adj_token_ids: frozenset[int],
    enc,
) -> tuple[float, float, int, list[list[int]], list[list[str]]]:
    """Returns (top1_adj_acc, held_class_acc, n_unique, top5_ids, top5_tokens).

    `top5_ids` and `top5_tokens` are per-prompt lists for audit storage —
    rejected archs can be reviewed by inspecting what they actually
    predicted for each test prompt.
    """
    model.eval()
    logits = model(prompt_ids)
    n_top1_adj = 0
    n_held_top1_adj = 0
    n_held = 0
    top1_tokens: list[int] = []
    top5_ids_per_prompt: list[list[int]] = []
    top5_tokens_per_prompt: list[list[str]] = []
    for i, lp in enumerate(last_pos):
        top5_vals = torch.topk(logits[i, lp, :], k=5)
        ids5 = [int(t) for t in top5_vals.indices.tolist()]
        top5_ids_per_prompt.append(ids5)
        top5_tokens_per_prompt.append([enc.decode([t]).strip() for t in ids5])
        top1 = ids5[0]
        top1_tokens.append(top1)
        is_adj = top1 in adj_token_ids
        if is_adj:
            n_top1_adj += 1
        prompt_noun = prompt_sentences[i].split()[1]
        if prompt_noun in held_out:
            n_held += 1
            if is_adj:
                n_held_top1_adj += 1
    return (
        n_top1_adj / len(last_pos),
        n_held_top1_adj / max(n_held, 1),
        len(set(top1_tokens)),
        top5_ids_per_prompt,
        top5_tokens_per_prompt,
    )


def nano_bind(
    graph_json: str,
    *,
    device: str = "cuda",
    seed: int = 0,
    held_out: tuple[str, ...] = DEFAULT_HELD_OUT,
    test_nouns: tuple[str, ...] = DEFAULT_TEST_NOUNS,
    checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH,
    n_a: int = 80,
    n_b: int = 120,
    n_c: int = 80,
    n_adj_per_noun: int = DEFAULT_N_ADJ_PER_NOUN,
    n_adjectives: int = DEFAULT_N_ADJECTIVES,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> NanoBindResult:
    """Run NanoBind on a graph spec.

    Returns ``NanoBindResult`` with ``is_no_go=True`` iff slot-ending
    accuracy is exactly 0.00 at every checkpoint.  The caller's S0
    pipeline should reject (set ``tier='screened_out'``,
    ``failure_op='nano_bind'``) on no-go and continue evaluation
    otherwise.  Passing this test does NOT imply the architecture is
    good; it only excludes the persistent-zero failure mode.
    """

    t0 = time.perf_counter()
    enc = _get_encoder()
    setup = _setup_run(
        graph_json,
        device=device,
        seed=seed,
        t0=t0,
        enc=enc,
        held_out=held_out,
        test_nouns=test_nouns,
        n_a=n_a,
        n_b=n_b,
        n_c=n_c,
        n_adj_per_noun=n_adj_per_noun,
        n_adjectives=n_adjectives,
    )
    if isinstance(setup, NanoBindResult):
        return setup
    model, train_ids, prompt_ids, last_pos, prompts, dev = setup

    adj_token_ids = _get_adj_token_ids(enc, int(n_adjectives))
    sweep_state = _SweepState(checkpoints=checkpoints)
    deadline = t0 + float(timeout_s)
    try:
        _run_sweep(
            model,
            train_ids,
            prompt_ids,
            last_pos,
            prompts,
            adj_token_ids=adj_token_ids,
            held_set=frozenset(held_out),
            enc=enc,
            seed=seed,
            lr=lr,
            batch_size=batch_size,
            deadline=deadline,
            state=sweep_state,
        )
    finally:
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "checkpoints": list(checkpoints),
        "lr": lr,
        "batch_size": batch_size,
        "n_a": n_a,
        "n_b": n_b,
        "n_c": n_c,
        "n_adj_per_noun": n_adj_per_noun,
        "n_adjectives": n_adjectives,
        "held_out": list(held_out),
        "test_nouns": list(test_nouns),
        "seed": seed,
        "tokenizer": TIKTOKEN_ENCODING,
    }
    return _build_result(sweep_state, prompts=prompts, metadata=metadata, t0=t0)


def _setup_run(
    graph_json: str,
    *,
    device: str,
    seed: int,
    t0: float,
    enc,
    held_out: tuple[str, ...],
    test_nouns: tuple[str, ...],
    n_a: int,
    n_b: int,
    n_c: int,
    n_adj_per_noun: int,
    n_adjectives: int,
):
    """Compile model + build corpus tensors. Returns NanoBindResult on
    failure, or (model, train_ids, prompt_ids, last_pos, prompts, dev)."""
    dev = torch.device(device)
    try:
        graph = graph_from_json(graph_json)
        torch.manual_seed(int(seed))
        model = compile_model([graph]).to(dev)
    except Exception as exc:  # noqa: BLE001
        return NanoBindResult(
            status="compile_failed",
            error=str(exc),
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
    try:
        train_ids, prompt_ids, last_pos, prompts = _build_corpus_tensors(
            enc=enc,
            device=dev,
            held_out=held_out,
            test_nouns=test_nouns,
            n_a=n_a,
            n_b=n_b,
            n_c=n_c,
            n_adj_per_noun=n_adj_per_noun,
            n_adjectives=n_adjectives,
            seed=seed,
        )
    except Exception as exc:  # noqa: BLE001
        return NanoBindResult(
            status="error",
            error=f"corpus_build:{exc}",
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
    return model, train_ids, prompt_ids, last_pos, prompts, dev


def _build_result(
    state: _SweepState, *, prompts: list[str], metadata: dict, t0: float
) -> NanoBindResult:
    top5_ids = tuple(tuple(tuple(p) for p in cp) for cp in state.top5_ids)
    top5_tokens = tuple(tuple(tuple(p) for p in cp) for cp in state.top5_toks)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    if state.error_status is not None:
        return NanoBindResult(
            status=state.error_status,
            error=state.error_msg,
            scores=tuple(state.scores),
            held_acc=tuple(state.held_accs),
            n_unique=tuple(state.n_uniques),
            top5_token_ids=top5_ids,
            top5_tokens=top5_tokens,
            prompt_sentences=tuple(prompts),
            elapsed_ms=elapsed,
        )
    is_no_go = bool(state.scores) and all(s == 0.0 for s in state.scores)
    return NanoBindResult(
        is_no_go=is_no_go,
        scores=tuple(state.scores),
        held_acc=tuple(state.held_accs),
        n_unique=tuple(state.n_uniques),
        top5_token_ids=top5_ids,
        top5_tokens=top5_tokens,
        prompt_sentences=tuple(prompts),
        sweep_metadata=metadata,
        elapsed_ms=elapsed,
        status="ok",
    )


@dataclass(slots=True)
class _SweepState:
    checkpoints: tuple[int, ...]
    scores: list[float] = None  # type: ignore[assignment]
    held_accs: list[float] = None  # type: ignore[assignment]
    n_uniques: list[int] = None  # type: ignore[assignment]
    top5_ids: list[list[list[int]]] = None  # type: ignore[assignment]
    top5_toks: list[list[list[str]]] = None  # type: ignore[assignment]
    error_status: str | None = None
    error_msg: str | None = None

    def __post_init__(self) -> None:
        if self.scores is None:
            self.scores = []
        if self.held_accs is None:
            self.held_accs = []
        if self.n_uniques is None:
            self.n_uniques = []
        if self.top5_ids is None:
            self.top5_ids = []
        if self.top5_toks is None:
            self.top5_toks = []


def _run_sweep(
    model: torch.nn.Module,
    train_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    last_pos: list[int],
    prompts: list[str],
    *,
    adj_token_ids: frozenset[int],
    held_set: frozenset[str],
    enc,
    seed: int,
    lr: float,
    batch_size: int,
    deadline: float,
    state: _SweepState,
) -> None:
    from research.eval.utils import make_adamw

    rng = torch.Generator(device=train_ids.device)
    rng.manual_seed(int(seed))
    opt = make_adamw(model.parameters(), lr=lr)
    prev_step = 0
    for cp in state.checkpoints:
        delta = max(0, int(cp) - prev_step)
        for _ in range(delta):
            if time.perf_counter() > deadline:
                state.error_status = "timeout"
                return
            model.train()
            if not _train_one_step(model, train_ids, opt, rng, batch_size):
                state.error_status = "error"
                state.error_msg = "non_finite_loss"
                return
        top1_adj, held_acc, n_unique, top5_ids, top5_toks = _eval_checkpoint(
            model,
            prompt_ids,
            last_pos,
            prompts,
            held_out=held_set,
            adj_token_ids=adj_token_ids,
            enc=enc,
        )
        state.scores.append(round(top1_adj, 4))
        state.held_accs.append(round(held_acc, 4))
        state.n_uniques.append(int(n_unique))
        state.top5_ids.append(top5_ids)
        state.top5_toks.append(top5_toks)
        prev_step = int(cp)


__all__ = [
    "NANO_BIND_METRIC_VERSION",
    "NanoBindResult",
    "nano_bind",
    "DEFAULT_CHECKPOINTS",
    "DEFAULT_HELD_OUT",
    "DEFAULT_TEST_NOUNS",
]
