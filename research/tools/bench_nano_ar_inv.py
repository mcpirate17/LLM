#!/usr/bin/env python
"""Micro-benchmark for nano_ar_inv hot paths (CPU).

Three sections:
  A. class_token_ids — the inline 45-encode loop in nano_ar_inv()
  B. corpus tensor build — tokenize 595 sentences + 15 prompts, pack to tensors
  C. end-to-end nano_ar_inv on a tiny graph (mirrors test_probe_smoke_on_tiny_graph)

Reports median over N reps. Each section also reports per-sub-step counters.

Usage:
    python -m research.tools.bench_nano_ar_inv --reps 5
    python -m research.tools.bench_nano_ar_inv --reps 5 --no-e2e   # skip C
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def _bench_class_token_ids_inline(reps: int) -> list[float]:
    """OLD path: 45 individual _tokenize_words(enc, [w])[0] calls per probe."""
    from research.eval.nano_ar_inv import _tokenize_words
    from research.eval.nano_ar_inv_corpus import OBJECTS
    from research.eval.utils import _get_tiktoken_encoder
    from research.tools.nano_corpus_v0 import ADJECTIVES

    enc = _get_tiktoken_encoder("cl100k_base")
    n_adj, n_obj = 20, 25
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        adj_token_ids = {a: _tokenize_words(enc, [a])[0] for a in ADJECTIVES[:n_adj]}
        obj_token_ids = {o: _tokenize_words(enc, [o])[0] for o in OBJECTS[:n_obj]}
        times.append((time.perf_counter() - t0) * 1000)
        assert len(adj_token_ids) == n_adj
        assert len(obj_token_ids) == n_obj
    return times


def _bench_class_token_ids_cached(reps: int) -> list[float]:
    """NEW path: _get_class_token_ids() with hash-key cache."""
    from research.eval import nano_ar_inv as _mod
    from research.eval.nano_ar_inv import NanoARInvConfig, _get_class_token_ids
    from research.eval.utils import _get_tiktoken_encoder

    enc = _get_tiktoken_encoder("cl100k_base")
    cfg = NanoARInvConfig()
    _mod._TOKEN_ID_CACHE.clear()
    # First call populates the cache; rep 0 = cold, reps 1..N = warm.
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        adj_token_ids, obj_token_ids = _get_class_token_ids(enc, cfg)
        times.append((time.perf_counter() - t0) * 1000)
        assert len(adj_token_ids) == 20
        assert len(obj_token_ids) == 25
    return times


def _bench_corpus_tensor_build(reps: int, device: str) -> list[float]:
    from research.eval.nano_ar_inv import (
        NanoARInvConfig,
        _build_prompt_tensor,
        _build_train_tensor,
    )
    from research.eval.nano_ar_inv_corpus import build_corpus
    from research.eval.utils import _get_tiktoken_encoder

    enc = _get_tiktoken_encoder("cl100k_base")
    cfg = NanoARInvConfig(seed=0)
    spec = build_corpus(
        seed=cfg.seed,
        n_pairs_per_noun=cfg.n_pairs_per_noun,
        reps=cfg.reps,
        n_distractors=cfg.n_distractors,
        held_out_nouns=cfg.held_out_nouns,
        n_adjectives=cfg.n_adjectives,
        n_objects=cfg.n_objects,
    )
    dev = torch.device(device)
    print(
        f"  corpus: {len(spec.train_sentences)} train, "
        f"{len(spec.test_facts)} prompts (device={device})"
    )
    times = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_ids = _build_train_tensor(enc, list(spec.train_sentences), dev)
        prompt_ids, _last_pos = _build_prompt_tensor(enc, spec.test_facts, dev)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        assert train_ids.shape[0] == len(spec.train_sentences)
        assert prompt_ids.shape[0] == len(spec.test_facts)
    return times


def _bench_nano_bind_corpus(reps: int, device: str) -> list[float]:
    """nano_bind._build_corpus_tensors — same per-row torch.tensor pattern."""
    from research.eval.nano_bind import (
        DEFAULT_HELD_OUT,
        DEFAULT_N_ADJ_PER_NOUN,
        DEFAULT_N_ADJECTIVES,
        DEFAULT_TEST_NOUNS,
        _build_corpus_tensors,
    )
    from research.eval.utils import _get_tiktoken_encoder

    enc = _get_tiktoken_encoder("cl100k_base")
    dev = torch.device(device)
    times = []
    for rep in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_ids, prompt_ids, _last_pos, _prompts = _build_corpus_tensors(
            enc=enc,
            device=dev,
            held_out=DEFAULT_HELD_OUT,
            test_nouns=DEFAULT_TEST_NOUNS,
            n_a=80,
            n_b=120,
            n_c=80,
            n_adj_per_noun=DEFAULT_N_ADJ_PER_NOUN,
            n_adjectives=DEFAULT_N_ADJECTIVES,
            seed=rep,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        assert train_ids.shape[0] > 0
        assert prompt_ids.shape[0] > 0
    return times


def _bench_nano_bind_adj_token_ids(reps: int) -> list[float]:
    """The inline adj_token_ids set comprehension at nano_bind.py:317-319."""
    from research.eval.nano_bind import TIKTOKEN_ENCODING
    from research.eval.utils import _get_tiktoken_encoder
    from research.tools.nano_corpus_v0 import ADJECTIVES, DEFAULT_N_ADJECTIVES

    enc = _get_tiktoken_encoder(TIKTOKEN_ENCODING)
    n_adj = int(DEFAULT_N_ADJECTIVES)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        adj_token_ids = {
            int(enc.encode(" " + a, allowed_special=set())[0])
            for a in ADJECTIVES[:n_adj]
        }
        times.append((time.perf_counter() - t0) * 1000)
        assert len(adj_token_ids) > 0
    return times


def _bench_end_to_end(reps: int, device: str) -> list[float]:
    from research.eval.nano_ar_inv import NanoARInvConfig, nano_ar_inv
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.serializer import graph_to_json

    g = ComputationGraph(model_dim=64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    attn = g.add_op("softmax_attention", [norm])
    fix = g.add_op("linear_proj", [attn], config={"out_dim": 64})
    out = g.add_op("add", [inp, fix])
    g.set_output(out)
    cfg = NanoARInvConfig(
        seed=0,
        finetune_steps=20,
        wikitext_warmup_steps=0,
        timeout_s=60.0,
        from_s1=False,
        n_distractors=20,
        n_pairs_per_noun=2,
        reps=2,
    )
    times = []
    for _ in range(reps):
        from research.eval import nano_ar_inv as _mod

        _mod._TENSOR_CACHE.clear()
        _mod._TOKEN_ID_CACHE.clear()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = nano_ar_inv(graph_json=graph_to_json(g), device=device, cfg=cfg)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        assert result.status in ("ok", "timeout"), result.error
    return times


def _summarize(label: str, times_ms: list[float]) -> None:
    print(
        f"  {label:32s} median={statistics.median(times_ms):8.2f} ms  "
        f"min={min(times_ms):8.2f}  max={max(times_ms):8.2f}  "
        f"n={len(times_ms)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--no-e2e", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    print("=" * 70)
    print(f"nano_ar_inv benchmark (device={args.device}, reps={args.reps})")
    print("=" * 70)

    print("\n[A1] class_token_ids OLD inline loop (45 encodes per probe)")
    a1 = _bench_class_token_ids_inline(args.reps)
    _summarize("inline_loop_per_probe", a1)

    print("\n[A2] class_token_ids NEW cached path (1st call cold, rest warm)")
    a2 = _bench_class_token_ids_cached(args.reps)
    _summarize("cached_per_probe", a2)
    if len(a2) >= 2:
        warm = a2[1:]
        _summarize("cached (warm only)", warm)

    print(f"\n[B] corpus tensor build (~3500 encodes + pack, {args.device})")
    b = _bench_corpus_tensor_build(args.reps, args.device)
    _summarize("build_train+prompt_tensor", b)

    if not args.no_e2e:
        print(
            f"\n[C] end-to-end nano_ar_inv (tiny graph, 20 finetune steps, "
            f"{args.device})"
        )
        c = _bench_end_to_end(args.reps, args.device)
        _summarize("nano_ar_inv full", c)

    print(f"\n[D] nano_bind corpus tensor build ({args.device})")
    d = _bench_nano_bind_corpus(args.reps, args.device)
    _summarize("nb_build_corpus_tensors", d)

    print("\n[E] nano_bind adj_token_ids inline set comprehension")
    e = _bench_nano_bind_adj_token_ids(args.reps)
    _summarize("nb_adj_token_ids_inline", e)


if __name__ == "__main__":
    main()
