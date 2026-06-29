"""Multi-token-out (+1..+7) on the structured NANO corpus — is the corpus the problem?

On real web text (100k vocab) the exact 2nd/3rd token is non-deterministic, so EVERY model
(incl. softmax) floored at multi-token-out. The nano corpus (`nano_corpus_v4`, ~35-word vocab,
frame `the {noun} was {adj}`) has DETERMINISTIC skeleton structure (the->noun->was->adj->eos
->the...): `was`, `eos`, `the` are predictable multi-token-out; only noun/adj are random.

This trains loss monsters on the nano corpus as a continuous stream and measures free-rollout
exact-match at +1..+7 vs the unigram floor. If they now predict multi-token well above floor
(hitting the deterministic skeleton), the web-text floor was a CORPUS artifact, not a model or
scale limit. Reuses W0 helpers; word-level tokenizer; train-mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.scientist.native_runner import compile_model_native_first
from research.synthesis.serializer import graph_from_json
from research.tools.loss_monster_screen import (
    _OUT_DIR,
    _RUNS_DB,
    _sample_batch,
    evaluate,
    select_family_champions,
)
from research.tools.loss_monster_horizon import rollout_horizon

_NANO = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "data"
    / "nano_corpus"
    / "nano_corpus_v4.txt"
)


def load_nano_stream() -> tuple[np.ndarray, np.ndarray, int]:
    """Word-level tokenize nano_corpus_v4 into one continuous stream (sentences joined by <eos>)."""
    lines = [
        ln.split()
        for ln in _NANO.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    words = sorted({w for s in lines for w in s})
    vocab = ["<eos>", *words]
    stoi = {w: i for i, w in enumerate(vocab)}
    toks: list[int] = []
    for s in lines:
        toks.extend(stoi[w] for w in s)
        toks.append(0)  # <eos>
    arr = np.array(toks, dtype=np.int64)
    n_val = max(400, len(arr) // 5)
    return arr[:-n_val], arr[-n_val:], len(vocab)


_BABI = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "data"
    / "babiqa_for_sft"
    / "train.csv"
)


def load_babi_stream(max_rows: int = 6000) -> tuple[np.ndarray, np.ndarray, int]:
    """Word-tokenize bAbI (HF) query+answer into one stream (rows joined by <eos>).

    bAbI is templated spatial-reasoning QA, so multi-token-out tests the deterministic
    template AND the reasoned answer — a harder, bigger structured corpus than nano_v4.
    """
    import csv

    rows: list[list[str]] = []
    with open(_BABI, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            text = f"{row.get('query', '')} {row.get('answer', '')}".replace("\n", " ")
            toks = text.split()
            if toks:
                rows.append(toks)
    words = sorted({w for s in rows for w in s})
    stoi = {w: i for i, w in enumerate(["<eos>", *words])}
    stream: list[int] = []
    for s in rows:
        stream.extend(stoi[w] for w in s)
        stream.append(0)
    arr = np.array(stream, dtype=np.int64)
    n_val = max(2000, len(arr) // 10)
    return arr[:-n_val], arr[-n_val:], len(stoi)


def unigram_top1(stream: np.ndarray, vocab: int) -> float:
    c = np.bincount(stream, minlength=vocab)
    return float(c.max() / c.sum())


def _train_eval(
    model: torch.nn.Module,
    family: str,
    train: np.ndarray,
    val: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    gen = np.random.default_rng(0)
    for _ in range(args.steps):
        x, y = _sample_batch(train, args.batch, args.seq, gen, args.device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    ev = evaluate(
        model,
        val,
        batch=args.batch,
        seq=args.seq,
        n_batches=args.eval_batches,
        device=args.device,
    )
    hor = rollout_horizon(
        model,
        val,
        ctx_len=args.ctx_len,
        horizon=7,
        n_contexts=args.n_contexts,
        batch=args.batch,
        device=args.device,
    )
    rec = {
        "family": family,
        "top1": round(ev["top1_acc"], 3),
        "ppl": round(ev["val_ppl"], 2),
        "horizon": [round(h, 3) for h in hor],
    }
    print(
        f"  {family:24s} top1={ev['top1_acc']:.3f}  +1..+7 "
        + " ".join(f"{h:.2f}" for h in hor),
        flush=True,
    )
    return rec


def run_family(family, graph_json, train, val, vocab, args) -> dict:
    graph = graph_from_json(graph_json)
    model = compile_model_native_first(
        [graph] * args.n_layers, vocab_size=vocab, max_seq_len=args.seq
    ).to(args.device)
    return _train_eval(model, family, train, val, args)


def run_softmax(train, val, vocab, args) -> dict:
    """Softmax-attention control on the SAME structured corpus (the multi-token baseline)."""
    from research.tools.softmax_control import TinyGPT

    model = TinyGPT(vocab, 256, args.n_layers, 4, args.seq).to(args.device)
    return _train_eval(model, "softmax_control", train, val, args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--families",
        nargs="*",
        default=["recursive_depth_router", "parallel_split", "residual_block"],
    )
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--ctx-len", type=int, default=32)
    ap.add_argument("--n-contexts", type=int, default=512)
    ap.add_argument("--corpus", default="nano", choices=["nano", "babi"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "nano_multitoken.json"))
    args = ap.parse_args()

    train, val, vocab = (
        load_babi_stream() if args.corpus == "babi" else load_nano_stream()
    )
    floor = unigram_top1(np.concatenate([train, val]), vocab)
    chance = 1.0 / vocab
    print(
        f"corpus={args.corpus}: vocab={vocab}  train_toks={len(train)} val_toks={len(val)}  "
        f"unigram_floor={floor:.3f}  uniform_chance={chance:.3f}"
    )
    print("(web-text reference: every model incl softmax floored at +2~.05 +3+~.02)\n")

    champs = {c.family: c for c in select_family_champions(_RUNS_DB)}
    results = []
    for fam in args.families:
        if fam == "softmax":
            results.append(run_softmax(train, val, vocab, args))
            continue
        if fam not in champs:
            print(f"  {fam}: no champion graph")
            continue
        results.append(run_family(fam, champs[fam].graph_json, train, val, vocab, args))

    Path(args.out).write_text(
        json.dumps(
            {
                "config": vars(args),
                "vocab": vocab,
                "unigram_floor": floor,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nfloor={floor:.3f} | Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
