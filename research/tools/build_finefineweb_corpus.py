"""Build a cl100k-tokenized FineFineWeb corpus (.npy) for mixer_fingerprint.

Controlled-experiment data: identical tokenizer (cl100k_base, vocab 100277) and
format as the existing ``research/corpus/wikitext103_train.npy``, so swapping it
into mixer_fingerprint changes ONLY the training corpus content (real diverse
web text) — architecture, tokenizer, vocab, schedule, evals all unchanged.

Round-robins across FineFineWeb domains for topical diversity, inserts the
cl100k EOT (100257) between documents, clips to vocab-1 (mirrors
``eval.utils.tokenize_string``), and writes int32 train + val arrays sized so a
533M–688M-token Chinchilla run does NOT repeat data (removing the multi-epoch
confound the 50M-token WikiText-103 cache had).

Usage:
    python -m research.tools.build_finefineweb_corpus \
        --target-train-tokens 750_000_000 --target-val-tokens 10_000_000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research.defaults import VOCAB_SIZE
from research.eval.utils import _get_tiktoken_encoder

_FFW_ROOT = Path("/mnt/data/hf_finefineweb")
_OUT_DIR = Path("research/corpus")
_EOT = 100257  # cl100k <|endoftext|>


def _round_robin_shards(root: Path) -> list[Path]:
    """All domain shards, interleaved by domain so a prefix is topically diverse."""
    by_domain = []
    for domain in sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        shards = sorted(domain.glob(f"{domain.name}_*.jsonl"))
        if shards:
            by_domain.append(shards)
    out: list[Path] = []
    i = 0
    while any(i < len(s) for s in by_domain):
        for shards in by_domain:
            if i < len(shards):
                out.append(shards[i])
        i += 1
    return out


def _encode_until(
    shards: list[Path], target_tokens: int, enc, *, start_shard: int = 0
) -> tuple[np.ndarray, int]:
    """Encode documents round-robin across shards until target_tokens reached.

    Returns (int32 token array, index of next unused shard)."""
    chunks: list[np.ndarray] = []
    total = 0
    vmax = int(VOCAB_SIZE) - 1
    si = start_shard
    t0 = time.perf_counter()
    while total < target_tokens and si < len(shards):
        texts: list[str] = []
        with shards[si].open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    txt = json.loads(line).get("text")
                except json.JSONDecodeError:
                    continue
                if txt:
                    texts.append(txt)
            # batch-encode this shard's docs, EOT-separated
        for ids in enc.encode_ordinary_batch(texts, num_threads=8):
            arr = np.asarray(ids, dtype=np.int64)
            if arr.size:
                np.minimum(arr, vmax, out=arr)
            chunks.append(arr.astype(np.int32))
            chunks.append(np.array([_EOT], dtype=np.int32))
            total += arr.size + 1
        print(
            f"  shard {si} ({shards[si].parent.name}) → {total / 1e6:.1f}M tok "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
        si += 1
        if total >= target_tokens:
            break
    return np.concatenate(chunks)[:target_tokens], si


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_FFW_ROOT)
    ap.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    ap.add_argument("--target-train-tokens", type=int, default=750_000_000)
    ap.add_argument("--target-val-tokens", type=int, default=10_000_000)
    ap.add_argument("--prefix", type=str, default="finefineweb")
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"FineFineWeb root not found: {args.root}", file=__import__("sys").stderr)
        return 2
    enc = _get_tiktoken_encoder("cl100k_base")
    shards = _round_robin_shards(args.root)
    print(f"{len(shards)} shards across domains; encoding cl100k…", flush=True)

    print(f"== train target {args.target_train_tokens / 1e6:.0f}M ==", flush=True)
    train, next_shard = _encode_until(shards, args.target_train_tokens, enc)
    print(
        f"== val target {args.target_val_tokens / 1e6:.0f}M (held-out shards) ==",
        flush=True,
    )
    val, _ = _encode_until(shards, args.target_val_tokens, enc, start_shard=next_shard)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / f"{args.prefix}_train.npy"
    val_path = args.out_dir / f"{args.prefix}_val.npy"
    np.save(train_path, train)
    np.save(val_path, val)
    print(
        f"wrote {train_path} ({train.size / 1e6:.1f}M tok, {train.nbytes / 1e9:.2f} GB)"
    )
    print(f"wrote {val_path} ({val.size / 1e6:.1f}M tok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
