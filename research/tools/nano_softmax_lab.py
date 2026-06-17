"""Hands-on nano lab: train a multi-head softmax LM on the FULL nano_corpus_v4,
checkpoint it, watch loss/ppl, and inspect generations visually.

Corpus: research/data/nano_corpus/nano_corpus_v4.txt — Buckets A (the {noun} was
{adj}), B (same nouns, other frames: sat/ran/jumped/slept, "I see the {noun}"),
C (same adjectives, other constructions). Training on A+B+C (the Test-prompts
section is held out as inference examples). Word-level tokenizer over the ~34-word
vocab, so generations are human-readable. Lane = multi-head causal softmax attention.

Commands::

    # train (you set the steps), checkpoint + loss/ppl logging
    python -m research.tools.nano_softmax_lab train \
        --steps 4000 --dim 128 --n-blocks 2 --log-every 200 \
        --ckpt research/reports/nano_softmax.pt

    # inspect generations (top-k next-token after a prompt) from the checkpoint
    python -m research.tools.nano_softmax_lab generate \
        --ckpt research/reports/nano_softmax.pt --prompt "the cat was" --top-k 8
    # held-out noun (cat/book/child/lamp/ship were NOT in training) -> does it
    # still put adjectives on top? that's binding generalization, not memorization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from component_fab.harness.tiny_lm import lane_factory_for_baseline
from component_fab.harness.training_probe import build_tiny_lm

_REPO = Path(__file__).resolve().parents[1]
_CORPUS = _REPO / "data" / "nano_corpus" / "nano_corpus_v4.txt"
_PAD = "<pad>"


def _load_corpus(
    path: Path = _CORPUS,
) -> tuple[list[list[str]], list[str], dict[str, int]]:
    """Lines from Buckets A/B/C (before the Test-prompts header) + vocab."""
    train_lines: list[str] = []
    in_test = False
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if s.startswith("#"):
            in_test = s.lower().startswith("# test")
            continue
        if s and not in_test:
            train_lines.append(s)
    sentences = [ln.split() for ln in train_lines]
    words = sorted({w for s in sentences for w in s})
    vocab = [_PAD, *words]
    stoi = {w: i for i, w in enumerate(vocab)}
    return sentences, vocab, stoi


def _batchify(sentences, stoi, device):
    """Right-pad to max length; returns (ids, pad_id)."""
    pad = stoi[_PAD]
    maxlen = max(len(s) for s in sentences)
    rows = [[stoi[w] for w in s] + [pad] * (maxlen - len(s)) for s in sentences]
    return torch.tensor(rows, device=device), pad


def _build(dim, n_blocks, vocab_size, max_seq_len, device, *, lane_factory=None):
    # Default lane = multi-head causal softmax attention; pass lane_factory to swap
    # in a candidate mechanism (e.g. a fab lane via generate_module).
    if lane_factory is None:
        lane_factory = lane_factory_for_baseline("gpt2")
    return build_tiny_lm(
        lane_factory,
        vocab_size=vocab_size,
        dim=dim,
        n_blocks=n_blocks,
        max_seq_len=max_seq_len,
        use_position_embedding=True,
        use_ffn=True,
        ffn_mult=4,
    ).to(device)


def train(args) -> int:
    device = args.device
    sentences, vocab, stoi = _load_corpus(Path(args.corpus))
    print(
        f"corpus: {Path(args.corpus).name} — {len(sentences)} sentences, vocab={len(vocab)}"
    )
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(sentences), generator=g).tolist()
    n_val = max(8, len(sentences) // 10)
    val = [sentences[i] for i in perm[:n_val]]
    train_s = [sentences[i] for i in perm[n_val:]]
    train_ids, pad = _batchify(train_s, stoi, device)
    val_ids, _ = _batchify(val, stoi, device)
    maxlen = train_ids.shape[1]

    torch.manual_seed(args.seed)
    model = _build(args.dim, args.n_blocks, len(vocab), maxlen, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"multi-head softmax LM: dim={args.dim} n_blocks={args.n_blocks} params={n_params:,}"
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(ignore_index=pad)

    def _loss(ids):
        logits = model(ids)
        return ce(logits[:, :-1, :].reshape(-1, len(vocab)), ids[:, 1:].reshape(-1))

    n = train_ids.shape[0]
    for step in range(1, args.steps + 1):
        model.train()
        idx = torch.randint(0, n, (min(args.batch_size, n),), device=device)
        loss = _loss(train_ids[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vloss = _loss(val_ids).item()
            print(
                f"  step {step:>6} train_loss={loss.item():.4f} "
                f"val_loss={vloss:.4f} val_ppl={torch.tensor(vloss).exp().item():.2f}"
            )

    ckpt = Path(args.ckpt)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "dim": args.dim,
            "n_blocks": args.n_blocks,
            "max_seq_len": maxlen,
        },
        ckpt,
    )
    print(f"saved checkpoint -> {ckpt}")
    return 0


def generate(args) -> int:
    device = args.device
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    vocab = ck["vocab"]
    stoi = {w: i for i, w in enumerate(vocab)}
    model = _build(ck["dim"], ck["n_blocks"], len(vocab), ck["max_seq_len"], device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    toks = args.prompt.split()
    unknown = [w for w in toks if w not in stoi]
    if unknown:
        print(f"WARNING: not in vocab (ignored): {unknown}")
        toks = [w for w in toks if w in stoi]
    ids = torch.tensor([[stoi[w] for w in toks]], device=device)
    with torch.no_grad():
        probs = torch.softmax(model(ids)[0, -1], dim=-1)
    topv, topi = probs.topk(args.top_k)
    print(f"\nprompt: {args.prompt!r}")
    print(f"top-{args.top_k} next tokens:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  {vocab[i]:<10} {v:.3f}")
    # greedy continuation for a few tokens (visual)
    cont = list(toks)
    cur = ids
    for _ in range(args.n_continue):
        with torch.no_grad():
            nxt = int(model(cur)[0, -1].argmax())
        cont.append(vocab[nxt])
        cur = torch.tensor([[stoi[w] for w in cont]], device=device)
    print(f"greedy continuation: {' '.join(cont)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--dim", type=int, default=128)
    t.add_argument("--n-blocks", type=int, default=2)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--log-every", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument(
        "--corpus", default=str(_CORPUS), help="corpus .txt (swap in bigger ones)"
    )
    t.add_argument("--ckpt", default=str(_REPO / "reports" / "nano_softmax.pt"))
    t.set_defaults(func=train)

    g = sub.add_parser("generate")
    g.add_argument("--ckpt", default=str(_REPO / "reports" / "nano_softmax.pt"))
    g.add_argument("--prompt", default="the cat was")
    g.add_argument("--top-k", type=int, default=8)
    g.add_argument("--n-continue", type=int, default=2)
    g.set_defaults(func=generate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
