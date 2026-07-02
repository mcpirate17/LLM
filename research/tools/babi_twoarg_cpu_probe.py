"""CPU sandbox: train a nano TinyLM on bAbI two-arg-relations and measure
memorization vs rule-learning, with basic-induction as a transfer control.

Answers both questions:
  1. Does training (N passes) MEMORIZE? -> compare accuracy on a DISJOINT
     held-out two-arg split vs the train split. train>>heldout = memorized;
     train~=heldout = learned the binding rule. (The dataset's own test split
     leaks 18% exact prompts into train, so we build our own clean split.)
  2. Spurious transfer? -> score basic-induction (a DIFFERENT mechanism) before
     and after. It should stay at its 25% floor if the model only learned
     two-arg binding rather than an answer-position/frequency shortcut.

Scoring is single-token: every room (" office"/" kitchen"/...) and every
induction color (" gray"/...) encodes to ONE cl100k token with a leading space
and there are no first-token collisions, so we read 6 (or 4) logits at the
answer position and argmax. Chance 16.7% (two-arg) / 25% (induction); the
majority-class baseline (~17.5% / ~28%) is the real "did it actually bind" bar.

Read-only on data; writes a JSON report. No DB writes. CPU-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import tiktoken
import torch
import torch.nn.functional as F

from research.tools.scaling_blimp_study import _build_lane_factory, _build_tinylm

ENC = tiktoken.get_encoding("cl100k_base")
DATA = Path("research/data/babiqa_for_sft")


def _answer_token(word: str) -> int:
    """Single cl100k id for ' word' (verified 1 token, no collisions for our sets)."""
    ids = ENC.encode(" " + word)
    if len(ids) != 1:
        raise ValueError(f"answer {word!r} is not a single token: {ids}")
    return ids[0]


def _load_category(cat: str) -> pl.DataFrame:
    tr = pl.read_csv(DATA / "train.csv")
    test_df = pl.read_csv(DATA / "test.csv")
    df = pl.concat([tr, test_df], how="vertical")
    df = df.filter(pl.col("query_type") == cat)
    # dedupe on the full prompt so train/test can't share identical rows
    return df.unique(subset=["query"], keep="first", maintain_order=True)


def _majority_fraction(answers: pl.Series) -> float:
    """Fraction of the most frequent answer (the majority-class baseline)."""
    return answers.value_counts()["count"].max() / len(answers)


def _split(df: pl.DataFrame, frac_test: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(df), generator=g).tolist()
    n_test = int(len(df) * frac_test)
    test_idx = set(perm[:n_test])
    is_test = pl.Series([i in test_idx for i in range(len(df))])
    return df.filter(~is_test), df.filter(is_test)


def _encode_rows(df: pl.DataFrame, max_len: int):
    """Return (input_ids[B,L] padded, answer_pos[B], answer_tok[B]).

    The model is trained/scored to predict the answer token at the position of
    the LAST prompt token (next-token prediction of the answer).
    """
    # guardrail: allow-complexity - tiny CSV tokenizer loop; model training dominates.
    seqs, ans_pos, ans_tok = [], [], []
    for row in df.iter_rows(named=True):
        prompt = ENC.encode(str(row["query"]))
        a = _answer_token(str(row["answer"]))
        # sequence = prompt + answer; predict answer at position len(prompt)-1
        seq = prompt + [a]
        if len(seq) > max_len:
            seq = seq[-max_len:]
        seqs.append(seq)
        ans_pos.append(len(prompt) - 1)
        ans_tok.append(a)
    L = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return ids, torch.tensor(ans_pos), torch.tensor(ans_tok)


@torch.no_grad()
def _accuracy(model, ids, ans_pos, ans_tok, candidate_toks, batch=64) -> float:
    model.eval()
    cand = torch.tensor(candidate_toks)
    correct = 0
    for i in range(0, len(ids), batch):
        b_ids = ids[i : i + batch]
        b_pos = ans_pos[i : i + batch]
        b_ans = ans_tok[i : i + batch]
        logits = model(b_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        # logits at the prompt's last position predict the answer
        at = logits[torch.arange(len(b_ids)), b_pos]  # [b, vocab]
        # restrict to the candidate answer set (clean closed-vocab scoring)
        cand_logits = at[:, cand]  # [b, n_cand]
        pred = cand[cand_logits.argmax(dim=-1)]
        correct += (pred == b_ans).sum().item()
    return correct / len(ids)


def _mtp_loss(logits, ids, depth: int) -> torch.Tensor:
    losses = []
    for k in range(1, depth + 1):
        if ids.shape[1] <= k:
            continue
        target = ids[:, k:].clone()
        target[target == 0] = -100
        losses.append(
            F.cross_entropy(
                logits[:, :-k, :].reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                ignore_index=-100,
            )
        )
    return sum(losses) / len(losses) if losses else logits.sum() * 0.0


def _train(
    model,
    ids,
    ans_pos,
    ans_tok,
    passes,
    lr,
    batch,
    seed,
    mtp_depth=0,
    mtp_weight=0.0,
):
    # guardrail: allow-complexity - explicit CPU probe loop over small nano batches.
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(passes):
        order = torch.randperm(len(ids), generator=g).tolist()
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            b_ids = ids[idx]
            b_pos = ans_pos[idx]
            b_ans = ans_tok[idx]
            logits = model(b_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            at = logits[torch.arange(len(idx)), b_pos]
            loss = F.cross_entropy(at, b_ans)
            if mtp_depth > 0 and mtp_weight > 0:
                loss = loss + float(mtp_weight) * _mtp_loss(logits, b_ids, mtp_depth)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    return losses


def main() -> None:
    # guardrail: allow-god-function - standalone CLI assembles one JSON report.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane", default="softmax_attention")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--frac-test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/babi_twoarg_cpu_probe.json")
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # --- data ---
    two = _load_category("two-arg-relations")
    ind = _load_category("basic-induction")
    rooms = sorted(two["answer"].unique().to_list())
    colors = sorted(ind["answer"].unique().to_list())
    room_toks = [_answer_token(r) for r in rooms]
    color_toks = [_answer_token(c) for c in colors]

    tr, test_df = _split(two, args.frac_test, args.seed)
    tr_ids, tr_pos, tr_ans = _encode_rows(tr, args.max_len)
    test_ids, test_pos, test_ans = _encode_rows(test_df, args.max_len)
    ind_ids, ind_pos, ind_ans = _encode_rows(ind, args.max_len + 40)

    # majority baselines (on train answer distribution)
    two_major = _majority_fraction(tr["answer"])
    ind_major = _majority_fraction(ind["answer"])

    # --- model ---
    model = _build_tinylm(
        _build_lane_factory(args.lane),
        dim=args.dim,
        n_blocks=args.n_blocks,
        use_ffn=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_emb = sum(p.numel() for n, p in model.named_parameters() if "embed" in n)

    # --- before training ---
    pre_tr = _accuracy(model, tr_ids, tr_pos, tr_ans, room_toks)
    pre_test = _accuracy(model, test_ids, test_pos, test_ans, room_toks)
    pre_ind = _accuracy(model, ind_ids, ind_pos, ind_ans, color_toks)

    losses = _train(
        model, tr_ids, tr_pos, tr_ans, args.passes, args.lr, args.batch, args.seed
    )

    # --- after training ---
    post_tr = _accuracy(model, tr_ids, tr_pos, tr_ans, room_toks)
    post_test = _accuracy(model, test_ids, test_pos, test_ans, room_toks)
    post_ind = _accuracy(model, ind_ids, ind_pos, ind_ans, color_toks)

    report = {
        "config": vars(args) | {"out": str(args.out)},
        "model": {
            "lane": args.lane,
            "dim": args.dim,
            "n_blocks": args.n_blocks,
            "params": n_params,
            "non_embed_params": n_params - n_emb,
        },
        "data": {
            "two_arg_train": len(tr),
            "two_arg_test": len(test_df),
            "induction": len(ind),
            "rooms": rooms,
            "colors": colors,
        },
        "baselines": {
            "two_arg_chance": round(1 / len(rooms), 4),
            "two_arg_majority": round(two_major, 4),
            "induction_chance": round(1 / len(colors), 4),
            "induction_majority": round(ind_major, 4),
        },
        "before": {
            "two_arg_train": round(pre_tr, 4),
            "two_arg_test": round(pre_test, 4),
            "induction": round(pre_ind, 4),
        },
        "after": {
            "two_arg_train": round(post_tr, 4),
            "two_arg_test": round(post_test, 4),
            "induction": round(post_ind, 4),
        },
        "loss": {
            "first": round(losses[0], 4),
            "last": round(losses[-1], 4),
            "n_steps": len(losses),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(
        f"=== {args.lane} dim{args.dim} nb{args.n_blocks} "
        f"({n_params / 1e6:.1f}M, {(n_params - n_emb) / 1e6:.2f}M non-emb), "
        f"{args.passes} passes ==="
    )
    print(f"two-arg: chance {1 / len(rooms):.3f}  majority {two_major:.3f}")
    print(f"  train  {pre_tr:.3f} -> {post_tr:.3f}")
    print(
        f"  test   {pre_test:.3f} -> {post_test:.3f}   "
        f"(gap train-test = {post_tr - post_test:+.3f})"
    )
    print(
        f"induction (CONTROL, untrained): chance {1 / len(colors):.3f}  "
        f"majority {ind_major:.3f}"
    )
    print(f"  {pre_ind:.3f} -> {post_ind:.3f}")
    print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f} ({len(losses)} steps)")
    print("\nMEMORIZATION: ", end="")
    if post_test >= two_major + 0.10 and (post_tr - post_test) < 0.15:
        print("LEARNED THE RULE (test well above majority, small train-test gap)")
    elif post_tr - post_test > 0.25:
        print("MEMORIZED (large train-test gap)")
    else:
        print("INCONCLUSIVE / didn't learn (test near majority)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
