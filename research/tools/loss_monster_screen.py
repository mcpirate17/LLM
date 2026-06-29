"""Workstream 0b — rebuild loss-monster family champions and validate next-token quality.

The original loss-monster checkpoints were deleted, but ``program_results.graph_json``
still holds each architecture's exact graph. This tool, per family (template_name):

1. picks the family champion (lowest ``screening_loss_ratio`` among dead-induction,
   >=1M-param rows that still have a ``graph_json``),
2. rebuilds it exactly (``graph_from_json`` -> ``compile_model_native_first``),
3. trains it briefly on FineFineWeb (cl100k) with a next-token LM objective,
4. measures next-token **top-1 / top-2** accuracy vs the **unigram floor**, plus loss/ppl,
5. saves a checkpoint + a metrics row for the Workstream-A/B seed roster.

A champion that cannot beat the unigram floor on top-1 is not a usable scaffold and is
flagged ``usable=False``. Monsters are graded on next-token *as a scaffold-fitness check
only* — they never enter the capability leaderboard on loss (see
``tasks/loss_monster_scaffolding_plan.md``).

Reuse-only: no training/runner files are edited; this is a self-contained driver under
``research/tools/`` (the safe scripts zone). Gemini's harness is imported, not modified.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from research.defaults import VOCAB_SIZE, MAX_SEQ_LEN, N_LAYERS
from research.scientist.native_runner import compile_model_native_first
from research.synthesis.serializer import graph_from_json

_REPO = Path(__file__).resolve().parents[2]
_RUNS_DB = _REPO / "research" / "runs.db"
_CORPUS_TRAIN = _REPO / "research" / "corpus" / "finefineweb_train.npy"
_CORPUS_VAL = _REPO / "research" / "corpus" / "finefineweb_val.npy"
_OUT_DIR = _REPO / "research" / "reports" / "loss_monsters"


@dataclass(frozen=True, slots=True)
class Champion:
    family: str
    result_id: str
    arch_desc: str
    param_count: float
    screening_loss_ratio: float
    induction_auc: float
    graph_json: str


def select_family_champions(
    db_path: Path,
    *,
    min_params: float = 1_000_000.0,
    max_loss: float = 0.15,
    max_induction: float = 0.05,
    families: tuple[str, ...] | None = None,
) -> list[Champion]:
    """One champion per template family: lowest loss_ratio, dead induction, has a graph."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT f.template_name AS family, l.result_id AS result_id,
                   COALESCE(l.architecture_desc, l.entry_id) AS arch_desc,
                   l.param_count AS param_count,
                   l.screening_loss_ratio AS screening_loss_ratio,
                   l.induction_screening_auc AS induction_auc,
                   p.graph_json AS graph_json
            FROM leaderboard l
            JOIN program_graph_features f ON l.result_id = f.result_id
            JOIN program_results p ON l.result_id = p.result_id
            WHERE l.param_count >= ?
              AND l.screening_loss_ratio < ?
              AND l.induction_screening_auc < ?
              AND f.template_name IS NOT NULL AND f.template_name <> ''
              AND p.graph_json IS NOT NULL AND p.graph_json <> ''
            ORDER BY l.screening_loss_ratio ASC
            """,
            (min_params, max_loss, max_induction),
        ).fetchall()
    finally:
        conn.close()

    best: dict[str, Champion] = {}
    for r in rows:
        fam = str(r["family"])
        if families and fam not in families:
            continue
        if fam in best:  # rows are loss-ascending, so first seen is the champion
            continue
        best[fam] = Champion(
            family=fam,
            result_id=str(r["result_id"]),
            arch_desc=str(r["arch_desc"]),
            param_count=float(r["param_count"] or 0.0),
            screening_loss_ratio=float(r["screening_loss_ratio"] or 0.0),
            induction_auc=float(r["induction_auc"] or 0.0),
            graph_json=str(r["graph_json"]),
        )
    # Cheapest (lowest-loss) families first.
    return sorted(best.values(), key=lambda c: c.screening_loss_ratio)


def unigram_floor(tokens: np.ndarray, vocab_size: int) -> dict[str, float]:
    """Top-1 / top-2 next-token accuracy achievable by always guessing the modal token."""
    counts = np.bincount(tokens, minlength=vocab_size).astype(np.float64)
    total = counts.sum()
    top = np.sort(counts)[::-1]
    return {
        "unigram_top1": float(top[0] / total),
        "unigram_top2": float((top[0] + top[1]) / total),
    }


def _sample_batch(
    tokens: np.ndarray, batch: int, seq: int, gen: np.random.Generator, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    hi = tokens.shape[0] - seq - 1
    starts = gen.integers(0, hi, size=batch)
    idx = starts[:, None] + np.arange(seq + 1)[None, :]
    chunk = torch.as_tensor(  # pyright: ignore[reportPrivateImportUsage]
        np.ascontiguousarray(tokens[idx]),
        dtype=torch.int64,  # pyright: ignore[reportPrivateImportUsage]
        device=device,
    )
    return chunk[:, :-1], chunk[:, 1:]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val: np.ndarray,
    *,
    batch: int,
    seq: int,
    n_batches: int,
    device: str,
) -> dict[str, float]:
    # NOTE: do NOT switch to model.eval(). These synthesis graphs (esp. adaptive
    # depth-router / halt ops) have a divergent eval-mode branch and were originally
    # scored in train mode; .eval() sends val_loss off a cliff (25.6 vs train 6.0).
    # We measure next-token in the same mode the monster was trained/scored in.
    gen = np.random.default_rng(0)
    tot_loss = tot_tok = top1 = top2 = 0.0
    for _ in range(n_batches):
        x, y = _sample_batch(val, batch, seq, gen, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        top = logits.topk(2, dim=-1).indices
        c1 = top[..., 0] == y
        c2 = c1 | (top[..., 1] == y)
        n = float(y.numel())
        tot_loss += float(loss) * n
        top1 += float(c1.sum())
        top2 += float(c2.sum())
        tot_tok += n
    mean_loss = tot_loss / tot_tok
    return {
        "val_loss": mean_loss,
        "val_ppl": float(np.exp(min(mean_loss, 20.0))),
        "top1_acc": top1 / tot_tok,
        "top2_acc": top2 / tot_tok,
    }


def screen_champion(
    champ: Champion,
    train: np.ndarray,
    val: np.ndarray,
    *,
    n_layers: int,
    seq: int,
    batch: int,
    steps: int,
    lr: float,
    device: str,
    eval_batches: int,
) -> dict[str, Any]:
    t0 = time.time()
    graph = graph_from_json(champ.graph_json)
    model = compile_model_native_first(
        [graph] * n_layers, vocab_size=VOCAB_SIZE, max_seq_len=seq
    ).to(device)
    model.train()  # train/scored mode throughout (eval-mode breaks halt graphs)
    n_params = sum(p.numel() for p in model.parameters())

    pre = evaluate(
        model, val, batch=batch, seq=seq, n_batches=eval_batches, device=device
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    gen = np.random.default_rng(1234)
    model.train()
    last = 0.0
    for step in range(steps):
        x, y = _sample_batch(train, batch, seq, gen, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss)
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(
                f"    [{champ.family}] step {step}/{steps} train_loss={last:.4f}",
                flush=True,
            )

    post = evaluate(
        model, val, batch=batch, seq=seq, n_batches=eval_batches, device=device
    )
    floor = unigram_floor(val[:2_000_000].astype(np.int64), VOCAB_SIZE)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = _OUT_DIR / f"{champ.family}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "family": champ.family,
            "result_id": champ.result_id,
            "n_layers": n_layers,
            "vocab": VOCAB_SIZE,
            "dim": graph.model_dim,
        },
        ckpt,
    )

    beats_floor_top1 = post["top1_acc"] > floor["unigram_top1"]
    return {
        **asdict(champ),
        "graph_json": f"<{len(champ.graph_json)} chars>",  # don't dump the whole graph
        "compiled_params_m": round(n_params / 1e6, 2),
        "n_layers": n_layers,
        "pre_train": pre,
        "post_train": post,
        **floor,
        "final_train_loss": last,
        # scaffold-fitness gate: must beat the unigram floor on top-1
        "usable": bool(beats_floor_top1),
        "top1_lift_over_floor": round(post["top1_acc"] - floor["unigram_top1"], 4),
        "checkpoint": str(ckpt),
        "elapsed_s": round(time.time() - t0, 1),
    }


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="restrict to these template families (default: all)",
    )
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    ap.add_argument("--seq", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-families", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=str(_OUT_DIR / "loss_monster_screen.json"))
    return ap


def _screen_one(
    champ: Champion, train: np.ndarray, val: np.ndarray, args: argparse.Namespace
) -> dict[str, Any]:
    print(
        f"\n=== {champ.family}  (loss_ratio={champ.screening_loss_ratio:.3f}, "
        f"hist_ind={champ.induction_auc:.4f}, {champ.param_count / 1e6:.1f}M) ===",
        flush=True,
    )
    try:
        res = screen_champion(
            champ,
            train,
            val,
            n_layers=args.n_layers,
            seq=args.seq,
            batch=args.batch,
            steps=args.steps,
            lr=args.lr,
            device=args.device,
            eval_batches=args.eval_batches,
        )
    except Exception as exc:  # loud per-candidate failure, screen continues
        res = {
            **asdict(champ),
            "graph_json": f"<{len(champ.graph_json)} chars>",
            "error": f"{type(exc).__name__}: {exc}",
            "usable": False,
        }
        print(f"    FAILED: {res['error']}", flush=True)
        return res
    p = res["post_train"]
    print(
        f"    -> top1={p['top1_acc']:.4f} top2={p['top2_acc']:.4f} "
        f"floor_top1={res['unigram_top1']:.4f} "
        f"lift={res['top1_lift_over_floor']:+.4f} usable={res['usable']}",
        flush=True,
    )
    return res


def main() -> int:
    args = _build_argparser().parse_args()

    fams = tuple(args.families) if args.families else None
    champs = select_family_champions(_RUNS_DB, families=fams)
    if args.max_families > 0:
        champs = champs[: args.max_families]
    if not champs:
        print("No family champions matched the filters.")
        return 1

    print(
        f"Screening {len(champs)} family champion(s) on {args.device}: "
        f"{[c.family for c in champs]}"
    )
    train = np.load(_CORPUS_TRAIN, mmap_mode="r")
    val = np.load(_CORPUS_VAL, mmap_mode="r")

    results = [_screen_one(champ, train, val, args) for champ in champs]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": vars(args),
                "results": results,
                "n_usable": sum(1 for r in results if r.get("usable")),
            },
            indent=2,
        )
    )
    print(
        f"\nWrote {out}  ({sum(1 for r in results if r.get('usable'))}/{len(results)} usable)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
