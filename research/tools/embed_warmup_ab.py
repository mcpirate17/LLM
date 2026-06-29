"""Workstream A — does warming the embedding table with a loss monster train faster?

Headline hypothesis (user): pre-train / warm the embedding table with a loss monster, hand
it to the broader (induction-capable) model, and training converges faster and/or reaches
higher capability per step.

Clean matched A/B. For each seed, an induction-capable non-QKV **carrier**
(``latent_compress_block``, rebuilt from ``program_results.graph_json``) is built with an
*identical* random init, then run under three conditions that differ ONLY in the embedding:

- ``cold``           : random embedding, trainable (control)
- ``warm_frozen``    : embedding <- a loss monster's trained cl100k table, FROZEN
- ``warm_unfrozen``  : embedding <- monster table, trainable (fine-tuned)

We log val-loss / next-token-top1 vs steps and report **steps-to-threshold** (the cold
condition's final loss as the target) — the direct "does it train faster" measure — plus
loss at the fixed step budget.

Mission note: loss is the warm-up objective only. The monster never enters the capability
leaderboard; the carrier is the on-mission non-QKV mechanism we want to accelerate. Follow-up
adds an induction-probe pass to test whether warm-up also lifts capability, not just loss.

Reuse-only driver under ``research/tools/`` (reuses Workstream-0 helpers; no harness edits).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from research.defaults import VOCAB_SIZE, MAX_SEQ_LEN, N_LAYERS
from research.scientist.native_runner import compile_model_native_first
from research.synthesis.serializer import graph_from_json
from research.tools.loss_monster_screen import (
    _CORPUS_TRAIN,
    _CORPUS_VAL,
    _RUNS_DB,
    _sample_batch,
    evaluate,
)

# latent_compress_block champion, induction_screening_auc = 1.0 (non-QKV carrier).
_DEFAULT_CARRIER_RID = "4b69e623-3ea"
_DEFAULT_WARM_SOURCE = "recursive_depth_router"  # best next-token monster (W0 roster)
_OUT_DIR = (
    Path(__file__).resolve().parents[2] / "research" / "reports" / "loss_monsters"
)
_CONDITIONS = ("cold", "warm_frozen", "warm_unfrozen")


def _carrier_graph_json(db_path: Path, rid_prefix: str) -> str:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT graph_json FROM program_results WHERE result_id LIKE ? "
            "AND graph_json IS NOT NULL AND graph_json <> '' LIMIT 1",
            (f"{rid_prefix}%",),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise ValueError(f"No carrier graph_json for result_id prefix {rid_prefix!r}")
    return str(row[0])


def _build_carrier(
    graph_json: str, n_layers: int, seq: int, device: str, seed: int
) -> torch.nn.Module:
    torch.manual_seed(seed)  # identical init across conditions for a given seed
    graph = graph_from_json(graph_json)
    model = compile_model_native_first(
        [graph] * n_layers, vocab_size=VOCAB_SIZE, max_seq_len=seq
    ).to(device)
    model.train()  # train/scored mode (eval-mode breaks halt graphs — see W0 note)
    return model


def _apply_warmup(
    model: torch.nn.Module, monster_ckpt: Path, *, trainable: bool
) -> None:
    # SynthesizedModel ties lm_head.weight = embed.weight, so warming `embed` also warms
    # the output head, and (un)freezing `embed` (un)freezes both. This is the tied-table
    # warm-up case from the plan; the monster's embed table IS its next-token geometry.
    emb = model.embed
    assert isinstance(emb, torch.nn.Embedding), (
        f"carrier.embed is {type(emb)}, expected Embedding"
    )
    sd = torch.load(monster_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    with torch.no_grad():
        emb.weight.copy_(sd["embed.weight"].to(emb.weight.device))
    emb.weight.requires_grad_(trainable)


def _train_curve(
    model: torch.nn.Module,
    train: np.ndarray,
    val: np.ndarray,
    *,
    seq: int,
    batch: int,
    steps: int,
    lr: float,
    device: str,
    eval_every: int,
    eval_batches: int,
) -> list[dict[str, float]]:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    gen = np.random.default_rng(1234)
    curve: list[dict[str, float]] = []
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            m = evaluate(
                model, val, batch=batch, seq=seq, n_batches=eval_batches, device=device
            )
            curve.append({"step": step, **m})
        if step == steps:
            break
        x, y = _sample_batch(train, batch, seq, gen, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    return curve


def _steps_to_threshold(curve: list[dict[str, float]], target: float) -> int | None:
    for pt in curve:
        if pt["val_loss"] <= target:
            return int(pt["step"])
    return None


def run_condition(
    graph_json: str,
    monster_ckpt: Path,
    condition: str,
    seed: int,
    train: np.ndarray,
    val: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    t0 = time.time()
    model = _build_carrier(graph_json, args.n_layers, args.seq, args.device, seed)
    if condition != "cold":
        _apply_warmup(model, monster_ckpt, trainable=(condition == "warm_unfrozen"))
    curve = _train_curve(
        model,
        train,
        val,
        seq=args.seq,
        batch=args.batch,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
    )
    final = curve[-1]
    print(
        f"    [{condition} seed{seed}] final val_loss={final['val_loss']:.4f} "
        f"top1={final['top1_acc']:.4f}  ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return {"condition": condition, "seed": seed, "curve": curve, "final": final}


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-condition mean final loss + steps-to-(cold-final) convergence speed."""
    cold_finals = [r["final"]["val_loss"] for r in results if r["condition"] == "cold"]
    target = float(np.mean(cold_finals)) if cold_finals else float("inf")
    summary: dict[str, Any] = {
        "convergence_target_val_loss": round(target, 4),
        "by_condition": {},
    }
    for cond in _CONDITIONS:
        runs = [r for r in results if r["condition"] == cond]
        if not runs:
            continue
        finals = [r["final"]["val_loss"] for r in runs]
        top1s = [r["final"]["top1_acc"] for r in runs]
        s2t = [
            s
            for r in runs
            if (s := _steps_to_threshold(r["curve"], target)) is not None
        ]
        summary["by_condition"][cond] = {
            "mean_final_val_loss": round(float(np.mean(finals)), 4),
            "mean_final_top1": round(float(np.mean(top1s)), 4),
            "mean_steps_to_cold_final": (
                round(float(np.mean(s2t)), 1) if s2t else None
            ),
            "n_reached_target": f"{len(s2t)}/{len(runs)}",
        }
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--carrier-rid", default=_DEFAULT_CARRIER_RID)
    ap.add_argument(
        "--warm-source",
        default=_DEFAULT_WARM_SOURCE,
        help="loss-monster family name (ckpt under reports/loss_monsters/)",
    )
    ap.add_argument("--conditions", nargs="*", default=list(_CONDITIONS))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    ap.add_argument("--seq", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "embed_warmup_ab.json"))
    return ap


def main() -> int:
    args = _build_argparser().parse_args()
    graph_json = _carrier_graph_json(_RUNS_DB, args.carrier_rid)
    monster_ckpt = _OUT_DIR / f"{args.warm_source}.pt"
    if not monster_ckpt.exists():
        print(
            f"Missing warm-source checkpoint {monster_ckpt} — run loss_monster_screen first."
        )
        return 1
    train = np.load(_CORPUS_TRAIN, mmap_mode="r")
    val = np.load(_CORPUS_VAL, mmap_mode="r")

    print(
        f"Carrier rid={args.carrier_rid} warm_source={args.warm_source} "
        f"conditions={args.conditions} seeds={args.seeds} steps={args.steps}"
    )
    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        for cond in args.conditions:
            results.append(
                run_condition(graph_json, monster_ckpt, cond, seed, train, val, args)
            )

    summary = _summarize(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"config": vars(args), "summary": summary, "results": results}, indent=2
        )
    )
    print("\n=== SUMMARY ===")
    print(
        f"convergence target (cold final val_loss) = {summary['convergence_target_val_loss']}"
    )
    for cond, s in summary["by_condition"].items():
        print(
            f"  {cond:14s} final_loss={s['mean_final_val_loss']:.4f} "
            f"top1={s['mean_final_top1']:.4f} steps_to_target={s['mean_steps_to_cold_final']} "
            f"({s['n_reached_target']})"
        )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
