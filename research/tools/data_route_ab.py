"""Capability A/B for data-pipeline routes (Workstream D, increment 2).

Does *how the token stream is folded/ordered* change what a non-QKV carrier
learns at a fixed token budget? For each ``DataRouteSpec`` condition we train the
SAME carrier (identical init per seed) on the SAME corpus, transforming each
training window with the route, then evaluate on the **natural** (unrouted) val
stream — so the score measures whether the route produced a better model on the
real next-token task, not whether it fit its own scrambled objective.

Mission framing: the route is graded on capability at a fixed token budget
(val next-token top-1 + convergence speed), never on training loss alone. Routes
are a search dimension to beat the natural-order baseline, not an end in
themselves.

Build is dependency-light; RUNNING trains models and is GPU/user-gated. Reuses the
W0/A carrier + corpus + eval helpers — no reimplementation of the training core.

The ``order``/``fold`` routes are pure window permutations and apply to any
carrier. The ``surprisal_split`` route needs a paired (monster+carrier) model
that consumes ``route_prior`` (Workstream B's LossMonsterPairedBlock); it is
covered by the grammar unit tests and wired in a later increment (in-loop
surprisal), so it is intentionally not one of the conditions here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from research.defaults import MAX_SEQ_LEN, N_LAYERS
from dataclasses import replace

from research.synthesis.data_pipeline_grammar import (
    DataRouteSpec,
    apply_data_route,
    batchable_data_route_specs,
    data_route_to_axes,
)
from research.tools.embed_warmup_ab import (
    _DEFAULT_CARRIER_RID,
    _OUT_DIR,
    _build_carrier,
    _carrier_graph_json,
)
from research.tools.loss_monster_screen import (
    _CORPUS_TRAIN,
    _CORPUS_VAL,
    _RUNS_DB,
    evaluate,
)
from research.training.window_packing import (
    DEFAULT_EOT_ID,
    find_doc_boundaries,
    pack_window_starts,
)

# Route conditions: the natural-order baseline plus the batch-applicable
# permutations and packings. order/fold permute the sampled window; doc_boundary
# changes which window is sampled (never crossing a document boundary). Segment
# routes such as surprisal_split need a route-prior-aware paired block and stay
# out of this generic next-token A/B so they are never silent no-ops.
_ROUTE_CONDITIONS: dict[str, DataRouteSpec] = {
    "natural": DataRouteSpec(),
    **batchable_data_route_specs(),
    "doc_boundary_reverse": DataRouteSpec(pack="doc_boundary", order="reverse"),
}


def _sample_routed_batch(
    tokens: np.ndarray,
    batch: int,
    seq: int,
    gen: np.random.Generator,
    device: str,
    spec: DataRouteSpec,
    boundaries: np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a ``seq + 1`` window under ``spec.pack``, apply the order/fold route
    to the FULL window, then split into (input, next-token) so the route changes
    which token is "next"."""
    starts = pack_window_starts(
        tokens.shape[0], batch, seq + 1, spec.pack, gen, boundaries=boundaries
    )
    idx = starts[:, None] + np.arange(seq + 1)[None, :]
    chunk = torch.as_tensor(
        np.ascontiguousarray(tokens[idx]), dtype=torch.int64, device=device
    )
    # pack is already applied (window selection); apply only order/fold here.
    chunk = apply_data_route(chunk, replace(spec, pack="contiguous"))
    return chunk[:, :-1], chunk[:, 1:]


def _train_curve_routed(
    model: torch.nn.Module,
    train: np.ndarray,
    val: np.ndarray,
    spec: DataRouteSpec,
    *,
    seq: int,
    batch: int,
    steps: int,
    lr: float,
    device: str,
    eval_every: int,
    eval_batches: int,
    boundaries: np.ndarray | None,
) -> list[dict[str, float]]:
    """Train on ROUTED windows; evaluate on the NATURAL val stream."""
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
        x, y = _sample_routed_batch(train, batch, seq, gen, device, spec, boundaries)
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
    condition: str,
    spec: DataRouteSpec,
    seed: int,
    train: np.ndarray,
    val: np.ndarray,
    args: argparse.Namespace,
    boundaries: np.ndarray | None,
) -> dict[str, Any]:
    t0 = time.time()
    model = _build_carrier(graph_json, args.n_layers, args.seq, args.device, seed)
    curve = _train_curve_routed(
        model,
        train,
        val,
        spec,
        seq=args.seq,
        batch=args.batch,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        boundaries=boundaries,
    )
    final = curve[-1]
    print(
        f"    [{condition} seed{seed}] final val_loss={final['val_loss']:.4f} "
        f"top1={final['top1_acc']:.4f}  ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return {
        "condition": condition,
        "data_route": data_route_to_axes(spec),
        "seed": seed,
        "curve": curve,
        "final": final,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-route seed-robust capability + convergence speed vs natural."""
    nat = [r["final"]["val_loss"] for r in results if r["condition"] == "natural"]
    nat_top1 = [r["final"]["top1_acc"] for r in results if r["condition"] == "natural"]
    target = float(np.median(nat)) if nat else float("inf")
    baseline_top1 = float(np.median(nat_top1)) if nat_top1 else float("nan")
    summary: dict[str, Any] = {
        "baseline_natural_median_val_loss": round(target, 4),
        "baseline_natural_median_top1": round(baseline_top1, 4),
        "by_condition": {},
    }
    conditions = dict.fromkeys(r["condition"] for r in results)
    for cond in conditions:
        runs = [r for r in results if r["condition"] == cond]
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
            "median_final_val_loss": round(float(np.median(finals)), 4),
            "median_final_top1": round(float(np.median(top1s)), 4),
            "delta_median_val_loss_vs_natural": round(
                float(np.median(finals)) - target, 4
            ),
            "delta_median_top1_vs_natural": round(
                float(np.median(top1s)) - baseline_top1, 4
            ),
            "mean_steps_to_natural_final": (
                round(float(np.mean(s2t)), 1) if s2t else None
            ),
            "median_steps_to_natural_final": (
                round(float(np.median(s2t)), 1) if s2t else None
            ),
            "n_reached_baseline": f"{len(s2t)}/{len(runs)}",
        }
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--carrier-rid", default=_DEFAULT_CARRIER_RID)
    ap.add_argument(
        "--conditions",
        nargs="*",
        default=list(_ROUTE_CONDITIONS),
        help=f"subset of {list(_ROUTE_CONDITIONS)}",
    )
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    ap.add_argument("--seq", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--eot-id",
        type=int,
        default=DEFAULT_EOT_ID,
        help="document separator token id for doc_boundary packing (cl100k eot)",
    )
    ap.add_argument("--out", default=str(_OUT_DIR / "data_route_ab.json"))
    return ap


def main() -> int:
    args = _build_argparser().parse_args()
    unknown = [c for c in args.conditions if c not in _ROUTE_CONDITIONS]
    if unknown:
        print(f"Unknown conditions {unknown}; valid={list(_ROUTE_CONDITIONS)}")
        return 1
    graph_json = _carrier_graph_json(_RUNS_DB, args.carrier_rid)
    train = np.load(_CORPUS_TRAIN, mmap_mode="r")
    val = np.load(_CORPUS_VAL, mmap_mode="r")

    # Document boundaries are only needed when a doc_boundary pack is requested.
    needs_boundaries = any(
        _ROUTE_CONDITIONS[c].pack == "doc_boundary" for c in args.conditions
    )
    boundaries = (
        find_doc_boundaries(np.asarray(train), args.eot_id)
        if needs_boundaries
        else None
    )

    print(
        f"Carrier rid={args.carrier_rid} conditions={args.conditions} "
        f"seeds={args.seeds} steps={args.steps}"
        + (f" docs={len(boundaries) + 1}" if boundaries is not None else "")
    )
    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        for cond in args.conditions:
            results.append(
                run_condition(
                    graph_json,
                    cond,
                    _ROUTE_CONDITIONS[cond],
                    seed,
                    train,
                    val,
                    args,
                    boundaries,
                )
            )

    summary = _summarize(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
