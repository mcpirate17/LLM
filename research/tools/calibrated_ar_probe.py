"""Calibrated AR probe — Zoology-aligned gMQAR with a softmax positive-control gate.

Reuses research/eval/gmqar.py (the already-aligned zero-shot MQAR: multi-query,
disjoint key/value vocab, multi-position answer masking, candidate-restricted
scoring, difficulty grid -> AUDC/D50). The legacy associative_recall.py probe is
NOT used or modified.

Modes
-----
capacity : train a FRESH small model on gMQAR to convergence (Zoology-style
           architectural-capacity test). softmax_ffn is the POSITIVE CONTROL: if
           attention does not clear the gate (AUDC >= --gate-audc), the harness/
           config is the problem, not the model under test -- so a non-QKV floor
           in that case is meaningless.
zeroshot : score an already-trained checkpoint zero-shot (in-context recall of
           the real pretrained model, e.g. hyper_mor_b step-40k).

GPU is hard-capped via set_per_process_memory_fraction so this can never starve
the live training job; capacity training falls back to CPU on OOM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from research.eval.gmqar import (
    GMQARConfig,
    default_grid,
    make_gmqar_batch,
    score_model_gmqar,
)
from research.tools._scaling_lanes import _build_lane_factory
from research.tools.scaling_blimp_study import _build_tinylm

HYPER_LANE = (
    "hyper_mor_surprise_refine_mlp258_native_semiring_adapt_bilane"
    "_m32_g0_t1_b1_l0_h2_r7_surprise_memory"
)
CKPT = (
    "research/reports/hyper_mor_b_chin_ckpts/hyper_mor_b_chin_"
    + HYPER_LANE
    + "_step040000.pt"
)


def _cap_gpu(fraction: float, device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, 0)


def _train_on_gmqar(model, *, vocab, steps, lr, device, train_pairs, token_pool):
    """Zoology-style: train a fresh model on gMQAR (multi-position CE) to convergence."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator(device=device).manual_seed(0)
    curve = []
    for step in range(1, steps + 1):
        n_pairs = int(train_pairs[step % len(train_pairs)])
        cfg = GMQARConfig(
            vocab_size=vocab,
            n_pairs=n_pairs,
            n_queries=min(4, n_pairs),
            batch_size=32,
            token_pool=token_pool,
        )
        ids, tgt, _ = make_gmqar_batch(cfg, g, device)
        opt.zero_grad(set_to_none=True)
        out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100
        )
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 8) == 0 or step == steps:
            res = score_model_gmqar(
                model, vocab_size=vocab, device=device, token_pool=token_pool
            )
            curve.append(
                {
                    "step": step,
                    "audc": res.audc,
                    "d50": res.d50,
                    "loss": round(float(loss.detach()), 4),
                }
            )
            model.train()
    return curve


def _capacity(label, lane, args):
    """Train a fresh small model on gMQAR; report convergence curve + final grid."""
    for dev in (args.device, "cpu"):
        try:
            m = _build_tinylm(
                _build_lane_factory(lane),
                dim=args.dim,
                n_blocks=args.n_blocks,
                vocab_size=args.cap_vocab,
            )
            curve = _train_on_gmqar(
                m,
                vocab=args.cap_vocab,
                steps=args.steps,
                lr=args.lr,
                device=dev,
                train_pairs=(4, 8, 16),
                token_pool=0,
            )
            final = score_model_gmqar(m, vocab_size=args.cap_vocab, device=dev)
            row = {
                "mode": "capacity",
                "model": label,
                "device": dev,
                "audc": final.audc,
                "d50": final.d50,
                "chance": final.chance,
                "curve": curve,
                "cells": final.cells,
            }
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)
            return row
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "mode": "capacity",
                        "model": label,
                        "note": f"OOM on {dev}, falling back",
                    }
                ),
                flush=True,
            )
            continue
        except Exception as e:  # noqa: BLE001
            print(
                json.dumps({"mode": "capacity", "model": label, "error": str(e)[:200]}),
                flush=True,
            )
            return None


def _zeroshot_grid(vocab_size, token_pool, max_pairs):
    """default_grid extended to larger n_pairs to find the true breaking point."""
    grid = default_grid(vocab_size=vocab_size, token_pool=token_pool)
    base_pairs = {c.n_pairs for c in grid}
    for n_pairs in (64, 128, 256):
        if n_pairs <= max_pairs and n_pairs not in base_pairs:
            for distract in (0, 128):
                grid.append(
                    GMQARConfig(
                        vocab_size=vocab_size,
                        n_pairs=n_pairs,
                        n_queries=min(4, n_pairs),
                        distractor_tokens=distract,
                        token_pool=token_pool,
                    )
                )
    return grid


def _zeroshot_ckpt(args):
    """Zero-shot gMQAR on a real pretrained hyper_mor_b checkpoint."""
    ckpt = args.checkpoint or CKPT
    payload = torch.load(ckpt, map_location="cpu")  # nosec B614 - local ckpt
    for dev in (args.device, "cpu"):
        try:
            m = _build_tinylm(
                _build_lane_factory(HYPER_LANE), dim=736, n_blocks=8, vocab_size=100277
            )
            m.load_state_dict(payload["model_state_dict"])
            m.to(dev)
            grid = _zeroshot_grid(100277, args.token_pool, args.max_pairs)
            res = score_model_gmqar(
                m, grid=grid, vocab_size=100277, device=dev, token_pool=args.token_pool
            )
            row = {
                "mode": "zeroshot",
                "model": args.ckpt_label,
                "device": dev,
                "audc": res.audc,
                "d50": res.d50,
                "chance": res.chance,
                "cells": res.cells,
            }
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)
            return row
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(
                json.dumps({"mode": "zeroshot", "note": f"OOM on {dev}, falling back"}),
                flush=True,
            )
            continue
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"mode": "zeroshot", "error": str(e)[:200]}), flush=True)
            return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["capacity", "zeroshot", "both"], default="both")
    ap.add_argument("--dim", type=int, default=128)  # small capacity models
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument(
        "--cap-vocab", type=int, default=512
    )  # capacity-mode synthetic vocab
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--token-pool", type=int, default=2048
    )  # zero-shot: well-trained ids
    ap.add_argument(
        "--gate-audc",
        type=float,
        default=0.30,
        help="softmax control must clear this AUDC or results are void",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="zeroshot: path to the hyper_mor_b checkpoint (default step-40k)",
    )
    ap.add_argument("--ckpt-label", type=str, default="hyper_mor_b_CKPT@40k")
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=32,
        help="zeroshot: extend the difficulty grid up to this n_pairs (64/128/256)",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-frac", type=float, default=0.08)  # hard cap vs live training
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/calibrated_ar_probe.jsonl")
    )
    args = ap.parse_args()

    _cap_gpu(args.gpu_frac, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.mode in ("capacity", "both"):
        ctrl = _capacity("softmax_ffn[POSITIVE]", "softmax_ffn", args)
        rows.append(ctrl)
        rows.append(_capacity("hyper_mor[non-QKV]", HYPER_LANE, args))
        gate = ctrl and ctrl.get("audc", 0) >= args.gate_audc
        print(
            json.dumps(
                {
                    "gate": "PASS" if gate else "FAIL",
                    "softmax_audc": (ctrl or {}).get("audc"),
                    "threshold": args.gate_audc,
                    "verdict": "harness valid; non-QKV result is trustworthy"
                    if gate
                    else "harness/config under-powered; results VOID",
                }
            ),
            flush=True,
        )

    if args.mode in ("zeroshot", "both"):
        rows.append(_zeroshot_ckpt(args))

    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            if r:
                f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len([r for r in rows if r])} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
