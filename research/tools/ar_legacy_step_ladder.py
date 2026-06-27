"""ar_legacy positive-control + step-ladder.

Question: is the universal ar_legacy floor (every scale model ~chance) a real
architectural ceiling, or a probe-under-training artifact? The runs.db scores
were taken at n_train_steps=300, batch=8 — *below* the probe's own documented
pass spec ("full attention passes >15% at step 500"), and the learning curves
are flat→declining at chance with NO positive control.

This runs `associative_recall_score` (the exact ar_legacy probe) on:
  - softmax_ffn  (GPT2-style attention) -> POSITIVE CONTROL: must rise if the
    harness can detect AR-learning at all.
  - the hyper_mor non-QKV lane          -> the mission mechanism, fresh init.
  - optionally the hyper_mor_b step-40k CHECKPOINT (the real pretrained model).
across a step ladder so we can read WHERE each architecture's curve lifts off.

Small fresh models run on GPU (tiny footprint, safe alongside the live training
job). The 144M checkpoint is GPU-guarded with a CPU/skip fallback so it can
never starve the training run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from research.eval.associative_recall import associative_recall_score
from research.tools._scaling_lanes import _build_lane_factory
from research.tools.scaling_blimp_study import _build_tinylm

VOCAB = 100277
HYPER_LANE = (
    "hyper_mor_surprise_refine_mlp258_native_semiring_adapt_bilane"
    "_m32_g0_t1_b1_l0_h2_r7_surprise_memory"
)
CKPT = (
    "research/reports/hyper_mor_b_chin_ckpts/hyper_mor_b_chin_"
    + HYPER_LANE
    + "_step040000.pt"
)


def _build(lane_name: str, dim: int, n_blocks: int) -> torch.nn.Module:
    factory = _build_lane_factory(lane_name)
    return _build_tinylm(factory, dim=dim, n_blocks=n_blocks, vocab_size=VOCAB)


def _run(model, label, steps, batch, lr, device, timeout_s):
    t0 = time.perf_counter()
    res = associative_recall_score(
        model,
        n_pairs=20,
        n_train_steps=steps,
        n_eval=200,
        eval_every=max(100, steps // 5),
        lr=lr,
        batch_size=batch,
        device=device,
        timeout_s=timeout_s,
    )
    row = {
        "model": label,
        "steps": steps,
        "batch": batch,
        "lr": lr,
        "final_acc": round(res.final_acc, 4),
        "above_chance": res.above_chance,
        "auc": res.auc,
        "curve": res.learning_curve,
        "status": res.status,
        "timed_out": res.timed_out,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    print(json.dumps(row), flush=True)
    return row


def _empty_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()


def _err(label, steps, exc, **extra) -> None:
    print(
        json.dumps({"model": label, "steps": steps, "error": str(exc)[:200], **extra}),
        flush=True,
    )


def _ladder_controls(args) -> list[dict]:
    """Fresh small attention (positive) vs non-QKV (mission) over the step ladder."""
    rows = []
    controls = [
        ("softmax_ffn", "softmax_ffn[POSITIVE]"),
        (HYPER_LANE, "hyper_mor[non-QKV]"),
    ]
    # (steps, batch): the ladder at batch 8 + the probe's documented 500/batch-16 spec point
    points = [(s, 8) for s in args.ladder] + [(500, 16)]
    for lane, label in controls:
        for steps, batch in points:
            try:
                m = _build(lane, args.dim, args.n_blocks)
                rows.append(
                    _run(m, label, steps, batch, args.lr, args.device, args.timeout_s)
                )
                del m
                _empty_cache(args.device)
            except Exception as e:  # noqa: BLE001 - report and continue the ladder
                _err(label, steps, e, batch=batch)
    return rows


def _ladder_checkpoint(args) -> list[dict]:
    """The real pretrained 144M checkpoint, GPU-first with CPU fallback (never OOMs training)."""
    rows = []
    payload = torch.load(CKPT, map_location="cpu")  # nosec B614 - local ckpt
    for steps in (300, 1000):
        for dev in (args.device, "cpu"):
            try:
                m = _build(HYPER_LANE, 736, 8)
                m.load_state_dict(payload["model_state_dict"])
                rows.append(
                    _run(
                        m,
                        "hyper_mor_b_CKPT@40k",
                        steps,
                        8,
                        args.lr,
                        dev,
                        args.timeout_s,
                    )
                )
                del m
                _empty_cache(dev)
                break  # succeeded on this device
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(
                    json.dumps(
                        {
                            "model": "hyper_mor_b_CKPT@40k",
                            "steps": steps,
                            "note": f"OOM on {dev}, falling back",
                        }
                    ),
                    flush=True,
                )
                continue
            except Exception as e:  # noqa: BLE001
                _err("hyper_mor_b_CKPT@40k", steps, e)
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--ladder", type=int, nargs="+", default=[300, 500, 1000, 2000])
    ap.add_argument("--lr", type=float, default=1e-3)  # harness default
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/ar_legacy_step_ladder.jsonl")
    )
    ap.add_argument(
        "--with-checkpoint",
        action="store_true",
        help="also ladder the real hyper_mor_b step-40k checkpoint",
    )
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = _ladder_controls(args)
    if args.with_checkpoint:
        rows += _ladder_checkpoint(args)

    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
