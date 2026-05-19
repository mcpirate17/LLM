"""Autonomous scaling campaign orchestrator.

Runs the full multi-model x multi-scale BLiMP study with the user's
early-stop rule:

1. FIRST cell: block_gated_parallel @ 30M, train to 10K. If BLiMP < threshold,
   HALT the entire campaign and write a report.
2. Otherwise, for the same model continue 20K, 40K. Then 60M, 120M.
3. After block_gated_parallel: do other models (softmax_ffn, simplified_mamba,
   recursive_depth_router, hetero_moe_block, tropical_attention).
4. Result: a per-cell JSON + a top-level scaling_campaign_summary.json.

Each cell is run as a fresh subprocess via ``scaling_blimp_study``, so a
crash in one cell doesn't kill the campaign.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]
_OUT_DIR = _REPO / "component_fab" / "catalog"


MODELS = (
    "block_gated_parallel",  # GATE — must clear first cell or campaign halts
    "softmax_ffn",
    "simplified_mamba",
    "recursive_depth_router",
    "hetero_moe_block",
    "tropical_attention",
)

SIZES = ("30M", "60M", "120M")


def _cell_path(lane: str, size: str) -> Path:
    return _OUT_DIR / f"scaling_blimp_{lane}_{size}.json"


def _run_cell(
    lane: str,
    size: str,
    n_train_steps: int,
    checkpoints: tuple[int, ...],
    early_stop_blimp: float | None,
    quiet: bool,
) -> dict:
    """Run one cell as a subprocess; return parsed JSON result."""
    out = _cell_path(lane, size)
    cmd = [
        sys.executable,
        "-m",
        "research.tools.scaling_blimp_study",
        "--lane-name",
        lane,
        "--size",
        size,
        "--n-train-steps",
        str(n_train_steps),
        "--checkpoint-steps",
        ",".join(str(c) for c in checkpoints),
        "--output",
        str(out),
    ]
    if early_stop_blimp is not None:
        cmd.extend(["--early-stop-blimp", str(early_stop_blimp)])
    log_path = _REPO / "research" / "reports" / f"scaling_log_{lane}_{size}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"\n>>> Running {lane} @ {size} (log: {log_path})", flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    if not out.exists():
        return {
            "status": f"subprocess_failed_no_output rc={rc.returncode}",
            "elapsed_s": elapsed,
        }
    result = json.loads(out.read_text())
    result["_cell_wall_clock_s"] = round(elapsed, 1)
    return result


def _best_blimp_in_result(result: dict) -> float:
    ckpts = result.get("checkpoints") or []
    if not ckpts:
        return 0.0
    return max(c.get("blimp_overall", 0.0) for c in ckpts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.55,
        help="If the gating cell's BLiMP < this, abort the campaign.",
    )
    parser.add_argument(
        "--gate-lane",
        type=str,
        default="block_gated_parallel",
        help="Which lane to use as the gate cell.",
    )
    parser.add_argument(
        "--gate-size",
        type=str,
        default="30M",
        help="Which size to use as the gate cell.",
    )
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--checkpoints", type=str, default="10000,20000,40000")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=_OUT_DIR / "scaling_campaign_summary.json",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    ckpts = tuple(int(x) for x in args.checkpoints.split(",") if x.strip())
    started = time.monotonic()
    summary: dict = {
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gate_threshold": args.gate_threshold,
        "checkpoints": list(ckpts),
        "cells": {},
        "aborted": False,
    }

    # ── Gating cell ─────────────────────────────────────────────────────
    gate_existing = _cell_path(args.gate_lane, args.gate_size)
    if gate_existing.exists():
        if not args.quiet:
            print(f"Gating cell already exists, reusing: {gate_existing}", flush=True)
        gate_result = json.loads(gate_existing.read_text())
    else:
        gate_result = _run_cell(
            args.gate_lane,
            args.gate_size,
            n_train_steps=ckpts[0],
            checkpoints=(ckpts[0],),
            early_stop_blimp=args.gate_threshold,
            quiet=args.quiet,
        )
    gate_key = f"{args.gate_lane}_{args.gate_size}"
    summary["cells"][gate_key] = gate_result
    best_gate = _best_blimp_in_result(gate_result)
    if not args.quiet:
        print(
            f"\nGate cell {gate_key} best BLiMP: {best_gate:.4f} (threshold {args.gate_threshold})",
            flush=True,
        )

    if best_gate < args.gate_threshold:
        summary["aborted"] = True
        summary["abort_reason"] = (
            f"gate_cell_blimp={best_gate:.4f} below threshold {args.gate_threshold}"
        )
        summary["elapsed_total_s"] = round(time.monotonic() - started, 1)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        if not args.quiet:
            print(f"\nCAMPAIGN ABORTED. Summary: {args.summary_output}", flush=True)
        return 0

    # ── Continue with the full grid ──────────────────────────────────────
    if not args.quiet:
        print("\nGate cleared. Running full grid.", flush=True)

    for lane in MODELS:
        for size in SIZES:
            cell_key = f"{lane}_{size}"
            if cell_key in summary["cells"]:
                continue
            existing = _cell_path(lane, size)
            if existing.exists():
                summary["cells"][cell_key] = json.loads(existing.read_text())
                if not args.quiet:
                    print(f"reusing existing cell: {cell_key}", flush=True)
                continue
            # No early-stop on subsequent cells (gate already passed).
            result = _run_cell(
                lane,
                size,
                n_train_steps=args.max_steps,
                checkpoints=ckpts,
                early_stop_blimp=None,
                quiet=args.quiet,
            )
            summary["cells"][cell_key] = result
            # Save incrementally so a crash doesn't lose work.
            summary["elapsed_total_s"] = round(time.monotonic() - started, 1)
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            args.summary_output.write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
            best = _best_blimp_in_result(result)
            if not args.quiet:
                print(f"  cell {cell_key} best BLiMP: {best:.4f}", flush=True)

    summary["elapsed_total_s"] = round(time.monotonic() - started, 1)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    if not args.quiet:
        print(f"\nCampaign complete. Summary: {args.summary_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
