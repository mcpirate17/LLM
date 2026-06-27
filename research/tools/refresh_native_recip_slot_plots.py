"""Refresh native_recip_slot plots every N training steps."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from research.tools.native_gate_floor_utils import DEFAULT_NATIVE_GATE_FLOORS_CSV


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS = PROJECT_ROOT / "research" / "reports"
CKPT_DIR = REPORTS / "native_adaptive_hydra_ckpts"
RUN_JSONL = REPORTS / "native_recip_slot_chin_floor25_gateaux_ckpt1k_2026-06-13.jsonl"
RESUME_CHECKPOINT = (
    CKPT_DIR
    / "native_recip_slot_chin_native_adaptive_reciprocal_slot_delta_step050000.pt"
)
CHECKPOINT_GLOBS = (
    "native_recip_slot_chin_floor25_gateaux_ckpt1k_native_adaptive_reciprocal_slot_delta_step*.pt",
    "native_recip_slot_chin_floor25_native_adaptive_reciprocal_slot_delta_step*.pt",
    "native_recip_slot_chin_corrected_native_adaptive_reciprocal_slot_delta_step*.pt",
)
STATE_PATH = REPORTS / "native_recip_slot_plot_refresh_state.json"


def _latest_step(path: Path) -> int | None:
    if not path.exists():
        return None
    latest: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = row.get("step")
            if isinstance(step, int):
                latest = max(latest or step, step)
    return latest


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"_step(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _latest_probe_checkpoint() -> Path | None:
    for pattern in CHECKPOINT_GLOBS:
        checkpoints = sorted(
            CKPT_DIR.glob(pattern), key=lambda p: (_checkpoint_step(p), p.name)
        )
        if checkpoints:
            return checkpoints[-1]
    return RESUME_CHECKPOINT if RESUME_CHECKPOINT.exists() else None


def _render_gate_report(*, probe_json: Path, suffix: str) -> None:
    _run(
        [
            sys.executable,
            "research/reports/mor_histo.py",
            "--mode",
            "gate",
            "--input",
            str(probe_json),
            "--table-output",
            str(REPORTS / f"native_recip_slot_gate_probe_{suffix}_table.txt"),
            "--gate-plot-output",
            str(REPORTS / f"native_recip_slot_gate_probe_{suffix}_gate.png"),
        ]
    )


def refresh(args: argparse.Namespace, *, force_probe: bool = False) -> dict[str, Any]:
    latest = _latest_step(args.run_jsonl)
    if latest is None:
        raise RuntimeError(f"No step rows found in {args.run_jsonl}")

    _run(
        [
            sys.executable,
            "research/reports/mor_histo.py",
            "--mode",
            "mor",
            "--input",
            str(args.run_jsonl),
            "--out",
            str(args.depth_out),
        ]
    )
    for extra_run, extra_out in zip(args.extra_depth_runs, args.extra_depth_outs):
        if not extra_run.exists():
            continue
        _run(
            [
                sys.executable,
                "research/reports/mor_histo.py",
                "--mode",
                "mor",
                "--input",
                str(extra_run),
                "--out",
                str(extra_out),
            ]
        )

    dashboard_paths = [path for path in args.dashboard_runs] + [args.run_jsonl]
    seen_dashboard_paths: set[Path] = set()
    dashboard_runs = []
    for path in dashboard_paths:
        resolved = path.resolve()
        if resolved in seen_dashboard_paths:
            continue
        seen_dashboard_paths.add(resolved)
        dashboard_runs.append(str(path))
    _run(
        [
            sys.executable,
            "research/reports/mor_histo.py",
            "--mode",
            "curves",
            "--runs",
            *dashboard_runs,
            "--out",
            str(args.dashboard_out),
            "--ppl",
            "--lr",
            "--grad",
            "--depth",
            "--title",
            "native_recip_slot: original vs 10pct floor vs 25pct floor vs gate-aux",
        ]
    )

    state = _load_state(args.state)
    checkpoint = _latest_probe_checkpoint()
    probed_checkpoint = None
    if checkpoint is not None and (
        force_probe or state.get("last_probe_checkpoint") != str(checkpoint)
    ):
        step = _checkpoint_step(checkpoint)
        suffix = f"step{step:06d}" if step >= 0 else "resume"
        probe_json = REPORTS / f"native_recip_slot_gate_probe_{suffix}_latest.json"
        _run(
            [
                sys.executable,
                "-m",
                "research.tools.native_recip_slot_gate_probe",
                "--checkpoint",
                str(checkpoint),
                "--out-json",
                str(probe_json),
                "--out-plot",
                str(REPORTS / f"native_recip_slot_gate_probe_{suffix}_latest.png"),
                "--batches",
                str(args.probe_batches),
                "--batch",
                str(args.probe_batch),
                "--seq-len",
                str(args.probe_seq_len),
                "--device",
                args.device,
                "--native-gate-floors",
                args.native_gate_floors,
            ]
        )
        _render_gate_report(probe_json=probe_json, suffix=f"{suffix}_latest")
        probed_checkpoint = str(checkpoint)

    return {
        "latest_step": latest,
        "step_bucket": latest // args.every_steps,
        "probed_checkpoint": probed_checkpoint,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-jsonl", type=Path, default=RUN_JSONL)
    parser.add_argument(
        "--depth-out",
        type=Path,
        default=REPORTS
        / "native_recip_slot_floor25_gateaux_ckpt1k_depth_hist_2026-06-13.png",
    )
    parser.add_argument(
        "--extra-depth-runs",
        type=Path,
        nargs="*",
        default=[
            REPORTS / "native_recip_slot_chin_floor25_2026-06-13.jsonl",
            REPORTS / "native_recip_slot_chin_corrected_2026-06-13.jsonl",
            REPORTS / "native_recip_slot_chin_2026-06-12.jsonl",
        ],
    )
    parser.add_argument(
        "--extra-depth-outs",
        type=Path,
        nargs="*",
        default=[
            REPORTS / "native_recip_slot_floor25_depth_hist_2026-06-13.png",
            REPORTS / "native_recip_slot_corrected_depth_hist_2026-06-13.png",
            REPORTS / "native_recip_slot_depth_hist_2026-06-12.png",
        ],
    )
    parser.add_argument(
        "--dashboard-out",
        type=Path,
        default=REPORTS / "native_recip_slot_original_corrected_floor25_curve.png",
    )
    parser.add_argument(
        "--dashboard-runs",
        type=Path,
        nargs="*",
        default=[
            REPORTS / "native_recip_slot_chin_2026-06-12.jsonl",
            REPORTS / "native_recip_slot_chin_corrected_2026-06-13.jsonl",
            REPORTS / "native_recip_slot_chin_floor25_2026-06-13.jsonl",
        ],
    )
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--every-steps", type=int, default=2000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--probe-batches", type=int, default=2)
    parser.add_argument("--probe-batch", type=int, default=1)
    parser.add_argument("--probe-seq-len", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--native-gate-floors",
        default=DEFAULT_NATIVE_GATE_FLOORS_CSV,
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-initial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dashboard_runs = [path for path in args.dashboard_runs if path.exists()]
    state = _load_state(args.state)

    while True:
        latest = _latest_step(args.run_jsonl)
        if latest is None:
            print(f"Waiting for step rows in {args.run_jsonl}", flush=True)
        else:
            active_run = str(args.run_jsonl.resolve())
            run_changed = state.get("active_run") != active_run
            bucket = latest // args.every_steps
            last_bucket = state.get("last_step_bucket")
            should_run = args.force or run_changed or (
                not args.skip_initial and last_bucket is None
            ) or (last_bucket is not None and bucket > int(last_bucket))
            if should_run:
                result = refresh(args, force_probe=args.force)
                state.update(
                    {
                        "active_run": active_run,
                        "last_step": result["latest_step"],
                        "last_step_bucket": result["step_bucket"],
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                )
                if result["probed_checkpoint"] is not None:
                    state["last_probe_checkpoint"] = result["probed_checkpoint"]
                _save_state(args.state, state)
                print(
                    f"Refreshed plots at step {result['latest_step']} "
                    f"(bucket {result['step_bucket']})",
                    flush=True,
                )
            else:
                print(
                    f"No refresh needed: step {latest}, bucket {bucket}, "
                    f"last bucket {last_bucket}, active run {args.run_jsonl.name}",
                    flush=True,
                )
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
