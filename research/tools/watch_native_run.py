#!/usr/bin/env python3
"""Live monitor for native_adaptive_hydra_train JSONL runs.

Usage:
    python3 research/tools/watch_native_run.py                 # latest run, one-shot
    python3 research/tools/watch_native_run.py --interval 30    # refresh every 30s
    python3 research/tools/watch_native_run.py --jsonl <path>   # a specific run
    python3 research/tools/watch_native_run.py --follow         # stream every step
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

REPORTS = Path("research/reports")


def _latest_jsonl() -> str:
    cands = sorted(
        glob.glob(str(REPORTS / "native_adaptive_*.jsonl"))
        + glob.glob(str(REPORTS / "mor_*.jsonl")),
        key=lambda p: Path(p).stat().st_mtime,
    )
    if not cands:
        raise SystemExit("no native_adaptive_*/mor_*.jsonl found in research/reports/")
    return cands[-1]


def _last_record(path: str) -> dict | None:
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    return json.loads(last) if last else None


def _meta(path: str) -> dict:
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("event") == "start":
                return r
    return {}


def _fmt(path: str) -> str:
    r = _last_record(path)
    if r is None:
        return "(no records yet)"
    meta = _meta(path)
    total = meta.get("steps", "?")
    step = r.get("step") or r.get("first_step") or 0
    loss = r.get("loss")
    lr = r.get("lr")
    elapsed = r.get("elapsed_sec")
    d = r.get("depth") or {}
    frac = (d.get("histogram_fraction") or {}).get("4")
    eta = ""
    first = meta.get("first_step", 1) or 1
    done = (step - first + 1) if isinstance(step, int) else 0
    if elapsed and done > 0 and isinstance(total, int):
        per = elapsed / done  # per-step over THIS session (resume-aware)
        rem = (total - step) * per / 3600.0
        eta = f"  eta~{rem:.1f}h"
    parts = [
        f"step {step}/{total}",
        f"loss {loss:.4f}" if isinstance(loss, (int, float)) else "loss -",
        f"lr {lr:.2e}" if isinstance(lr, (int, float)) else "",
        f"mean_depth {d.get('mean_depth')}" if d else "",
        f"depth4 {frac:.3f}" if isinstance(frac, (int, float)) else "",
        eta,
    ]
    return "  ".join(p for p in parts if p)


def _f(x: object, nd: int = 4) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def _fmt_step(r: dict) -> str | None:
    """One live line for a single step row: loss, ppl, grad, MoR mean depth, depth-4 %.

    ``ppl`` only appears on eval rows (every ``--eval-every`` steps) and renders
    as ``-`` on log-only rows. ``mor4%`` is the fraction of tokens the router sent
    to depth 4 (``depth.histogram_fraction["4"]``), shown as a percentage.
    """
    if r.get("event") != "step":
        return None
    d = r.get("depth") or {}
    m4 = (d.get("histogram_fraction") or {}).get("4")
    ppl = (r.get("eval") or {}).get("ppl")
    avg = d.get("mean_depth")
    return (
        f"step {str(r.get('step')):>7}"
        f" | loss {_f(r.get('loss'))}"
        f" | ppl {('-' if ppl is None else f'{ppl:g}'):>8}"
        f" | grad {_f(r.get('grad_norm'))}"
        f" | mor_avg {'-' if avg is None else avg}"
        f" | mor4% {'-' if m4 is None else f'{m4 * 100:.1f}'}"
    )


# I/O idle-wait between checks for the next appended line. NOT a display refresh
# timer: output is strictly one line per step row the trainer writes (cadence =
# the trainer's --log-every), this only governs how promptly a new row is picked up.
_IDLE_WAIT_SEC = 0.25


def _follow(path: str, *, from_start: bool):
    """Yield JSONL records as they are appended (tail -f), waiting for the file."""
    while not Path(path).exists():
        time.sleep(_IDLE_WAIT_SEC)
    with open(path) as f:
        if not from_start:
            f.seek(0, 2)  # jump to EOF; only stream rows written from now on
        while True:
            line = f.readline()
            if not line:
                time.sleep(_IDLE_WAIT_SEC)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=None, help="run JSONL path (default: latest)")
    ap.add_argument(
        "--interval", type=float, default=0.0, help="refresh seconds (0=once)"
    )
    ap.add_argument(
        "--follow",
        action="store_true",
        help="stream every step row live (tail -f): loss/ppl/grad/mor_avg/mor4%%",
    )
    ap.add_argument(
        "--from-start",
        action="store_true",
        help="with --follow, replay from the first row instead of the tail",
    )
    args = ap.parse_args()
    path = args.jsonl or _latest_jsonl()
    print(f"# {path}")
    if args.follow:
        try:
            for r in _follow(path, from_start=args.from_start):
                line = _fmt_step(r)
                if line:
                    print(line, flush=True)
        except KeyboardInterrupt:
            pass
        return
    if args.interval <= 0:
        print(_fmt(path))
        return
    try:
        while True:
            print(_fmt(path), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
