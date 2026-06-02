#!/usr/bin/env python3
"""Live monitor for native_adaptive_hydra_train JSONL runs.

Usage:
    python3 research/tools/watch_native_run.py                 # latest run, one-shot
    python3 research/tools/watch_native_run.py --interval 30    # refresh every 30s
    python3 research/tools/watch_native_run.py --jsonl <path>   # a specific run
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=None, help="run JSONL path (default: latest)")
    ap.add_argument(
        "--interval", type=float, default=0.0, help="refresh seconds (0=once)"
    )
    args = ap.parse_args()
    path = args.jsonl or _latest_jsonl()
    print(f"# {path}")
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
