"""Watch floor-25 checkpoints and run CPU synthetic gate probes."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

from research.tools.native_gate_floor_utils import DEFAULT_NATIVE_GATE_FLOORS_CSV


REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "research" / "reports"
CKPT_DIR = REPORTS / "native_adaptive_hydra_ckpts"
LANE = "native_adaptive_reciprocal_slot_delta"
LABEL = "native_recip_slot_chin_floor25_gateaux_ckpt1k"
NATIVE_GATE_FLOORS = DEFAULT_NATIVE_GATE_FLOORS_CSV
LOG_PATH = REPORTS / "native_recip_slot_synthetic_gate_watch_2026-06-13.log"
LEDGER_PATH = REPORTS / "native_recip_slot_synthetic_gate_watch_gateaux_processed.json"
PYTHON = Path("/home/tim/venvs/llm/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def _log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _load_processed() -> set[str]:
    if not LEDGER_PATH.exists():
        return set()
    try:
        return set(json.loads(LEDGER_PATH.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return set()


def _save_processed(processed: set[str]) -> None:
    LEDGER_PATH.write_text(
        json.dumps(sorted(processed), indent=2),
        encoding="utf-8",
    )


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    marker = "_step"
    if marker not in stem:
        return -1
    return int(stem.rsplit(marker, 1)[1])


def _is_stable(path: Path) -> bool:
    if not path.exists():
        return False
    first = path.stat().st_size
    time.sleep(20)
    return path.exists() and path.stat().st_size == first and first > 0


def _probe(path: Path) -> bool:
    step = _checkpoint_step(path)
    out_base = REPORTS / f"native_recip_slot_synthetic_gate_probe_gateaux_step{step:06d}"
    cmd = [
        str(PYTHON),
        "-m",
        "research.tools.native_recip_slot_synthetic_gate_probe",
        str(path),
        "--device",
        "cpu",
        "--batch",
        "1",
        "--difficulties",
        "3",
        "--out-jsonl",
        str(out_base.with_suffix(".jsonl")),
        "--out-csv",
        str(out_base.with_suffix(".csv")),
        "--out-summary",
        str(out_base.with_name(out_base.name + "_summary.json")),
        "--native-gate-floors",
        NATIVE_GATE_FLOORS,
    ]
    _log(f"probing checkpoint step {step}: {path}")
    result = subprocess.run(
        cmd,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
        check=False,
    )
    _log(f"probe step {step} exit={result.returncode}")
    if result.stdout:
        _log(result.stdout.strip()[-4000:])
    return result.returncode == 0


def main() -> None:
    processed = _load_processed()
    _log("synthetic gate checkpoint watcher starting")
    pattern = f"{LABEL}_{LANE}_step*.pt"
    while True:
        checkpoints = sorted(CKPT_DIR.glob(pattern), key=_checkpoint_step)
        for checkpoint in checkpoints:
            key = str(checkpoint)
            if key in processed:
                continue
            if not _is_stable(checkpoint):
                continue
            if _probe(checkpoint):
                processed.add(key)
                _save_processed(processed)
        time.sleep(300)


if __name__ == "__main__":
    main()
