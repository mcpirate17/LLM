"""Watch the corrected run, switch to 25% branch floor at the 60k checkpoint."""

from __future__ import annotations

import datetime as dt
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "research" / "reports"
CKPT_DIR = REPORTS / "native_adaptive_hydra_ckpts"
OLD_LABEL = "native_recip_slot_chin_corrected"
NEW_LABEL = "native_recip_slot_chin_floor25"
LANE = "native_adaptive_reciprocal_slot_delta"
TARGET_STEP = 60_000
TARGET_CKPT = (
    CKPT_DIR / f"{OLD_LABEL}_{LANE}_step{TARGET_STEP:06d}.pt"
)
NEW_JSONL = REPORTS / f"{NEW_LABEL}_2026-06-13.jsonl"
NEW_LOG = REPORTS / f"{NEW_LABEL}_2026-06-13.log"
WATCHDOG_LOG = REPORTS / "native_recip_slot_floor25_watchdog_2026-06-13.log"
PYTHON = Path("/home/tim/venvs/llm/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def _log(message: str) -> None:
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    line = f"{stamp} {message}"
    print(line, flush=True)
    with WATCHDOG_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _run(cmd: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def _pgrep(pattern: str) -> list[int]:
    result = _run(["pgrep", "-af", pattern])
    pids: list[int] = []
    self_pid = os.getpid()
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        command = parts[1] if len(parts) > 1 else ""
        if pid == self_pid or "native_recip_slot_floor25_watchdog.py" in command:
            continue
        pids.append(pid)
    return pids


def _terminate(pattern: str, *, name: str) -> None:
    pids = _pgrep(pattern)
    if not pids:
        _log(f"No {name} processes matched {pattern!r}")
        return
    _log(f"Stopping {name} PIDs: {pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 90
    while time.time() < deadline:
        if not any(_pid_alive(pid) for pid in pids):
            _log(f"{name} stopped cleanly")
            return
        time.sleep(3)
    live = [pid for pid in pids if _pid_alive(pid)]
    _log(f"Escalating {name} PIDs with SIGKILL: {live}")
    for pid in live:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_stable_checkpoint(path: Path) -> None:
    _log(f"Waiting for checkpoint: {path}")
    last_size = -1
    stable_checks = 0
    while True:
        if path.exists():
            size = path.stat().st_size
            if size > 0 and size == last_size:
                stable_checks += 1
            else:
                stable_checks = 0
                last_size = size
            if stable_checks >= 2:
                _log(f"Checkpoint stable: {path} ({size} bytes)")
                return
        time.sleep(60)


def _verify_floor() -> None:
    source = REPO / "research" / "tools" / "_scaling_lanes.py"
    text = source.read_text(encoding="utf-8")
    if "GATE_FLOOR = 0.25" not in text:
        raise RuntimeError("Expected GATE_FLOOR = 0.25 before restart")
    _log("Verified source floor is 0.25")


def _probe_checkpoint() -> None:
    out_json = REPORTS / "native_recip_slot_gate_probe_step060000_floor25_2026-06-13.json"
    out_plot = REPORTS / "native_recip_slot_gate_probe_step060000_floor25_2026-06-13.png"
    cmd = [
        str(PYTHON),
        "research/tools/native_recip_slot_gate_probe.py",
        "--checkpoint",
        str(TARGET_CKPT),
        "--out-json",
        str(out_json),
        "--out-plot",
        str(out_plot),
        "--device",
        "cuda",
        "--batch",
        "1",
        "--seq-len",
        "128",
        "--batches",
        "1",
    ]
    _log("Running 60k floor-25 gate probe")
    try:
        result = _run(cmd, timeout=900)
    except subprocess.TimeoutExpired:
        _log("Gate probe timed out; continuing to restart training")
        return
    _log(f"Gate probe exit={result.returncode}")
    if result.stdout:
        _log(result.stdout.strip()[-4000:])


def _start_training() -> None:
    cmd = [
        str(PYTHON),
        "research/tools/native_adaptive_hydra_train.py",
        "--lane",
        LANE,
        "--run-label",
        NEW_LABEL,
        "--dataset",
        "codex_ffw60_chat30_pleias10_local",
        "--val-dataset",
        "codex_ffw60_chat30_pleias10_local",
        "--dim",
        "640",
        "--n-blocks",
        "8",
        "--steps",
        "810000",
        "--batch",
        "4",
        "--seq-len",
        "512",
        "--optimizer",
        "muon",
        "--muon-lr",
        "0.02",
        "--lr",
        "3e-4",
        "--warmup-steps",
        "800",
        "--min-lr-frac",
        "0.1",
        "--device",
        "cuda",
        "--log-every",
        "25",
        "--eval-every",
        "200",
        "--eval-batches",
        "32",
        "--save-every",
        "10000",
        "--grad-spike-threshold",
        "10.0",
        "--max-recoveries",
        "5",
        "--load-checkpoint",
        str(TARGET_CKPT),
        "--out",
        str(NEW_JSONL),
        "--checkpoint-dir",
        str(CKPT_DIR),
    ]
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = env.get("TORCH_CUDA_ARCH_LIST", "12.0")
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    NEW_LOG.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Starting floor-25 run: {' '.join(cmd)}")
    with NEW_LOG.open("ab") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _log(f"Started floor-25 training PID {proc.pid}")


def _start_pruner() -> None:
    pattern = (
        f"{CKPT_DIR}/{NEW_LABEL}_{LANE}_step*.pt"
    )
    cmd = (
        f"while true; do ls -1t {pattern} 2>/dev/null | "
        "tail -n +4 | xargs -r rm --; sleep 600; done"
    )
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _log(f"Started floor-25 checkpoint pruner PID {proc.pid}")


def main() -> None:
    _log("watchdog starting")
    _verify_floor()
    _wait_for_stable_checkpoint(TARGET_CKPT)
    _terminate(
        "native_adaptive_hydra_train.py.*native_recip_slot_chin_corrected",
        name="corrected training",
    )
    _terminate(
        "native_recip_slot_chin_corrected_native_adaptive_reciprocal_slot_delta_step",
        name="corrected checkpoint pruner",
    )
    _probe_checkpoint()
    _start_training()
    _start_pruner()
    _log("watchdog finished")


if __name__ == "__main__":
    main()
