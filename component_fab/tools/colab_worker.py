"""Colab-first runner for component_fab workloads.

This is intended for remote/iPad operation where the user's local workstation is
not available. It runs from a cloned GitHub checkout inside Colab, mounts Drive,
streams logs, writes status JSON, and places reports under Drive.

Example from Colab after cloning the repo::

    python -m component_fab.tools.colab_worker --mode smoke
    python -m component_fab.tools.colab_worker --mode surrogate
    python -m component_fab.tools.colab_worker --mode deep_probe -- --top-k 12 --steps 3000 --seed-count 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_DRIVE_DIR = Path("/content/drive/MyDrive/Colab Notebooks/component_fab")


_MODE_COMMANDS: dict[str, list[str]] = {
    "smoke": [
        sys.executable,
        "-c",
        "import component_fab; "
        "from component_fab.state.ledger import Ledger; "
        "from pathlib import Path; "
        "p=Path('component_fab/catalog/colab_smoke_ledger.jsonl'); "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        "Ledger(p).close(); "
        "print('component_fab smoke ok')",
    ],
    "surrogate": [sys.executable, "-m", "component_fab.tools.run_surrogate"],
    "fidelity": [sys.executable, "-m", "component_fab.tools.run_fidelity"],
    "deep_probe": [sys.executable, "-m", "component_fab.tools.run_deep_probe"],
    "lm_probe": [sys.executable, "-m", "component_fab.tools.run_lm_probe"],
    "probe_bench": [sys.executable, "-m", "component_fab.tools.run_probe_bench"],
    "autonomous": [sys.executable, "-m", "component_fab.tools.run_autonomous"],
    "invention": [sys.executable, "-m", "component_fab.tools.run_invention"],
}


_DEFAULT_MODE_ARGS: dict[str, list[str]] = {
    "surrogate": ["--out", "{report_dir}/surrogate_report.json"],
    "fidelity": [
        "--store",
        "{report_dir}/fidelity_scores.jsonl",
        "--out",
        "{report_dir}/fidelity_report.json",
        "--max-candidates",
        "4",
    ],
    "deep_probe": [
        "--output",
        "{report_dir}/deep_probe_report.json",
        "--top-k",
        "8",
        "--steps",
        "2000",
        "--seed-count",
        "3",
        "--statuses",
        "promoted+pending",
    ],
    "probe_bench": ["--out", "{report_dir}/probe_costs.json"],
    "autonomous": [
        "--cycles",
        "1",
        "--max-graded-per-cycle",
        "8",
        "--paired-seeds",
        "3",
        "--emit-run-summary",
    ],
    "invention": ["--max-specs", "4", "--output", "{report_dir}/invention_report.json"],
}


_DEPENDENCIES = (
    "xxhash",
    "zstandard",
    "pyyaml",
    "flask-cors",
    "lightgbm",
    "ninja",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _mount_drive_if_available() -> None:
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return
    drive.mount("/content/drive")


def _install_dependencies(skip_install: bool) -> None:
    if skip_install:
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *_DEPENDENCIES])


def _format_args(raw: Iterable[str], *, report_dir: Path, ledger: Path) -> list[str]:
    return [
        str(part).format(report_dir=str(report_dir), ledger=str(ledger))
        for part in raw
    ]


def _build_command(args: argparse.Namespace, report_dir: Path, ledger: Path) -> list[str]:
    if args.mode not in _MODE_COMMANDS:
        raise ValueError(f"unknown mode: {args.mode}")
    cmd = list(_MODE_COMMANDS[args.mode])
    if args.extra_args:
        cmd.extend(_format_args(args.extra_args, report_dir=report_dir, ledger=ledger))
    else:
        cmd.extend(_format_args(_DEFAULT_MODE_ARGS.get(args.mode, ()), report_dir=report_dir, ledger=ledger))
    if args.mode in {"surrogate", "deep_probe", "lm_probe", "trust_audit"}:
        # These runners accept ledger-style args with different names. Preserve
        # runner-specific defaults unless the caller explicitly supplies one.
        joined = " ".join(cmd)
        if args.mode == "deep_probe" and "--ledger-path" not in joined:
            cmd.extend(["--ledger-path", str(ledger)])
        elif args.mode in {"surrogate"} and "--ledger" not in joined:
            cmd.extend(["--ledger", str(ledger)])
    return cmd


def _run_streamed(cmd: list[str], *, log_path: Path, status_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    latest = ""
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== component_fab colab start {_utc_now()} ===\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        _write_json(status_path, {"state": "running", "updated_at": _utc_now(), "cmd": cmd})
        proc = subprocess.Popen(
            cmd,
            cwd=str(_REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONPATH": str(_REPO)},
        )
        assert proc.stdout is not None
        last_status = time.monotonic()
        for line in proc.stdout:
            latest = line.rstrip()
            print(line, end="")
            log.write(line)
            log.flush()
            if time.monotonic() - last_status >= 5:
                _write_json(
                    status_path,
                    {
                        "state": "running",
                        "updated_at": _utc_now(),
                        "latest_line": latest[-500:],
                        "cmd": cmd,
                    },
                )
                last_status = time.monotonic()
        rc = proc.wait()
        state = "complete" if rc == 0 else "failed"
        _write_json(
            status_path,
            {
                "state": state,
                "updated_at": _utc_now(),
                "returncode": rc,
                "latest_line": latest[-500:],
                "cmd": cmd,
                "log": str(log_path),
            },
        )
        log.write(f"=== component_fab colab end {_utc_now()} rc={rc} ===\n")
        return rc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run component_fab workloads from Colab")
    parser.add_argument("--mode", choices=sorted(_MODE_COMMANDS), required=True)
    parser.add_argument("--drive-dir", type=Path, default=_DEFAULT_DRIVE_DIR)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the selected runner after --. Use {report_dir} and {ledger} placeholders.",
    )
    args = parser.parse_args(argv)
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _mount_drive_if_available()
    _install_dependencies(args.skip_install)

    drive_dir = args.drive_dir
    report_dir = drive_dir / "reports"
    log_dir = drive_dir / "logs"
    status_dir = drive_dir / "status"
    for path in (report_dir, log_dir, status_dir):
        path.mkdir(parents=True, exist_ok=True)

    ledger = args.ledger or (drive_dir / "ledger.jsonl")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_command(args, report_dir, ledger)
    log_path = log_dir / f"{args.mode}.log"
    status_path = status_dir / f"{args.mode}.json"
    _write_json(
        status_path,
        {
            "state": "setup",
            "updated_at": _utc_now(),
            "mode": args.mode,
            "repo": str(_REPO),
            "drive_dir": str(drive_dir),
            "ledger": str(ledger),
        },
    )
    return _run_streamed(cmd, log_path=log_path, status_path=status_path)


if __name__ == "__main__":
    raise SystemExit(main())
