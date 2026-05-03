from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKUP_ROOTS = (
    PROJECT_ROOT / "research/db_backups",
    Path("/home/tim/GoogleDrive/Backups/LLM_Research"),
)


def _recent_backups(roots: list[Path], max_age_hours: float) -> list[Path]:
    cutoff = time.time() - max_age_hours * 3600.0
    recent: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.db"):
            try:
                if path.stat().st_mtime >= cutoff:
                    recent.append(path)
            except OSError:
                continue
    return sorted(recent, key=lambda p: p.stat().st_mtime, reverse=True)


def _notify(message: str) -> None:
    try:
        subprocess.run(
            ["notify-send", "LLM research backup audit", message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that research DB backups exist within a freshness window."
    )
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    parser.add_argument("--notify", action="store_true")
    parser.add_argument(
        "--backup-root",
        action="append",
        type=Path,
        dest="backup_roots",
        help="Backup root to inspect. May be passed more than once.",
    )
    args = parser.parse_args(argv)

    roots = args.backup_roots or list(DEFAULT_BACKUP_ROOTS)
    recent = _recent_backups(roots, args.max_age_hours)
    if recent:
        newest = recent[0]
        age_hours = (time.time() - newest.stat().st_mtime) / 3600.0
        print(f"Newest backup: {newest} ({age_hours:.2f}h old)")
        return 0

    roots_text = ", ".join(str(root) for root in roots)
    message = (
        f"No .db backup found in the last {args.max_age_hours:g}h under {roots_text}"
    )
    print(f"BLOCKED: {message}", file=sys.stderr)
    if args.notify:
        _notify(message)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
