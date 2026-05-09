"""Restore ``research/lab_notebook.db`` from a healthy snapshot or backup.

The restore is intentionally conservative:

* dry-run by default
* refuses to run while another writer holds ``<db>.writer-lock``
* verifies the source with ``PRAGMA quick_check``
* moves the current DB and sidecars aside instead of deleting them
* verifies the restored DB before reporting success
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from research.tools._db_maintenance import check_writer_lock
from research.tools.db_health import assert_sqlite_health


DEFAULT_DB = Path("research/lab_notebook.db")


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def _move_aside(path: Path, *, suffix: str) -> Path | None:
    if not path.exists():
        return None
    target = path.with_name(f"{path.name}.{suffix}")
    path.replace(target)
    return target


def restore_lab_notebook(
    *,
    source: Path,
    db_path: Path = DEFAULT_DB,
    apply: bool = False,
) -> dict[str, str]:
    source = source.resolve()
    db_path = db_path.resolve()
    if source == db_path:
        raise ValueError("restore source must be different from destination DB")
    if not source.is_file():
        raise FileNotFoundError(source)

    assert_sqlite_health(source, label="restore source")
    check_writer_lock(Path(f"{db_path}.writer-lock"))

    suffix = f"corrupt_{_timestamp()}"
    plan = {
        "source": str(source),
        "destination": str(db_path),
        "moved_current_db": str(db_path.with_name(f"{db_path.name}.{suffix}")),
    }
    if not apply:
        return plan

    db_path.parent.mkdir(parents=True, exist_ok=True)
    moved = _move_aside(db_path, suffix=suffix)
    if moved is not None:
        plan["moved_current_db"] = str(moved)

    for sidecar_suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(f"{db_path}{sidecar_suffix}")
        moved_sidecar = _move_aside(sidecar, suffix=suffix)
        if moved_sidecar is not None:
            plan[f"moved{sidecar_suffix}"] = str(moved_sidecar)

    shutil.copy2(source, db_path)
    assert_sqlite_health(db_path, label="restored notebook")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="healthy snapshot/backup DB to restore",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--apply", action="store_true", help="perform the restore")
    args = parser.parse_args(argv)

    plan = restore_lab_notebook(
        source=args.source,
        db_path=args.db,
        apply=args.apply,
    )
    action = "restored" if args.apply else "dry-run"
    print(f"{action}:")
    for key, value in plan.items():
        print(f"  {key}: {value}")
    if not args.apply:
        print("Re-run with --apply after stopping the writer process.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
