#!/usr/bin/env python3
"""Backfill fingerprint-level leaderboard aggregates across repeated runs."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scientist.notebook import LabNotebook


def main() -> None:
    default_db = _root / "lab_notebook.db"
    db_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_db
    nb = LabNotebook(str(db_path))
    try:
        synced = nb.backfill_fingerprint_aggregates()
        nb.conn.commit()
        print(
            f"Synchronized fingerprint aggregates for {synced} fingerprints in {db_path}."
        )
    finally:
        nb.close()


if __name__ == "__main__":
    main()
