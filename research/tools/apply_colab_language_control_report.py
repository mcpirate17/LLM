"""Apply a Colab NB05/NB10 language-control backfill report to the local DB.

The Colab tool (``colab_language_control_nb_backfill``) writes each scored row
as a ``{"result_id", "status", "updates", ...}`` line in a JSONL report.
Colab updates its own DB during the run; this tool replays the report against
the local DB so the scores end up wherever the report is unpacked.

Idempotent: each row becomes a single ``UPDATE graph_runs SET <cols> WHERE
result_id=?``; re-applying the same row is a no-op. Backs up the DB before
writing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable

from research.tools.language_control_backfill import _write_row

_DEFAULT_DB = Path("research/runs.db")
_DEFAULT_BACKUPS = Path("research/db_backups")


def _iter_updates(report: Path) -> Iterable[tuple[str, dict]]:
    """Yield (result_id, updates) for every row that carries a non-empty
    updates dict. Later occurrences of the same result_id override earlier
    ones (the report is append-only across Colab restarts)."""
    latest: Dict[str, dict] = {}
    statuses: Counter[str] = Counter()
    with report.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                statuses["json_error"] += 1
                continue
            statuses[str(rec.get("status", "missing"))] += 1
            rid = rec.get("result_id")
            upd = rec.get("updates")
            if not rid or not isinstance(upd, dict) or not upd:
                continue
            # Skip rows whose only update is the version stamp — they carry
            # no scores (Colab logs them when the row was already scored).
            score_cols = [k for k in upd if k.endswith("_score") or k.endswith("_acc")]
            if not score_cols:
                continue
            latest[rid] = upd
    sys.stderr.write(f"  report statuses: {dict(statuses)}\n")
    for rid, upd in latest.items():
        yield rid, upd


def _backup_db(db_path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = backup_dir / f"{db_path.stem}_pre_nb05_apply_{ts}.db"
    shutil.copy2(db_path, dst)
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=_DEFAULT_DB)
    ap.add_argument("--backup-dir", type=Path, default=_DEFAULT_BACKUPS)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="parse the report and count applicable rows without writing",
    )
    args = ap.parse_args(argv)

    if not args.report.exists():
        sys.exit(f"report not found: {args.report}")
    if not args.db.exists():
        sys.exit(f"db not found: {args.db}")

    pending = list(_iter_updates(args.report))
    print(f"applicable rows (deduped by result_id): {len(pending)}")
    if args.dry_run or not pending:
        return 0

    backup = _backup_db(args.db, args.backup_dir)
    print(f"backed up DB to {backup}")

    con = sqlite3.connect(str(args.db))
    try:
        n_applied = 0
        for rid, upd in pending:
            n_applied += _write_row(con, rid, upd)
        con.commit()
    finally:
        con.close()
    print(f"applied {n_applied} updates to {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
