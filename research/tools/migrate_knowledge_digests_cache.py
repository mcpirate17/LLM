from __future__ import annotations

"""Best-effort migration of legacy knowledge digests to the side cache."""

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict

from research.scientist.notebook.knowledge_digest_store import (
    KnowledgeDigestStore,
    default_cache_path,
)

LOGGER = logging.getLogger(__name__)


def migrate(db_path: Path, cache_path: Path | None = None) -> Dict[str, Any]:
    target = cache_path or default_cache_path(db_path)
    store = KnowledgeDigestStore(target)
    store.ensure_schema()
    copied = 0
    failed = 0
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            table_exists = conn.execute(
                """
                SELECT 1
                FROM sqlite_schema
                WHERE type = 'table' AND name = 'knowledge_digests'
                LIMIT 1
                """
            ).fetchone()
            if table_exists is None:
                return {
                    "status": "ok",
                    "source": str(db_path),
                    "cache": str(target),
                    "copied": 0,
                    "failed": 0,
                    "legacy_table": False,
                }
            rows = conn.execute(
                """
                SELECT digest_json
                FROM knowledge_digests
                ORDER BY timestamp ASC
                """
            )
            for row in rows:
                try:
                    payload = json.loads(row["digest_json"])
                    store.store(payload)
                    copied += 1
                except Exception as exc:
                    failed += 1
                    LOGGER.warning("Skipping malformed knowledge digest row: %s", exc)
    except (sqlite3.DatabaseError, OSError) as exc:
        LOGGER.warning(
            "Legacy knowledge_digests migration skipped; source is unreadable: %s",
            exc,
        )
        return {
            "status": "source_unreadable",
            "source": str(db_path),
            "cache": str(target),
            "copied": copied,
            "failed": failed,
            "error": str(exc),
        }
    return {
        "status": "ok",
        "source": str(db_path),
        "cache": str(target),
        "copied": copied,
        "failed": failed,
        "legacy_table": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("research/lab_notebook.db"))
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = migrate(args.db, args.cache)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
