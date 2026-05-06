from __future__ import annotations

"""Side-cache storage for derived knowledge digests.

Knowledge digests are derived narrative summaries. They should not live in the
source-of-truth notebook database, because cache corruption must not make
program_results, leaderboard, or dashboard reads unavailable.
"""

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_digests (
    digest_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    cycle_number INTEGER,
    digest_json TEXT NOT NULL,
    narrative_summary TEXT,
    n_experiments_analyzed INTEGER,
    n_curves_analyzed INTEGER
);

CREATE INDEX IF NOT EXISTS idx_knowledge_digests_ts
ON knowledge_digests(timestamp DESC);
"""


def default_cache_path(notebook_db_path: str | Path) -> Path:
    db_path = Path(notebook_db_path)
    if str(db_path) == ":memory:":
        return Path("research/cache/knowledge_digests.db").resolve()
    return db_path.resolve().parent / "cache" / "knowledge_digests.db"


class KnowledgeDigestStore:
    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path)

    def _connect(self) -> sqlite3.Connection:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.cache_path), timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.executescript(SCHEMA)
        return conn

    def ensure_schema(self) -> None:
        with self._connect():
            pass

    def _quarantine_corrupt_cache(self, exc: BaseException) -> None:
        if not self.cache_path.exists():
            return
        stamp = time.strftime("%Y%m%dT%H%M%S")
        target = self.cache_path.with_name(f"{self.cache_path.name}.corrupt_{stamp}")
        try:
            self.cache_path.replace(target)
            for suffix in ("-wal", "-shm"):
                sidecar = self.cache_path.with_name(f"{self.cache_path.name}{suffix}")
                if sidecar.exists():
                    sidecar.replace(target.with_name(f"{target.name}{suffix}"))
            LOGGER.warning(
                "Quarantined corrupt knowledge digest cache %s -> %s: %s",
                self.cache_path,
                target,
                exc,
            )
        except OSError as quarantine_exc:
            LOGGER.warning(
                "Failed to quarantine corrupt knowledge digest cache %s after %s: %s",
                self.cache_path,
                exc,
                quarantine_exc,
            )

    def store(self, digest_dict: Dict[str, Any]) -> str:
        digest_id = str(digest_dict.get("digest_id") or uuid.uuid4())
        ts = digest_dict.get("timestamp", time.time())
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO knowledge_digests
                       (digest_id, timestamp, cycle_number, digest_json,
                        narrative_summary, n_experiments_analyzed, n_curves_analyzed)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        digest_id,
                        ts,
                        digest_dict.get("cycle_number"),
                        json.dumps(digest_dict),
                        str(digest_dict.get("narrative") or "")[:2000],
                        digest_dict.get("n_experiments_analyzed"),
                        digest_dict.get("n_curves_analyzed"),
                    ),
                )
            return digest_id
        except sqlite3.DatabaseError as exc:
            self._quarantine_corrupt_cache(exc)
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO knowledge_digests
                       (digest_id, timestamp, cycle_number, digest_json,
                        narrative_summary, n_experiments_analyzed, n_curves_analyzed)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        digest_id,
                        ts,
                        digest_dict.get("cycle_number"),
                        json.dumps(digest_dict),
                        str(digest_dict.get("narrative") or "")[:2000],
                        digest_dict.get("n_experiments_analyzed"),
                        digest_dict.get("n_curves_analyzed"),
                    ),
                )
            return digest_id

    def get_latest(self) -> Optional[Dict[str, Any]]:
        if not self.cache_path.exists():
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT digest_json
                    FROM knowledge_digests
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """
                ).fetchone()
            if row and row[0]:
                return json.loads(row[0])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            LOGGER.warning("Failed to decode latest knowledge digest cache: %s", exc)
        except sqlite3.DatabaseError as exc:
            self._quarantine_corrupt_cache(exc)
        except OSError as exc:
            LOGGER.warning(
                "Failed to read knowledge digest cache %s: %s",
                self.cache_path,
                exc,
            )
        return None
