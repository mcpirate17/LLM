"""SQLite persistence layer for Aria Designer.

Uses WAL mode for concurrent reads. Tables: components, component_versions,
workflows, workflow_runs, aria_proposals.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).resolve().parent.parent / "aria_designer.db"
_local = threading.local()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS components (
    id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS component_versions (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    component_id TEXT NOT NULL,
    version TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (component_id) REFERENCES components(id)
);

CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    graph_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    author TEXT NOT NULL DEFAULT 'user',
    parent_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_runs (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL,
    run_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'pending',
    results_json TEXT,
    perf_json TEXT,
    started_at TEXT,
    completed_at TEXT,
    FOREIGN KEY (workflow_id) REFERENCES workflows(id)
);

CREATE TABLE IF NOT EXISTS aria_proposals (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    patch_json TEXT NOT NULL,
    rationale TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolved_by TEXT,
    FOREIGN KEY (workflow_id) REFERENCES workflows(id)
);

CREATE INDEX IF NOT EXISTS idx_components_category ON components(category);
CREATE INDEX IF NOT EXISTS idx_components_status ON components(status);
CREATE INDEX IF NOT EXISTS idx_workflows_author ON workflows(author);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON aria_proposals(status);
"""


def _get_conn() -> sqlite3.Connection:
    """Get thread-local database connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        logger.debug("Opening new connection to %s", _DB_PATH)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return _local.conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database schema."""
    global _DB_PATH
    if db_path is not None:
        _DB_PATH = db_path
    
    logger.debug("Initializing DB at %s", _DB_PATH)
    # Clear thread-local connection if it exists to ensure we use the new path
    if hasattr(_local, "conn") and _local.conn:
        logger.debug("Closing existing thread-local connection")
        _local.conn.close()
        _local.conn = None
        
    conn = _get_conn()
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    logger.debug("DB initialization complete")


# ── Component CRUD ────────────────────────────────────────────────────

def upsert_component(manifest: Dict[str, Any], created_at: str, updated_at: str) -> None:
    """Insert or update a component from its manifest dict."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO components (id, version, name, category, status, manifest_json, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             version=excluded.version, name=excluded.name, category=excluded.category,
             status=excluded.status, manifest_json=excluded.manifest_json, updated_at=excluded.updated_at""",
        (
            manifest["id"],
            manifest.get("version", "1.0.0"),
            manifest["name"],
            manifest["category"],
            manifest.get("status", "draft"),
            json.dumps(manifest),
            created_at,
            updated_at,
        ),
    )
    # Also record version history
    conn.execute(
        """INSERT INTO component_versions (component_id, version, manifest_json, created_at)
           VALUES (?, ?, ?, ?)""",
        (manifest["id"], manifest.get("version", "1.0.0"), json.dumps(manifest), created_at),
    )
    conn.commit()


def list_components(
    category: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List components, optionally filtered."""
    conn = _get_conn()
    sql = "SELECT manifest_json FROM components WHERE 1=1"
    params: list = []
    if category:
        sql += " AND category = ?"
        params.append(category)
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY category, name"
    rows = conn.execute(sql, params).fetchall()
    return [json.loads(row["manifest_json"]) for row in rows]


def get_component(component_id: str) -> Optional[Dict[str, Any]]:
    """Get a single component by ID."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT manifest_json FROM components WHERE id = ?", (component_id,)
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["manifest_json"])


def update_component_status(component_id: str, status: str, updated_at: str) -> bool:
    """Update component status (approve, deprecate, quarantine)."""
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE components SET status = ?, updated_at = ? WHERE id = ?",
        (status, updated_at, component_id),
    )
    conn.commit()
    return cur.rowcount > 0


def count_components() -> Dict[str, int]:
    """Count components by status."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM components GROUP BY status"
    ).fetchall()
    return {row["status"]: row["cnt"] for row in rows}


# ── Workflow CRUD ─────────────────────────────────────────────────────

def save_workflow(
    workflow_id: str, name: str, graph_json: str,
    author: str = "user", parent_id: Optional[str] = None,
    created_at: str = "", updated_at: str = "",
) -> int:
    """Save or update a workflow. Returns version number."""
    conn = _get_conn()
    existing = conn.execute(
        "SELECT version FROM workflows WHERE id = ?", (workflow_id,)
    ).fetchone()

    if existing:
        new_version = existing["version"] + 1
        conn.execute(
            """UPDATE workflows SET name=?, graph_json=?, version=?, updated_at=?
               WHERE id=?""",
            (name, graph_json, new_version, updated_at, workflow_id),
        )
    else:
        new_version = 1
        conn.execute(
            """INSERT INTO workflows (id, name, graph_json, version, author, parent_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (workflow_id, name, graph_json, new_version, author, parent_id, created_at, updated_at),
        )
    conn.commit()
    return new_version


def get_workflow(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Get a workflow by ID."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM workflows WHERE id = ?", (workflow_id,)
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def list_workflows() -> List[Dict[str, Any]]:
    """List all workflows."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, name, version, author, created_at, updated_at FROM workflows ORDER BY updated_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


# ── Aria Proposals ────────────────────────────────────────────────────

def save_proposal(
    proposal_id: str, workflow_id: str, patch_json: str,
    rationale: str, created_at: str,
) -> None:
    """Save an Aria patch proposal."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO aria_proposals (id, workflow_id, patch_json, rationale, status, created_at)
           VALUES (?, ?, ?, ?, 'pending', ?)""",
        (proposal_id, workflow_id, patch_json, rationale, created_at),
    )
    conn.commit()


def resolve_proposal(
    proposal_id: str, status: str, resolved_by: str, resolved_at: str,
) -> bool:
    """Approve or reject a proposal."""
    conn = _get_conn()
    cur = conn.execute(
        """UPDATE aria_proposals SET status=?, resolved_by=?, resolved_at=?
           WHERE id=? AND status='pending'""",
        (status, resolved_by, resolved_at, proposal_id),
    )
    conn.commit()
    return cur.rowcount > 0


def list_proposals(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """List proposals, optionally filtered by status."""
    conn = _get_conn()
    if status:
        rows = conn.execute(
            "SELECT * FROM aria_proposals WHERE status = ? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM aria_proposals ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_proposal(proposal_id: str) -> Optional[Dict[str, Any]]:
    """Get a single proposal."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM aria_proposals WHERE id = ?", (proposal_id,)
    ).fetchone()
    if row is None:
        return None
    return dict(row)
