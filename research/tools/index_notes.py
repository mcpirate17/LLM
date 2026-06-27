"""Index research notes + task docs into runs.db for durable, fast search.

Two complementary indexes, both queryable straight from `research/runs.db`:

  1. notes_fts   — SQLite FTS5 full-text index over every .md (prose search).
  2. note_tables — every markdown table extracted as structured rows
                   (headers_json + rows_json), so result matrices buried in
                   notes (e.g. cross_axis_architecture_matrix_2026-06-07.md)
                   stay queryable after the prose is forgotten.

FTS5 ships with Python's stdlib sqlite3 — nothing to install.

Sources: research/notes/**.md (source='notes'), tasks/**.md (source='tasks').
Idempotent full rebuild each run (a few hundred small files, sub-second).

Usage:
  python research/tools/index_notes.py                 # rebuild both indexes
  python research/tools/index_notes.py search "binding wall semiring"
  python research/tools/index_notes.py tables "cross_axis"   # list tables in matching notes
"""

from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(REPO, "research", "runs.db")

SOURCES = (
    ("notes", os.path.join(REPO, "research", "notes")),
    ("tasks", os.path.join(REPO, "tasks")),
)

_SEP_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")


def _clean_cell(cell: str) -> str:
    return cell.strip().strip("`").replace("**", "").strip()


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [_clean_cell(c) for c in s.split("|")]


def _is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def extract_tables(text: str) -> list[dict]:
    """Return list of {section, table_idx, headers, rows} markdown tables."""
    lines = text.splitlines()
    tables: list[dict] = []
    section = ""
    i = 0
    tidx = 0
    while i < len(lines):
        m = _HEADING_RE.match(lines[i])
        if m:
            section = m.group(2).strip()
            i += 1
            continue
        # A markdown table = header row, separator row, then >=0 data rows.
        if (
            _is_table_row(lines[i])
            and i + 1 < len(lines)
            and _SEP_RE.match(lines[i + 1])
            and "-" in lines[i + 1]
        ):
            headers = _split_row(lines[i])
            j = i + 2
            rows: list[list[str]] = []
            while j < len(lines) and _is_table_row(lines[j]):
                rows.append(_split_row(lines[j]))
                j += 1
            tables.append(
                {
                    "section": section,
                    "table_idx": tidx,
                    "headers": headers,
                    "rows": rows,
                }
            )
            tidx += 1
            i = j
            continue
        i += 1
    return tables


def _title_of(text: str, path: str) -> str:
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            return m.group(2).strip()
    return os.path.basename(path)


def _ddl(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
            path UNINDEXED, source UNINDEXED, title, body, mtime UNINDEXED
        );
        CREATE TABLE IF NOT EXISTS note_tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            path TEXT NOT NULL,
            note TEXT NOT NULL,
            table_idx INTEGER NOT NULL,
            section_heading TEXT,
            n_cols INTEGER,
            n_rows INTEGER,
            headers_json TEXT NOT NULL,
            rows_json TEXT NOT NULL,
            ingested_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_note_tables_note ON note_tables(note);
        """
    )


def rebuild(conn: sqlite3.Connection) -> tuple[int, int]:
    _ddl(conn)
    now = time.time()
    conn.execute("DELETE FROM notes_fts")
    conn.execute("DELETE FROM note_tables")
    n_files = 0
    n_tables = 0
    for source, root in SOURCES:
        for path in sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True)):
            with open(path, errors="ignore") as fh:
                text = fh.read()
            rel = os.path.relpath(path, REPO)
            note = os.path.basename(path)
            mtime = os.path.getmtime(path)
            conn.execute(
                "INSERT INTO notes_fts (path, source, title, body, mtime) VALUES (?,?,?,?,?)",
                (rel, source, _title_of(text, path), text, mtime),
            )
            n_files += 1
            for t in extract_tables(text):
                conn.execute(
                    """INSERT INTO note_tables
                       (source, path, note, table_idx, section_heading, n_cols,
                        n_rows, headers_json, rows_json, ingested_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        source,
                        rel,
                        note,
                        t["table_idx"],
                        t["section"],
                        len(t["headers"]),
                        len(t["rows"]),
                        json.dumps(t["headers"]),
                        json.dumps(t["rows"]),
                        now,
                    ),
                )
                n_tables += 1
    conn.commit()
    return n_files, n_tables


def cmd_search(conn: sqlite3.Connection, query: str) -> None:
    cur = conn.execute(
        """SELECT path, title, snippet(notes_fts, 3, '[', ']', ' … ', 12) AS snip
           FROM notes_fts WHERE notes_fts MATCH ? ORDER BY rank LIMIT 20""",
        (query,),
    )
    rows = cur.fetchall()
    if not rows:
        print(f"no matches for: {query}")
        return
    for path, title, snip in rows:
        print(f"\n# {title}\n  {path}\n  {snip}")


def cmd_tables(conn: sqlite3.Connection, note_like: str) -> None:
    cur = conn.execute(
        """SELECT note, table_idx, section_heading, n_cols, n_rows, headers_json
           FROM note_tables WHERE note LIKE ? ORDER BY note, table_idx""",
        (f"%{note_like}%",),
    )
    for note, idx, section, ncol, nrow, headers in cur.fetchall():
        hdr = ", ".join(json.loads(headers))
        print(f"{note} [#{idx}] ({nrow}x{ncol}) {section!r}\n    cols: {hdr}")


def main(argv: list[str]) -> None:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"runs.db not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        if len(argv) >= 2 and argv[1] == "search":
            cmd_search(conn, " ".join(argv[2:]))
        elif len(argv) >= 2 and argv[1] == "tables":
            cmd_tables(conn, " ".join(argv[2:]))
        else:
            n_files, n_tables = rebuild(conn)
            print(f"indexed {n_files} notes, {n_tables} tables into runs.db")
            print("  search: python research/tools/index_notes.py search '<query>'")
            print("  tables: python research/tools/index_notes.py tables '<note>'")
    finally:
        conn.close()


if __name__ == "__main__":
    main(sys.argv)
