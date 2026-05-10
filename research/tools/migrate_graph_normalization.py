"""Graph-fingerprint normalization migration (additive, strangler-fig).

Adds two tables alongside the existing `program_results`:

  graphs       — one row per architecture identity (graph_fingerprint PK)
  graph_runs   — one row per observation, FK to graphs.graph_fingerprint

A view `program_results_compat` exposes the legacy column shape for read sites
during cutover. The original `program_results` table is left intact; subsequent
migration phases switch the write path to dual-write, then single-write to the
new tables, then drop the legacy table.

Plan: /home/tim/.claude/plans/zesty-wibbling-hippo.md
Source plan: tasks/graph_fingerprint_normalization.md

Usage:
    python -m research.tools.migrate_graph_normalization --dry-run \
        [--target /tmp/dryrun_lab_notebook.db]
    python -m research.tools.migrate_graph_normalization --apply \
        [--db research/lab_notebook.db]
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "research" / "lab_notebook.db"
WRITER_LOCK_SUFFIX = ".writer-lock"

LOGGER = logging.getLogger("migrate_graph_normalization")

# Per-architecture columns moved into `graphs`. Everything else stays per-run.
# Kept minimal: only what is structurally constant per fingerprint by definition
# (graph_json IS what gets fingerprinted; arch_spec_json is its sibling).
# novelty_*, param_count, graph_n_*, routing_mode are kept on graph_runs:
# they get recalibrated/recomputed and are per-run for historical drift.
GRAPH_COLUMNS_FROM_PROGRAM_RESULTS: Tuple[str, ...] = (
    "graph_json",
    "arch_spec_json",
)

LEADING_RUN_COLS = ["result_id", "experiment_id", "timestamp", "graph_fingerprint"]


def _writer_lock_active(db_path: Path) -> bool:
    """Return True iff a live process is holding the writer lock."""
    lock = db_path.with_name(db_path.name + WRITER_LOCK_SUFFIX)
    if not lock.exists():
        return False
    try:
        pid = int(lock.read_text().strip() or "0")
    except (OSError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False  # stale


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({_quote_ident(table)})")]


def _column_types(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    return {
        r[1]: r[2] for r in conn.execute(f"PRAGMA table_info({_quote_ident(table)})")
    }


def _create_graphs_table(cur: sqlite3.Cursor) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS graphs (
            graph_fingerprint TEXT PRIMARY KEY,
            graph_json        TEXT NOT NULL,
            arch_spec_json    TEXT,
            first_seen_ts     REAL NOT NULL,
            last_seen_ts      REAL NOT NULL,
            graph_json_is_placeholder INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        INSERT OR IGNORE INTO graphs
            (graph_fingerprint, graph_json, arch_spec_json,
             first_seen_ts, last_seen_ts, graph_json_is_placeholder)
        SELECT
            graph_fingerprint,
            COALESCE(
                (SELECT graph_json FROM program_results pr2
                 WHERE pr2.graph_fingerprint = pr.graph_fingerprint
                   AND pr2.graph_json IS NOT NULL
                   AND pr2.graph_json <> ''
                   AND pr2.graph_json <> '{}'
                 ORDER BY pr2.timestamp DESC LIMIT 1),
                '{}'
            ),
            (SELECT arch_spec_json FROM program_results pr3
             WHERE pr3.graph_fingerprint = pr.graph_fingerprint
               AND pr3.arch_spec_json IS NOT NULL
             ORDER BY pr3.timestamp DESC LIMIT 1),
            MIN(pr.timestamp),
            MAX(pr.timestamp),
            CASE WHEN MAX(
                CASE WHEN pr.graph_json IS NOT NULL
                          AND pr.graph_json <> ''
                          AND pr.graph_json <> '{}' THEN 1 ELSE 0 END
            ) = 1 THEN 0 ELSE 1 END
        FROM program_results pr
        WHERE TRIM(COALESCE(pr.graph_fingerprint, '')) <> ''
        GROUP BY pr.graph_fingerprint
        """
    )
    n_graphs = cur.execute("SELECT COUNT(*) FROM graphs").fetchone()[0]
    n_placeholder = cur.execute(
        "SELECT COUNT(*) FROM graphs WHERE graph_json_is_placeholder = 1"
    ).fetchone()[0]
    LOGGER.info("graphs: %d rows (%d placeholder)", n_graphs, n_placeholder)


def _build_run_col_definitions(
    run_cols: List[str], type_map: dict[str, str]
) -> List[str]:
    defs: List[str] = []
    for c in run_cols:
        ctype = type_map.get(c, "")
        if c == "result_id":
            constraints = " PRIMARY KEY"
        elif c == "graph_fingerprint":
            constraints = (
                " NOT NULL REFERENCES graphs(graph_fingerprint) ON DELETE CASCADE"
            )
        elif c == "timestamp":
            constraints = " NOT NULL"
        else:
            constraints = ""
        defs.append(f"{_quote_ident(c)} {ctype}{constraints}".strip())
    return defs


def _create_graph_runs_table(
    cur: sqlite3.Cursor, run_cols: List[str], type_map: dict[str, str]
) -> None:
    col_defs = _build_run_col_definitions(run_cols, type_map)
    cur.execute(f"CREATE TABLE IF NOT EXISTS graph_runs ({', '.join(col_defs)})")
    col_list = ", ".join(_quote_ident(c) for c in run_cols)
    cur.execute(
        f"INSERT OR IGNORE INTO graph_runs ({col_list}) "
        f"SELECT {col_list} FROM program_results "
        "WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''"
    )
    n_runs = cur.execute("SELECT COUNT(*) FROM graph_runs").fetchone()[0]
    n_legacy = cur.execute("SELECT COUNT(*) FROM program_results").fetchone()[0]
    LOGGER.info("graph_runs: %d rows (legacy: %d)", n_runs, n_legacy)


def _create_indexes(cur: sqlite3.Cursor) -> None:
    for stmt in (
        "CREATE INDEX IF NOT EXISTS idx_graph_runs_fp ON graph_runs(graph_fingerprint)",
        "CREATE INDEX IF NOT EXISTS idx_graph_runs_trust ON graph_runs(trust_label)",
        "CREATE INDEX IF NOT EXISTS idx_graph_runs_exp ON graph_runs(experiment_id)",
        "CREATE INDEX IF NOT EXISTS idx_graph_runs_ts ON graph_runs(timestamp)",
    ):
        cur.execute(stmt)


def _create_compat_view(cur: sqlite3.Cursor, legacy_cols: List[str]) -> None:
    arch_set = set(GRAPH_COLUMNS_FROM_PROGRAM_RESULTS)
    select_cols: List[str] = []
    for c in legacy_cols:
        prefix = "g" if c in arch_set else "r"
        select_cols.append(f"{prefix}.{_quote_ident(c)}")
    cur.execute(
        "CREATE VIEW IF NOT EXISTS program_results_compat AS\n"
        f"  SELECT {', '.join(select_cols)}\n"
        "  FROM graph_runs r\n"
        "  LEFT JOIN graphs g USING (graph_fingerprint)"
    )


def _check_fk_violations(cur: sqlite3.Cursor) -> None:
    """Check FK integrity on tables this migration created.

    Pre-existing FK violations in legacy tables (program_graph_pairs,
    program_graph_features, etc.) are out of scope and intentionally ignored —
    they exist on the live DB before migration.
    """
    cur.execute("PRAGMA foreign_keys = ON")
    new_violations: list = []
    for table in ("graphs", "graph_runs"):
        rows = cur.execute(
            f"PRAGMA foreign_key_check({_quote_ident(table)})"
        ).fetchall()
        new_violations.extend(rows)
    if new_violations:
        LOGGER.error("FK violations on new tables: %s", new_violations[:10])
        raise RuntimeError(f"FK check failed: {len(new_violations)} violations")


def run_migration(conn: sqlite3.Connection) -> None:
    """Execute the additive migration in a single transaction. Idempotent."""
    cur = conn.cursor()
    legacy_cols = _table_columns(conn, "program_results")
    if not legacy_cols:
        raise RuntimeError("program_results table not found")

    arch_set = set(GRAPH_COLUMNS_FROM_PROGRAM_RESULTS)
    run_cols = LEADING_RUN_COLS + [
        c for c in legacy_cols if c not in LEADING_RUN_COLS and c not in arch_set
    ]
    type_map = _column_types(conn, "program_results")
    LOGGER.info(
        "schema: legacy=%d arch=%d run=%d",
        len(legacy_cols),
        len(arch_set),
        len(run_cols),
    )

    cur.execute("PRAGMA foreign_keys = OFF")
    cur.execute("BEGIN")
    try:
        _create_graphs_table(cur)
        _create_graph_runs_table(cur, run_cols, type_map)
        _create_indexes(cur)
        _create_compat_view(cur, legacy_cols)
        cur.execute("ANALYZE graphs")
        cur.execute("ANALYZE graph_runs")
        cur.execute("COMMIT")
    except Exception:
        cur.execute("ROLLBACK")
        raise

    _check_fk_violations(cur)


def _sample_compat_parity(conn: sqlite3.Connection, n: int = 50) -> Tuple[int, int]:
    """Compare n random rows. Returns (compared, diffs)."""
    sample_ids = [
        r[0]
        for r in conn.execute(
            "SELECT result_id FROM program_results "
            "WHERE TRIM(COALESCE(graph_fingerprint,'')) <> '' "
            "ORDER BY RANDOM() LIMIT ?",
            (n,),
        ).fetchall()
    ]
    if not sample_ids:
        return 0, 0
    cmp_cols = (
        "result_id, graph_fingerprint, experiment_id, timestamp, "
        "graph_json, arch_spec_json, "
        "stage0_passed, stage1_passed, loss_ratio, "
        "wikitext_perplexity, hellaswag_acc, blimp_overall_accuracy, "
        "induction_screening_auc, binding_screening_auc, ar_legacy_auc, "
        "trust_label, comparability_label, model_source"
    )
    placeholders = ",".join(["?"] * len(sample_ids))
    legacy = {
        r[0]: tuple(r)
        for r in conn.execute(
            f"SELECT {cmp_cols} FROM program_results WHERE result_id IN ({placeholders})",
            sample_ids,
        )
    }
    compat = {
        r[0]: tuple(r)
        for r in conn.execute(
            f"SELECT {cmp_cols} FROM program_results_compat WHERE result_id IN ({placeholders})",
            sample_ids,
        )
    }
    diffs = 0
    for rid, lrow in legacy.items():
        crow = compat.get(rid)
        if crow != lrow:
            diffs += 1
            if diffs <= 3:
                LOGGER.error(
                    "parity diff rid=%s\n  legacy=%s\n  compat=%s", rid, lrow, crow
                )
    return len(legacy), diffs


def _gate_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "graphs": conn.execute("SELECT COUNT(*) FROM graphs").fetchone()[0],
        "graph_runs": conn.execute("SELECT COUNT(*) FROM graph_runs").fetchone()[0],
        "legacy": conn.execute("SELECT COUNT(*) FROM program_results").fetchone()[0],
        "legacy_with_fp": conn.execute(
            "SELECT COUNT(*) FROM program_results "
            "WHERE TRIM(COALESCE(graph_fingerprint,'')) <> ''"
        ).fetchone()[0],
        "unique_fp": conn.execute(
            "SELECT COUNT(DISTINCT graph_fingerprint) FROM program_results "
            "WHERE TRIM(COALESCE(graph_fingerprint,'')) <> ''"
        ).fetchone()[0],
        "compat": conn.execute(
            "SELECT COUNT(*) FROM program_results_compat"
        ).fetchone()[0],
    }


def acceptance_gate(conn: sqlite3.Connection) -> bool:
    """Return True iff all acceptance criteria pass."""
    c = _gate_counts(conn)
    LOGGER.info("--- acceptance gate ---")
    LOGGER.info(
        "legacy=%(legacy)d (with_fp=%(legacy_with_fp)d) unique_fp=%(unique_fp)d", c
    )
    LOGGER.info("graphs=%(graphs)d graph_runs=%(graph_runs)d compat=%(compat)d", c)

    ok = True
    if c["graphs"] != c["unique_fp"]:
        LOGGER.error("FAIL graphs(%d) != unique_fp(%d)", c["graphs"], c["unique_fp"])
        ok = False
    if c["graph_runs"] != c["legacy_with_fp"]:
        LOGGER.error(
            "FAIL graph_runs(%d) != legacy_with_fp(%d)",
            c["graph_runs"],
            c["legacy_with_fp"],
        )
        ok = False
    if c["compat"] != c["graph_runs"]:
        LOGGER.error("FAIL compat(%d) != graph_runs(%d)", c["compat"], c["graph_runs"])
        ok = False

    fk: list = []
    for table in ("graphs", "graph_runs"):
        fk.extend(
            conn.execute(f"PRAGMA foreign_key_check({_quote_ident(table)})").fetchall()
        )
    if fk:
        LOGGER.error("FAIL foreign_key_check (new tables): %d violations", len(fk))
        ok = False
    else:
        LOGGER.info("PASS foreign_key_check (new tables)")

    compared, diffs = _sample_compat_parity(conn, n=50)
    if diffs > 0:
        LOGGER.error("FAIL parity: %d/%d rows diverged", diffs, compared)
        ok = False
    else:
        LOGGER.info("PASS parity (%d rows byte-equal)", compared)
    return ok


def _copy_db_with_sidecars(src: Path, dst: Path) -> None:
    shutil.copy2(src, dst)
    for suffix in ("-wal", "-shm"):
        sidecar = src.with_name(src.name + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, dst.with_name(dst.name + suffix))


def _prepare_target(args: argparse.Namespace, src: Path) -> Path:
    if args.dry_run:
        target = Path(args.target).resolve()
        if target.exists():
            target.unlink()
        LOGGER.info("DRY-RUN: copying %s -> %s", src, target)
        _copy_db_with_sidecars(src, target)
        return target
    backup = (
        Path(args.backup)
        if args.backup
        else src.with_name(src.stem + ".pre_graphnorm" + src.suffix)
    )
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    LOGGER.info("APPLY: backing up %s -> %s", src, backup)
    _copy_db_with_sidecars(src, backup)
    return src


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--target", default="/tmp/dryrun_lab_notebook.db")
    p.add_argument("--backup", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    src = Path(args.db).resolve()
    if not src.exists():
        LOGGER.error("source DB not found: %s", src)
        return 2
    if _writer_lock_active(src):
        LOGGER.error("writer lock ACTIVE on %s — refuse to migrate", src)
        return 3

    try:
        db_path = _prepare_target(args, src)
    except FileExistsError as e:
        LOGGER.error("%s — refuse to overwrite", e)
        return 4

    t0 = time.time()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        run_migration(conn)
        gate_ok = acceptance_gate(conn)
    LOGGER.info("migration finished in %.1fs", time.time() - t0)

    if not gate_ok:
        LOGGER.error("acceptance gate FAILED")
        return 5
    LOGGER.info("acceptance gate PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
