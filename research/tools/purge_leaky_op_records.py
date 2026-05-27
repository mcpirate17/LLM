#!/usr/bin/env python3
"""Purge all DB records whose computation graph instantiates a given op as a NODE.

Built 2026-05-23 to remove `gated_linear_attention` records after it was proven
anti-causal (leaks the next token → inflates binding_range + wikitext_ppl). The
op is NOT being fixed; its records are being deleted.

Matching is EXACT node-level (parse graph_json.nodes[].op_name) — NOT a
`LIKE '%op%'` substring, which false-positives on metadata/config/registry
mentions (4363 substring vs 48 real node-hits for gated_linear_attention).

Cascade model (verified against schema):
  - Canonical graph_runs rows are deleted directly. The legacy
    program_results mirror is intentionally left for the Phase 5b Stage 3
    drop instead of adding new writes to the retired table surface.
  - leaderboard has NO cascade -> deleted explicitly by result_id.
  - graph_runs rows with no legacy mirror are covered by the same canonical
    delete.
  - now-orphan graphs rows (no remaining graph_runs for that fingerprint) -> deleted.
  - meta_analysis.db op-keyed catalogs -> GLA rows deleted; pair/triplet/template
    aggregates that included it are flagged stale (need a meta rebuild).

Dry-run by default. Pass --execute to commit. Take a DB backup first.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS_DB = REPO / "runs.db"
META_DB = REPO / "meta_analysis.db"


def _graph_has_op_node(graph_json: str, op: str) -> bool:
    try:
        g = json.loads(graph_json)
        nodes = g["nodes"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return False
    seq = nodes.values() if isinstance(nodes, dict) else nodes
    return any((n.get("op_name") or n.get("op")) == op for n in seq)


def find_target_result_ids(
    con: sqlite3.Connection, op: str
) -> tuple[set[str], set[str]]:
    """Return (result_ids, fingerprints) whose graph instantiates `op` as a node."""
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT result_id, graph_fingerprint AS fp, graph_json "
        "FROM program_results_compat WHERE graph_json LIKE ?",
        (f"%{op}%",),
    ).fetchall()
    rids: set[str] = set()
    fps: set[str] = set()
    for r in rows:
        if r["graph_json"] and _graph_has_op_node(r["graph_json"], op):
            rids.add(r["result_id"])
            fps.add(r["fp"])
    return rids, fps


def _count(con: sqlite3.Connection, table: str, col: str, ids: set[str]) -> int:
    if not ids:
        return 0
    ph = ",".join("?" * len(ids))
    return con.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {col} IN ({ph})", tuple(ids)
    ).fetchone()[0]


def purge_runs_db(op: str, execute: bool) -> dict:
    con = sqlite3.connect(RUNS_DB)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    rids, fps = find_target_result_ids(con, op)
    report = {
        "op": op,
        "target_result_ids": len(rids),
        "target_fingerprints": len(fps),
        "pre": {
            "program_results": _count(con, "program_results", "result_id", rids),
            "graph_runs": _count(con, "graph_runs", "result_id", rids),
            "leaderboard": _count(con, "leaderboard", "result_id", rids),
            "training_curves": _count(con, "training_curves", "result_id", rids),
            "program_graph_ops": _count(con, "program_graph_ops", "result_id", rids),
            "program_graph_features": _count(
                con, "program_graph_features", "result_id", rids
            ),
            "program_graph_pairs": _count(
                con, "program_graph_pairs", "result_id", rids
            ),
        },
    }
    # leaderboard tier breakdown for the targets
    if rids:
        ph = ",".join("?" * len(rids))
        tiers = con.execute(
            f"SELECT tier, COUNT(*) c FROM leaderboard WHERE result_id IN ({ph}) GROUP BY tier",
            tuple(rids),
        ).fetchall()
        report["leaderboard_tiers"] = {r["tier"]: r["c"] for r in tiers}

    if not execute or not rids:
        con.close()
        return report

    ph = ",".join("?" * len(rids))
    t = tuple(rids)
    cur = con.cursor()
    cur.execute("BEGIN")
    # 1) leaderboard (no cascade)
    cur.execute(f"DELETE FROM leaderboard WHERE result_id IN ({ph})", t)
    # 2) graph-feature tables — delete EXPLICITLY by the target result_ids. FK
    #    cascade only fires for rows that have a program_results parent; the
    #    graph_runs-only records (no program_results mirror) would otherwise
    #    orphan their feature rows. Targeting the exact result_ids (never a
    #    broad "orphan" sweep) keeps the blast radius to these records only.
    for ft in (
        "program_graph_ops",
        "program_graph_features",
        "program_graph_pairs",
        "training_curves",
    ):
        cur.execute(f"DELETE FROM {ft} WHERE result_id IN ({ph})", t)
    # 3) graph_runs is the canonical result table post-Phase-5b.
    cur.execute(f"DELETE FROM graph_runs WHERE result_id IN ({ph})", t)
    # 4) now-orphan graphs rows (fingerprints with no remaining graph_runs)
    if fps:
        fph = ",".join("?" * len(fps))
        cur.execute(
            f"DELETE FROM graphs WHERE graph_fingerprint IN ({fph}) "
            f"AND graph_fingerprint NOT IN (SELECT graph_fingerprint FROM graph_runs)",
            tuple(fps),
        )
    con.commit()

    report["post"] = {
        "program_results": _count(con, "program_results", "result_id", rids),
        "graph_runs": _count(con, "graph_runs", "result_id", rids),
        "leaderboard": _count(con, "leaderboard", "result_id", rids),
        "training_curves": _count(con, "training_curves", "result_id", rids),
        "program_graph_ops": _count(con, "program_graph_ops", "result_id", rids),
        "program_graph_features": _count(
            con, "program_graph_features", "result_id", rids
        ),
        "program_graph_pairs": _count(con, "program_graph_pairs", "result_id", rids),
    }
    con.close()
    return report


def purge_meta_db(op: str, execute: bool) -> dict:
    if not META_DB.exists():
        return {"meta_analysis": "absent"}
    con = sqlite3.connect(META_DB)
    con.row_factory = sqlite3.Row
    tables = {
        r["name"]
        for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    op_keyed = []  # (table, op_col)
    for t in tables:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
        for c in ("op_name", "op"):
            if c in cols:
                op_keyed.append((t, c))
                break
    report = {"op_keyed_tables": {}}
    for t, c in op_keyed:
        n = con.execute(f"SELECT COUNT(*) FROM {t} WHERE {c}=?", (op,)).fetchone()[0]
        if n:
            report["op_keyed_tables"][t] = n
    if execute:
        cur = con.cursor()
        cur.execute("BEGIN")
        for t, c in op_keyed:
            cur.execute(f"DELETE FROM {t} WHERE {c}=?", (op,))
        con.commit()
        report["deleted"] = True
    con.close()
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="gated_linear_attention")
    ap.add_argument(
        "--execute", action="store_true", help="commit deletes (default dry-run)"
    )
    ap.add_argument("--skip-meta", action="store_true")
    args = ap.parse_args()

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"=== purge_leaky_op_records: op={args.op!r} mode={mode} ===")
    runs = purge_runs_db(args.op, args.execute)
    print(json.dumps(runs, indent=2))
    if not args.skip_meta:
        meta = purge_meta_db(args.op, args.execute)
        print("meta_analysis:", json.dumps(meta, indent=2))
    if not args.execute:
        print("\n(dry-run — nothing deleted. Re-run with --execute to commit.)")


if __name__ == "__main__":
    main()
