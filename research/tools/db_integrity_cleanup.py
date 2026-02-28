#!/usr/bin/env python3
"""Database integrity cleanup.

Removes orphaned records, deprecated-op models, stale metadata,
and recalculates experiment summary counts.

Usage:
    # Dry run (report only)
    python -m research.tools.db_integrity_cleanup --dry-run

    # Execute cleanup
    python -m research.tools.db_integrity_cleanup

    # Verbose (show each affected record)
    python -m research.tools.db_integrity_cleanup --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DB_PATH = os.environ.get(
    "LAB_NOTEBOOK_DB",
    os.path.join(os.path.dirname(__file__), "..", "lab_notebook.db"),
)

DEPRECATED_OPS = {"rfft_seq", "sort_seq", "argsort_seq", "token_pool_restore", "softmax_seq"}


def _graph_uses_deprecated_ops(graph_json: str) -> Tuple[bool, List[str]]:
    """Check if graph_json references any deprecated ops."""
    try:
        g = json.loads(graph_json)
    except (json.JSONDecodeError, TypeError):
        return False, []
    nodes = g.get("nodes", {})
    found = []
    for node in nodes.values():
        op = node.get("op_name", "")
        if op in DEPRECATED_OPS:
            found.append(op)
    return bool(found), found


def _graph_has_empty_nodes(graph_json: str) -> bool:
    """Check if graph_json has zero nodes."""
    try:
        g = json.loads(graph_json)
    except (json.JSONDecodeError, TypeError):
        return False
    return len(g.get("nodes", {})) == 0


def cleanup(db_path: str, dry_run: bool = False, verbose: bool = False) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    c = conn.cursor()

    stats = {}

    # ── Phase 1: Deprecated-op program_results ────────────────────────
    print("Phase 1: Removing program_results with deprecated ops...")

    rows = c.execute(
        "SELECT result_id, graph_json FROM program_results"
        " WHERE graph_json IS NOT NULL AND graph_json != ''"
    ).fetchall()

    dep_result_ids: List[str] = []
    for r in rows:
        has_dep, ops = _graph_uses_deprecated_ops(r["graph_json"])
        if has_dep:
            dep_result_ids.append(r["result_id"])
            if verbose:
                print(f"  {r['result_id']}: {', '.join(set(ops))}")

    # Also find the empty-nodes graph
    empty_ids: List[str] = []
    for r in rows:
        if _graph_has_empty_nodes(r["graph_json"]):
            empty_ids.append(r["result_id"])
            if verbose:
                print(f"  {r['result_id']}: empty nodes (corrupt)")

    all_bad_ids = list(set(dep_result_ids + empty_ids))

    if all_bad_ids:
        ph = ",".join("?" for _ in all_bad_ids)
        tc_del = c.execute(
            f"SELECT COUNT(*) FROM training_curves WHERE result_id IN ({ph})",
            all_bad_ids,
        ).fetchone()[0]
        print(f"  Found {len(dep_result_ids)} deprecated-op results, {len(empty_ids)} empty-node results")
        print(f"  Cascading: {tc_del} training_curves rows")

        if not dry_run:
            c.execute("BEGIN")
            c.execute(f"DELETE FROM training_curves WHERE result_id IN ({ph})", all_bad_ids)
            c.execute(f"DELETE FROM leaderboard WHERE result_id IN ({ph})", all_bad_ids)
            c.execute(f"DELETE FROM program_results WHERE result_id IN ({ph})", all_bad_ids)
            c.execute("COMMIT")
            print(f"  Deleted {len(all_bad_ids)} program_results + cascades")
        else:
            print(f"  [DRY RUN] Would delete {len(all_bad_ids)} program_results")
    else:
        print("  No deprecated-op or empty-node results found")

    stats["deprecated_op_results"] = len(dep_result_ids)
    stats["empty_node_results"] = len(empty_ids)

    # ── Phase 2: Orphaned leaderboard entries ─────────────────────────
    print("\nPhase 2: Removing orphaned leaderboard entries...")

    orphan_lb = c.execute(
        "SELECT entry_id, result_id, reference_name FROM leaderboard"
        " WHERE result_id NOT IN (SELECT result_id FROM program_results)"
    ).fetchall()

    if orphan_lb:
        for r in orphan_lb:
            print(f"  {r['entry_id']}: ref={r['reference_name']} (orphaned)")
        if not dry_run:
            ph = ",".join("?" for _ in orphan_lb)
            ids = [r["entry_id"] for r in orphan_lb]
            c.execute(f"DELETE FROM leaderboard WHERE entry_id IN ({ph})", ids)
            print(f"  Deleted {len(orphan_lb)} orphaned leaderboard entries")
        else:
            print(f"  [DRY RUN] Would delete {len(orphan_lb)} orphaned leaderboard entries")
    else:
        print("  No orphaned leaderboard entries")

    stats["orphaned_leaderboard"] = len(orphan_lb)

    # ── Phase 3: Orphaned training curves ─────────────────────────────
    print("\nPhase 3: Removing orphaned training curves...")

    orphan_tc = c.execute(
        "SELECT COUNT(*) FROM training_curves"
        " WHERE result_id NOT IN (SELECT result_id FROM program_results)"
    ).fetchone()[0]

    if orphan_tc:
        if not dry_run:
            c.execute(
                "DELETE FROM training_curves"
                " WHERE result_id NOT IN (SELECT result_id FROM program_results)"
            )
            print(f"  Deleted {orphan_tc} orphaned training curve rows")
        else:
            print(f"  [DRY RUN] Would delete {orphan_tc} orphaned training curve rows")
    else:
        print("  No orphaned training curves")

    stats["orphaned_training_curves"] = orphan_tc

    # ── Phase 4: Orphaned entries, hypotheses, preregistrations, healer_tasks
    print("\nPhase 4: Removing orphaned metadata records...")

    tables_with_experiment_fk = [
        ("entries", "experiment_id"),
        ("hypotheses", "experiment_id"),
        ("hypothesis_preregistrations", "experiment_id"),
        ("healer_tasks", "experiment_id"),
    ]

    for table, fk in tables_with_experiment_fk:
        orphan_count = c.execute(
            f"SELECT COUNT(*) FROM {table}"
            f" WHERE {fk} IS NOT NULL"
            f" AND {fk} NOT IN (SELECT experiment_id FROM experiments)"
        ).fetchone()[0]

        if orphan_count:
            if not dry_run:
                # For healer_tasks, also cascade to healer_task_events
                if table == "healer_tasks":
                    orphan_task_ids = [
                        r[0] for r in c.execute(
                            f"SELECT task_id FROM healer_tasks"
                            f" WHERE {fk} IS NOT NULL"
                            f" AND {fk} NOT IN (SELECT experiment_id FROM experiments)"
                        ).fetchall()
                    ]
                    if orphan_task_ids:
                        ph = ",".join("?" for _ in orphan_task_ids)
                        evt_del = c.execute(
                            f"SELECT COUNT(*) FROM healer_task_events WHERE task_id IN ({ph})",
                            orphan_task_ids,
                        ).fetchone()[0]
                        c.execute(
                            f"DELETE FROM healer_task_events WHERE task_id IN ({ph})",
                            orphan_task_ids,
                        )
                        print(f"  Cascaded: {evt_del} healer_task_events")

                c.execute(
                    f"DELETE FROM {table}"
                    f" WHERE {fk} IS NOT NULL"
                    f" AND {fk} NOT IN (SELECT experiment_id FROM experiments)"
                )
                print(f"  Deleted {orphan_count} orphaned {table} rows")
            else:
                print(f"  [DRY RUN] Would delete {orphan_count} orphaned {table} rows")
        else:
            print(f"  No orphaned {table}")

        stats[f"orphaned_{table}"] = orphan_count

    # ── Phase 5: Recalculate experiment counts ────────────────────────
    print("\nPhase 5: Recalculating experiment counts...")

    exps = c.execute(
        "SELECT experiment_id, n_programs_generated, n_stage0_passed,"
        " n_stage05_passed, n_stage1_passed FROM experiments"
    ).fetchall()

    fixed = 0
    for exp in exps:
        eid = exp["experiment_id"]
        actual_gen = c.execute(
            "SELECT COUNT(*) FROM program_results WHERE experiment_id = ?", (eid,)
        ).fetchone()[0]
        actual_s0 = c.execute(
            "SELECT COUNT(*) FROM program_results WHERE experiment_id = ? AND stage0_passed = 1",
            (eid,),
        ).fetchone()[0]
        actual_s05 = c.execute(
            "SELECT COUNT(*) FROM program_results WHERE experiment_id = ? AND stage05_passed = 1",
            (eid,),
        ).fetchone()[0]
        actual_s1 = c.execute(
            "SELECT COUNT(*) FROM program_results WHERE experiment_id = ? AND stage1_passed = 1",
            (eid,),
        ).fetchone()[0]

        needs_fix = (
            exp["n_programs_generated"] != actual_gen
            or exp["n_stage0_passed"] != actual_s0
            or (exp["n_stage05_passed"] or 0) != actual_s05
            or exp["n_stage1_passed"] != actual_s1
        )

        if needs_fix:
            if verbose:
                print(
                    f"  {eid}: gen {exp['n_programs_generated']}→{actual_gen},"
                    f" s0 {exp['n_stage0_passed']}→{actual_s0},"
                    f" s05 {exp['n_stage05_passed']}→{actual_s05},"
                    f" s1 {exp['n_stage1_passed']}→{actual_s1}"
                )
            if not dry_run:
                # Also recalculate best_loss_ratio
                best_lr = c.execute(
                    "SELECT MIN(loss_ratio) FROM program_results"
                    " WHERE experiment_id = ? AND loss_ratio IS NOT NULL",
                    (eid,),
                ).fetchone()[0]
                best_nov = c.execute(
                    "SELECT MAX(novelty_score) FROM program_results"
                    " WHERE experiment_id = ? AND novelty_score IS NOT NULL",
                    (eid,),
                ).fetchone()[0]
                c.execute(
                    "UPDATE experiments SET"
                    " n_programs_generated = ?,"
                    " n_stage0_passed = ?,"
                    " n_stage05_passed = ?,"
                    " n_stage1_passed = ?,"
                    " best_loss_ratio = ?,"
                    " best_novelty_score = ?"
                    " WHERE experiment_id = ?",
                    (actual_gen, actual_s0, actual_s05, actual_s1, best_lr, best_nov, eid),
                )
            fixed += 1

    if fixed:
        if not dry_run:
            conn.commit()
            print(f"  Fixed {fixed} experiment count mismatches")
        else:
            print(f"  [DRY RUN] Would fix {fixed} experiment count mismatches")
    else:
        print("  All experiment counts are correct")

    stats["experiment_count_fixes"] = fixed

    # ── Phase 6: Purge deprecated ops from op_success_rates ───────────
    print("\nPhase 6: Purging deprecated ops from op_success_rates...")

    dep_ops = c.execute(
        "SELECT op_name, n_used FROM op_success_rates WHERE op_name IN ({})".format(
            ",".join("?" for _ in DEPRECATED_OPS)
        ),
        list(DEPRECATED_OPS),
    ).fetchall()

    if dep_ops:
        for r in dep_ops:
            print(f"  {r['op_name']}: n_used={r['n_used']}")
        if not dry_run:
            c.execute(
                "DELETE FROM op_success_rates WHERE op_name IN ({})".format(
                    ",".join("?" for _ in DEPRECATED_OPS)
                ),
                list(DEPRECATED_OPS),
            )
            print(f"  Deleted {len(dep_ops)} deprecated op entries")
        else:
            print(f"  [DRY RUN] Would delete {len(dep_ops)} deprecated op entries")
    else:
        print("  No deprecated ops in op_success_rates")

    stats["deprecated_op_success_rates"] = len(dep_ops)

    # ── Phase 7: Remove empty experiments ─────────────────────────────
    print("\nPhase 7: Checking for empty experiments (0 program_results)...")

    empty_exps = c.execute(
        "SELECT e.experiment_id, e.n_programs_generated FROM experiments e"
        " WHERE NOT EXISTS ("
        "   SELECT 1 FROM program_results pr WHERE pr.experiment_id = e.experiment_id"
        ")"
    ).fetchall()

    if empty_exps:
        print(f"  Found {len(empty_exps)} experiments with 0 program_results")
        if verbose:
            for r in empty_exps[:10]:
                print(f"    {r['experiment_id']}: originally {r['n_programs_generated']} programs")
        if not dry_run:
            # Cascade: delete decisions, insights, entries linked to these experiments
            empty_eids = [r["experiment_id"] for r in empty_exps]
            ph = ",".join("?" for _ in empty_eids)

            # Delete linked preregistrations
            c.execute(
                f"DELETE FROM hypothesis_preregistrations WHERE experiment_id IN ({ph})",
                empty_eids,
            )
            # Delete linked hypotheses
            c.execute(
                f"DELETE FROM hypotheses WHERE experiment_id IN ({ph})",
                empty_eids,
            )
            # Delete linked entries
            c.execute(
                f"DELETE FROM entries WHERE experiment_id IN ({ph})",
                empty_eids,
            )
            # Delete linked insights
            c.execute(
                f"DELETE FROM insights WHERE experiment_id IN ({ph})",
                empty_eids,
            )
            # Delete healer_task_events for healer_tasks of these experiments
            task_ids = [
                r[0]
                for r in c.execute(
                    f"SELECT task_id FROM healer_tasks WHERE experiment_id IN ({ph})",
                    empty_eids,
                ).fetchall()
            ]
            if task_ids:
                tph = ",".join("?" for _ in task_ids)
                c.execute(f"DELETE FROM healer_task_events WHERE task_id IN ({tph})", task_ids)
                c.execute(f"DELETE FROM healer_tasks WHERE task_id IN ({tph})", task_ids)

            c.execute(f"DELETE FROM experiments WHERE experiment_id IN ({ph})", empty_eids)
            conn.commit()
            print(f"  Deleted {len(empty_exps)} empty experiments + cascaded metadata")
        else:
            print(f"  [DRY RUN] Would delete {len(empty_exps)} empty experiments")
    else:
        print("  No empty experiments")

    stats["empty_experiments"] = len(empty_exps)

    # ── Final summary ─────────────────────────────────────────────────
    conn.commit()

    remaining_pr = c.execute("SELECT COUNT(*) FROM program_results").fetchone()[0]
    remaining_lb = c.execute("SELECT COUNT(*) FROM leaderboard").fetchone()[0]
    remaining_tc = c.execute("SELECT COUNT(*) FROM training_curves").fetchone()[0]
    remaining_exp = c.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    remaining_entries = c.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    remaining_insights = c.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
    remaining_hyp = c.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]

    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    print(f"  experiments:       {remaining_exp}")
    print(f"  program_results:   {remaining_pr}")
    print(f"  leaderboard:       {remaining_lb}")
    print(f"  training_curves:   {remaining_tc}")
    print(f"  entries:           {remaining_entries}")
    print(f"  insights:          {remaining_insights}")
    print(f"  hypotheses:        {remaining_hyp}")

    if dry_run:
        print("\n[DRY RUN] No changes were written.")

    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Database integrity cleanup")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Report only, no deletes")
    parser.add_argument("--verbose", action="store_true", help="Show each affected record")
    args = parser.parse_args()

    print("=== Database Integrity Cleanup ===\n")
    cleanup(args.db, dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
