#!/usr/bin/env python3
"""Aria health monitor — watches for degenerate experiment patterns.

Run:  python -m research.tools.aria_monitor [--interval 30]

Checks:
  1. Consecutive 0-S1 experiments (alert at 3+)
  2. Repeated investigation of same fingerprints
  3. Config diversity (are configs actually varying?)
  4. S0 pass rate collapse
  5. Stale leaderboard (investigation-ready but already investigated)
"""

import sqlite3
import json
import time
import argparse
from datetime import datetime

DB_PATH = "research/lab_notebook.db"

RED = "\033[91m"
YEL = "\033[93m"
GRN = "\033[92m"
RST = "\033[0m"
BOLD = "\033[1m"


def alert(msg):
    print(f"{RED}{BOLD}[ALERT]{RST} {RED}{msg}{RST}")


def warn(msg):
    print(f"{YEL}[WARN]{RST}  {msg}")


def ok(msg):
    print(f"{GRN}[OK]{RST}    {msg}")


def info(msg):
    print(f"[INFO]  {msg}")


def check_consecutive_failures(db):
    """Alert if recent experiments have 0 S1 survivors in a row."""
    rows = db.execute(
        "SELECT experiment_id, experiment_type, n_stage1_passed "
        "FROM experiments ORDER BY timestamp DESC LIMIT 10"
    ).fetchall()
    streak = 0
    for r in rows:
        if r[2] == 0:
            streak += 1
        else:
            break
    if streak >= 5:
        alert(f"{streak} consecutive experiments with 0 S1 survivors!")
    elif streak >= 3:
        warn(f"{streak} consecutive experiments with 0 S1 survivors")
    else:
        ok(f"S1 streak OK (last failure streak: {streak})")
    return streak


def check_investigation_loop(db):
    """Alert if same fingerprints are being investigated repeatedly."""
    rows = db.execute(
        "SELECT pr.graph_fingerprint, COUNT(*) as cnt "
        "FROM program_results pr "
        "JOIN experiments e ON e.experiment_id = pr.experiment_id "
        "WHERE e.experiment_type = 'investigation' "
        "GROUP BY pr.graph_fingerprint "
        "HAVING cnt > 3 "
        "ORDER BY cnt DESC"
    ).fetchall()
    if rows:
        for r in rows:
            alert(f"Fingerprint {r[0][:10]} investigated {r[1]}x!")
    else:
        ok("No investigation loops detected")
    return len(rows)


def check_config_diversity(db):
    """Check if recent experiment configs are varied."""
    rows = db.execute(
        "SELECT config_json FROM experiments ORDER BY timestamp DESC LIMIT 5"
    ).fetchall()
    if len(rows) < 2:
        info("Not enough experiments to check config diversity")
        return

    configs = []
    for r in rows:
        cfg = json.loads(r[0]) if r[0] else {}
        key = (
            cfg.get("max_depth"),
            cfg.get("max_ops"),
            cfg.get("residual_prob"),
            cfg.get("model_source"),
            cfg.get("n_programs"),
        )
        configs.append(key)

    unique = len(set(configs))
    if unique <= 1:
        alert(f"Last {len(configs)} experiments have IDENTICAL configs!")
    elif unique <= 2:
        warn(f"Low config diversity: {unique} unique configs in last {len(configs)}")
    else:
        ok(f"Config diversity OK: {unique} unique configs in last {len(configs)}")


def check_s0_pass_rate(db):
    """Alert if S0 pass rate is critically low."""
    row = db.execute(
        "SELECT COUNT(*), "
        "SUM(CASE WHEN stage0_passed=1 THEN 1 ELSE 0 END) "
        "FROM program_results pr "
        "JOIN experiments e ON e.experiment_id = pr.experiment_id "
        "WHERE e.experiment_id IN ("
        "  SELECT experiment_id FROM experiments ORDER BY timestamp DESC LIMIT 5"
        ")"
    ).fetchone()
    total, s0 = row[0], row[1] or 0
    if total == 0:
        info("No recent programs to check S0 rate")
        return
    rate = s0 / total
    if rate < 0.05:
        alert(
            f"S0 pass rate is {rate:.1%} ({s0}/{total}) — grammar is generating broken architectures"
        )
    elif rate < 0.15:
        warn(f"S0 pass rate is low: {rate:.1%} ({s0}/{total})")
    else:
        ok(f"S0 pass rate: {rate:.1%} ({s0}/{total})")


def check_stale_leaderboard(db):
    """Check for leaderboard entries stuck in screening that were already investigated."""
    investigated_fps = set()
    try:
        rows = db.execute(
            "SELECT DISTINCT pr.graph_fingerprint "
            "FROM program_results pr "
            "JOIN experiments e ON e.experiment_id = pr.experiment_id "
            "WHERE e.experiment_type = 'investigation'"
        ).fetchall()
        investigated_fps = {r[0] for r in rows if r[0]}
    except Exception:
        pass

    if not investigated_fps:
        ok("No investigated fingerprints to check")
        return

    stale = db.execute(
        "SELECT l.entry_id, pr.graph_fingerprint "
        "FROM leaderboard l "
        "JOIN program_results pr ON pr.result_id = l.result_id "
        "WHERE l.tier = 'screening' "
        "AND pr.graph_fingerprint IN ({})".format(
            ",".join("?" for _ in investigated_fps)
        ),
        list(investigated_fps),
    ).fetchall()

    if stale:
        warn(
            f"{len(stale)} leaderboard entries stuck in screening despite failed investigation"
        )
        for s in stale[:3]:
            info(f"  {s[1][:10]} still in screening tier")
    else:
        ok("No stale leaderboard entries")


def check_experiment_stats(db):
    """Show current experiment stats."""
    total_exp = db.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    total_prog = db.execute("SELECT COUNT(*) FROM program_results").fetchone()[0]
    total_s1 = (
        db.execute("SELECT SUM(n_stage1_passed) FROM experiments").fetchone()[0] or 0
    )
    lb_size = db.execute("SELECT COUNT(*) FROM leaderboard").fetchone()[0]
    info(
        f"DB: {total_exp} experiments, {total_prog} programs, {total_s1} total S1 survivors, {lb_size} leaderboard"
    )


def run_checks(db):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 60}")
    print(f"{BOLD}Aria Health Check — {now}{RST}")
    print(f"{'=' * 60}")
    check_experiment_stats(db)
    print()
    check_consecutive_failures(db)
    check_investigation_loop(db)
    check_config_diversity(db)
    check_s0_pass_rate(db)
    check_stale_leaderboard(db)
    print()


def main():
    parser = argparse.ArgumentParser(description="Monitor Aria for degenerate patterns")
    parser.add_argument(
        "--interval", type=int, default=60, help="Check interval in seconds (0=once)"
    )
    parser.add_argument("--db", default=DB_PATH, help="Path to lab_notebook.db")
    args = parser.parse_args()

    if args.interval == 0:
        db = sqlite3.connect(args.db)
        run_checks(db)
        db.close()
        return

    print(f"Monitoring Aria every {args.interval}s (Ctrl+C to stop)")
    try:
        while True:
            db = sqlite3.connect(args.db)
            run_checks(db)
            db.close()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
