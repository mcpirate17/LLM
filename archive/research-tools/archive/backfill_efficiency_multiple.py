#!/usr/bin/env python3
"""Backfill efficiency_multiple for all leaderboard entries that have data."""

import sys
from pathlib import Path

# Ensure research/ is on sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scientist.notebook import LabNotebook


def main():
    nb = LabNotebook()
    rows = nb.conn.execute(
        "SELECT l.entry_id, l.result_id, l.efficiency_multiple, "
        "pr.loss_ratio, pr.param_count, pr.flops_forward, "
        "pr.throughput_tok_s, pr.peak_memory_mb, pr.forward_time_ms "
        "FROM leaderboard l "
        "LEFT JOIN program_results pr ON pr.result_id = l.result_id"
    ).fetchall()

    updated = 0
    skipped = 0
    insufficient = 0

    for row in rows:
        d = dict(row)
        # Skip if already computed
        if d.get("efficiency_multiple") is not None:
            skipped += 1
            continue

        eff = LabNotebook.compute_efficiency_multiple(
            loss_ratio=d.get("loss_ratio"),
            param_count=d.get("param_count"),
            flops_forward=d.get("flops_forward"),
            throughput_tok_s=d.get("throughput_tok_s"),
            peak_memory_mb=d.get("peak_memory_mb"),
            forward_time_ms=d.get("forward_time_ms"),
        )
        if eff is None:
            insufficient += 1
            continue

        geomean = eff["geomean"]
        nb.conn.execute(
            "UPDATE leaderboard SET efficiency_multiple = ? WHERE entry_id = ?",
            (geomean, d["entry_id"]),
        )
        # Also store in program_results if column exists
        try:
            nb.conn.execute(
                "UPDATE program_results SET efficiency_multiple = ? WHERE result_id = ?",
                (geomean, d["result_id"]),
            )
        except Exception:
            pass
        updated += 1

    nb.conn.commit()
    print(
        f"Backfilled {updated} entries, skipped {skipped} (already set), "
        f"{insufficient} had insufficient data"
    )


if __name__ == "__main__":
    main()
