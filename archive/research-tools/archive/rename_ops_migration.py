#!/usr/bin/env python3
"""Migrate old routing op names to new names in lab_notebook.db.

Walks all serialized graph JSON in programs/experiments tables and replaces
old op names with their new canonical names.  Also updates op name columns
in leaderboard/stats tables.

Usage:
    python -m research.tools.rename_ops_migration          # dry-run (default)
    python -m research.tools.rename_ops_migration --apply  # commit changes
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

# Old name → New name mapping (matches OP_NAME_ALIASES in primitives.py)
RENAMES = {
    "route_topk": "feature_sparsity",
    "route_lanes": "gated_lane_blend",
    "route_recursion": "depth_gated_transform",
    "routing_conditioned_compression": "signal_conditioned_compression",
    "relu_gate_routing": "relu_gated_moe",
    "adaptive_lane_mixer": "difficulty_blend_3way",
    "adaptive_recursion": "depth_weighted_proj",
    "cascade": "learned_token_gate",
    "compression_mixture_experts": "dual_compression_blend",
    "difficulty_scorer": "token_difficulty_proj",
    "early_exit": "confidence_token_gate",
    "entropy_score": "token_entropy",
    "mixed_recursion_gate": "score_depth_blend",
    "mod_topk": "depth_token_mask",
    "progressive_compression_gate": "adaptive_rank_gate",
    "speculative": "cheap_verify_blend",
    "token_merge": "adjacent_token_merge",
    "token_type_classifier": "token_class_proj",
    "n_way_sparse_router": "sparse_bottleneck_moe",
}

# Pre-compile regex for whole-word matching in JSON strings.
# Matches "old_name" as a JSON string value (surrounded by quotes).
_PATTERNS = {old: re.compile(rf'"{re.escape(old)}"') for old in RENAMES}


def _rename_in_json(text: str) -> tuple[str, int]:
    """Replace old op names in a JSON string. Returns (new_text, n_replacements)."""
    count = 0
    for old, new in RENAMES.items():
        pat = _PATTERNS[old]
        matches = len(pat.findall(text))
        if matches:
            text = pat.sub(f'"{new}"', text)
            count += matches
    return text, count


def _find_db() -> Path:
    candidates = [
        Path("lab_notebook.db"),
        Path("research/lab_notebook.db"),
        Path(__file__).parent.parent / "lab_notebook.db",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find lab_notebook.db. Run from project root or research/ directory."
    )


def migrate(db_path: Path, apply: bool = False) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    total_rows = 0
    total_replacements = 0

    # --- JSON columns in programs table ---
    for table in ("programs", "program_results", "experiments"):
        cur.execute(f"SELECT name FROM pragma_table_info('{table}')")
        cols = [r[0] for r in cur.fetchall()]
        if not cols:
            continue

        # Find columns likely to contain serialized graphs (JSON text)
        json_cols = [
            c
            for c in cols
            if any(
                k in c.lower()
                for k in ("graph", "config", "program", "spec", "json", "metadata")
            )
        ]
        if not json_cols:
            continue

        cur.execute(f"SELECT rowid, {', '.join(json_cols)} FROM {table}")
        rows = cur.fetchall()
        for row in rows:
            rowid = row[0]
            updates = {}
            for i, col in enumerate(json_cols):
                val = row[i + 1]
                if not isinstance(val, str) or len(val) < 5:
                    continue
                new_val, n = _rename_in_json(val)
                if n > 0:
                    updates[col] = new_val
                    total_replacements += n
            if updates:
                total_rows += 1
                if apply:
                    set_clause = ", ".join(f"{c} = ?" for c in updates)
                    cur.execute(
                        f"UPDATE {table} SET {set_clause} WHERE rowid = ?",
                        [*updates.values(), rowid],
                    )

    # --- Op name columns in leaderboard/stats ---
    for table in ("leaderboard", "template_stats", "op_stats"):
        cur.execute(f"SELECT name FROM pragma_table_info('{table}')")
        cols = [r[0] for r in cur.fetchall()]
        if not cols:
            continue

        # Look for columns that store op names directly
        name_cols = [
            c for c in cols if c in ("op_name", "name", "model_source", "tags")
        ]
        for col in name_cols:
            for old, new in RENAMES.items():
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (old,))
                count = cur.fetchone()[0]
                if count > 0:
                    total_rows += count
                    total_replacements += count
                    if apply:
                        cur.execute(
                            f"UPDATE {table} SET {col} = ? WHERE {col} = ?",
                            (new, old),
                        )

        # Handle tags column (comma-separated or JSON list)
        if "tags" in cols:
            for old, new in RENAMES.items():
                cur.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE tags LIKE ?",
                    (f"%{old}%",),
                )
                count = cur.fetchone()[0]
                if count > 0:
                    total_rows += count
                    total_replacements += count
                    if apply:
                        cur.execute(
                            f"UPDATE {table} SET tags = REPLACE(tags, ?, ?) WHERE tags LIKE ?",
                            (old, new, f"%{old}%"),
                        )

    if apply:
        conn.commit()
        print(
            f"APPLIED: {total_replacements} replacements across {total_rows} rows in {db_path}"
        )
    else:
        print(
            f"DRY RUN: would make {total_replacements} replacements across {total_rows} rows in {db_path}"
        )
        print("Re-run with --apply to commit changes.")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate old routing op names in lab_notebook.db"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually commit changes (default: dry-run)",
    )
    parser.add_argument("--db", type=Path, default=None, help="Path to lab_notebook.db")
    args = parser.parse_args()

    db_path = args.db or _find_db()
    print(f"Database: {db_path}")
    migrate(db_path, apply=args.apply)


if __name__ == "__main__":
    main()
