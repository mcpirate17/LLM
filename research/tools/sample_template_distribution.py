"""Phase 3 deployment validation: sample template-pick distribution.

Loads DB-blended template weights (fires `_fetch_template_weight_rows` CTE +
`_template_dynamic_weight` sa_factor + ndjson emission), then samples N
templates via `pick_template`. Tabulates per-bucket pick share to confirm:

  - Bucket B (cull, weight floor 0.5) → near-zero pick share
  - Bucket A/A+ → boosted share
  - Bucket D new templates (tropical_attn_conv1d_seq_block etc.) → non-zero

Pure read on lab_notebook.db (only `_fetch_template_weight_rows` SELECT). Safe
to run while the dashboard server holds its writer flock.

Usage:
  python research/tools/sample_template_distribution.py --n 10000
"""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.synthesis.grammar_support import (  # noqa: E402
    _build_db_template_weights,
    _fetch_template_weight_rows,
)
from research.synthesis.templates import (  # noqa: E402
    DEFAULT_TEMPLATE_WEIGHTS,
    TEMPLATES,
    pick_template,
)


CLASSIFICATION_CSV = REPO / "research/reports/template_classification.csv"
LAB_DB = REPO / "research/lab_notebook.db"


def load_bucket_map() -> dict[str, str]:
    if not CLASSIFICATION_CSV.exists():
        return {}
    with CLASSIFICATION_CSV.open() as f:
        return {r["template_name"]: r["bucket"] for r in csv.DictReader(f)}


def load_db_weights() -> dict[str, float]:
    conn = sqlite3.connect(f"file:{LAB_DB}?mode=ro&immutable=0", uri=True)
    rows = _fetch_template_weight_rows(conn)
    weights = _build_db_template_weights(rows, template_slot_context={})
    conn.close()
    return weights


def sample(weights: dict[str, float] | None, n: int) -> Counter[str]:
    rng = random.Random(0)
    counts: Counter[str] = Counter()
    for _ in range(n):
        name, _, _ = pick_template(rng, weights=weights, exploration_budget=0.0)
        counts[name] += 1
    return counts


def report(counts: Counter[str], buckets: dict[str, str], n: int, label: str) -> None:
    by_bucket: Counter[str] = Counter()
    for name, c in counts.items():
        by_bucket[buckets.get(name, "UNKNOWN")] += c

    print(f"\n=== {label} (n={n} picks) ===", file=sys.stderr)
    print("Per-bucket share:", file=sys.stderr)
    for b in ("A+", "A", "C", "D", "HOLD", "E", "B", "UNKNOWN"):
        share = by_bucket.get(b, 0) / n
        print(f"  {b:8s}  {by_bucket.get(b, 0):>5d}  {share:.3%}", file=sys.stderr)

    # Bucket B drill-down: which culled templates still got picked?
    print("\nBucket B picks (cull-floor):", file=sys.stderr)
    b_picks = sorted(
        ((name, c) for name, c in counts.items() if buckets.get(name) == "B"),
        key=lambda kv: -kv[1],
    )
    for name, c in b_picks[:10]:
        print(f"  {name:42s} {c:>4d}", file=sys.stderr)
    if not b_picks:
        print("  (none — clean)", file=sys.stderr)

    # New Phase 3.1 Bucket D templates
    print("\nNew Bucket D templates (Phase 3.1):", file=sys.stderr)
    for name in (
        "tropical_attn_conv1d_seq_block",
        "rwkv_channel_conv1d_seq_block",
        "matmul_conv1d_seq_block",
    ):
        c = counts.get(name, 0)
        share = c / n
        print(f"  {name:42s} {c:>4d}  {share:.3%}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10000)
    args = parser.parse_args()

    buckets = load_bucket_map()
    print(f"Loaded {len(buckets)} bucket assignments", file=sys.stderr)
    print(f"TEMPLATES registry: {len(TEMPLATES)} entries", file=sys.stderr)

    # Static-only baseline (no DB blender)
    static_counts = sample(None, args.n)
    report(
        static_counts, buckets, args.n, "STATIC weights (DEFAULT_TEMPLATE_WEIGHTS only)"
    )

    # DB-blended (fires CTE + sa_factor + ndjson)
    db_weights = load_db_weights()
    print(
        f"\nDB-blended weights computed for {len(db_weights)} templates",
        file=sys.stderr,
    )
    blended_counts = sample(db_weights, args.n)
    report(
        blended_counts, buckets, args.n, "DB-BLENDED weights (sa_factor + final clamp)"
    )

    # Sanity: DEFAULT_TEMPLATE_WEIGHTS at floor for B bucket?
    print("\nFloor sanity (DEFAULT_TEMPLATE_WEIGHTS for Bucket B):", file=sys.stderr)
    b_at_floor = 0
    b_total = 0
    for name, b in buckets.items():
        if b != "B":
            continue
        b_total += 1
        w = DEFAULT_TEMPLATE_WEIGHTS.get(name)
        if w == 0.5:
            b_at_floor += 1
    print(
        f"  {b_at_floor}/{b_total} Bucket-B templates at static weight 0.5",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
