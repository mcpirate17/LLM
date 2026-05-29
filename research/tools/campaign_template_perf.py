"""Per-template performance summary for the novel-mixer optimization campaign.

Joins ``graph_runs`` to ``experiments`` (via ``experiment_id``) to recover the
backfill template, then aggregates the nano-scale metrics that matter at <20k
steps (BLiMP is the live signal; induction/binding/ar sit at noise floor).

Usage:
    python -m research.tools.campaign_template_perf --since-min 120
    python -m research.tools.campaign_template_perf --templates clifford_geometric_mixer_block
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from typing import Any

# Priority nano metrics (user: ignore ppl; BLiMP + capability probes matter).
# Capability columns (ar_gate / nano_induction_nearest / language_control_s05·s10)
# are NULL unless the dedicated probe tools have been run on the fingerprints.
_METRICS = (
    "blimp_overall_accuracy",
    "ar_gate_score",
    "nano_induction_nearest_max_accuracy",
    "language_control_s05_binding_score",
    "language_control_s10_binding_score",
    "induction_screening_auc",
    "loss_ratio",
)


def _template_by_experiment(conn: sqlite3.Connection) -> dict[str, str]:
    """experiment_id -> backfill template name (config_json wins, hypothesis fallback)."""
    out: dict[str, str] = {}
    for row in conn.execute(
        "SELECT experiment_id, hypothesis, config_json FROM experiments"
    ):
        eid, hyp, cfg_json = row
        name = None
        if cfg_json:
            try:
                name = json.loads(cfg_json).get("backfill_template")
            except (ValueError, TypeError):
                name = None
        if not name and hyp and "template '" in hyp:
            name = hyp.split("template '", 1)[1].rstrip("'")
        if name:
            out[eid] = name
    return out


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def collect(
    db: str, since_min: float | None, only: set[str] | None
) -> dict[str, dict[str, Any]]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    tpl_map = _template_by_experiment(conn)
    cutoff = (time.time() - since_min * 60) if since_min else None
    sel = "experiment_id, timestamp, stage1_passed, graph_fingerprint, " + ", ".join(
        _METRICS
    )
    rows = conn.execute(
        f"SELECT {sel} FROM graph_runs ORDER BY timestamp ASC"
    ).fetchall()
    agg: dict[str, dict[str, Any]] = {}
    for r in rows:
        if cutoff and (r["timestamp"] or 0) < cutoff:
            continue
        tpl = tpl_map.get(r["experiment_id"])
        if not tpl or (only and tpl not in only):
            continue
        a = agg.setdefault(
            tpl, {"n": 0, "s1": 0, "rows": [], **{m: [] for m in _METRICS}}
        )
        a["n"] += 1
        a["s1"] += int(r["stage1_passed"] or 0)
        for m in _METRICS:
            if r[m] is not None:
                a[m].append(float(r[m]))
        a["rows"].append((r["graph_fingerprint"], r["blimp_overall_accuracy"]))
    conn.close()
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="research/runs.db")
    ap.add_argument(
        "--since-min", type=float, default=None, help="only rows newer than N minutes"
    )
    ap.add_argument("--templates", nargs="*", default=None)
    args = ap.parse_args()

    agg = collect(
        args.db, args.since_min, set(args.templates) if args.templates else None
    )
    if not agg:
        print("(no matching graph_runs rows)")
        return

    # Priority nano columns: BLiMP (primary) + capability probes. ppl dropped.
    # ARg=ar_gate_score, indN=nano_induction_nearest, nb05/nb10=language_control binding.
    hdr = (
        f"{'template':42s} {'n':>3s} {'s1%':>4s} {'BLiMP med':>9s} {'BLiMP max':>9s} "
        f"{'ARgate':>6s} {'indNear':>7s} {'nb05':>5s} {'nb10':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))

    def med(a, m):
        return _median(a[m])

    for tpl in sorted(
        agg, key=lambda t: -(_median(agg[t]["blimp_overall_accuracy"]) or 0)
    ):
        a = agg[tpl]
        bl = a["blimp_overall_accuracy"]
        s1pct = 100 * a["s1"] / a["n"] if a["n"] else 0

        def cell(metric):
            v = med(a, metric)
            return f"{v:.3f}" if v is not None else "  -  "

        print(
            f"{tpl:42s} {a['n']:3d} {s1pct:4.0f} "
            f"{(_median(bl) or 0):9.4f} {(max(bl) if bl else 0):9.4f} "
            f"{cell('ar_gate_score'):>6s} {cell('nano_induction_nearest_max_accuracy'):>7s} "
            f"{cell('language_control_s05_binding_score'):>5s} "
            f"{cell('language_control_s10_binding_score'):>5s}"
        )
    print("(ARgate/indNear/nb05/nb10 = '-' until dedicated probe tools are run)")
    print("\nbest BLiMP fingerprint per template:")
    for tpl, a in agg.items():
        best = max(
            (r for r in a["rows"] if r[1] is not None), key=lambda r: r[1], default=None
        )
        if best:
            print(f"  {tpl:42s} {best[0]}  BLiMP={best[1]:.4f}")


if __name__ == "__main__":
    main()
