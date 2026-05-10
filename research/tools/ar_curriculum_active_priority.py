#!/usr/bin/env python
"""Decision-tree-driven active priority list for ar_curriculum backfill.

Trains a decision tree on existing backfilled rows, identifies leaves with
high outcome variance (= regions where the predictor can't reliably guess
the curriculum AUC), then routes un-backfilled archs through the tree and
priority-orders them by leaf uncertainty.

Premise: when adding more data, you learn the most by sampling regions
where your current model is most uncertain. This is the simplest form of
active learning — entropy-based selection without retraining a full
ensemble for variance.

Output:
  research/runtime/ar_curriculum_experiment/active_priority_<run_id>.jsonl
  research/runtime/ar_curriculum_experiment/active_priority_<run_id>.md
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from research.scientist.notebook import LabNotebook
from research.tools.ar_curriculum_predictor import to_matrix
from research.tools.ar_curriculum_trends import UPSTREAM_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / "research/runtime/ar_curriculum_experiment"
DEFAULT_DB = REPO_ROOT / "research/runs.db"
TARGET = "ar_curriculum_auc_pair_final"


def _load_done(nb: LabNotebook) -> list[dict[str, Any]]:
    feature_cols = ", ".join(f"pr.{f}" for f in UPSTREAM_FEATURES)
    sql = f"""
        SELECT pr.graph_fingerprint, pr.result_id, l.tier, l.composite_score,
               {feature_cols}, pr.{TARGET}
        FROM program_results_compat pr
        JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE pr.{TARGET} IS NOT NULL
    """
    rows = nb.conn.execute(sql).fetchall()
    return [
        dict(r) if hasattr(r, "keys") else {k: r[i] for i, k in enumerate(r.keys())}
        for r in rows
    ]


def _load_remaining(nb: LabNotebook, tiers: tuple[str, ...]) -> list[dict[str, Any]]:
    feature_cols = ", ".join(f"pr.{f}" for f in UPSTREAM_FEATURES)
    placeholders = ",".join("?" for _ in tiers)
    sql = f"""
        SELECT pr.graph_fingerprint, pr.result_id, l.tier, l.composite_score,
               {feature_cols}
        FROM program_results_compat pr
        JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE pr.{TARGET} IS NULL
          AND l.tier IN ({placeholders})
          AND pr.graph_json IS NOT NULL
    """
    rows = nb.conn.execute(sql, tiers).fetchall()
    return [
        dict(r) if hasattr(r, "keys") else {k: r[i] for i, k in enumerate(r.keys())}
        for r in rows
    ]


def compute_leaf_stats(
    tree: DecisionTreeRegressor, X: np.ndarray, y: np.ndarray
) -> dict[int, dict[str, float]]:
    leaf_ids = tree.apply(X)
    bucket: dict[int, list[float]] = defaultdict(list)
    for lid, target in zip(leaf_ids, y):
        bucket[int(lid)].append(float(target))
    stats: dict[int, dict[str, float]] = {}
    for lid, vals in bucket.items():
        stats[lid] = {
            "n": float(len(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "range": float(np.max(vals) - np.min(vals)),
        }
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--tiers", default="validation")
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--min-samples-leaf", type=int, default=10)
    p.add_argument("--limit", type=int, default=100, help="Top-K archs to write")
    p.add_argument(
        "--uncertainty",
        default="std",
        choices=("std", "range"),
        help="Leaf uncertainty proxy: std or range of training targets in leaf.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tiers = tuple(t.strip() for t in str(args.tiers).split(",") if t.strip())
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    nb = LabNotebook(str(args.db), read_only=True)
    done_rows = _load_done(nb)
    remaining_rows = _load_remaining(nb, tiers)
    nb.close()
    logger.info("done=%d remaining=%d", len(done_rows), len(remaining_rows))
    if len(done_rows) < 30 or not remaining_rows:
        logger.info("Insufficient data to build active priority list")
        return 0

    Xtr, ytr, feature_names = to_matrix(done_rows, TARGET)
    tree = DecisionTreeRegressor(
        max_depth=int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.seed),
    )
    tree.fit(Xtr, ytr)
    leaf_stats = compute_leaf_stats(tree, Xtr, ytr)

    # Build remaining arch matrix using the SAME feature columns + medians
    from research.tools.ar_curriculum_predictor import (
        to_matrix as _to_matrix,
    )  # reuse imputation

    Xte, _, feature_names_te = _to_matrix(
        [{**r, TARGET: 0.0} for r in remaining_rows], TARGET
    )
    if feature_names_te != feature_names:
        logger.warning(
            "Feature columns differ between done and remaining (%d vs %d). "
            "Aligning by intersection.",
            len(feature_names),
            len(feature_names_te),
        )

    leaf_ids_te = tree.apply(Xte)
    scored: list[dict[str, Any]] = []
    for arch, lid in zip(remaining_rows, leaf_ids_te):
        stats = leaf_stats.get(int(lid), {})
        if not stats:
            continue
        uncertainty = stats.get(args.uncertainty, 0.0)
        scored.append(
            {
                "result_id": arch["result_id"],
                "graph_fingerprint": arch["graph_fingerprint"],
                "tier": arch["tier"],
                "composite_score": float(arch.get("composite_score") or 0),
                "leaf_id": int(lid),
                "leaf_n": int(stats["n"]),
                "leaf_mean": round(stats["mean"], 3),
                "leaf_std": round(stats["std"], 3),
                "leaf_range": round(stats["range"], 3),
                "uncertainty": round(uncertainty, 4),
            }
        )

    scored.sort(key=lambda d: (d["uncertainty"], -d["composite_score"]), reverse=True)
    top = scored[: int(args.limit)]

    out_dir = RUNTIME_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"active_priority_{run_id}.jsonl"
    md_path = out_dir / f"active_priority_{run_id}.md"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for entry in top:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    leaves_sorted = sorted(
        leaf_stats.items(), key=lambda p: p[1].get(args.uncertainty, 0), reverse=True
    )
    lines: list[str] = [
        f"# AR curriculum active priority — {run_id}",
        "",
        f"Trained DecisionTreeRegressor(max_depth={args.max_depth}, "
        f"min_samples_leaf={args.min_samples_leaf}) on n_train={len(Xtr)}",
        f"Wrote top-{int(args.limit)} of {len(scored)} candidates by leaf {args.uncertainty}.",
        "",
        "## Leaf statistics (sorted by uncertainty)",
        "",
        "| leaf | n_train | mean | std | range | min | max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lid, st_d in leaves_sorted[:15]:
        lines.append(
            f"| {lid} | {int(st_d['n'])} | {st_d['mean']:.3f} | {st_d['std']:.3f} | "
            f"{st_d['range']:.3f} | {st_d['min']:.3f} | {st_d['max']:.3f} |"
        )

    leaf_to_remaining: dict[int, int] = defaultdict(int)
    for s in scored:
        leaf_to_remaining[s["leaf_id"]] += 1
    lines += [
        "",
        "## Distribution of remaining archs across tree leaves",
        "",
        "| leaf | n_remaining | leaf uncertainty |",
        "|---:|---:|---:|",
    ]
    for lid, n in sorted(leaf_to_remaining.items(), key=lambda p: -p[1])[:10]:
        u = leaf_stats[lid].get(args.uncertainty, 0.0)
        lines.append(f"| {lid} | {n} | {u:.3f} |")

    lines += [
        "",
        f"## Top {len(top)} priority archs",
        "",
        "| rank | fp | tier | composite | leaf | leaf_std | leaf_range | leaf_mean |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, p in enumerate(top[:30], 1):
        lines.append(
            f"| {i} | {p['graph_fingerprint'][:12]} | {p['tier']} | "
            f"{p['composite_score']:.0f} | {p['leaf_id']} | "
            f"{p['leaf_std']:.3f} | {p['leaf_range']:.3f} | {p['leaf_mean']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Wrote %s (%d entries)", jsonl_path, len(top))
    logger.info("Wrote %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
