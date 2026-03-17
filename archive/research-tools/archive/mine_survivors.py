#!/usr/bin/env python3
"""Mine stage-1 survivors for common op motifs and seed templates."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, List


def _safe_graph(graph_json: str) -> Dict:
    try:
        parsed = json.loads(graph_json or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def mine_experiment(db_path: Path, experiment_id: str) -> Dict:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT graph_json, novelty_score, loss_ratio FROM program_results "
        "WHERE experiment_id = ? AND stage1_passed = 1",
        (experiment_id,),
    )
    rows = cur.fetchall()
    conn.close()

    op_counts: Counter = Counter()
    bigrams: Counter = Counter()
    trigrams: Counter = Counter()
    seed_templates: List[Dict] = []

    for graph_json, novelty, loss_ratio in rows:
        graph = _safe_graph(graph_json)
        nodes = graph.get("nodes") if isinstance(graph.get("nodes"), dict) else {}
        if not nodes:
            continue

        for node in nodes.values():
            op = str(node.get("op_name") or "").strip()
            if not op or op == "input":
                continue
            op_counts[op] += 1

        for node in nodes.values():
            op = str(node.get("op_name") or "").strip()
            if not op or op == "input":
                continue
            for parent_id in node.get("input_ids") or []:
                parent = nodes.get(str(parent_id)) or {}
                parent_op = str(parent.get("op_name") or "").strip()
                if not parent_op or parent_op == "input":
                    continue
                bigrams[(parent_op, op)] += 1
                for gp_id in parent.get("input_ids") or []:
                    gp = nodes.get(str(gp_id)) or {}
                    gp_op = str(gp.get("op_name") or "").strip()
                    if not gp_op or gp_op == "input":
                        continue
                    trigrams[(gp_op, parent_op, op)] += 1

        # Track seed-worthy survivors only.
        if float(novelty or 0.0) >= 0.75:
            seed_templates.append(
                {
                    "novelty_score": float(novelty or 0.0),
                    "loss_ratio": float(loss_ratio or 1.0),
                    "ops": sorted(
                        {
                            str(n.get("op_name") or "").strip()
                            for n in nodes.values()
                            if str(n.get("op_name") or "").strip()
                            and str(n.get("op_name") or "").strip() != "input"
                        }
                    ),
                }
            )

    top_ops = [{"op": op, "count": count} for op, count in op_counts.most_common(20)]
    top_bigrams = [
        {"from": pair[0], "to": pair[1], "count": count}
        for pair, count in bigrams.most_common(20)
    ]
    top_trigrams = [
        {"a": tri[0], "b": tri[1], "c": tri[2], "count": count}
        for tri, count in trigrams.most_common(20)
    ]
    best_seeds = sorted(
        seed_templates, key=lambda x: (x["loss_ratio"], -x["novelty_score"])
    )[:25]

    return {
        "experiment_id": experiment_id,
        "n_survivors": len(rows),
        "top_ops": top_ops,
        "top_bigrams": top_bigrams,
        "top_trigrams": top_trigrams,
        "seed_templates": best_seeds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="dba351f2-d9e")
    parser.add_argument("--db-path", default="research/lab_notebook.db")
    parser.add_argument(
        "--output",
        default="research/reports/survivor_motifs_dba351f2-d9e.json",
    )
    args = parser.parse_args()

    report = mine_experiment(
        db_path=Path(args.db_path), experiment_id=str(args.experiment_id)
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Mined {report['n_survivors']} survivors from {report['experiment_id']} "
        f"-> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
