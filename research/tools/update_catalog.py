"""Populate empty fields in component_catalog.csv from lab_notebook.db.

Fields updated:
  - template_weight: normalized DB-computed performance weight for this op
  - motif_class: most common motif this op appears in
  - motif_support: number of evaluations containing this op
  - motif_lift: S1 pass rate relative to global average
  - motif_avg_loss: mean loss_ratio across experiments containing this op
  - paradigm: inferred from op category + performance cluster

Usage:
    python -m research.tools.update_catalog [--db research/lab_notebook.db]
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from pathlib import Path
from typing import Dict, Optional

CATALOG_PATH = Path("research/profiling/component_catalog.csv")

# Paradigm inference from category and op name patterns
_PARADIGM_MAP = {
    "mixing": "attention",
    "sequence": "recurrent",
    "frequency": "spectral",
    "math_space": "geometric",
    "reduction": "pooling",
    "linear_algebra": "linear",
    "functional": "routing",
}

_OP_PARADIGM_OVERRIDES = {
    "softmax_attention": "attention",
    "diff_attention": "attention",
    "local_window_attn": "attention",
    "graph_attention": "attention",
    "ultrametric_attention": "attention",
    "state_space": "state-space",
    "selective_scan": "state-space",
    "rwkv_time_mixing": "state-space",
    "conv1d_seq": "convolution",
    "conv_only": "convolution",
    "depthwise_conv1d": "convolution",
    "moe_topk": "mixture-of-experts",
    "moe_2expert": "mixture-of-experts",
    "n_way_sparse_router": "mixture-of-experts",
    "tropical_moe": "mixture-of-experts",
    "lif_neuron": "spiking",
    "stdp_attention": "spiking",
    "spike_rate_code": "spiking",
    "sparse_threshold": "spiking",
    "swiglu_mlp": "ffn",
    "fused_linear_gelu": "ffn",
    "gated_linear": "ffn",
    "adaptive_recursion": "routing",
    "cascade": "routing",
    "early_exit": "routing",
    "token_merge": "routing",
}


def _infer_paradigm(op_name: str, category: str) -> str:
    """Infer paradigm from op name and category."""
    if op_name in _OP_PARADIGM_OVERRIDES:
        return _OP_PARADIGM_OVERRIDES[op_name]
    return _PARADIGM_MAP.get(category, "general")


def update_catalog(
    db_path: str = "research/lab_notebook.db",
    catalog_path: Optional[Path] = None,
) -> int:
    """Update catalog CSV with DB-computed fields. Returns rows updated."""
    catalog_path = catalog_path or CATALOG_PATH

    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.execute("PRAGMA busy_timeout=10000")

    # Load op_stats
    op_stats: Dict[str, dict] = {}
    try:
        for row in conn.execute(
            "SELECT op_name, eval_count, s1_pass_count, mean_loss, min_loss, co_occurrence_json FROM op_stats"
        ).fetchall():
            op_stats[row[0]] = {
                "eval_count": row[1],
                "s1_pass_count": row[2],
                "mean_loss": row[3],
                "min_loss": row[4],
                "co_occurrence": row[5],
            }
    except sqlite3.OperationalError:
        print("op_stats table not found — run backfill_stats.py first")
        conn.close()
        return 0

    # Load motif membership: which motif each op appears in most
    motif_membership: Dict[str, str] = {}
    try:
        # Parse motif_stats for best_template per motif, then map ops
        from research.synthesis.motifs import ALL_MOTIFS

        for motif in ALL_MOTIFS:
            for step in motif.steps:
                op_name = step.op_name
                # Assign op to its highest-lift motif
                if op_name not in motif_membership:
                    motif_membership[op_name] = motif.name
                else:
                    current_motif = next(
                        (m for m in ALL_MOTIFS if m.name == motif_membership[op_name]),
                        None,
                    )
                    if current_motif and motif.lift > current_motif.lift:
                        motif_membership[op_name] = motif.name
    except ImportError:
        pass

    conn.close()

    # Compute global S1 rate for lift calculation
    total_evals = sum(d["eval_count"] for d in op_stats.values())
    total_s1 = sum(d["s1_pass_count"] for d in op_stats.values())
    global_s1_rate = total_s1 / max(total_evals, 1)

    # Read CSV
    rows = []
    with open(catalog_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # Update fields
    updated = 0
    for row in rows:
        op_name = row.get("maps_to_primitive") or row["name"]
        stats = op_stats.get(op_name)
        category = row.get("category", "")

        if stats:
            eval_count = stats["eval_count"]
            s1_count = stats["s1_pass_count"]
            mean_loss = stats["mean_loss"]
            s1_rate = s1_count / max(eval_count, 1)

            # template_weight: exp(-3 * mean_loss) * (1 + s1_rate), normalized
            if mean_loss is not None and math.isfinite(mean_loss):
                row["template_weight"] = (
                    f"{math.exp(-3.0 * mean_loss) * (1.0 + s1_rate):.4f}"
                )
            else:
                row["template_weight"] = ""

            # motif_support: number of evaluations
            row["motif_support"] = str(eval_count)

            # motif_avg_loss
            if mean_loss is not None and math.isfinite(mean_loss):
                row["motif_avg_loss"] = f"{mean_loss:.4f}"
            else:
                row["motif_avg_loss"] = ""

            # motif_lift: S1 rate relative to global
            if global_s1_rate > 0:
                row["motif_lift"] = f"{s1_rate / global_s1_rate:.3f}"
            else:
                row["motif_lift"] = ""

            updated += 1
        else:
            row["template_weight"] = ""
            row["motif_support"] = "0"
            row["motif_avg_loss"] = ""
            row["motif_lift"] = ""

        # motif_class: best motif this op belongs to
        row["motif_class"] = motif_membership.get(op_name, "")

        # paradigm
        row["paradigm"] = _infer_paradigm(op_name, category)

    # Write updated CSV
    with open(catalog_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {updated}/{len(rows)} rows in {catalog_path}")
    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate catalog CSV from DB stats")
    parser.add_argument("--db", default="research/lab_notebook.db")
    args = parser.parse_args()
    update_catalog(args.db)
