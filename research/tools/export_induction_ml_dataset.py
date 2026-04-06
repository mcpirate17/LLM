#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = "research/lab_notebook.db"
DEFAULT_OUT = "tasks/induction_native_probe/induction_ml_dataset.csv"


def _load_json(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_template(graph: dict[str, Any]) -> str:
    metadata = graph.get("metadata") or {}
    primary = metadata.get("primary_template")
    if primary:
        return str(primary)
    used = metadata.get("templates_used") or []
    if isinstance(used, list) and used:
        return str(used[0])
    return ""


def _extract_templates(graph: dict[str, Any]) -> str:
    metadata = graph.get("metadata") or {}
    used = metadata.get("templates_used") or []
    if not isinstance(used, list):
        return ""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in used:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return "|".join(ordered)


def _extract_ops(graph: dict[str, Any]) -> tuple[str, int, int]:
    nodes = graph.get("nodes") or {}
    if isinstance(nodes, dict):
        node_iter = nodes.values()
    elif isinstance(nodes, list):
        node_iter = nodes
    else:
        return "", 0, 0
    ops: list[str] = []
    seen: set[str] = set()
    for node in node_iter:
        if not isinstance(node, dict):
            continue
        op = str(node.get("op_name") or "").strip()
        if not op or op in {"input", "output"}:
            continue
        ops.append(op)
        seen.add(op)
    return "|".join(sorted(seen)), len(ops), len(seen)


def _learner_bucket(auc: float | None) -> str:
    value = float(auc or 0.0)
    if value >= 0.05:
        return "learner"
    if value >= 0.02:
        return "weak_learner"
    if value > 0.0:
        return "trace_learner"
    return "zero"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export one-row-per-fingerprint induction ML dataset."
    )
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT pr.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY pr.graph_fingerprint
                       ORDER BY COALESCE(pr.induction_probe_train_steps, 0) DESC,
                                pr.timestamp DESC
                   ) AS rn
            FROM program_results pr
            WHERE COALESCE(pr.graph_json, '') NOT IN ('', '{}')
              AND COALESCE(pr.graph_fingerprint, '') != ''
        )
        SELECT
            l.graph_fingerprint,
            l.result_id,
            l.experiment_id,
            l.timestamp,
            l.graph_json,
            l.model_source,
            l.stage0_passed,
            l.stage05_passed,
            l.stage1_passed,
            l.loss_ratio,
            l.validation_loss_ratio,
            l.discovery_loss_ratio,
            l.wikitext_perplexity,
            l.param_count,
            l.graph_n_ops,
            l.graph_depth,
            l.graph_n_edges,
            l.graph_n_unique_ops,
            l.graph_n_params_estimate,
            l.graph_category_histogram,
            l.graph_uses_math_spaces,
            l.fingerprint_json,
            l.arch_spec_json,
            l.routing_mode,
            l.routing_expert_count,
            l.activation_sparsity_score,
            l.dead_neuron_ratio,
            l.routing_collapse_score,
            l.error_type,
            l.stage_at_death,
            im.metric_version,
            im.speed_mode,
            im.train_steps,
            im.eval_examples,
            im.batch_size,
            im.pool_size,
            im.auc,
            im.gap_4,
            im.gap_8,
            im.gap_16,
            im.gap_32,
            im.gap_64,
            im.wall_ms,
            im.source_cohort
        FROM latest l
        JOIN induction_metrics_v2 im ON im.graph_fingerprint = l.graph_fingerprint
        WHERE l.rn = 1
        ORDER BY im.auc DESC, l.graph_fingerprint ASC
        """
    ).fetchall()
    conn.close()

    enriched: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        graph = _load_json(record.get("graph_json"))
        op_names, parsed_n_ops, parsed_n_unique_ops = _extract_ops(graph)
        record["primary_template"] = _extract_template(graph)
        record["templates_used"] = _extract_templates(graph)
        record["op_names"] = op_names
        record["parsed_graph_n_ops"] = parsed_n_ops
        record["parsed_graph_n_unique_ops"] = parsed_n_unique_ops
        record["learner_bucket"] = _learner_bucket(record.get("auc"))
        enriched.append(record)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(enriched[0].keys()) if enriched else []
        )
        if enriched:
            writer.writeheader()
            writer.writerows(enriched)

    print(f"rows={len(enriched)} out={out_path}")


if __name__ == "__main__":
    main()
