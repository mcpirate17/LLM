"""Export label-quality triage queues for ambiguous duplicates and near misses.

Usage:
    python -m research.tools.export_label_triage
    python -m research.tools.export_label_triage --top-ambiguous 50 --top-near-miss 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from research.scientist.intelligence.predictor import load_runtime_ensemble
from research.synthesis.graph_features import (
    enrich_with_op_stats,
    extract_graph_features,
    load_op_stats,
)

_DEFAULT_DB = Path("research/lab_notebook.db")
_DEFAULT_REPORT_DIR = Path("research/reports")


def _graph_signature(graph_json: Any) -> Optional[str]:
    if isinstance(graph_json, str):
        try:
            graph_json = json.loads(graph_json)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(graph_json, dict):
        return None
    try:
        canonical = json.dumps(graph_json, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _extract_ops_templates(graph_json: Dict[str, Any]) -> Dict[str, Any]:
    nodes = graph_json.get("nodes") or {}
    ops = sorted(
        {
            node.get("op_name", "")
            for node in nodes.values()
            if node.get("op_name", "") and node.get("op_name", "") != "input"
        }
    )
    metadata = graph_json.get("metadata") or {}
    return {
        "ops": ops,
        "templates_used": list(metadata.get("templates_used") or []),
        "motifs_used": list(metadata.get("motifs_used") or []),
    }


def _load_rows(db_path: Path) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    rows = conn.execute(
        """SELECT result_id, timestamp, graph_json, stage0_passed, stage05_passed,
                  stage1_passed, loss_ratio, wikitext_perplexity
           FROM program_results
           WHERE graph_json IS NOT NULL"""
    ).fetchall()
    conn.close()

    parsed: List[Dict[str, Any]] = []
    for row in rows:
        gj = row["graph_json"]
        try:
            graph_json = json.loads(gj) if isinstance(gj, str) else gj
        except (json.JSONDecodeError, TypeError):
            continue
        signature = _graph_signature(graph_json)
        if signature is None:
            continue
        parsed.append(
            {
                "result_id": row["result_id"],
                "timestamp": row["timestamp"],
                "graph_json": graph_json,
                "signature": signature,
                "stage0_passed": bool(row["stage0_passed"]),
                "stage05_passed": bool(row["stage05_passed"]),
                "stage1_passed": bool(row["stage1_passed"]),
                "loss_ratio": row["loss_ratio"],
                "wikitext_perplexity": row["wikitext_perplexity"],
            }
        )
    return parsed


def build_report(
    db_path: Path,
    top_ambiguous: int,
    top_near_miss: int,
    profiling_db: str,
) -> Dict[str, Any]:
    rows = _load_rows(db_path)
    op_stats_cache = load_op_stats(str(db_path))
    ensemble = load_runtime_ensemble(profiling_db=profiling_db)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["signature"]].append(row)

    ambiguous_groups: List[Dict[str, Any]] = []
    near_misses: List[Dict[str, Any]] = []

    for signature, items in groups.items():
        s1_vals = [1 if it["stage1_passed"] else 0 for it in items]
        s1_rate = sum(s1_vals) / len(s1_vals)
        sample_graph = items[0]["graph_json"]
        graph_meta = _extract_ops_templates(sample_graph)
        feats = extract_graph_features(sample_graph)
        ensemble_score = None
        gbm_score = None
        graph_score = None
        induction_score = None
        induction_auc = None
        if feats:
            enrich_with_op_stats(feats, graph_meta["ops"], preloaded=op_stats_cache)
            if ensemble.gbm is not None and ensemble.gbm.is_fitted():
                gbm_score = float(ensemble.gbm.predict_gate(feats))
            if ensemble.is_fitted():
                ensemble_score = float(
                    ensemble.predict_gate(sample_graph, graph_features=feats)
                )
                induction_score = float(
                    ensemble.predict_induction_learner_prob(
                        sample_graph, graph_features=feats
                    )
                )
                induction_auc = float(
                    ensemble.predict_induction_auc(sample_graph, graph_features=feats)
                )
        if ensemble.graph_pred is not None and ensemble.graph_pred.is_fitted():
            graph_score = float(ensemble.graph_pred.predict_gate(sample_graph))

        if len(items) > 1 and 0.0 < s1_rate < 1.0:
            ambiguous_groups.append(
                {
                    "signature": signature,
                    "n_rows": len(items),
                    "s1_rate": s1_rate,
                    "stage05_rate": sum(
                        1 if it["stage05_passed"] else 0 for it in items
                    )
                    / len(items),
                    "stage0_rate": sum(1 if it["stage0_passed"] else 0 for it in items)
                    / len(items),
                    "ensemble_score": ensemble_score,
                    "gbm_score": gbm_score,
                    "graph_score": graph_score,
                    "predicted_induction_learner": induction_score,
                    "predicted_induction_auc": induction_auc,
                    "ops": graph_meta["ops"],
                    "templates_used": graph_meta["templates_used"],
                    "motifs_used": graph_meta["motifs_used"],
                    "result_ids": [it["result_id"] for it in items],
                }
            )

        for it in items:
            if it["stage1_passed"] or not it["stage05_passed"]:
                continue
            near_misses.append(
                {
                    "result_id": it["result_id"],
                    "timestamp": it["timestamp"],
                    "signature": signature,
                    "ensemble_score": ensemble_score,
                    "gbm_score": gbm_score,
                    "graph_score": graph_score,
                    "predicted_induction_learner": induction_score,
                    "predicted_induction_auc": induction_auc,
                    "loss_ratio": it["loss_ratio"],
                    "wikitext_perplexity": it["wikitext_perplexity"],
                    "dup_group_size": len(items),
                    "dup_group_s1_rate": s1_rate if len(items) > 1 else 0.0,
                    "ops": graph_meta["ops"],
                    "templates_used": graph_meta["templates_used"],
                    "motifs_used": graph_meta["motifs_used"],
                }
            )

    ambiguous_groups.sort(
        key=lambda item: (
            -(item["ensemble_score"] if item["ensemble_score"] is not None else -1.0),
            -item["s1_rate"],
            -item["n_rows"],
        )
    )
    near_misses.sort(
        key=lambda item: (
            -(item["ensemble_score"] if item["ensemble_score"] is not None else -1.0),
            -(item["gbm_score"] if item["gbm_score"] is not None else -1.0),
            -(item["graph_score"] if item["graph_score"] is not None else -1.0),
            -item["dup_group_s1_rate"],
        )
    )

    return {
        "summary": {
            "n_rows": len(rows),
            "n_unique_graphs": len(groups),
            "n_ambiguous_groups": len(ambiguous_groups),
            "n_stage05_near_misses": len(near_misses),
            "ensemble_loaded": bool(ensemble.is_fitted()),
        },
        "ambiguous_groups": ambiguous_groups[:top_ambiguous],
        "near_misses": near_misses[:top_near_miss],
    }


def _format_md(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Label Triage Report",
        "",
        f"- Rows scanned: {summary['n_rows']}",
        f"- Unique exact graphs: {summary['n_unique_graphs']}",
        f"- Ambiguous duplicate groups: {summary['n_ambiguous_groups']}",
        f"- Stage05 near misses: {summary['n_stage05_near_misses']}",
        f"- Ensemble loaded: {summary['ensemble_loaded']}",
        "",
        "## Top Ambiguous Duplicate Groups",
        "",
    ]
    if not report["ambiguous_groups"]:
        lines.append("None.")
    else:
        for idx, item in enumerate(report["ambiguous_groups"], start=1):
            lines.extend(
                [
                    f"### {idx}. `{item['signature'][:12]}`",
                    f"- rows: {item['n_rows']}",
                    f"- s1_rate: {item['s1_rate']:.3f}",
                    f"- stage05_rate: {item['stage05_rate']:.3f}",
                    f"- ensemble_score: {item['ensemble_score']}",
                    f"- gbm_score: {item['gbm_score']}",
                    f"- graph_score: {item['graph_score']}",
                    f"- templates: {', '.join(item['templates_used'][:6]) or 'none'}",
                    f"- motifs: {', '.join(item['motifs_used'][:6]) or 'none'}",
                    f"- ops: {', '.join(item['ops'][:12]) or 'none'}",
                    f"- result_ids: {item['result_ids'][:12]}",
                    "",
                ]
            )
    lines.extend(["## Top Stage05 Near Misses", ""])
    if not report["near_misses"]:
        lines.append("None.")
    else:
        for idx, item in enumerate(report["near_misses"], start=1):
            lines.extend(
                [
                    f"### {idx}. result `{item['result_id']}`",
                    f"- ensemble_score: {item['ensemble_score']}",
                    f"- gbm_score: {item['gbm_score']}",
                    f"- graph_score: {item['graph_score']}",
                    f"- duplicate_group_size: {item['dup_group_size']}",
                    f"- duplicate_group_s1_rate: {item['dup_group_s1_rate']:.3f}",
                    f"- templates: {', '.join(item['templates_used'][:6]) or 'none'}",
                    f"- motifs: {', '.join(item['motifs_used'][:6]) or 'none'}",
                    f"- ops: {', '.join(item['ops'][:12]) or 'none'}",
                    "",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export label-quality triage queues")
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB)
    parser.add_argument("--report-dir", type=Path, default=_DEFAULT_REPORT_DIR)
    parser.add_argument("--top-ambiguous", type=int, default=50)
    parser.add_argument("--top-near-miss", type=int, default=100)
    parser.add_argument(
        "--profiling-db",
        type=str,
        default="research/profiling/component_profiles.db",
    )
    args = parser.parse_args()

    report = build_report(
        db_path=args.db,
        top_ambiguous=args.top_ambiguous,
        top_near_miss=args.top_near_miss,
        profiling_db=args.profiling_db,
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.report_dir / f"label_triage_{stamp}.json"
    md_path = args.report_dir / f"label_triage_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_format_md(report), encoding="utf-8")
    print(f"[label_triage] JSON report: {json_path}")
    print(f"[label_triage] Markdown report: {md_path}")


if __name__ == "__main__":
    main()
