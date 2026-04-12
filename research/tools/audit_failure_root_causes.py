from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.scientist.runner.failure_provenance import (
    infer_graph_failure_provenance,
)
from research.synthesis.graph import ComputationGraph

TARGET_LABELS = (
    "insufficient_learning",
    "RuntimeError",
    "rapid_screening_error",
    "forward_error",
    "unstable_dynamics",
    "inflight_no_progress",
    "causality_violation",
    "failed_convergence",
    "cuda_fatal",
)

_ROI_ROOT_CAUSES = frozenset(
    {
        "dtype_mismatch",
        "shape_or_residual_mismatch",
        "selective_scan_contract",
        "token_merge_contract",
        "hybrid_routing_assembly",
        "causality_contract",
        "routing_telemetry_state_mismatch",
        "stale_routing_bias_state",
        "residual_dominant_no_learning",
    }
)


@dataclass(slots=True)
class FailureRow:
    raw_error_type: str
    error_type: str
    error_message: str
    graph: ComputationGraph


def _normalize_label(error_type: str) -> str:
    normalized = str(error_type or "unknown").strip()
    if normalized.startswith("s1_"):
        normalized = normalized[3:]
    return normalized


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit graph-build and screening root causes"
    )
    p.add_argument("--db", default="research/lab_notebook.db")
    p.add_argument(
        "--markdown-out", default="research/reports/failure_root_cause_audit.md"
    )
    p.add_argument(
        "--json-out", default="research/reports/failure_root_cause_audit.json"
    )
    return p.parse_args()


def _load_graph(graph_json: str) -> ComputationGraph | None:
    if not graph_json or not graph_json.strip():
        return None
    try:
        payload = json.loads(graph_json)
    except Exception:
        return None
    try:
        return ComputationGraph.from_dict(payload)
    except Exception:
        return None


def _load_rows(db_path: str) -> list[FailureRow]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT error_type, COALESCE(error_message, '') AS error_message, graph_json
        FROM program_results
        WHERE graph_json IS NOT NULL
          AND TRIM(graph_json) != ''
        """
    ).fetchall()
    loaded: list[FailureRow] = []
    for row in rows:
        graph = _load_graph(row["graph_json"])
        if graph is None:
            continue
        raw_error_type = str(row["error_type"] or "")
        error_type = _normalize_label(raw_error_type)
        if error_type not in TARGET_LABELS:
            continue
        loaded.append(
            FailureRow(
                raw_error_type=raw_error_type,
                error_type=error_type,
                error_message=str(row["error_message"] or ""),
                graph=graph,
            )
        )
    return loaded


def _cluster_row(row: FailureRow) -> dict[str, Any]:
    provenance = infer_graph_failure_provenance(
        row.graph,
        error_type=row.raw_error_type,
        error_message=row.error_message,
    )
    details = json.loads(provenance["failure_details_json"])
    return {
        "label": row.error_type,
        "root_cause": str(details.get("root_cause_code") or row.error_type),
        "failure_op": provenance.get("failure_op"),
        "source_op": details.get("source_op"),
        "validator_errors": list(details.get("validator_errors") or []),
        "dim_flow_errors": list(details.get("dim_flow_errors") or []),
        "error_message": row.error_message,
    }


def _severity(count: int) -> str:
    if count >= 200:
        return "high"
    if count >= 50:
        return "medium"
    return "low"


def _roi(root_cause: str, count: int) -> str:
    if root_cause in _ROI_ROOT_CAUSES:
        return "high"
    return _severity(count)


def _build_summary(rows: list[FailureRow]) -> dict[str, Any]:
    before_counts = Counter(row.error_type for row in rows)
    root_counts = Counter()
    source_counts = Counter()
    victim_counts = Counter()
    prevented_after = Counter()
    cluster_examples: dict[str, list[str]] = defaultdict(list)
    source_victim_pairs = Counter()
    source_victim_pairs_by_root: dict[str, Counter] = defaultdict(Counter)
    false_blame = Counter()

    for row in rows:
        clustered = _cluster_row(row)
        root = clustered["root_cause"]
        root_counts[root] += 1
        failure_op = clustered["failure_op"]
        source_op = clustered["source_op"]
        if source_op:
            source_counts[source_op] += 1
        if failure_op:
            victim_counts[failure_op] += 1
        if source_op and failure_op:
            source_victim_pairs[(str(source_op), str(failure_op))] += 1
            source_victim_pairs_by_root[root][(str(source_op), str(failure_op))] += 1
            if source_op != failure_op:
                false_blame[failure_op] += 1
        trigger = (
            clustered["validator_errors"][:1]
            or clustered["dim_flow_errors"][:1]
            or [clustered["error_message"][:160]]
        )[0]
        if len(cluster_examples[root]) < 3 and trigger:
            cluster_examples[root].append(trigger)
        if clustered["validator_errors"] or clustered["dim_flow_errors"]:
            prevented_after[row.error_type] += 1

    after_counts = {
        label: max(0, before_counts.get(label, 0) - prevented_after.get(label, 0))
        for label in TARGET_LABELS
    }
    ranked_root_causes = []
    for root_cause, count in root_counts.most_common():
        top_pair = None
        if source_victim_pairs_by_root[root_cause]:
            (source, victim), pair_count = source_victim_pairs_by_root[
                root_cause
            ].most_common(1)[0]
            top_pair = {
                "source_op": source,
                "victim_op": victim,
                "count": pair_count,
            }
        ranked_root_causes.append(
            {
                "root_cause": root_cause,
                "count": count,
                "severity": _severity(count),
                "roi": _roi(root_cause, count),
                "example_triggers": cluster_examples.get(root_cause, []),
                "top_source_op": top_pair["source_op"] if top_pair else None,
                "top_victim_op": top_pair["victim_op"] if top_pair else None,
            }
        )

    return {
        "labels": list(TARGET_LABELS),
        "before_counts": {
            label: before_counts.get(label, 0) for label in TARGET_LABELS
        },
        "after_counts_preflight_replay": after_counts,
        "prevented_by_current_validation": {
            label: prevented_after.get(label, 0) for label in TARGET_LABELS
        },
        "ranked_root_causes": ranked_root_causes,
        "top_source_ops": source_counts.most_common(15),
        "top_terminal_victims": victim_counts.most_common(15),
        "frequently_falsely_blamed_terminal_ops": false_blame.most_common(15),
        "top_source_victim_pairs": [
            {
                "source_op": source,
                "victim_op": victim,
                "count": count,
            }
            for (source, victim), count in source_victim_pairs.most_common(15)
        ],
        "notes": [
            "after_counts_preflight_replay is a counterfactual replay over the historical notebook corpus using current graph validation",
            "root causes are normalized across raw and s1_* labels",
            "terminal victims are not treated as root causes when upstream source_op differs",
        ],
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Failure Root-Cause Audit",
        "",
        "## Before / After Failure Counts",
        "",
        "| Label | Before | Prevented By Preflight Replay | After |",
        "| --- | ---: | ---: | ---: |",
    ]
    before = summary["before_counts"]
    prevented = summary["prevented_by_current_validation"]
    after = summary["after_counts_preflight_replay"]
    for label in TARGET_LABELS:
        lines.append(
            f"| {label} | {before.get(label, 0)} | {prevented.get(label, 0)} | {after.get(label, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Ranked Root Causes",
            "",
            "| Root Cause | Count | Severity | ROI | Source Op | Terminal Victim | Example Trigger |",
            "| --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for item in summary["ranked_root_causes"]:
        example = "; ".join(item["example_triggers"][:1])
        lines.append(
            "| "
            f"{item['root_cause']} | {item['count']} | {item['severity']} | {item['roi']} | "
            f"{item.get('top_source_op') or ''} | {item.get('top_victim_op') or ''} | {example} |"
        )

    lines.extend(["", "## Source vs Victim Ops", ""])
    for pair in summary["top_source_victim_pairs"]:
        lines.append(
            f"- `{pair['source_op']}` -> `{pair['victim_op']}`: {pair['count']}"
        )

    lines.extend(["", "## Frequently Falsely Blamed Terminal Ops", ""])
    for op_name, count in summary["frequently_falsely_blamed_terminal_ops"]:
        lines.append(f"- `{op_name}`: {count}")

    lines.extend(["", "## Notes", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    rows = _load_rows(args.db)
    summary = _build_summary(rows)
    md_path = Path(args.markdown_out)
    json_path = Path(args.json_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(md_path, summary)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
