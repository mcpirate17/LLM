"""Audit template attribution from empirical generation stats.

The report separates global template evidence, slot/class rescue evidence, and
co-occurring op evidence so steering failures are easier to diagnose.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.synthesis.grammar_support import (
    DBOpWeightCache,
    DBTemplateWeightCache,
    _capability_score,
)
from research.synthesis.templates import DEFAULT_TEMPLATE_WEIGHTS

REPORT_DIR = Path("research/reports")
METRIC_COLUMNS = (
    "avg_induction_screening_auc",
    "avg_binding_screening_auc",
    "avg_binding_screening_composite",
    "avg_ar_legacy_auc",
    "avg_hellaswag_acc",
    "avg_blimp_overall_accuracy",
    "avg_induction_intermediate_auc",
    "avg_binding_intermediate_auc",
    "math_space_rate",
)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bounded(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})")}


def _metric_select(columns: set[str]) -> str:
    parts = []
    for column in METRIC_COLUMNS:
        parts.append(column if column in columns else f"NULL AS {column}")
    return ", ".join(parts)


def _capability_from_values(values: list[Any]) -> float:
    return _capability_score(*values)


def _capability_from_slot_payload(payload: dict[str, Any]) -> float:
    return _capability_score(
        payload.get("mean_induction_screening_auc"),
        payload.get("mean_binding_screening_auc"),
        payload.get("mean_binding_screening_composite"),
        payload.get("mean_ar_legacy_auc"),
        payload.get("mean_hellaswag_acc"),
        payload.get("mean_blimp_overall_accuracy"),
        payload.get("mean_induction_intermediate_auc"),
        payload.get("mean_binding_intermediate_auc"),
        payload.get("math_space_rate"),
    )


def _support_confidence(n: int, prior: float = 8.0) -> float:
    n = max(int(n or 0), 0)
    return n / max(n + prior, 1.0)


def _decode_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_slot_context(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    columns = _columns(conn, "slot_stats")
    rows = conn.execute(
        f"""SELECT slot_key, template_name, slot_index, slot_classes, eval_count,
                   s1_pass_count, mean_loss, {_metric_select(columns)},
                   class_outcomes, wildcard_class_outcomes
            FROM slot_stats
            WHERE eval_count >= 1"""
    ).fetchall()
    out: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "slots": [],
            "slot_count": 0,
            "mean_slot_capability": 0.0,
            "mean_supported_class_capability": 0.0,
            "best_supported_class": None,
            "supported_rescue_count": 0,
            "weak_slot_fraction": 0.0,
        }
    )
    for row in rows:
        slot_key = str(row["slot_key"] or "")
        template_name = str(row["template_name"] or "")
        if not template_name:
            continue
        eval_count = int(row["eval_count"] or 0)
        s1_rate = float(row["s1_pass_count"] or 0) / max(eval_count, 1)
        metric_values = [row[column] for column in METRIC_COLUMNS]
        slot_capability = _capability_from_values(metric_values)
        best_class: dict[str, Any] | None = None
        best_supported_score = slot_capability
        for payload_column in ("class_outcomes", "wildcard_class_outcomes"):
            for class_name, vals in _decode_json_dict(row[payload_column]).items():
                if not isinstance(vals, dict):
                    continue
                n = int(vals.get("n") or 0)
                if n < 3:
                    continue
                raw_capability = _capability_from_slot_payload(vals)
                supported_score = raw_capability * _support_confidence(n)
                if supported_score > best_supported_score:
                    best_supported_score = supported_score
                    best_class = {
                        "slot_key": slot_key,
                        "class_name": str(class_name),
                        "n": n,
                        "s1_rate": float(vals.get("s1") or 0) / max(n, 1),
                        "raw_capability": raw_capability,
                        "supported_capability": supported_score,
                        "rescue_gap": supported_score - slot_capability,
                    }
        ctx = out[template_name]
        ctx["slots"].append(
            {
                "slot_key": slot_key,
                "slot_index": int(row["slot_index"] or 0),
                "slot_classes": _decode_json_dict(row["slot_classes"])
                if isinstance(row["slot_classes"], dict)
                else row["slot_classes"],
                "eval_count": eval_count,
                "s1_rate": s1_rate,
                "mean_loss": _float_or_none(row["mean_loss"]),
                "slot_capability": slot_capability,
                "best_supported_class": best_class,
                "best_supported_capability": best_supported_score,
            }
        )

    for template_name, ctx in out.items():
        slots = list(ctx["slots"])
        slot_caps = [float(slot["slot_capability"]) for slot in slots]
        class_caps = [float(slot["best_supported_capability"]) for slot in slots]
        rescue_slots = [
            slot
            for slot in slots
            if float(slot["best_supported_capability"])
            > float(slot["slot_capability"]) + 0.05
        ]
        best_slot = max(
            slots,
            key=lambda item: float(item.get("best_supported_capability") or 0.0),
            default=None,
        )
        ctx["slot_count"] = len(slots)
        ctx["mean_slot_capability"] = (
            sum(slot_caps) / len(slot_caps) if slot_caps else 0.0
        )
        ctx["mean_supported_class_capability"] = (
            sum(class_caps) / len(class_caps) if class_caps else 0.0
        )
        ctx["supported_rescue_count"] = len(rescue_slots)
        ctx["weak_slot_fraction"] = sum(
            1 for score in slot_caps if score <= 0.15
        ) / max(len(slot_caps), 1)
        ctx["best_supported_class"] = (
            best_slot.get("best_supported_class") if best_slot else None
        )
        ctx["top_rescue_slots"] = sorted(
            rescue_slots,
            key=lambda item: float(item.get("best_supported_capability") or 0.0),
            reverse=True,
        )[:5]
    return dict(out)


def _parse_graph_template_ops(graph_json: str) -> tuple[list[str], list[str]]:
    try:
        graph = json.loads(graph_json)
    except (TypeError, json.JSONDecodeError):
        return [], []
    if not isinstance(graph, dict):
        return [], []
    metadata = graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {}
    templates = [
        str(item)
        for item in (metadata.get("templates_used") or [])
        if str(item).strip()
    ]
    ops: list[str] = []
    nodes = graph.get("nodes") or {}
    node_iter = nodes.values() if isinstance(nodes, dict) else nodes
    for node in node_iter:
        if not isinstance(node, dict):
            continue
        op_name = str(node.get("op_name") or "").strip()
        if op_name and op_name != "input":
            ops.append(op_name)
    return templates, ops


def _load_template_op_context(
    conn: sqlite3.Connection,
    db_path: str,
    op_weights: dict[str, float],
) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """SELECT graph_json
           FROM program_results_compat
           WHERE TRIM(COALESCE(graph_json, '')) <> ''
             AND graph_json <> '{}'"""
    ).fetchall()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        graph_json = resolve_graph_json_value(
            conn,
            db_path,
            row["graph_json"],
        )
        templates, ops = _parse_graph_template_ops(graph_json)
        if not templates or not ops:
            continue
        unique_ops = set(ops)
        for template_name in templates:
            counts[template_name].update(unique_ops)

    out: dict[str, dict[str, Any]] = {}
    for template_name, counter in counts.items():
        top_ops = [
            {
                "op_name": op_name,
                "count": count,
                "db_weight": float(op_weights.get(op_name, 1.0)),
            }
            for op_name, count in counter.most_common(8)
        ]
        weighted_counts = sum(item["count"] for item in top_ops)
        avg_weight = sum(item["count"] * item["db_weight"] for item in top_ops) / max(
            weighted_counts, 1
        )
        out[template_name] = {
            "top_ops": top_ops,
            "top_op_weight_mean": avg_weight,
        }
    return out


def _label_attribution(
    *,
    eval_count: int,
    weight_ratio: float,
    capability: float,
    slot_ctx: dict[str, Any],
) -> str:
    if eval_count < 10:
        return "low_support"
    mean_supported = float(slot_ctx.get("mean_supported_class_capability") or 0.0)
    best_class = slot_ctx.get("best_supported_class") or {}
    best_supported = float(best_class.get("supported_capability") or 0.0)
    if best_supported > capability + 0.10 and mean_supported > capability + 0.03:
        return "slot_rescued"
    if capability >= 0.35 and weight_ratio >= 1.0:
        return "global_positive"
    if weight_ratio < 0.80 and best_supported <= capability + 0.05:
        return "globally_weak"
    if weight_ratio > 1.20 and capability < 0.20:
        return "loss_or_s1_driven"
    return "mixed"


def build_attribution_audit(
    db_path: str,
    *,
    limit: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        template_columns = _columns(conn, "template_stats")
        rows = conn.execute(
            f"""SELECT template_name, eval_count, s0_pass_count, s1_pass_count,
                       mean_loss, min_loss, mean_novelty, {_metric_select(template_columns)}
                FROM template_stats
                WHERE eval_count >= 1"""
        ).fetchall()
        template_weights = DBTemplateWeightCache(ttl=0.0).get(db_path) or {}
        op_weights = DBOpWeightCache(ttl=0.0).get(db_path) or {}
        slot_context = _load_slot_context(conn)
        op_context = _load_template_op_context(conn, db_path, op_weights)
    finally:
        conn.close()

    report_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    for row in rows:
        template_name = str(row["template_name"] or "")
        eval_count = int(row["eval_count"] or 0)
        s1_rate = float(row["s1_pass_count"] or 0) / max(eval_count, 1)
        capability = _capability_from_values([row[column] for column in METRIC_COLUMNS])
        default_weight = float(DEFAULT_TEMPLATE_WEIGHTS.get(template_name, 1.0) or 1.0)
        db_weight = float(template_weights.get(template_name, default_weight))
        weight_ratio = db_weight / max(default_weight, 0.01)
        slot_ctx = slot_context.get(template_name) or {}
        label = _label_attribution(
            eval_count=eval_count,
            weight_ratio=weight_ratio,
            capability=capability,
            slot_ctx=slot_ctx,
        )
        risks: list[str] = []
        best_class = slot_ctx.get("best_supported_class") or {}
        best_supported = float(best_class.get("supported_capability") or 0.0)
        mean_supported = float(slot_ctx.get("mean_supported_class_capability") or 0.0)
        if best_supported > capability + 0.15 and mean_supported <= capability + 0.02:
            risks.append("single_slot_over_rescue")
        if weight_ratio < 0.75 and best_supported > capability + 0.10:
            risks.append("broad_template_penalty_may_hide_slot_signal")
        if weight_ratio > 1.30 and capability < 0.15:
            risks.append("weight_boost_not_capability_driven")
        if eval_count < 10:
            risks.append("low_template_support")
        for risk in risks:
            risk_counts[risk] += 1
        label_counts[label] += 1
        report_rows.append(
            {
                "template_name": template_name,
                "attribution_label": label,
                "risks": risks,
                "eval_count": eval_count,
                "s1_rate": s1_rate,
                "mean_loss": _float_or_none(row["mean_loss"]),
                "mean_novelty": _float_or_none(row["mean_novelty"]),
                "capability_score": capability,
                "default_weight": default_weight,
                "db_weight": db_weight,
                "weight_ratio": weight_ratio,
                "slot_count": int(slot_ctx.get("slot_count") or 0),
                "mean_slot_capability": float(
                    slot_ctx.get("mean_slot_capability") or 0.0
                ),
                "mean_supported_class_capability": mean_supported,
                "weak_slot_fraction": float(slot_ctx.get("weak_slot_fraction") or 0.0),
                "supported_rescue_count": int(
                    slot_ctx.get("supported_rescue_count") or 0
                ),
                "best_supported_class": best_class or None,
                "top_rescue_slots": slot_ctx.get("top_rescue_slots") or [],
                "op_context": op_context.get(template_name) or {},
            }
        )

    report_rows.sort(
        key=lambda item: (
            len(item.get("risks") or []),
            abs(float(item.get("weight_ratio") or 1.0) - 1.0),
            float(item.get("eval_count") or 0),
        ),
        reverse=True,
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "db_path": db_path,
        "n_templates": len(report_rows),
        "limit": int(limit),
        "label_counts": dict(label_counts),
        "risk_counts": dict(risk_counts),
        "uses_screening_ensemble_gate": False,
        "uses_learned_generation_influence": False,
    }
    return report_rows[:limit], summary


def write_reports(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    output_prefix: Path,
) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    jsonl_path = output_prefix.with_suffix(".jsonl")
    json_path.write_text(
        json.dumps({"summary": summary, "templates": rows}, indent=2),
        encoding="utf-8",
    )
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return json_path, jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="research/runs.db")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Output path without suffix. Defaults to research/reports/generation_attribution_audit_YYYY-MM-DD.",
    )
    args = parser.parse_args()

    rows, summary = build_attribution_audit(args.db, limit=max(1, int(args.limit)))
    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else REPORT_DIR
        / f"generation_attribution_audit_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    )
    json_path, jsonl_path = write_reports(rows, summary, output_prefix=prefix)
    print(f"Wrote {len(rows)} template attribution rows")
    print(f"JSON:  {json_path}")
    print(f"JSONL: {jsonl_path}")
    for row in rows[:10]:
        print(
            f"{row['template_name']}: label={row['attribution_label']} "
            f"ratio={float(row['weight_ratio']):.2f} "
            f"cap={float(row['capability_score']):.3f} "
            f"risks={','.join(row.get('risks') or []) or '-'}"
        )


if __name__ == "__main__":
    main()
