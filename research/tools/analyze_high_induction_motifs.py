#!/usr/bin/env python3
"""Analyze local graph motifs behind high-induction ablation effects.

This is an evidence report, not a policy update. It joins recent ablation
child observations to their parent graphs, extracts local path context around
the edited node, and summarizes which non-attention motifs protect or preserve
induction.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.synthesis.serializer import graph_from_json  # noqa: E402


DB_PATH = PROJECT_ROOT / "research/lab_notebook.db"
RUNTIME_DIR = PROJECT_ROOT / "research/runtime"

ATTENTION_TOKENS = (
    "attention",
    "softmax_attention",
    "linear_attention",
    "diff_attention",
    "graph_attention",
    "latent_attention",
)
ROUTING_TOKENS = ("gate", "route", "routing", "topk", "gather", "token_type")
NORM_TOKENS = ("norm", "rmsnorm", "layernorm")
PROJECTION_TOKENS = ("proj", "linear", "matmul", "basis")
SEQUENCE_TOKENS = ("conv", "rope", "merge", "spectral", "scan", "state_space")


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(child: Any, parent: Any) -> float | None:
    child_f = _float(child)
    parent_f = _float(parent)
    if child_f is None or parent_f is None:
        return None
    return child_f - parent_f


def _advantage(parent: Any, child: Any) -> float | None:
    parent_f = _float(parent)
    child_f = _float(child)
    if parent_f is None or child_f is None:
        return None
    return parent_f - child_f


def _pct_advantage(parent: Any, child: Any) -> float | None:
    parent_f = _float(parent)
    child_f = _float(child)
    if parent_f is None or child_f is None or parent_f <= 0:
        return None
    return (parent_f - child_f) / parent_f


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _contains_token(*values: Any, tokens: tuple[str, ...]) -> bool:
    text = " ".join(str(value or "").lower() for value in values)
    return any(token in text for token in tokens)


def _classify_op(op_name: str) -> str:
    name = str(op_name or "").lower()
    if _contains_token(name, tokens=ATTENTION_TOKENS):
        return "attention"
    if _contains_token(name, tokens=NORM_TOKENS):
        return "normalization"
    if _contains_token(name, tokens=ROUTING_TOKENS):
        return "routing"
    if _contains_token(name, tokens=PROJECTION_TOKENS):
        return "projection"
    if _contains_token(name, tokens=SEQUENCE_TOKENS):
        return "sequence"
    if name in {"add", "mul", "sub", "maximum", "minimum"}:
        return "merge"
    if name in {"gelu", "relu", "silu", "sigmoid", "tanh", "swiglu_mlp"}:
        return "activation"
    return "other"


def _node_id_from_observation(
    row: sqlite3.Row, provenance: dict[str, Any]
) -> int | None:
    raw = provenance.get("node_id")
    if raw is None:
        parts = str(row["rule_key"] or "").split(":")
        if parts:
            raw = (
                parts[1]
                if row["rule_type"] == "component_replace" and len(parts) > 1
                else parts[0]
            )
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _ops_for_ids(graph: Any, ids: list[int]) -> list[str]:
    out: list[str] = []
    for node_id in ids:
        node = graph.nodes.get(node_id)
        if node is not None and not node.is_input:
            out.append(str(node.op_name))
    return sorted(out)


def _input_ops(graph: Any, node_id: int) -> list[str]:
    node = graph.nodes.get(node_id)
    if node is None:
        return []
    return _ops_for_ids(graph, list(node.input_ids))


def _consumer_ops(graph: Any, node_id: int) -> list[str]:
    return _ops_for_ids(graph, graph.children_map().get(node_id, []))


def _two_hop_input_ops(graph: Any, node_id: int) -> list[str]:
    node = graph.nodes.get(node_id)
    if node is None:
        return []
    ids: list[int] = []
    for input_id in node.input_ids:
        parent = graph.nodes.get(input_id)
        if parent is not None:
            ids.extend(parent.input_ids)
    return _ops_for_ids(graph, ids)


def _two_hop_consumer_ops(graph: Any, node_id: int) -> list[str]:
    ids: list[int] = []
    children = graph.children_map()
    for child_id in children.get(node_id, []):
        ids.extend(children.get(child_id, []))
    return _ops_for_ids(graph, ids)


def _position_bucket(graph: Any, node_id: int) -> str:
    order = [
        nid
        for nid in graph.topological_order()
        if nid in graph.nodes and not graph.nodes[nid].is_input
    ]
    if not order or node_id not in order:
        return "unknown"
    frac = (order.index(node_id) + 1) / max(1, len(order))
    if frac <= 0.33:
        return "early"
    if frac <= 0.66:
        return "middle"
    return "late"


def _nearest_upstream(graph: Any, node_id: int, predicate) -> str | None:
    seen: set[int] = set()
    stack = list(getattr(graph.nodes.get(node_id), "input_ids", []) or [])
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        node = graph.nodes.get(current)
        if node is None:
            continue
        if not node.is_input and predicate(str(node.op_name)):
            return str(node.op_name)
        stack.extend(node.input_ids)
    return None


def _nearest_downstream(graph: Any, node_id: int, predicate) -> str | None:
    children = graph.children_map()
    seen: set[int] = set()
    stack = list(children.get(node_id, []))
    while stack:
        current = stack.pop(0)
        if current in seen:
            continue
        seen.add(current)
        node = graph.nodes.get(current)
        if node is None:
            continue
        if not node.is_input and predicate(str(node.op_name)):
            return str(node.op_name)
        stack.extend(children.get(current, []))
    return None


def _local_context(graph: Any, node_id: int | None) -> dict[str, Any]:
    if node_id is None or node_id not in graph.nodes:
        return {
            "node_id": node_id,
            "node_op": "",
            "position_bucket": "unknown",
            "signature": "unknown",
        }
    node = graph.nodes[node_id]
    input_ops = _input_ops(graph, node_id)
    consumer_ops = _consumer_ops(graph, node_id)
    two_in = _two_hop_input_ops(graph, node_id)
    two_out = _two_hop_consumer_ops(graph, node_id)
    upstream_norm = _nearest_upstream(
        graph, node_id, lambda name: _contains_token(name, tokens=NORM_TOKENS)
    )
    upstream_routing = _nearest_upstream(
        graph, node_id, lambda name: _contains_token(name, tokens=ROUTING_TOKENS)
    )
    downstream_merge = _nearest_downstream(
        graph, node_id, lambda name: name in {"add", "mul", "sub"}
    )
    downstream_projection = _nearest_downstream(
        graph, node_id, lambda name: _contains_token(name, tokens=PROJECTION_TOKENS)
    )
    signature = (
        f"{'+'.join(two_in or ['input'])}>"
        f"{'+'.join(input_ops or ['input'])}>"
        f"{node.op_name}>"
        f"{'+'.join(consumer_ops or ['output'])}>"
        f"{'+'.join(two_out or ['output'])}"
    )
    return {
        "node_id": node_id,
        "node_op": str(node.op_name),
        "node_class": _classify_op(str(node.op_name)),
        "depth": int(node.depth),
        "position_bucket": _position_bucket(graph, node_id),
        "input_ops": input_ops,
        "consumer_ops": consumer_ops,
        "two_hop_input_ops": two_in,
        "two_hop_consumer_ops": two_out,
        "upstream_norm": upstream_norm,
        "upstream_routing": upstream_routing,
        "downstream_merge": downstream_merge,
        "downstream_projection": downstream_projection,
        "signature": signature,
        "role_signature": (
            f"{_classify_op(str(node.op_name))}|"
            f"pos={_position_bucket(graph, node_id)}|"
            f"in={'+'.join(_classify_op(op) for op in input_ops) or 'input'}|"
            f"out={'+'.join(_classify_op(op) for op in consumer_ops) or 'output'}|"
            f"up_norm={upstream_norm or 'none'}|"
            f"up_route={upstream_routing or 'none'}|"
            f"down_merge={downstream_merge or 'none'}"
        ),
    }


def _fetch_rows(
    conn: sqlite3.Connection, parent_ids: list[str], lookback_hours: int
) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in parent_ids)
    return conn.execute(
        f"""
        SELECT ev.evidence_id, ev.timestamp, ev.rule_type, ev.rule_key, ev.outcome,
               obs.provenance_json,
               pp.result_id AS parent_result_id,
               pp.graph_fingerprint AS parent_fingerprint,
               pp.graph_json AS parent_graph_json,
               pp.loss_ratio AS parent_loss,
               pp.wikitext_perplexity AS parent_ppl,
               pp.hellaswag_acc AS parent_hellaswag,
               pp.blimp_overall_accuracy AS parent_blimp,
               pp.induction_auc AS parent_induction,
               pp.binding_composite AS parent_binding,
               pp.ar_auc AS parent_ar,
               cp.result_id AS child_result_id,
               cp.loss_ratio AS child_loss,
               cp.wikitext_perplexity AS child_ppl,
               cp.hellaswag_acc AS child_hellaswag,
               cp.blimp_overall_accuracy AS child_blimp,
               cp.induction_auc AS child_induction,
               cp.binding_composite AS child_binding,
               cp.ar_auc AS child_ar
        FROM causal_rule_evidence ev
        JOIN causal_ablation_child_observations obs ON obs.evidence_id = ev.evidence_id
        JOIN program_results pp ON pp.result_id = obs.parent_result_id
        JOIN program_results cp ON cp.result_id = obs.child_result_id
        WHERE ev.parent_result_id IN ({placeholders})
          AND ev.timestamp > strftime('%s', 'now') - ?
        ORDER BY ev.timestamp ASC, ev.evidence_id ASC
        """,
        tuple(parent_ids) + (max(1, int(lookback_hours)) * 3600,),
    ).fetchall()


def _default_parent_ids(conn: sqlite3.Connection, lookback_hours: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT ev.parent_result_id, pp.induction_auc
        FROM causal_rule_evidence ev
        JOIN program_results pp ON pp.result_id = ev.parent_result_id
        WHERE ev.timestamp > strftime('%s', 'now') - ?
          AND pp.induction_auc IS NOT NULL
        ORDER BY pp.induction_auc DESC
        LIMIT 20
        """,
        (max(1, int(lookback_hours)) * 3600,),
    ).fetchall()
    return [str(row["parent_result_id"]) for row in rows if row["parent_result_id"]]


def _observation_from_row(
    row: sqlite3.Row, graph_cache: dict[str, Any]
) -> dict[str, Any] | None:
    provenance = json.loads(row["provenance_json"] or "{}")
    op_name = (
        provenance.get("deleted_op")
        or provenance.get("original_op")
        or str(row["rule_key"] or "").split(":")[-1]
    )
    replacement = provenance.get("replacement_op")
    component = provenance.get("component_class") or (
        "node_delete" if row["rule_type"] == "node_delete" else ""
    )
    graph = graph_cache.get(row["parent_result_id"])
    if graph is None:
        graph = graph_from_json(str(row["parent_graph_json"]))
        graph_cache[str(row["parent_result_id"])] = graph
    node_id = _node_id_from_observation(row, provenance)
    context = _local_context(graph, node_id)
    return {
        "evidence_id": str(row["evidence_id"]),
        "rule_type": str(row["rule_type"]),
        "rule_key": str(row["rule_key"]),
        "outcome": str(row["outcome"]),
        "parent_result_id": str(row["parent_result_id"]),
        "parent_fingerprint": str(row["parent_fingerprint"]),
        "child_result_id": str(row["child_result_id"]),
        "node_id": node_id,
        "op": str(op_name or context.get("node_op") or ""),
        "replacement_op": str(replacement) if replacement else None,
        "component_class": str(component or context.get("node_class") or ""),
        "is_attention_related": _contains_token(
            op_name,
            replacement,
            component,
            row["rule_key"],
            tokens=ATTENTION_TOKENS,
        ),
        "induction_delta": _delta(row["child_induction"], row["parent_induction"]),
        "binding_delta": _delta(row["child_binding"], row["parent_binding"]),
        "loss_advantage": _advantage(row["parent_loss"], row["child_loss"]),
        "ppl_advantage_pct": _pct_advantage(row["parent_ppl"], row["child_ppl"]),
        "hellaswag_delta": _delta(row["child_hellaswag"], row["parent_hellaswag"]),
        "blimp_delta": _delta(row["child_blimp"], row["parent_blimp"]),
        "ar_delta": _delta(row["child_ar"], row["parent_ar"]),
        "parent_induction": _float(row["parent_induction"]),
        "child_induction": _float(row["child_induction"]),
        "parent_binding": _float(row["parent_binding"]),
        "child_binding": _float(row["child_binding"]),
        "context": context,
    }


def _context_is_attention_related(obs: dict[str, Any]) -> bool:
    context = obs.get("context") or {}
    values: list[Any] = [
        context.get("signature"),
        context.get("role_signature"),
        context.get("upstream_norm"),
        context.get("upstream_routing"),
        context.get("downstream_merge"),
        context.get("downstream_projection"),
    ]
    for key in (
        "input_ops",
        "consumer_ops",
        "two_hop_input_ops",
        "two_hop_consumer_ops",
    ):
        values.extend(context.get(key) or [])
    return _contains_token(*values, tokens=ATTENTION_TOKENS)


def _aggregate(observations: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        groups[tuple(key_fn(obs))].append(obs)
    rows: list[dict[str, Any]] = []
    for key, items in groups.items():
        rows.append(
            {
                "key": key,
                "n": len(items),
                "parents": len({item["parent_result_id"] for item in items}),
                "avg_induction_delta": _mean(
                    [item["induction_delta"] for item in items]
                ),
                "avg_binding_delta": _mean([item["binding_delta"] for item in items]),
                "avg_loss_advantage": _mean([item["loss_advantage"] for item in items]),
                "avg_ppl_advantage_pct": _mean(
                    [item["ppl_advantage_pct"] for item in items]
                ),
                "induction_drop_gt_0_25": sum(
                    1 for item in items if (item["induction_delta"] or 0.0) <= -0.25
                ),
                "induction_drop_gt_0_50": sum(
                    1 for item in items if (item["induction_delta"] or 0.0) <= -0.50
                ),
                "loss_better_induction_drop_gt_0_50": sum(
                    1
                    for item in items
                    if (item["loss_advantage"] or 0.0) > 0.0
                    and (item["induction_delta"] or 0.0) <= -0.50
                ),
                "outcomes": dict(Counter(item["outcome"] for item in items)),
                "examples": [
                    {
                        "parent_result_id": item["parent_result_id"],
                        "rule_type": item["rule_type"],
                        "rule_key": item["rule_key"],
                        "induction_delta": item["induction_delta"],
                        "loss_advantage": item["loss_advantage"],
                        "context": item["context"],
                    }
                    for item in sorted(
                        items,
                        key=lambda item: (
                            item["induction_delta"]
                            if item["induction_delta"] is not None
                            else 999.0
                        ),
                    )[:3]
                ],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["avg_induction_delta"]
            if row["avg_induction_delta"] is not None
            else 999.0,
            -row["n"],
        ),
    )


def _parent_backbones(observations: list[dict[str, Any]]) -> dict[str, Any]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        by_parent[obs["parent_result_id"]].append(obs)
    out: dict[str, Any] = {}
    for parent_id, items in by_parent.items():
        critical = [
            item
            for item in items
            if (item["induction_delta"] or 0.0) <= -0.50
            and not item["is_attention_related"]
        ]
        preserving = [
            item
            for item in items
            if not item["is_attention_related"]
            and item["induction_delta"] is not None
            and item["induction_delta"] >= -0.10
        ]
        out[parent_id] = {
            "critical_non_attention_nodes": [
                {
                    "node_id": item["node_id"],
                    "op": item["op"],
                    "replacement_op": item["replacement_op"],
                    "rule_type": item["rule_type"],
                    "induction_delta": item["induction_delta"],
                    "binding_delta": item["binding_delta"],
                    "loss_advantage": item["loss_advantage"],
                    "signature": item["context"].get("signature"),
                    "role_signature": item["context"].get("role_signature"),
                }
                for item in sorted(
                    critical,
                    key=lambda item: (
                        item["induction_delta"]
                        if item["induction_delta"] is not None
                        else 999.0
                    ),
                )
            ],
            "non_attention_preservers": [
                {
                    "node_id": item["node_id"],
                    "op": item["op"],
                    "replacement_op": item["replacement_op"],
                    "rule_type": item["rule_type"],
                    "induction_delta": item["induction_delta"],
                    "binding_delta": item["binding_delta"],
                    "loss_advantage": item["loss_advantage"],
                    "signature": item["context"].get("signature"),
                }
                for item in sorted(
                    preserving,
                    key=lambda item: (
                        item["induction_delta"]
                        if item["induction_delta"] is not None
                        else -999.0
                    ),
                    reverse=True,
                )[:12]
            ],
        }
    return out


def build_report(
    conn: sqlite3.Connection,
    *,
    parent_ids: list[str],
    lookback_hours: int,
    exclude_attention: bool,
    exclude_attention_context: bool,
) -> dict[str, Any]:
    graph_cache: dict[str, Any] = {}
    observations = [
        obs
        for row in _fetch_rows(conn, parent_ids, lookback_hours)
        if (obs := _observation_from_row(row, graph_cache)) is not None
    ]
    scope = [
        obs
        for obs in observations
        if not (exclude_attention and obs["is_attention_related"])
        and not (exclude_attention_context and _context_is_attention_related(obs))
    ]
    return {
        "created_at": time.time(),
        "parent_ids": parent_ids,
        "lookback_hours": int(lookback_hours),
        "exclude_attention": bool(exclude_attention),
        "exclude_attention_context": bool(exclude_attention_context),
        "observation_count": len(scope),
        "all_observation_count": len(observations),
        "summary": {
            "outcomes": dict(Counter(obs["outcome"] for obs in scope)),
            "rule_types": dict(Counter(obs["rule_type"] for obs in scope)),
            "mean_induction_delta": _mean([obs["induction_delta"] for obs in scope]),
            "mean_binding_delta": _mean([obs["binding_delta"] for obs in scope]),
            "mean_loss_advantage": _mean([obs["loss_advantage"] for obs in scope]),
            "mean_ppl_advantage_pct": _mean(
                [obs["ppl_advantage_pct"] for obs in scope]
            ),
            "induction_drop_gt_0_25": sum(
                1 for obs in scope if (obs["induction_delta"] or 0.0) <= -0.25
            ),
            "induction_drop_gt_0_50": sum(
                1 for obs in scope if (obs["induction_delta"] or 0.0) <= -0.50
            ),
            "loss_better_induction_drop_gt_0_50": sum(
                1
                for obs in scope
                if (obs["loss_advantage"] or 0.0) > 0.0
                and (obs["induction_delta"] or 0.0) <= -0.50
            ),
        },
        "by_op_edit": _aggregate(
            scope,
            lambda obs: (
                obs["rule_type"],
                obs["component_class"],
                obs["op"],
                obs["replacement_op"] or "",
            ),
        ),
        "by_role_signature": _aggregate(
            scope,
            lambda obs: (obs["context"].get("role_signature") or "unknown",),
        ),
        "by_local_signature": _aggregate(
            scope,
            lambda obs: (obs["context"].get("signature") or "unknown",),
        ),
        "parent_backbones": _parent_backbones(scope),
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines: list[str] = []
    lines.append("# High-Induction Motif Analysis")
    lines.append("")
    lines.append(f"- Parents: {len(report['parent_ids'])}")
    lines.append(
        f"- Observations analyzed: {report['observation_count']} "
        f"of {report['all_observation_count']} total"
    )
    lines.append(f"- Exclude attention-like edits: {report['exclude_attention']}")
    lines.append(
        f"- Exclude local neighborhoods containing attention: "
        f"{report['exclude_attention_context']}"
    )
    lines.append(f"- Outcomes: {summary['outcomes']}")
    lines.append(f"- Rule types: {summary['rule_types']}")
    lines.append(f"- Mean induction delta: {_fmt(summary['mean_induction_delta'])}")
    lines.append(f"- Mean binding delta: {_fmt(summary['mean_binding_delta'])}")
    lines.append(f"- Mean loss advantage: {_fmt(summary['mean_loss_advantage'])}")
    lines.append(
        f"- Loss-better with induction drop > 0.50: "
        f"{summary['loss_better_induction_drop_gt_0_50']}"
    )
    lines.append("")

    lines.append("## Worst Non-Attention Edit Patterns")
    for row in report["by_op_edit"][:18]:
        rule_type, component, op, replacement = row["key"]
        label = op if not replacement else f"{op} -> {replacement}"
        lines.append(
            f"- {rule_type} {component} {label}: n={row['n']} "
            f"parents={row['parents']} avg_ind={_fmt(row['avg_induction_delta'])} "
            f"avg_bind={_fmt(row['avg_binding_delta'])} "
            f"avg_loss_adv={_fmt(row['avg_loss_advantage'])} "
            f"loss_better_bad_ind={row['loss_better_induction_drop_gt_0_50']} "
            f"outcomes={row['outcomes']}"
        )
    lines.append("")

    lines.append("## Repeated Role Signatures")
    repeated = [row for row in report["by_role_signature"] if row["n"] >= 2]
    for row in repeated[:18]:
        lines.append(
            f"- {row['key'][0]}: n={row['n']} parents={row['parents']} "
            f"avg_ind={_fmt(row['avg_induction_delta'])} "
            f"avg_bind={_fmt(row['avg_binding_delta'])} "
            f"loss_better_bad_ind={row['loss_better_induction_drop_gt_0_50']}"
        )
    lines.append("")

    lines.append("## Parent Critical Backbones")
    for parent_id, payload in report["parent_backbones"].items():
        critical = payload["critical_non_attention_nodes"]
        if not critical:
            continue
        lines.append(f"### {parent_id}")
        for item in critical[:10]:
            repl = f" -> {item['replacement_op']}" if item["replacement_op"] else ""
            lines.append(
                f"- node {item['node_id']} {item['op']}{repl} "
                f"{item['rule_type']} ind={_fmt(item['induction_delta'])} "
                f"bind={_fmt(item['binding_delta'])} "
                f"loss_adv={_fmt(item['loss_advantage'])}; "
                f"{item['signature']}"
            )
    lines.append("")

    lines.append("## Induction-Preserving Non-Attention Alternatives")
    preservers = [
        row
        for row in report["by_op_edit"]
        if row["avg_induction_delta"] is not None
        and row["avg_induction_delta"] >= -0.10
    ]
    preservers.sort(
        key=lambda row: (row["avg_induction_delta"], row["n"]), reverse=True
    )
    for row in preservers[:18]:
        rule_type, component, op, replacement = row["key"]
        label = op if not replacement else f"{op} -> {replacement}"
        lines.append(
            f"- {rule_type} {component} {label}: n={row['n']} "
            f"parents={row['parents']} avg_ind={_fmt(row['avg_induction_delta'])} "
            f"avg_bind={_fmt(row['avg_binding_delta'])} "
            f"avg_loss_adv={_fmt(row['avg_loss_advantage'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--lookback-hours", type=int, default=8)
    parser.add_argument("--parent-result-id", action="append", default=[])
    parser.add_argument("--include-attention", action="store_true")
    parser.add_argument(
        "--exclude-attention-context",
        action="store_true",
        help="Also exclude non-attention edits whose local neighborhood contains attention.",
    )
    parser.add_argument(
        "--output-json",
        default=str(RUNTIME_DIR / "high_induction_motif_analysis.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(RUNTIME_DIR / "high_induction_motif_analysis.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        parent_ids = list(dict.fromkeys(args.parent_result_id or []))
        if not parent_ids:
            parent_ids = _default_parent_ids(conn, int(args.lookback_hours))
        if not parent_ids:
            raise SystemExit("no recent ablation parent ids found")
        report = build_report(
            conn,
            parent_ids=parent_ids,
            lookback_hours=int(args.lookback_hours),
            exclude_attention=not bool(args.include_attention),
            exclude_attention_context=bool(args.exclude_attention_context),
        )
    finally:
        conn.close()

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(output_md, report)
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
