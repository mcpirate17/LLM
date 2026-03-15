#!/usr/bin/env python3
"""Bulk rescore workflow novelty after the bridge behavioral-novelty fix.

Default mode is dry-run: compute fixed novelty scores, write a report, and
leave databases untouched.

Optional apply mode updates `workflows.graph_json.metadata` in
`aria_designer/api/aria_designer.db` with the rescored novelty fields.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/home/tim/Projects/LLM/aria_designer")
DB_PATH = ROOT / "api" / "aria_designer.db"
DEFAULT_REPORT = ROOT / "workflows" / "generated" / "bulk_rescore_novelty_report.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.bridge import workflow_to_graph  # noqa: E402
from research.defaults import MODEL_DIM, VOCAB_SIZE  # noqa: E402
from research.synthesis.compiler import compile_model  # noqa: E402
from research.eval.sandbox import safe_eval  # noqa: E402
from research.eval.fingerprint import compute_fingerprint  # noqa: E402
from research.eval.metrics import novelty_score  # noqa: E402


@dataclass(slots=True)
class RescoreRow:
    workflow_id: str
    name: str
    author: str | None
    version: int | None
    result_id: str | None
    source_novelty: float | None
    old_bridge_novelty: float | None
    new_bridge_novelty: float | None
    delta_new_minus_old: float | None
    behavioral_novelty: float | None
    structural_novelty: float | None
    most_similar_to: str | None
    graph_fingerprint: str | None
    model_dim: int
    n_ops: int
    updated_metadata: bool = False


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_workflows(con: sqlite3.Connection, scope: str) -> list[tuple]:
    cur = con.cursor()
    if scope == "imported":
        cur.execute(
            """
            SELECT id, name, graph_json, version, author
            FROM workflows
            WHERE id LIKE 'imported_%'
            ORDER BY updated_at DESC
            """
        )
    else:
        cur.execute(
            """
            SELECT id, name, graph_json, version, author
            FROM workflows
            ORDER BY updated_at DESC
            """
        )
    return cur.fetchall()


def _pick_rows(rows: list[tuple], limit: int | None, seed: int, random_sample: bool) -> list[tuple]:
    items = list(rows)
    if random_sample:
        rng = random.Random(seed)
        rng.shuffle(items)
    if limit is not None:
        return items[: max(0, int(limit))]
    return items


def _rescore_workflow(workflow_id: str, name: str, graph_json: str, version: int | None, author: str | None) -> RescoreRow | None:
    workflow = json.loads(graph_json)
    meta = workflow.get("metadata") or {}
    model_dim = int(meta.get("model_dim") or MODEL_DIM)

    graph = workflow_to_graph(workflow, model_dim=model_dim)
    model = compile_model([graph], vocab_size=VOCAB_SIZE)
    sandbox = safe_eval(
        model,
        batch_size=1,
        seq_len=32,
        vocab_size=VOCAB_SIZE,
        device="cpu",
        run_stability_probe=False,
    )
    if not sandbox.passed:
        return None

    fp = compute_fingerprint(
        model,
        seq_len=32,
        model_dim=model_dim,
        vocab_size=VOCAB_SIZE,
        device="cpu",
    )
    old_metrics = novelty_score(graph)
    new_metrics = novelty_score(graph, fingerprint=fp)

    return RescoreRow(
        workflow_id=workflow_id,
        name=name,
        author=author,
        version=version,
        result_id=meta.get("result_id"),
        source_novelty=_f(meta.get("novelty_score")),
        old_bridge_novelty=_f(old_metrics.overall_novelty),
        new_bridge_novelty=_f(new_metrics.overall_novelty),
        delta_new_minus_old=_f(new_metrics.overall_novelty - old_metrics.overall_novelty),
        behavioral_novelty=_f(new_metrics.behavioral_novelty),
        structural_novelty=_f(new_metrics.structural_novelty),
        most_similar_to=new_metrics.most_similar_to or None,
        graph_fingerprint=graph.fingerprint(),
        model_dim=model_dim,
        n_ops=int(graph.n_ops()),
    )


def _apply_metadata_update(con: sqlite3.Connection, row: RescoreRow, original_graph_json: str) -> None:
    workflow = json.loads(original_graph_json)
    meta = workflow.setdefault("metadata", {})
    meta["novelty_score"] = row.new_bridge_novelty
    meta["structural_novelty"] = row.structural_novelty
    meta["behavioral_novelty"] = row.behavioral_novelty
    meta["most_similar_to"] = row.most_similar_to
    meta["graph_fingerprint"] = row.graph_fingerprint
    meta["novelty_rescored_at"] = _utc_now()
    meta["novelty_rescore_source"] = "bridge_behavioral_fix"
    meta["novelty_rescore_old_bridge"] = row.old_bridge_novelty
    cur = con.cursor()
    cur.execute(
        """
        UPDATE workflows
        SET graph_json = ?, updated_at = ?
        WHERE id = ?
        """,
        (json.dumps(workflow), _utc_now(), row.workflow_id),
    )
    row.updated_metadata = cur.rowcount > 0


def _summary(rows: list[RescoreRow], attempted: int) -> dict[str, Any]:
    deltas = [r.delta_new_minus_old for r in rows if r.delta_new_minus_old is not None]
    old_vals = [r.old_bridge_novelty for r in rows if r.old_bridge_novelty is not None]
    new_vals = [r.new_bridge_novelty for r in rows if r.new_bridge_novelty is not None]
    return {
        "attempted": attempted,
        "rescored": len(rows),
        "avg_old_bridge_novelty": _f(statistics.mean(old_vals)) if old_vals else None,
        "avg_new_bridge_novelty": _f(statistics.mean(new_vals)) if new_vals else None,
        "avg_delta": _f(statistics.mean(deltas)) if deltas else None,
        "median_delta": _f(statistics.median(deltas)) if deltas else None,
        "max_delta": _f(max(deltas)) if deltas else None,
        "min_delta": _f(min(deltas)) if deltas else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DB_PATH, help="Path to aria_designer sqlite DB")
    p.add_argument("--scope", choices=["imported", "all"], default="imported", help="Which workflows to consider")
    p.add_argument("--limit", type=int, default=None, help="Maximum workflows to inspect")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    p.add_argument("--ordered", action="store_true", help="Preserve DB ordering instead of random sampling")
    p.add_argument("--apply-workflow-metadata", action="store_true", help="Write rescored novelty fields back into workflows.graph_json metadata")
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="JSON report output path")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(args.db))
    try:
        db_rows = _load_workflows(con, args.scope)
        chosen = _pick_rows(db_rows, args.limit, args.seed, random_sample=not args.ordered)
        rescored: list[RescoreRow] = []
        errors: list[dict[str, Any]] = []

        for workflow_id, name, graph_json, version, author in chosen:
            try:
                row = _rescore_workflow(workflow_id, name, graph_json, version, author)
                if row is None:
                    errors.append({"workflow_id": workflow_id, "reason": "sandbox_failed_or_noncomparable"})
                    continue
                if args.apply_workflow_metadata:
                    _apply_metadata_update(con, row, graph_json)
                rescored.append(row)
            except Exception as exc:
                errors.append({"workflow_id": workflow_id, "reason": str(exc)})

        if args.apply_workflow_metadata:
            con.commit()

        report = {
            "generated_at": _utc_now(),
            "db_path": str(args.db),
            "scope": args.scope,
            "limit": args.limit,
            "ordered": bool(args.ordered),
            "apply_workflow_metadata": bool(args.apply_workflow_metadata),
            "summary": _summary(rescored, attempted=len(chosen)),
            "rescored": [asdict(r) for r in sorted(rescored, key=lambda x: x.delta_new_minus_old or 0.0, reverse=True)],
            "errors": errors,
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(args.report)
        print(json.dumps(report["summary"], indent=2))
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
