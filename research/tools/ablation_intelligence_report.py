#!/usr/bin/env python3
"""Build a multi-metric component-knockout ablation report.

The causal evidence table is an index/provenance table. This report dedupes
evidence by its semantic key and reads the real child measurements from
program_results so duplicated metadata cannot inflate confidence.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "research/lab_notebook.db"
RUNTIME_DIR = PROJECT_ROOT / "research/runtime"

CORE_METRICS = (
    "loss_ratio",
    "wikitext_perplexity",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "ar_auc",
    "induction_auc",
    "binding_auc",
    "binding_composite",
    "induction_v2_investigation_auc",
    "binding_v2_investigation_auc",
)


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    parent_result_id: str
    ablation_experiment_id: str
    rule_type: str
    rule_key: str
    outcome: str
    duplicate_count: int
    superseded_count: int
    child_result_id: str | None


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row(conn: sqlite3.Connection, result_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM program_results WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    return dict(row) if row else None


def _metric_snapshot(row: dict[str, Any] | None) -> dict[str, float | None]:
    return {metric: _f(row.get(metric)) if row else None for metric in CORE_METRICS}


def _parse_child_result_id(evidence_json: str | None) -> str | None:
    if not evidence_json:
        return None
    try:
        payload = json.loads(evidence_json)
    except json.JSONDecodeError:
        return None
    child_id = payload.get("child_result_id")
    return str(child_id) if child_id else None


def _deduped_evidence(
    conn: sqlite3.Connection,
    parent_ids: list[str],
) -> list[EvidenceRef]:
    placeholders = ",".join("?" for _ in parent_ids)
    rows = conn.execute(
        f"""
        SELECT *
        FROM causal_rule_evidence
        WHERE parent_result_id IN ({placeholders})
          AND rule_type IN ('node_delete_s1', 'node_delete_investigation')
        ORDER BY timestamp ASC
        """,
        tuple(parent_ids),
    ).fetchall()
    idempotency_groups: dict[tuple[str, str, str, str, str], list[sqlite3.Row]] = (
        defaultdict(list)
    )
    latest_groups: dict[tuple[str, str, str, str], list[sqlite3.Row]] = defaultdict(
        list
    )
    for row in rows:
        outcome = str(row["outcome"] or "")
        if outcome not in {"measured_s1", "measured_investigation"}:
            continue
        idempotency_groups[
            (
                str(row["parent_result_id"] or ""),
                str(row["ablation_experiment_id"] or ""),
                str(row["rule_type"] or ""),
                str(row["rule_key"] or ""),
                outcome,
            )
        ].append(row)
        latest_groups[
            (
                str(row["parent_result_id"] or ""),
                str(row["rule_type"] or ""),
                str(row["rule_key"] or ""),
                outcome,
            )
        ].append(row)

    refs: list[EvidenceRef] = []
    for latest_key, group_rows in latest_groups.items():
        latest = max(group_rows, key=lambda r: float(r["timestamp"] or 0.0))
        idempotency_key = (
            latest_key[0],
            str(latest["ablation_experiment_id"] or ""),
            latest_key[1],
            latest_key[2],
            latest_key[3],
        )
        same_run_duplicates = len(idempotency_groups.get(idempotency_key, [latest]))
        refs.append(
            EvidenceRef(
                evidence_id=str(latest["evidence_id"]),
                parent_result_id=str(latest["parent_result_id"]),
                ablation_experiment_id=str(latest["ablation_experiment_id"]),
                rule_type=str(latest["rule_type"]),
                rule_key=str(latest["rule_key"]),
                outcome=str(latest["outcome"]),
                duplicate_count=same_run_duplicates,
                superseded_count=max(0, len(group_rows) - same_run_duplicates),
                child_result_id=_parse_child_result_id(latest["evidence_json"]),
            )
        )
    return sorted(
        refs,
        key=lambda r: (
            r.parent_result_id,
            r.rule_type,
            int(r.rule_key.split(":", 1)[0])
            if r.rule_key.split(":", 1)[0].isdigit()
            else 9999,
            r.rule_key,
        ),
    )


def _classify(
    parent: dict[str, Any], child: dict[str, Any] | None
) -> tuple[str, list[str]]:
    if not child:
        return "missing_child_row", ["evidence did not resolve to program_results"]
    if not bool(child.get("stage1_passed")):
        return "failed_knockout", [str(child.get("error_type") or "stage1_not_passed")]

    parent_loss = _f(parent.get("loss_ratio"))
    child_loss = _f(child.get("loss_ratio"))
    parent_wiki = _f(parent.get("wikitext_perplexity"))
    child_wiki = _f(child.get("wikitext_perplexity"))
    parent_hs = _f(parent.get("hellaswag_acc"))
    child_hs = _f(child.get("hellaswag_acc"))
    parent_blimp = _f(parent.get("blimp_overall_accuracy"))
    child_blimp = _f(child.get("blimp_overall_accuracy"))
    parent_bc = _f(parent.get("binding_composite"))
    child_bc = _f(child.get("binding_composite"))
    parent_bind = _f(parent.get("binding_auc"))
    child_bind = _f(child.get("binding_auc"))
    parent_ind_v2 = _f(parent.get("induction_v2_investigation_auc"))
    child_ind_v2 = _f(child.get("induction_v2_investigation_auc"))
    parent_bind_v2 = _f(parent.get("binding_v2_investigation_auc"))
    child_bind_v2 = _f(child.get("binding_v2_investigation_auc"))

    reasons: list[str] = []
    loss_delta = (
        None if parent_loss is None or child_loss is None else child_loss - parent_loss
    )
    if loss_delta is not None:
        if loss_delta < -0.02:
            reasons.append(f"loss improved {loss_delta:+.4f}")
        elif loss_delta > 0.02:
            reasons.append(f"loss worsened {loss_delta:+.4f}")
        else:
            reasons.append(f"loss near-neutral {loss_delta:+.4f}")

    metric_conflict = False
    if parent_wiki and child_wiki and child_wiki > parent_wiki * 1.5:
        metric_conflict = True
        reasons.append(f"WikiText worsened {child_wiki / parent_wiki:.1f}x")
    if parent_hs is not None and child_hs is not None and child_hs < parent_hs - 0.03:
        metric_conflict = True
        reasons.append(f"HellaSwag down {child_hs - parent_hs:+.3f}")
    if (
        parent_blimp is not None
        and child_blimp is not None
        and child_blimp < parent_blimp - 0.03
    ):
        metric_conflict = True
        reasons.append(f"BLiMP down {child_blimp - parent_blimp:+.3f}")
    if parent_bc is not None and child_bc is not None and child_bc < parent_bc - 0.05:
        metric_conflict = True
        reasons.append(f"binding composite down {child_bc - parent_bc:+.3f}")
    if (
        parent_bind is not None
        and child_bind is not None
        and child_bind < parent_bind * 0.25
    ):
        metric_conflict = True
        reasons.append("binding_auc collapsed")
    if (
        parent_ind_v2 is not None
        and child_ind_v2 is not None
        and child_ind_v2 < parent_ind_v2 - 0.15
    ):
        metric_conflict = True
        reasons.append(f"induction-v2 down {child_ind_v2 - parent_ind_v2:+.3f}")
    if (
        parent_bind_v2 is not None
        and child_bind_v2 is not None
        and child_bind_v2 < parent_bind_v2 - 0.15
    ):
        metric_conflict = True
        reasons.append(f"binding-v2 down {child_bind_v2 - parent_bind_v2:+.3f}")

    if metric_conflict:
        return "metric_conflict_do_not_prune", reasons
    if loss_delta is not None and loss_delta > 0.02:
        return "harmful_deletion_keep_component", reasons
    if loss_delta is not None and loss_delta < -0.02:
        return "replication_candidate_not_yet_dead_weight", reasons
    return "near_neutral_needs_replication", reasons


def build_report(conn: sqlite3.Connection, parent_ids: list[str]) -> dict[str, Any]:
    refs = _deduped_evidence(conn, parent_ids)
    by_parent: dict[str, dict[str, Any]] = {}
    for parent_id in parent_ids:
        parent = _row(conn, parent_id)
        if not parent:
            continue
        by_parent[parent_id] = {
            "parent": {
                "result_id": parent_id,
                "experiment_id": parent.get("experiment_id"),
                "fingerprint": parent.get("graph_fingerprint"),
                "metrics": _metric_snapshot(parent),
            },
            "evidence_duplicate_rows": 0,
            "superseded_evidence_rows": 0,
            "observations": [],
            "verdict_counts": defaultdict(int),
        }

    for ref in refs:
        parent_bucket = by_parent.get(ref.parent_result_id)
        if parent_bucket is None:
            continue
        parent_row = _row(conn, ref.parent_result_id)
        child_row = _row(conn, ref.child_result_id) if ref.child_result_id else None
        verdict, reasons = _classify(parent_row or {}, child_row)
        parent_bucket["evidence_duplicate_rows"] += max(0, ref.duplicate_count - 1)
        parent_bucket["superseded_evidence_rows"] += ref.superseded_count
        parent_bucket["verdict_counts"][verdict] += 1
        parent_bucket["observations"].append(
            {
                "rule_key": ref.rule_key,
                "phase": ref.rule_type.removeprefix("node_delete_"),
                "ablation_experiment_id": ref.ablation_experiment_id,
                "evidence_id": ref.evidence_id,
                "duplicate_count": ref.duplicate_count,
                "superseded_count": ref.superseded_count,
                "child_result_id": ref.child_result_id,
                "child_stage1_passed": bool(
                    child_row and child_row.get("stage1_passed")
                ),
                "verdict": verdict,
                "reasons": reasons,
                "child_metrics": _metric_snapshot(child_row),
            }
        )

    for bucket in by_parent.values():
        bucket["verdict_counts"] = dict(bucket["verdict_counts"])
    return {"parents": by_parent}


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Ablation Intelligence Report",
        "",
        "Evidence rows are first deduped by the write idempotency key, then the latest measured row per parent, phase, and rule key is used for analysis. Verdicts use `program_results` metrics, not raw evidence counts.",
        "",
    ]
    for parent_id, bucket in report["parents"].items():
        parent = bucket["parent"]
        lines.extend(
            [
                f"## Parent {parent_id}",
                "",
                f"- experiment: `{parent['experiment_id']}`",
                f"- duplicate evidence rows ignored: {bucket['evidence_duplicate_rows']}",
                f"- superseded older evidence rows ignored: {bucket['superseded_evidence_rows']}",
                f"- verdict counts: `{json.dumps(bucket['verdict_counts'], sort_keys=True)}`",
                "",
                "| phase | rule | pass | loss | wiki | hs | blimp | ind | bind | bc | ind_v2 | bind_v2 | verdict |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        observations = sorted(
            bucket["observations"],
            key=lambda o: (
                o["phase"],
                int(str(o["rule_key"]).split(":", 1)[0])
                if str(o["rule_key"]).split(":", 1)[0].isdigit()
                else 9999,
            ),
        )
        for obs in observations:
            m = obs["child_metrics"]
            lines.append(
                "| {phase} | `{rule}` | {passed} | {loss} | {wiki} | {hs} | {blimp} | {ind} | {bind} | {bc} | {ind_v2} | {bind_v2} | {verdict} |".format(
                    phase=obs["phase"],
                    rule=obs["rule_key"],
                    passed="yes" if obs["child_stage1_passed"] else "no",
                    loss=_fmt(m["loss_ratio"], 4),
                    wiki=_fmt(m["wikitext_perplexity"], 1),
                    hs=_fmt(m["hellaswag_acc"], 3),
                    blimp=_fmt(m["blimp_overall_accuracy"], 3),
                    ind=_fmt(m["induction_auc"], 3),
                    bind=_fmt(m["binding_auc"], 3),
                    bc=_fmt(m["binding_composite"], 3),
                    ind_v2=_fmt(m["induction_v2_investigation_auc"], 3),
                    bind_v2=_fmt(m["binding_v2_investigation_auc"], 3),
                    verdict=obs["verdict"],
                )
            )
        lines.append("")
        lines.append("### Notes")
        for obs in observations:
            reason = "; ".join(obs["reasons"]) if obs["reasons"] else "no reason"
            lines.append(
                f"- `{obs['phase']} {obs['rule_key']}`: {obs['verdict']} - {reason}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument(
        "--parent-result-id",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--output",
        default=str(RUNTIME_DIR / "ablation_intelligence_report.md"),
    )
    parser.add_argument(
        "--json-output",
        default=str(RUNTIME_DIR / "ablation_intelligence_report.json"),
    )
    args = parser.parse_args()

    conn = _connect(Path(args.db))
    try:
        parent_ids = args.parent_result_id or ["574271ca-f37", "ec7025d7-338"]
        report = build_report(conn, parent_ids)
    finally:
        conn.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")
    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {output}")
    print(f"wrote {json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
