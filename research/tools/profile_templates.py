#!/usr/bin/env python
"""Template profiling and governance report.

Profiles every active template from live notebook evidence and can optionally
run targeted batches to backfill missing evidence before scoring governance.

Usage:
    python -m research.tools.profile_templates --db research/lab_notebook.db
    python -m research.tools.profile_templates --run --target-eval 10 --batch-size 6
    python -m research.tools.profile_templates --templates mamba_reference topk_retrieval
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from research.synthesis.templates import TEMPLATES
from research.tools.backfill_templates import (
    DB_PATH,
    get_template_stats,
    run_template_batch,
)

MIN_TEMPLATE_EVIDENCE_RUNS = 10


def _load_program_rows(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """
            SELECT
                result_id,
                timestamp,
                graph_json,
                stage0_passed,
                stage05_passed,
                stage1_passed,
                loss_ratio,
                validation_loss_ratio,
                discovery_loss_ratio,
                error_type,
                stage_at_death,
                failure_details_json
            FROM program_results
            WHERE graph_json IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()


def _safe_json_loads(raw: Any) -> Any:
    if not raw:
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _root_cause(row: sqlite3.Row) -> str:
    details = _safe_json_loads(row["failure_details_json"])
    return str(
        details.get("root_cause_code")
        or row["error_type"]
        or row["stage_at_death"]
        or "unknown"
    )


def build_template_profiles(
    db_path: Path,
    templates: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = _load_program_rows(db_path)
    active_templates = templates or set(TEMPLATES)
    by_template: dict[str, dict[str, Any]] = {}

    for row in rows:
        graph = _safe_json_loads(row["graph_json"])
        metadata = graph.get("metadata", {}) if isinstance(graph, dict) else {}
        used_templates = metadata.get("templates_used") or []
        slot_usage = metadata.get("template_slot_usage") or []
        if not isinstance(used_templates, list):
            continue
        for template_name in used_templates:
            name = str(template_name)
            if name not in active_templates:
                continue
            bucket = by_template.setdefault(
                name,
                {
                    "name": name,
                    "eval": 0,
                    "s0": 0,
                    "s05": 0,
                    "s1": 0,
                    "losses": [],
                    "validation_losses": [],
                    "discovery_losses": [],
                    "failure_reasons": Counter(),
                    "slot_count": 0,
                    "slot_uses": 0,
                    "slot_s1": 0,
                    "slot_failures": Counter(),
                    "slot_motifs": Counter(),
                },
            )
            bucket["eval"] += 1
            bucket["s0"] += 1 if row["stage0_passed"] else 0
            bucket["s05"] += 1 if row["stage05_passed"] else 0
            bucket["s1"] += 1 if row["stage1_passed"] else 0
            if row["loss_ratio"] is not None:
                bucket["losses"].append(float(row["loss_ratio"]))
            if row["validation_loss_ratio"] is not None:
                bucket["validation_losses"].append(float(row["validation_loss_ratio"]))
            if row["discovery_loss_ratio"] is not None:
                bucket["discovery_losses"].append(float(row["discovery_loss_ratio"]))
            if not row["stage1_passed"]:
                bucket["failure_reasons"][_root_cause(row)] += 1

            if not isinstance(slot_usage, list):
                continue
            template_slots = [
                slot
                for slot in slot_usage
                if isinstance(slot, dict) and str(slot.get("template_name")) == name
            ]
            bucket["slot_count"] = max(bucket["slot_count"], len(template_slots))
            for slot in template_slots:
                bucket["slot_uses"] += 1
                bucket["slot_s1"] += 1 if row["stage1_passed"] else 0
                selected = slot.get("selected_motif")
                if selected:
                    bucket["slot_motifs"][str(selected)] += 1
                if not row["stage1_passed"]:
                    bucket["slot_failures"][_root_cause(row)] += 1

    profiles: list[dict[str, Any]] = []
    stats = get_template_stats(db_path)
    for name in sorted(active_templates):
        bucket = by_template.get(
            name,
            {
                "name": name,
                "eval": 0,
                "s0": 0,
                "s05": 0,
                "s1": 0,
                "losses": [],
                "validation_losses": [],
                "discovery_losses": [],
                "failure_reasons": Counter(),
                "slot_count": 0,
                "slot_uses": 0,
                "slot_s1": 0,
                "slot_failures": Counter(),
                "slot_motifs": Counter(),
            },
        )
        eval_count = int(bucket["eval"] or 0)
        s0 = int(bucket["s0"] or 0)
        s05 = int(bucket["s05"] or 0)
        s1 = int(bucket["s1"] or 0)
        s1_rate = float(s1) / max(eval_count, 1)
        slot_uses = int(bucket["slot_uses"] or 0)
        slot_s1_rate = float(bucket["slot_s1"] or 0) / max(slot_uses, 1)
        val_losses = bucket["validation_losses"]
        train_losses = bucket["losses"]
        avg_val = sum(val_losses) / len(val_losses) if val_losses else None
        avg_loss = sum(train_losses) / len(train_losses) if train_losses else None
        top_failure = None
        if bucket["failure_reasons"]:
            top_failure = bucket["failure_reasons"].most_common(1)[0][0]
        top_slot_motif = None
        if bucket["slot_motifs"]:
            top_slot_motif = bucket["slot_motifs"].most_common(1)[0][0]

        evidence = stats.get(name, {})
        status = "promote"
        if eval_count < 3:
            status = "needs_data"
        elif s1 == 0:
            status = "quarantine"
        elif s1_rate < 0.10:
            status = "rehab"
        elif avg_val is not None and avg_val > 0.75:
            status = "rehab"

        profiles.append(
            {
                "name": name,
                "eval_count": eval_count,
                "s0_rate": float(s0) / max(eval_count, 1),
                "s05_rate": float(s05) / max(eval_count, 1),
                "s1_rate": s1_rate,
                "avg_loss_ratio": avg_loss,
                "avg_validation_loss_ratio": avg_val,
                "avg_discovery_loss_ratio": (
                    sum(bucket["discovery_losses"]) / len(bucket["discovery_losses"])
                    if bucket["discovery_losses"]
                    else None
                ),
                "top_failure_reason": top_failure,
                "failure_reasons": dict(bucket["failure_reasons"].most_common(3)),
                "slot_count": int(bucket["slot_count"] or 0),
                "slot_uses": slot_uses,
                "slot_s1_rate": slot_s1_rate if slot_uses else None,
                "top_slot_motif": top_slot_motif,
                "top_slot_failures": dict(bucket["slot_failures"].most_common(3)),
                "status": status,
                "live_stats": {
                    "eval": int(evidence.get("eval", 0)),
                    "s0": int(evidence.get("s0", 0)),
                    "s1": int(evidence.get("s1", 0)),
                },
            }
        )
    return sorted(
        profiles,
        key=lambda item: (
            {"quarantine": 0, "rehab": 1, "needs_data": 2, "promote": 3}[
                item["status"]
            ],
            item["s1_rate"],
            item["avg_validation_loss_ratio"]
            if item["avg_validation_loss_ratio"] is not None
            else 999.0,
            item["name"],
        ),
    )


def _run_targeted_profiles(
    db_path: Path,
    templates: list[str],
    device: str,
    target_eval: int,
    batch_size: int,
    weights: str,
) -> None:
    for name in templates:
        current_eval = int(get_template_stats(db_path).get(name, {}).get("eval", 0))
        while current_eval < target_eval:
            remaining = target_eval - current_eval
            run_template_batch(
                template_name=name,
                n_programs=min(batch_size, remaining),
                device=device,
                db_path=str(db_path),
                weight_mode=weights,
            )
            updated_eval = int(get_template_stats(db_path).get(name, {}).get("eval", 0))
            if updated_eval <= current_eval:
                break
            current_eval = updated_eval


def render_report(
    profiles: list[dict[str, Any]],
    *,
    top_k: int = 20,
) -> str:
    lines = []
    status_counts = Counter(profile["status"] for profile in profiles)
    lines.append(
        "Template governance: "
        + ", ".join(
            f"{status}={status_counts.get(status, 0)}"
            for status in ("quarantine", "rehab", "needs_data", "promote")
        )
    )
    lines.append("")
    lines.append(
        f"{'template':<30} {'status':<11} {'eval':>4} {'s1%':>6} {'val_lr':>7} {'slots':>5}  details"
    )
    lines.append("-" * 92)
    for item in profiles[:top_k]:
        avg_val = item["avg_validation_loss_ratio"]
        val_str = f"{avg_val:.3f}" if avg_val is not None else "n/a"
        details = item["top_failure_reason"] or item["top_slot_motif"] or "-"
        lines.append(
            f"{item['name']:<30} {item['status']:<11} {item['eval_count']:4d} "
            f"{item['s1_rate'] * 100:6.1f} {val_str:>7} {item['slot_count']:5d}  {details}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile active templates and slot governance"
    )
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--templates", nargs="*", default=None)
    parser.add_argument(
        "--run", action="store_true", help="Run targeted batches before profiling"
    )
    parser.add_argument("--target-eval", type=int, default=MIN_TEMPLATE_EVIDENCE_RUNS)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument(
        "--weights", choices=["uniform", "random", "default"], default="uniform"
    )
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    template_names = (
        sorted(set(args.templates) & set(TEMPLATES))
        if args.templates
        else sorted(TEMPLATES)
    )
    if args.run:
        _run_targeted_profiles(
            db_path=args.db,
            templates=template_names,
            device=args.device,
            target_eval=args.target_eval,
            batch_size=args.batch_size,
            weights=args.weights,
        )

    profiles = build_template_profiles(args.db, templates=set(template_names))
    print(render_report(profiles, top_k=args.top))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(profiles, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
