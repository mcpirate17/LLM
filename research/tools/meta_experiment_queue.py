#!/usr/bin/env python
"""Build experiment queues from profile-grounded meta-analysis findings.

The output is advisory: profiling candidates, compression safety probes, and
scaffold commands. It does not enqueue notebook follow-ups or alter grammar
weights, gates, or ablation settings.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.meta_analysis.metadata_db import DEFAULT_META_ANALYSIS_DB
from research.tools.meta_profile_ml_analysis import DEFAULT_REPORT_DIR
from research.tools.meta_report_helpers import markdown_table as _md_table
from research.tools.meta_report_helpers import write_csv
from research.tools.profile_component_scaffolds import recommended_scaffold_family


DEFAULT_MIN_SUPPORT = 10
DEFAULT_PROFILE_LIMIT = 24
DEFAULT_COMPRESSION_LIMIT = 24


def _connect_readonly(path: str | Path) -> sqlite3.Connection:
    db = Path(path)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_ml_summary(ml_report: str | Path | None) -> dict[str, Any]:
    if not ml_report:
        return {}
    path = Path(ml_report)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload.get("summary") or {}


def _scaffold_command(op_name: str, family: str) -> str:
    return (
        "python -m research.tools.profile_component_scaffolds "
        f"--family {family} --ops {op_name} --allow-arbitrary-ops "
        "--device cpu --data-mode random --stage1-steps 8 "
        "--model-dim 96 --n-layers 1 --batch-size 2 --top 10 "
        f"--json-out research/reports/scaffold_profile_{op_name}.json "
        "--no-persist --no-progress"
    )


def build_profile_refresh_queue(
    meta_db: str | Path,
    *,
    min_support: int = DEFAULT_MIN_SUPPORT,
    limit: int = DEFAULT_PROFILE_LIMIT,
) -> list[dict[str, Any]]:
    conn = _connect_readonly(meta_db)
    try:
        rows = conn.execute(
            """
            WITH graph_rows AS (
                SELECT
                    ranked.result_id,
                    ranked.failure_op,
                    ranked.routing_fast_lane_ppl_improvement,
                    ranked.wikitext_perplexity,
                    ranked.language_control_s05_sentence_assoc_score,
                    gp.profile_missing_op_count,
                    gp.profile_coverage_rate
                FROM (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY result_id
                            ORDER BY slot_count DESC, template_name ASC
                        ) AS rn
                    FROM template_observations
                ) ranked
                JOIN graph_profile_observations gp
                  ON gp.result_id = ranked.result_id
                WHERE ranked.rn = 1
            ),
            missing_ops AS (
                SELECT
                    oo.op_name,
                    COUNT(*) AS n_observations,
                    SUM(CASE WHEN gr.profile_missing_op_count >= 7 THEN 1 ELSE 0 END)
                        AS high_missing_graphs,
                    SUM(CASE WHEN gr.routing_fast_lane_ppl_improvement > 0 THEN 1 ELSE 0 END)
                        AS routing_improved_graphs,
                    SUM(CASE WHEN gr.failure_op = 'nano_bind' THEN 1 ELSE 0 END)
                        AS nano_bind_failures,
                    SUM(CASE WHEN gr.wikitext_perplexity < 200 THEN 1 ELSE 0 END)
                        AS good_wikitext_graphs,
                    AVG(gr.profile_coverage_rate) AS mean_profile_coverage,
                    AVG(COALESCE(gr.language_control_s05_sentence_assoc_score, 0.0)) AS mean_controlled_sa
                FROM op_observations oo
                JOIN graph_rows gr ON gr.result_id = oo.result_id
                LEFT JOIN op_profile_catalog opc ON opc.op_name = oo.op_name
                WHERE opc.op_name IS NULL
                GROUP BY oo.op_name
                HAVING n_observations >= ?
            )
            SELECT * FROM missing_ops
            ORDER BY
                routing_improved_graphs DESC,
                high_missing_graphs DESC,
                good_wikitext_graphs DESC,
                n_observations DESC
            LIMIT ?
            """,
            (int(min_support), int(limit * 3)),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for row in rows:
        op_name = str(row["op_name"] or "")
        family = recommended_scaffold_family(op_name) or ""
        routing_improved = int(row["routing_improved_graphs"] or 0)
        high_missing = int(row["high_missing_graphs"] or 0)
        nano = int(row["nano_bind_failures"] or 0)
        good_wikitext = int(row["good_wikitext_graphs"] or 0)
        n = int(row["n_observations"] or 0)
        priority = (
            routing_improved * 4.0
            + high_missing * 1.5
            + good_wikitext * 1.0
            + n * 0.05
            - nano * 0.25
        )
        item = {
            "op_name": op_name,
            "recommended_scaffold_family": family,
            "n_observations": n,
            "high_missing_graphs": high_missing,
            "routing_improved_graphs": routing_improved,
            "nano_bind_failures": nano,
            "good_wikitext_graphs": good_wikitext,
            "mean_profile_coverage": round(
                _safe_float(row["mean_profile_coverage"]), 6
            ),
            "mean_controlled_sa": round(_safe_float(row["mean_controlled_sa"]), 6),
            "priority_score": round(priority, 6),
            "action": (
                "run_scaffold_profile"
                if family
                else "add_scaffold_family_or_component_profile_harness"
            ),
            "scaffold_command": _scaffold_command(op_name, family) if family else "",
        }
        out.append(item)
    return sorted(out, key=lambda item: item["priority_score"], reverse=True)[:limit]


def build_compression_safety_queue(
    meta_db: str | Path,
    *,
    min_support: int = DEFAULT_MIN_SUPPORT,
    limit: int = DEFAULT_COMPRESSION_LIMIT,
) -> list[dict[str, Any]]:
    conn = _connect_readonly(meta_db)
    try:
        rows = conn.execute(
            """
            SELECT
                template_name,
                selected_motif,
                selected_motif_class,
                COUNT(*) AS n,
                SUM(CASE WHEN failure_op = 'nano_bind' THEN 1 ELSE 0 END)
                    AS nano_bind_failures,
                SUM(CASE WHEN wikitext_perplexity < 200 THEN 1 ELSE 0 END)
                    AS good_wikitext_graphs,
                AVG(frequency_collapse_risk) AS mean_frequency_risk,
                AVG(has_effective_positional_mixer) AS effective_pos_mixer_rate,
                AVG(COALESCE(language_control_s05_sentence_assoc_score, 0.0)) AS mean_controlled_sa
            FROM slot_observations
            WHERE has_compression_motif = 1
              AND selected_motif IS NOT NULL
            GROUP BY template_name, selected_motif, selected_motif_class
            HAVING n >= ?
            ORDER BY
                nano_bind_failures DESC,
                mean_frequency_risk DESC,
                good_wikitext_graphs DESC,
                n DESC
            LIMIT ?
            """,
            (int(min_support), int(limit)),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for row in rows:
        n = int(row["n"] or 0)
        nano = int(row["nano_bind_failures"] or 0)
        good_wikitext = int(row["good_wikitext_graphs"] or 0)
        mean_risk = _safe_float(row["mean_frequency_risk"])
        pos_rate = _safe_float(row["effective_pos_mixer_rate"])
        priority = (
            (nano * 3.0) + (mean_risk * n) + (good_wikitext * 0.5) - (pos_rate * n)
        )
        out.append(
            {
                "template_name": str(row["template_name"] or ""),
                "selected_motif": str(row["selected_motif"] or ""),
                "selected_motif_class": str(row["selected_motif_class"] or ""),
                "n": n,
                "nano_bind_failures": nano,
                "nano_bind_rate": round(nano / n, 6) if n else 0.0,
                "good_wikitext_graphs": good_wikitext,
                "mean_frequency_risk": round(mean_risk, 6),
                "effective_pos_mixer_rate": round(pos_rate, 6),
                "mean_controlled_sa": round(_safe_float(row["mean_controlled_sa"]), 6),
                "priority_score": round(priority, 6),
                "recommended_variant": (
                    "add_or_preserve_positional_or_content_mixer_after_compression"
                ),
                "evaluation_bundle": (
                    "NanoBind + language_control_s05 + WikiText + TinyStories + routing fast-lane"
                ),
            }
        )
    return sorted(out, key=lambda item: item["priority_score"], reverse=True)


def build_payload(
    meta_db: str | Path,
    *,
    ml_report: str | Path | None = None,
    min_support: int = DEFAULT_MIN_SUPPORT,
    profile_limit: int = DEFAULT_PROFILE_LIMIT,
    compression_limit: int = DEFAULT_COMPRESSION_LIMIT,
) -> dict[str, Any]:
    profile_queue = build_profile_refresh_queue(
        meta_db,
        min_support=min_support,
        limit=profile_limit,
    )
    compression_queue = build_compression_safety_queue(
        meta_db,
        min_support=min_support,
        limit=compression_limit,
    )
    scaffold_commands = [
        {
            "op_name": item["op_name"],
            "family": item["recommended_scaffold_family"],
            "command": item["scaffold_command"],
        }
        for item in profile_queue
        if item.get("scaffold_command")
    ]
    return {
        "summary": {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "meta_db": str(meta_db),
            "ml_report": str(ml_report or ""),
            "min_support": int(min_support),
            "profile_queue_count": len(profile_queue),
            "compression_queue_count": len(compression_queue),
            "scaffold_command_count": len(scaffold_commands),
            "ml_summary": _load_ml_summary(ml_report),
        },
        "profile_refresh_queue": profile_queue,
        "compression_safety_queue": compression_queue,
        "scaffold_commands": scaffold_commands,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        f"# Meta Experiment Queue - {summary['created_utc']}",
        "",
        "Advisory queue only. No notebook follow-ups, grammar weights, gates, or ablation settings are changed by this artifact.",
        "",
        "## Profile Refresh Queue",
        "",
        *_md_table(
            payload["profile_refresh_queue"],
            [
                "op_name",
                "recommended_scaffold_family",
                "n_observations",
                "routing_improved_graphs",
                "high_missing_graphs",
                "good_wikitext_graphs",
                "action",
                "priority_score",
            ],
            limit=30,
        ),
        "## Compression Safety Queue",
        "",
        *_md_table(
            payload["compression_safety_queue"],
            [
                "template_name",
                "selected_motif",
                "n",
                "nano_bind_rate",
                "good_wikitext_graphs",
                "mean_frequency_risk",
                "effective_pos_mixer_rate",
                "recommended_variant",
            ],
            limit=30,
        ),
        "## Scaffold Commands",
        "",
    ]
    for command in payload["scaffold_commands"][:30]:
        lines.extend(
            [
                f"### {command['op_name']}",
                "",
                "```bash",
                command["command"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-db", default=DEFAULT_META_ANALYSIS_DB)
    parser.add_argument("--ml-report", default="")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument("--profile-limit", type=int, default=DEFAULT_PROFILE_LIMIT)
    parser.add_argument(
        "--compression-limit", type=int, default=DEFAULT_COMPRESSION_LIMIT
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    prefix = args.output_prefix or f"meta_experiment_queue_{stamp}"
    payload = build_payload(
        args.meta_db,
        ml_report=args.ml_report or None,
        min_support=args.min_support,
        profile_limit=args.profile_limit,
        compression_limit=args.compression_limit,
    )

    json_path = report_dir / f"{prefix}.json"
    md_path = report_dir / f"{prefix}.md"
    profile_csv = report_dir / f"{prefix}_profile_refresh.csv"
    compression_csv = report_dir / f"{prefix}_compression_safety.csv"
    commands_jsonl = report_dir / f"{prefix}_scaffold_commands.jsonl"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(md_path, payload)
    write_csv(profile_csv, payload["profile_refresh_queue"])
    write_csv(compression_csv, payload["compression_safety_queue"])
    with commands_jsonl.open("w", encoding="utf-8") as handle:
        for command in payload["scaffold_commands"]:
            handle.write(json.dumps(command, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "profile_refresh_csv": str(profile_csv),
                "compression_safety_csv": str(compression_csv),
                "scaffold_commands_jsonl": str(commands_jsonl),
                "profile_queue_count": payload["summary"]["profile_queue_count"],
                "compression_queue_count": payload["summary"][
                    "compression_queue_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
