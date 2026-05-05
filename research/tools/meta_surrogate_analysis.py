#!/usr/bin/env python
"""Read-only surrogate analysis over the standalone meta-analysis DB.

This tool turns the materialized template/slot/op observations into reviewable
candidate rules. It does not mutate the notebook, scoring, gates, or grammar.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from research.meta_analysis.metadata_db import DEFAULT_META_ANALYSIS_DB


DEFAULT_REPORT_DIR = Path("research/reports")
NANO_BIND = "nano_bind"
MIN_RULE_SUPPORT = 25


@dataclass(frozen=True)
class GraphRow:
    result_id: str
    template_name: str
    failure_op: str
    wikitext_perplexity: float | None
    tinystories_score: float | None
    controlled_lang_s05_sa_score: float | None
    motif_count: int
    non_norm_motif_count: int
    norm_motif_count: int
    norm_dominance: float
    has_attention_motif: int
    has_ssm_motif: int
    has_conv_motif: int
    has_recurrent_motif: int
    has_routing_motif: int
    has_compression_motif: int
    has_effective_positional_mixer: int
    mixer_after_compression: int
    motif_thinness_score: float
    frequency_collapse_risk: float

    @property
    def nano_bind_failed(self) -> bool:
        return self.failure_op == NANO_BIND


@dataclass(frozen=True)
class Rule:
    name: str
    predicates: tuple[tuple[str, str, float], ...]


def _connect_readonly(path: str | Path) -> sqlite3.Connection:
    db = Path(path)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    return int(float(value))


def load_graph_rows(meta_db: str | Path) -> list[GraphRow]:
    """Load one graph-level row per result_id from template observations."""

    conn = _connect_readonly(meta_db)
    try:
        rows = conn.execute(
            """
            WITH ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY result_id
                        ORDER BY slot_count DESC, template_name ASC
                    ) AS rn
                FROM template_observations
            )
            SELECT
                result_id,
                template_name,
                COALESCE(failure_op, '') AS failure_op,
                wikitext_perplexity,
                tinystories_score,
                controlled_lang_s05_sa_score,
                motif_count,
                non_norm_motif_count,
                norm_motif_count,
                norm_dominance,
                has_attention_motif,
                has_ssm_motif,
                has_conv_motif,
                has_recurrent_motif,
                has_routing_motif,
                has_compression_motif,
                has_effective_positional_mixer,
                mixer_after_compression,
                motif_thinness_score,
                frequency_collapse_risk
            FROM ranked
            WHERE rn = 1
            """
        ).fetchall()
    finally:
        conn.close()
    return [
        GraphRow(
            result_id=str(row["result_id"]),
            template_name=str(row["template_name"] or ""),
            failure_op=str(row["failure_op"] or ""),
            wikitext_perplexity=_safe_float(row["wikitext_perplexity"]),
            tinystories_score=_safe_float(row["tinystories_score"]),
            controlled_lang_s05_sa_score=_safe_float(
                row["controlled_lang_s05_sa_score"]
            ),
            motif_count=_safe_int(row["motif_count"]),
            non_norm_motif_count=_safe_int(row["non_norm_motif_count"]),
            norm_motif_count=_safe_int(row["norm_motif_count"]),
            norm_dominance=float(row["norm_dominance"] or 0.0),
            has_attention_motif=_safe_int(row["has_attention_motif"]),
            has_ssm_motif=_safe_int(row["has_ssm_motif"]),
            has_conv_motif=_safe_int(row["has_conv_motif"]),
            has_recurrent_motif=_safe_int(row["has_recurrent_motif"]),
            has_routing_motif=_safe_int(row["has_routing_motif"]),
            has_compression_motif=_safe_int(row["has_compression_motif"]),
            has_effective_positional_mixer=_safe_int(
                row["has_effective_positional_mixer"]
            ),
            mixer_after_compression=_safe_int(row["mixer_after_compression"]),
            motif_thinness_score=float(row["motif_thinness_score"] or 0.0),
            frequency_collapse_risk=float(row["frequency_collapse_risk"] or 0.0),
        )
        for row in rows
    ]


def _rate(n_positive: int, n_total: int) -> float:
    return n_positive / n_total if n_total else 0.0


def _predicate(row: GraphRow, field: str, op: str, threshold: float) -> bool:
    value = getattr(row, field)
    if value is None:
        return False
    value = float(value)
    if op == "<=":
        return value <= threshold
    if op == ">=":
        return value >= threshold
    if op == "==":
        return value == threshold
    raise ValueError(f"unsupported op: {op}")


def _rule_matches(row: GraphRow, rule: Rule) -> bool:
    return all(
        _predicate(row, field, op, threshold)
        for field, op, threshold in rule.predicates
    )


def evaluate_rule(
    rows: list[GraphRow], rule: Rule, baseline_rate: float
) -> dict[str, Any]:
    matched = [row for row in rows if _rule_matches(row, rule)]
    positives = sum(row.nano_bind_failed for row in matched)
    total_positive = sum(row.nano_bind_failed for row in rows)
    precision = _rate(positives, len(matched))
    recall = _rate(positives, total_positive)
    lift = precision / baseline_rate if baseline_rate else 0.0
    return {
        "rule": rule.name,
        "predicates": [
            {"field": field, "op": op, "threshold": threshold}
            for field, op, threshold in rule.predicates
        ],
        "support": len(matched),
        "nano_bind_failures": positives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "lift": round(lift, 4),
    }


def candidate_rules() -> list[Rule]:
    singles: list[Rule] = [
        Rule("freq_risk>=0.50", (("frequency_collapse_risk", ">=", 0.50),)),
        Rule("freq_risk>=0.60", (("frequency_collapse_risk", ">=", 0.60),)),
        Rule("thinness>=0.50", (("motif_thinness_score", ">=", 0.50),)),
        Rule("thinness>=0.75", (("motif_thinness_score", ">=", 0.75),)),
        Rule("non_norm<=1", (("non_norm_motif_count", "<=", 1),)),
        Rule("non_norm<=2", (("non_norm_motif_count", "<=", 2),)),
        Rule("no_effective_pos_mixer", (("has_effective_positional_mixer", "==", 0),)),
        Rule("compression_motif", (("has_compression_motif", "==", 1),)),
        Rule("norm_dominance>=0.50", (("norm_dominance", ">=", 0.50),)),
    ]
    combos = [
        Rule(
            "thin_compressed_no_pos",
            (
                ("non_norm_motif_count", "<=", 2),
                ("has_compression_motif", "==", 1),
                ("has_effective_positional_mixer", "==", 0),
            ),
        ),
        Rule(
            "high_risk_no_pos",
            (
                ("frequency_collapse_risk", ">=", 0.60),
                ("has_effective_positional_mixer", "==", 0),
            ),
        ),
        Rule(
            "thin_high_risk",
            (
                ("motif_thinness_score", ">=", 0.50),
                ("frequency_collapse_risk", ">=", 0.50),
            ),
        ),
        Rule(
            "norm_dominated_compressed",
            (
                ("norm_dominance", ">=", 0.50),
                ("has_compression_motif", "==", 1),
            ),
        ),
        Rule(
            "routing_absent_high_risk",
            (
                ("frequency_collapse_risk", ">=", 0.50),
                ("has_routing_motif", "==", 0),
            ),
        ),
    ]
    return singles + combos


def analyze_rules(rows: list[GraphRow]) -> list[dict[str, Any]]:
    baseline = _rate(sum(row.nano_bind_failed for row in rows), len(rows))
    evaluated = [evaluate_rule(rows, rule, baseline) for rule in candidate_rules()]
    return sorted(
        (row for row in evaluated if row["support"] >= MIN_RULE_SUPPORT),
        key=lambda row: (row["precision"], row["recall"], row["support"]),
        reverse=True,
    )


def analyze_binary_features(rows: list[GraphRow]) -> list[dict[str, Any]]:
    fields = [
        "has_attention_motif",
        "has_ssm_motif",
        "has_conv_motif",
        "has_recurrent_motif",
        "has_routing_motif",
        "has_compression_motif",
        "has_effective_positional_mixer",
        "mixer_after_compression",
    ]
    baseline = _rate(sum(row.nano_bind_failed for row in rows), len(rows))
    out: list[dict[str, Any]] = []
    for field in fields:
        for value in (0, 1):
            subset = [row for row in rows if getattr(row, field) == value]
            if len(subset) < MIN_RULE_SUPPORT:
                continue
            positives = sum(row.nano_bind_failed for row in subset)
            precision = _rate(positives, len(subset))
            out.append(
                {
                    "feature": field,
                    "value": value,
                    "support": len(subset),
                    "nano_bind_failures": positives,
                    "failure_rate": round(precision, 4),
                    "lift": round(precision / baseline, 4) if baseline else 0.0,
                }
            )
    return sorted(out, key=lambda row: (row["lift"], row["support"]), reverse=True)


def analyze_templates(meta_db: str | Path) -> list[dict[str, Any]]:
    conn = _connect_readonly(meta_db)
    try:
        raw_rows = conn.execute(
            """
            SELECT
                template_name,
                COUNT(*) AS n,
                SUM(CASE WHEN failure_op = 'nano_bind' THEN 1 ELSE 0 END) AS nano_bind_failures,
                SUM(CASE WHEN wikitext_perplexity < 50 THEN 1 ELSE 0 END) AS ppl_lt_50,
                AVG(CASE WHEN frequency_collapse_risk >= 0.50 THEN 1.0 ELSE 0.0 END) AS high_freq_risk_rate,
                AVG(has_effective_positional_mixer) AS effective_pos_mixer_rate,
                AVG(non_norm_motif_count) AS mean_non_norm_motifs
            FROM template_observations
            GROUP BY template_name
            HAVING n >= 10
            """
        ).fetchall()
        ppl_rows = conn.execute(
            """
            SELECT template_name, wikitext_perplexity
            FROM template_observations
            WHERE wikitext_perplexity IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()
    ppls_by_template: dict[str, list[float]] = {}
    for row in ppl_rows:
        ppls_by_template.setdefault(str(row["template_name"] or ""), []).append(
            float(row["wikitext_perplexity"])
        )
    out: list[dict[str, Any]] = []
    for row in raw_rows:
        template = str(row["template_name"] or "")
        n = int(row["n"])
        failures = int(row["nano_bind_failures"] or 0)
        high_risk_rate = float(row["high_freq_risk_rate"] or 0.0)
        effective_rate = float(row["effective_pos_mixer_rate"] or 0.0)
        ppl_values = ppls_by_template.get(template, [])
        out.append(
            {
                "template_name": template,
                "n": n,
                "nano_bind_failures": failures,
                "nano_bind_rate": round(_rate(failures, n), 4),
                "ppl_lt_50": int(row["ppl_lt_50"] or 0),
                "median_wikitext_ppl": round(_median(ppl_values), 4)
                if ppl_values
                else None,
                "high_freq_risk_rate": round(high_risk_rate, 4),
                "effective_pos_mixer_rate": round(effective_rate, 4),
                "mean_non_norm_motifs": round(
                    float(row["mean_non_norm_motifs"] or 0.0), 4
                ),
                "salvageable_by_slot_fill": bool(
                    failures
                    and failures < n
                    and high_risk_rate > 0.0
                    and effective_rate > 0.0
                ),
            }
        )
    return sorted(out, key=lambda row: (row["nano_bind_rate"], row["n"]), reverse=True)


def analyze_slot_fills(meta_db: str | Path) -> list[dict[str, Any]]:
    conn = _connect_readonly(meta_db)
    try:
        rows = conn.execute(
            """
            SELECT
                template_name,
                slot_index,
                selected_motif,
                selected_motif_class,
                COUNT(*) AS n,
                SUM(CASE WHEN failure_op = 'nano_bind' THEN 1 ELSE 0 END) AS nano_bind_failures,
                AVG(COALESCE(controlled_lang_s05_sa_score, 0.0)) AS mean_sa,
                AVG(frequency_collapse_risk) AS mean_frequency_risk,
                AVG(has_effective_positional_mixer) AS effective_pos_mixer_rate
            FROM slot_observations
            WHERE selected_motif IS NOT NULL
            GROUP BY template_name, slot_index, selected_motif, selected_motif_class
            HAVING n >= 10
            """
        ).fetchall()
    finally:
        conn.close()
    out = []
    for row in rows:
        n = int(row["n"])
        failures = int(row["nano_bind_failures"] or 0)
        out.append(
            {
                "template_name": row["template_name"],
                "slot_index": int(row["slot_index"]),
                "selected_motif": row["selected_motif"],
                "selected_motif_class": row["selected_motif_class"],
                "n": n,
                "nano_bind_failures": failures,
                "nano_bind_rate": round(_rate(failures, n), 4),
                "mean_sa": round(float(row["mean_sa"] or 0.0), 4),
                "mean_frequency_risk": round(
                    float(row["mean_frequency_risk"] or 0.0), 4
                ),
                "effective_pos_mixer_rate": round(
                    float(row["effective_pos_mixer_rate"] or 0.0), 4
                ),
            }
        )
    return sorted(
        out,
        key=lambda row: (row["nano_bind_rate"], row["n"], row["mean_frequency_risk"]),
        reverse=True,
    )


def analyze_design_targets(rows: list[GraphRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int], list[GraphRow]] = {}
    for row in rows:
        if row.nano_bind_failed:
            continue
        key = (
            row.template_name,
            row.has_effective_positional_mixer,
            row.has_routing_motif,
            row.has_compression_motif,
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (template, has_pos, has_routing, has_compression), subset in grouped.items():
        if len(subset) < 10:
            continue
        ppl_values = [
            row.wikitext_perplexity
            for row in subset
            if row.wikitext_perplexity is not None
        ]
        ts_values = [
            row.tinystories_score for row in subset if row.tinystories_score is not None
        ]
        sa_values = [
            row.controlled_lang_s05_sa_score
            for row in subset
            if row.controlled_lang_s05_sa_score is not None
        ]
        score = (
            (1.0 / max(_median(ppl_values), 1.0) if ppl_values else 0.0)
            + (sum(ts_values) / len(ts_values) if ts_values else 0.0)
            + (sum(sa_values) / len(sa_values) if sa_values else 0.0)
            + (0.15 if has_pos else 0.0)
            + (0.10 if has_routing else 0.0)
            + (0.08 if has_compression else 0.0)
        )
        out.append(
            {
                "template_name": template,
                "has_effective_positional_mixer": has_pos,
                "has_routing_motif": has_routing,
                "has_compression_motif": has_compression,
                "n": len(subset),
                "median_wikitext_ppl": round(_median(ppl_values), 4)
                if ppl_values
                else None,
                "mean_tinystories_score": round(sum(ts_values) / len(ts_values), 4)
                if ts_values
                else None,
                "mean_controlled_lang_s05_sa": round(sum(sa_values) / len(sa_values), 4)
                if sa_values
                else None,
                "design_score": round(score, 4),
            }
        )
    return sorted(out, key=lambda row: (row["design_score"], row["n"]), reverse=True)


def _median(values: Iterable[float | None]) -> float:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return 0.0
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _md_table(
    rows: list[dict[str, Any]], fields: list[str], *, limit: int
) -> list[str]:
    if not rows:
        return ["_No rows._", ""]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows[:limit]:
        lines.append(
            "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        )
    lines.append("")
    return lines


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        f"# Meta Surrogate Analysis — {summary['created_utc']}",
        "",
        "Read-only analysis over `research/meta_analysis.db`. Candidate rules are review-only; no scoring or gate wiring is implied.",
        "",
        "## Summary",
        "",
        f"- Graph rows analyzed: {summary['n_graphs']}",
        f"- NanoBind failures: {summary['nano_bind_failures']} ({summary['baseline_nano_bind_rate']:.2%})",
        f"- Rows with WikiText PPL: {summary['wikitext_rows']}",
        f"- Rows with TinyStories score: {summary['tinystories_rows']}",
        "",
        "## Candidate Guardrail Rules",
        "",
        *_md_table(
            payload["rules"],
            ["rule", "support", "nano_bind_failures", "precision", "recall", "lift"],
            limit=12,
        ),
        "## Binary Feature Lifts",
        "",
        *_md_table(
            payload["binary_features"],
            ["feature", "value", "support", "failure_rate", "lift"],
            limit=14,
        ),
        "## Highest NanoBind-Risk Templates",
        "",
        *_md_table(
            payload["templates"],
            [
                "template_name",
                "n",
                "nano_bind_rate",
                "ppl_lt_50",
                "high_freq_risk_rate",
                "effective_pos_mixer_rate",
                "salvageable_by_slot_fill",
            ],
            limit=20,
        ),
        "## Risky Slot Fills",
        "",
        *_md_table(
            payload["slot_fills"],
            [
                "template_name",
                "slot_index",
                "selected_motif",
                "n",
                "nano_bind_rate",
                "mean_frequency_risk",
            ],
            limit=20,
        ),
        "## Design Targets To Explore",
        "",
        *_md_table(
            payload["design_targets"],
            [
                "template_name",
                "has_effective_positional_mixer",
                "has_routing_motif",
                "has_compression_motif",
                "n",
                "median_wikitext_ppl",
                "mean_tinystories_score",
                "mean_controlled_lang_s05_sa",
                "design_score",
            ],
            limit=20,
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


def build_payload(meta_db: str | Path) -> dict[str, Any]:
    rows = load_graph_rows(meta_db)
    nano_bind_failures = sum(row.nano_bind_failed for row in rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "meta_db": str(meta_db),
        "n_graphs": len(rows),
        "nano_bind_failures": nano_bind_failures,
        "baseline_nano_bind_rate": _rate(nano_bind_failures, len(rows)),
        "wikitext_rows": sum(row.wikitext_perplexity is not None for row in rows),
        "tinystories_rows": sum(row.tinystories_score is not None for row in rows),
    }
    return {
        "summary": summary,
        "rules": analyze_rules(rows),
        "binary_features": analyze_binary_features(rows),
        "templates": analyze_templates(meta_db),
        "slot_fills": analyze_slot_fills(meta_db),
        "design_targets": analyze_design_targets(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-db", default=DEFAULT_META_ANALYSIS_DB)
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    prefix = args.output_prefix or f"meta_surrogate_analysis_{stamp}"
    payload = build_payload(args.meta_db)

    json_path = report_dir / f"{prefix}.json"
    md_path = report_dir / f"{prefix}.md"
    rules_csv = report_dir / f"{prefix}_rules.csv"
    templates_csv = report_dir / f"{prefix}_templates.csv"
    slots_csv = report_dir / f"{prefix}_slot_fills.csv"
    targets_csv = report_dir / f"{prefix}_design_targets.csv"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(md_path, payload)
    write_csv(rules_csv, payload["rules"])
    write_csv(templates_csv, payload["templates"])
    write_csv(slots_csv, payload["slot_fills"])
    write_csv(targets_csv, payload["design_targets"])

    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "rules_csv": str(rules_csv),
                "templates_csv": str(templates_csv),
                "slot_fills_csv": str(slots_csv),
                "design_targets_csv": str(targets_csv),
                "n_graphs": payload["summary"]["n_graphs"],
                "nano_bind_failures": payload["summary"]["nano_bind_failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
