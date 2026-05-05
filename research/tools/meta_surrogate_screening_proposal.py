#!/usr/bin/env python
"""Generate a review-only screening proposal from meta-surrogate rules."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.meta_analysis.metadata_db import DEFAULT_META_ANALYSIS_DB
from research.tools.meta_surrogate_analysis import (
    DEFAULT_REPORT_DIR,
    GraphRow,
    Rule,
    analyze_rules,
    analyze_templates,
    candidate_rules,
    load_graph_rows,
    _rule_matches,
)


MAX_EXAMPLES_PER_CLASS = 12
GOOD_WIKITEXT_PPL = 200.0
GOOD_TINYSTORIES_SCORE = 0.55
GOOD_CONTROLLED_SA = 0.95


@dataclass(frozen=True)
class Example:
    result_id: str
    template_name: str
    outcome: str
    wikitext_perplexity: float | None
    tinystories_score: float | None
    controlled_lang_s05_sa_score: float | None
    non_norm_motif_count: int
    norm_dominance: float
    has_effective_positional_mixer: int
    has_routing_motif: int
    has_compression_motif: int
    frequency_collapse_risk: float
    escape_hatches: str


def _round_or_none(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def escape_hatches(row: GraphRow, salvageable_templates: set[str]) -> list[str]:
    hatches: list[str] = []
    if (
        row.wikitext_perplexity is not None
        and row.wikitext_perplexity < GOOD_WIKITEXT_PPL
    ):
        hatches.append("strong_wikitext")
    if (
        row.tinystories_score is not None
        and row.tinystories_score >= GOOD_TINYSTORIES_SCORE
    ):
        hatches.append("strong_tinystories")
    if (
        row.controlled_lang_s05_sa_score is not None
        and row.controlled_lang_s05_sa_score >= GOOD_CONTROLLED_SA
    ):
        hatches.append("strong_controlled_sa")
    if row.has_effective_positional_mixer:
        hatches.append("effective_positional_mixer")
    if row.has_routing_motif:
        hatches.append("routing_present")
    if row.has_attention_motif or row.has_ssm_motif or row.has_conv_motif:
        hatches.append("explicit_mixer_family")
    if row.template_name in salvageable_templates:
        hatches.append("salvageable_template_family")
    return hatches


def _example(row: GraphRow, outcome: str, salvageable_templates: set[str]) -> Example:
    return Example(
        result_id=row.result_id,
        template_name=row.template_name,
        outcome=outcome,
        wikitext_perplexity=_round_or_none(row.wikitext_perplexity),
        tinystories_score=_round_or_none(row.tinystories_score),
        controlled_lang_s05_sa_score=_round_or_none(row.controlled_lang_s05_sa_score),
        non_norm_motif_count=row.non_norm_motif_count,
        norm_dominance=round(row.norm_dominance, 4),
        has_effective_positional_mixer=row.has_effective_positional_mixer,
        has_routing_motif=row.has_routing_motif,
        has_compression_motif=row.has_compression_motif,
        frequency_collapse_risk=round(row.frequency_collapse_risk, 4),
        escape_hatches=",".join(escape_hatches(row, salvageable_templates)),
    )


def _false_positive_sort_key(row: GraphRow) -> tuple[float, float, float]:
    ppl = row.wikitext_perplexity if row.wikitext_perplexity is not None else 1e9
    tinystories = row.tinystories_score if row.tinystories_score is not None else -1.0
    sa = (
        row.controlled_lang_s05_sa_score
        if row.controlled_lang_s05_sa_score is not None
        else -1.0
    )
    return (ppl, -tinystories, -sa)


def sample_rule_examples(
    rows: list[GraphRow],
    rule: Rule,
    salvageable_templates: set[str],
) -> dict[str, list[Example]]:
    matched = [row for row in rows if _rule_matches(row, rule)]
    missed = [row for row in rows if not _rule_matches(row, rule)]
    true_positives = [row for row in matched if row.nano_bind_failed]
    false_positives = [row for row in matched if not row.nano_bind_failed]
    false_negatives = [row for row in missed if row.nano_bind_failed]

    true_positives.sort(key=lambda row: (-row.frequency_collapse_risk, row.result_id))
    false_positives.sort(key=_false_positive_sort_key)
    false_negatives.sort(key=lambda row: (-row.frequency_collapse_risk, row.result_id))

    return {
        "true_positives": [
            _example(row, "true_positive", salvageable_templates)
            for row in true_positives[:MAX_EXAMPLES_PER_CLASS]
        ],
        "false_positives": [
            _example(row, "false_positive", salvageable_templates)
            for row in false_positives[:MAX_EXAMPLES_PER_CLASS]
        ],
        "false_negatives": [
            _example(row, "false_negative", salvageable_templates)
            for row in false_negatives[:MAX_EXAMPLES_PER_CLASS]
        ],
    }


def _rule_by_name(name: str) -> Rule:
    for rule in candidate_rules():
        if rule.name == name:
            return rule
    raise ValueError(f"unknown rule: {name}")


def _template_policy_rows(template_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in template_rows:
        n = int(row["n"])
        nano_rate = float(row["nano_bind_rate"])
        high_risk = float(row["high_freq_risk_rate"])
        pos_rate = float(row["effective_pos_mixer_rate"])
        if n < 30:
            continue
        if nano_rate >= 0.25 and high_risk >= 0.50:
            action = "downweight_or_constrain"
        elif nano_rate >= 0.10 and pos_rate > 0.0:
            action = "rescue_by_slot_fill"
        elif nano_rate <= 0.02 and pos_rate >= 0.50:
            action = "mine"
        else:
            continue
        out.append(
            {
                "template_name": row["template_name"],
                "n": n,
                "nano_bind_rate": nano_rate,
                "high_freq_risk_rate": high_risk,
                "effective_pos_mixer_rate": pos_rate,
                "ppl_lt_50": row["ppl_lt_50"],
                "action": action,
            }
        )
    action_order = {"downweight_or_constrain": 0, "rescue_by_slot_fill": 1, "mine": 2}
    return sorted(
        out,
        key=lambda row: (
            action_order[row["action"]],
            -float(row["nano_bind_rate"]),
            -int(row["n"]),
        ),
    )


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


def _examples_table(examples: list[Example]) -> list[dict[str, Any]]:
    return [asdict(example) for example in examples]


def build_payload(meta_db: str | Path, *, rule_limit: int = 5) -> dict[str, Any]:
    rows = load_graph_rows(meta_db)
    rule_metrics = analyze_rules(rows)[:rule_limit]
    template_rows = analyze_templates(meta_db)
    salvageable_templates = {
        str(row["template_name"])
        for row in template_rows
        if bool(row.get("salvageable_by_slot_fill"))
    }
    rule_reviews: list[dict[str, Any]] = []
    for metrics in rule_metrics:
        rule = _rule_by_name(str(metrics["rule"]))
        examples = sample_rule_examples(rows, rule, salvageable_templates)
        false_positive_hatches: dict[str, int] = {}
        for example in examples["false_positives"]:
            for hatch in example.escape_hatches.split(","):
                if hatch:
                    false_positive_hatches[hatch] = (
                        false_positive_hatches.get(hatch, 0) + 1
                    )
        rule_reviews.append(
            {
                "metrics": metrics,
                "false_positive_escape_hatches": dict(
                    sorted(
                        false_positive_hatches.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ),
                "examples": {
                    key: _examples_table(value) for key, value in examples.items()
                },
            }
        )

    return {
        "summary": {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "meta_db": str(meta_db),
            "n_graphs": len(rows),
            "nano_bind_failures": sum(row.nano_bind_failed for row in rows),
            "recommendation": "soft triage only; no hard reject yet",
        },
        "rule_reviews": rule_reviews,
        "template_policy": _template_policy_rows(template_rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    best = payload["rule_reviews"][0]["metrics"] if payload["rule_reviews"] else {}
    lines = [
        f"# Meta Surrogate Screening Proposal — {summary['created_utc']}",
        "",
        "This is a review-only screening proposal. It does not recommend a hard reject gate yet.",
        "",
        "## Recommendation",
        "",
        "- Use the best rule as a soft triage priority and investigation flag.",
        "- Do not wire it as S0/S0.5 rejection without reviewing false positives.",
        "- Use escape hatches for strong real metrics and explicit positional mixers.",
        "- Downweight or constrain high-risk template/slot fills separately from hard rejection.",
        "",
        "## Best Candidate Rule",
        "",
        f"- Rule: `{best.get('rule', '')}`",
        f"- Support: {best.get('support', '')}",
        f"- Precision: {best.get('precision', '')}",
        f"- Recall: {best.get('recall', '')}",
        f"- Lift: {best.get('lift', '')}",
        "",
        "## Rule Reviews",
        "",
    ]
    for review in payload["rule_reviews"]:
        metrics = review["metrics"]
        lines.extend(
            [
                f"### {metrics['rule']}",
                "",
                f"- Support: {metrics['support']}",
                f"- NanoBind failures caught: {metrics['nano_bind_failures']}",
                f"- Precision: {metrics['precision']}",
                f"- Recall: {metrics['recall']}",
                f"- Lift: {metrics['lift']}",
                f"- False-positive escape hatches: {review['false_positive_escape_hatches']}",
                "",
                "False-positive examples with good metrics are the main blocker for hard gating.",
                "",
                "False positives:",
                "",
                *_md_table(
                    review["examples"]["false_positives"],
                    [
                        "result_id",
                        "template_name",
                        "wikitext_perplexity",
                        "tinystories_score",
                        "controlled_lang_s05_sa_score",
                        "escape_hatches",
                    ],
                    limit=8,
                ),
                "False negatives:",
                "",
                *_md_table(
                    review["examples"]["false_negatives"],
                    [
                        "result_id",
                        "template_name",
                        "wikitext_perplexity",
                        "frequency_collapse_risk",
                        "non_norm_motif_count",
                        "has_effective_positional_mixer",
                    ],
                    limit=8,
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Template Actions",
            "",
            *_md_table(
                payload["template_policy"],
                [
                    "template_name",
                    "n",
                    "nano_bind_rate",
                    "high_freq_risk_rate",
                    "effective_pos_mixer_rate",
                    "ppl_lt_50",
                    "action",
                ],
                limit=40,
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-db", default=DEFAULT_META_ANALYSIS_DB)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--rule-limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or (
        "meta_surrogate_screening_proposal_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    )
    payload = build_payload(args.meta_db, rule_limit=args.rule_limit)
    json_path = report_dir / f"{prefix}.json"
    md_path = report_dir / f"{prefix}.md"
    template_csv = report_dir / f"{prefix}_template_actions.csv"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(md_path, payload)
    write_csv(template_csv, payload["template_policy"])

    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "template_actions_csv": str(template_csv),
                "n_rule_reviews": len(payload["rule_reviews"]),
                "n_template_actions": len(payload["template_policy"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
