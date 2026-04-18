#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from research.scientist.analytics.model_strength import build_model_strength_report

DEFAULT_DB = "research/lab_notebook.db"
DEFAULT_OUT_DIR = "research/reports/model_strength"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _top_names(rows: list[dict[str, Any]], *, limit: int = 5) -> str:
    if not rows:
        return "none"
    return ", ".join(
        f"{row['name']} (effect={_fmt(row.get('adjusted_effect'))}, n={row.get('support_graphs')}, tier={row.get('confidence_tier', 'n/a')})"
        for row in rows[:limit]
    )


def _build_markdown(report: dict[str, Any]) -> str:
    support = report["support"]
    rankings = report["rankings"]
    drift = report["drift_analysis"]
    weight_bias = report["weight_bias"]
    lines: list[str] = []
    lines.append("# Model Strength Audit")
    lines.append("")
    lines.append("## Corpus")
    lines.append(
        f"- All runs: {support['all_runs']['runs']} rows / {support['all_runs']['unique_graphs']} graphs"
    )
    lines.append(
        f"- Trusted slice: {support['dedup_trusted']['runs']} deduped rows / {support['dedup_trusted']['unique_graphs']} graphs"
    )
    lines.append(
        f"- Promotable comparable slice: {support['dedup_promotable']['runs']} deduped rows / {support['dedup_promotable']['unique_graphs']} graphs"
    )
    lines.append(
        f"- Deep metric coverage: induction {support['dedup_promotable']['induction_coverage']}, binding {support['dedup_promotable']['binding_coverage']}, HellaSwag {support['dedup_promotable']['hellaswag_coverage']}, WikiText {support['dedup_promotable']['wikitext_coverage']}"
    )
    lines.append("")
    lines.append("## Great Model Definition")
    definition = report["great_model_definition"]
    lines.append(
        "- Primary metrics: "
        + ", ".join(
            f"{item['metric']} ({item['direction']}, w={_fmt(item['weight'], 2)})"
            for item in definition["primary_metrics"]
        )
    )
    lines.append(
        "- Secondary metrics: "
        + ", ".join(
            f"{item['metric']} ({item['direction']}, w={_fmt(item['weight'], 2)})"
            for item in definition["secondary_metrics"]
        )
    )
    lines.append("- Penalties: " + ", ".join(definition["penalties"]))
    lines.append("")
    lines.append("## Top Signals")
    lines.append("- Components: " + _top_names(rankings["best_components_overall"]))
    lines.append("- Pairs: " + _top_names(rankings["best_pairs_overall"]))
    lines.append(
        "- Slot/component combos: "
        + _top_names(rankings["best_slot_components_overall"])
    )
    lines.append("- Templates: " + _top_names(rankings["best_templates_overall"]))
    lines.append(
        "- Structural patterns: "
        + _top_names(rankings["best_structural_patterns_overall"])
    )
    lines.append("")
    lines.append("## Metric Families")
    for metric_name in (
        "quality_metric",
        "induction_auc",
        "binding_auc",
        "hellaswag_acc",
        "wikitext_quality",
        "stability_score",
        "efficiency_metric",
    ):
        section = rankings.get(metric_name) or {}
        lines.append(
            f"- {metric_name}: comps {_top_names(section.get('components', []), limit=3)}; templates {_top_names(section.get('templates', []), limit=3)}"
        )
    lines.append("")
    lines.append("## Loss-Ratio Drift")
    lines.append(
        f"- Trusted/promotable medians: trusted {_fmt(drift['distribution_checks']['trusted_loss_ratio_median'])}, promotable {_fmt(drift['distribution_checks']['promotable_loss_ratio_median'])}, search-all {_fmt(drift['distribution_checks']['search_all_loss_ratio_median'])}, search-promotable {_fmt(drift['distribution_checks']['search_promotable_loss_ratio_median'])}"
    )
    for key, value in drift["models"].items():
        if not value:
            continue
        lines.append(
            f"- {key}: time_coef={_fmt(value['time_coef'])}, p={_fmt(value['time_p_value'])}, R²={_fmt(value['r2'])}, n={value['n_obs']}"
        )
    lines.append("")
    lines.append("## Weighting Bias")
    if weight_bias["top_weighted_categories"]:
        lines.append(
            "- Mean category weights: "
            + ", ".join(
                f"{row['category']}={_fmt(row['mean_weight'])}"
                for row in weight_bias["top_weighted_categories"]
            )
        )
    if weight_bias["category_weight_vs_loss"]:
        lines.append(
            "- Strongest weight correlations: "
            + ", ".join(
                f"{row['category']} (corr loss={_fmt(row['corr_with_loss_ratio'])}, corr S1={_fmt(row['corr_with_stage1_passed'])})"
                for row in weight_bias["category_weight_vs_loss"][:5]
            )
        )
    lines.append("")
    lines.append("## Provenance")
    lines.append(
        "- Base query: `program_results` joined to `experiments` via `experiment_id`"
    )
    lines.append(
        "- Exact functions: "
        + ", ".join(report["query_provenance"]["source_functions"])
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate confounder-aware model strength report"
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-support", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    report = build_model_strength_report(
        args.db,
        min_support=max(args.min_support, 3),
        top_k=max(args.top_k, 5),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "model_strength_report.json", report)
    (out_dir / "model_strength_findings.md").write_text(
        _build_markdown(report),
        encoding="utf-8",
    )

    for table_name, rows in (
        ("best_components.csv", report["rankings"]["best_components_overall"]),
        ("best_component_pairs.csv", report["rankings"]["best_pairs_overall"]),
        (
            "best_slot_components.csv",
            report["rankings"]["best_slot_components_overall"],
        ),
        ("best_templates.csv", report["rankings"]["best_templates_overall"]),
        (
            "best_structural_patterns.csv",
            report["rankings"]["best_structural_patterns_overall"],
        ),
    ):
        _write_csv(out_dir / table_name, rows)


if __name__ == "__main__":
    main()
