"""Read-only audit of existing real-language metrics across architecture families.

This is intentionally not another probe.  It audits metrics that are already
persisted for thousands of graphs.  The primary report-only score is built from
real text metrics (WikiText + TinyStories when present).  HellaSwag and BLiMP
stay diagnostic because they are noisy at nano scale.

Usage:
    python -m research.tools.real_lm_quickcheck_audit \
        --out-md research/reports/real_lm_quickcheck_audit.md \
        --out-json research/reports/real_lm_quickcheck_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DB = Path("research/lab_notebook.db")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _family(template: str | None) -> str:
    t = (template or "").lower()
    if "token_merge" in t:
        return "token_merge"
    if "retrieval" in t:
        return "retrieval"
    if "ssm" in t or "mamba" in t or "rwkv" in t or "recurrent" in t:
        return "ssm_recurrent"
    if "attn" in t or "attention" in t:
        return "attention"
    if "conditional" in t:
        return "conditional_compute"
    return "other"


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def _std(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0 if vals else None
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _spearman(rows: list[dict[str, Any]], a: str, b: str) -> float | None:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        av = _safe_float(row.get(a))
        bv = _safe_float(row.get(b))
        if av is not None and bv is not None:
            pairs.append((av, bv))
    if len(pairs) < 3:
        return None
    ar = _rankdata([p[0] for p in pairs])
    br = _rankdata([p[1] for p in pairs])
    am = sum(ar) / len(ar)
    bm = sum(br) / len(br)
    num = sum((x - am) * (y - bm) for x, y in zip(ar, br))
    den_a = math.sqrt(sum((x - am) ** 2 for x in ar))
    den_b = math.sqrt(sum((y - bm) ** 2 for y in br))
    if den_a <= 0 or den_b <= 0:
        return None
    return num / (den_a * den_b)


def _percentile_scores(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    keyed = [
        (str(row["result_id"]), _safe_float(row.get(key)))
        for row in rows
        if _safe_float(row.get(key)) is not None
    ]
    if not keyed:
        return {}
    ranks = _rankdata([v for _rid, v in keyed if v is not None])
    n = max(len(ranks) - 1, 1)
    return {rid: (rank - 1) / n for (rid, _value), rank in zip(keyed, ranks)}


def _load_rows(db: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT l.result_id,
                   l.entry_id,
                   l.tier,
                   l.composite_score,
                   l.induction_auc,
                   l.binding_composite,
                   pr.wikitext_perplexity,
                   pr.wikitext_score,
                   pr.hellaswag_acc,
                   pr.blimp_overall_accuracy,
                   pr.tinystories_score,
                   pr.controlled_lang_s05_sa_score,
                   pr.controlled_lang_inv_sa_score,
                   pgf.template_name
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            LEFT JOIN program_graph_features pgf ON pgf.result_id = l.result_id
            WHERE pr.wikitext_perplexity IS NOT NULL
               OR pr.wikitext_score IS NOT NULL
               OR pr.hellaswag_acc IS NOT NULL
               OR pr.blimp_overall_accuracy IS NOT NULL
               OR pr.tinystories_score IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()
    out = [dict(row) for row in rows]
    for row in out:
        row["family"] = _family(row.get("template_name"))
        ppl = _safe_float(row.get("wikitext_perplexity"))
        row["neg_log_wikitext_ppl"] = -math.log(ppl) if ppl and ppl > 0 else None
    return out


def _attach_quickcheck_scores(rows: list[dict[str, Any]]) -> None:
    core_keys = ("wikitext_score", "tinystories_score")
    breadth_keys = ("hellaswag_acc", "blimp_overall_accuracy")
    percentiles = {
        key: _percentile_scores(rows, key) for key in (*core_keys, *breadth_keys)
    }
    for row in rows:
        rid = str(row["result_id"])
        core_vals = [
            percentiles[key][rid] for key in core_keys if rid in percentiles[key]
        ]
        breadth_vals = [
            percentiles[key][rid] for key in breadth_keys if rid in percentiles[key]
        ]
        all_vals = core_vals + breadth_vals
        row["real_lm_core_score"] = (
            round(sum(core_vals) / len(core_vals), 4) if core_vals else None
        )
        row["real_lm_breadth_score"] = (
            round(sum(breadth_vals) / len(breadth_vals), 4) if breadth_vals else None
        )
        row["real_lm_quickcheck"] = (
            round(sum(all_vals) / len(all_vals), 4) if all_vals else None
        )


def _family_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    summary = []
    for family, items in sorted(grouped.items()):
        summary.append(
            {
                "family": family,
                "n": len(items),
                "core_mean": _mean(
                    row["real_lm_core_score"]
                    for row in items
                    if row.get("real_lm_core_score") is not None
                ),
                "core_std": _std(
                    row["real_lm_core_score"]
                    for row in items
                    if row.get("real_lm_core_score") is not None
                ),
                "quickcheck_mean": _mean(
                    row["real_lm_quickcheck"]
                    for row in items
                    if row.get("real_lm_quickcheck") is not None
                ),
                "quickcheck_std": _std(
                    row["real_lm_quickcheck"]
                    for row in items
                    if row.get("real_lm_quickcheck") is not None
                ),
                "wikitext_ppl_mean": _mean(
                    row["wikitext_perplexity"]
                    for row in items
                    if row.get("wikitext_perplexity") is not None
                ),
                "tinystories_mean": _mean(
                    row["tinystories_score"]
                    for row in items
                    if row.get("tinystories_score") is not None
                ),
                "hellaswag_mean": _mean(
                    row["hellaswag_acc"]
                    for row in items
                    if row.get("hellaswag_acc") is not None
                ),
                "blimp_mean": _mean(
                    row["blimp_overall_accuracy"]
                    for row in items
                    if row.get("blimp_overall_accuracy") is not None
                ),
            }
        )
    return summary


def _fmt(value: Any, digits: int = 3) -> str:
    val = _safe_float(value)
    if val is None:
        return "-"
    return f"{val:.{digits}f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real LM Quickcheck Audit",
        "",
        "Read-only audit over already persisted leaderboard metrics. No new probe",
        "training, no DB writes, no scoring changes.",
        "",
        "Primary score: `real_lm_core_score`, a percentile average over existing",
        "WikiText and TinyStories scores when present. HellaSwag and BLiMP are",
        "reported separately as breadth diagnostics.",
        "",
        "## Coverage",
        "",
        f"- rows: {report['coverage']['rows']}",
        f"- families: {report['coverage']['families']}",
        "",
        "## Correlations",
        "",
        "| metric | Spearman vs composite | Spearman vs WikiText score |",
        "|---|---:|---:|",
    ]
    for item in report["correlations"]:
        lines.append(
            "| {metric} | {comp} | {wt} |".format(
                metric=item["metric"],
                comp=_fmt(item["spearman_vs_composite"]),
                wt=_fmt(item["spearman_vs_wikitext_score"]),
            )
        )
    lines.extend(
        [
            "",
            "## Family Means",
            "",
            "| family | n | core | quickcheck | ppl | tinystories | hs | blimp |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["family_summary"]:
        lines.append(
            "| {family} | {n} | {core} | {quick} | {ppl} | {tiny} | {hs} | {blimp} |".format(
                family=item["family"],
                n=item["n"],
                core=_fmt(item["core_mean"]),
                quick=_fmt(item["quickcheck_mean"]),
                ppl=_fmt(item["wikitext_ppl_mean"], 1),
                tiny=_fmt(item["tinystories_mean"]),
                hs=_fmt(item["hellaswag_mean"]),
                blimp=_fmt(item["blimp_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Top Core Rows",
            "",
            "| rank | result_id | family | template | core | quickcheck | composite | ppl | tiny | hs | blimp |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(report["top_core_rows"], start=1):
        lines.append(
            "| {idx} | `{rid}` | {family} | {template} | {core} | {quick} | {comp} | {ppl} | {tiny} | {hs} | {blimp} |".format(
                idx=idx,
                rid=row["result_id"],
                family=row["family"],
                template=row.get("template_name") or "-",
                core=_fmt(row["real_lm_core_score"]),
                quick=_fmt(row["real_lm_quickcheck"]),
                comp=_fmt(row["composite_score"], 1),
                ppl=_fmt(row["wikitext_perplexity"], 1),
                tiny=_fmt(row["tinystories_score"]),
                hs=_fmt(row["hellaswag_acc"]),
                blimp=_fmt(row["blimp_overall_accuracy"]),
            )
        )
    return "\n".join(lines) + "\n"


def build_report(db: Path) -> dict[str, Any]:
    rows = _load_rows(db)
    _attach_quickcheck_scores(rows)
    metric_keys = (
        "real_lm_core_score",
        "real_lm_quickcheck",
        "wikitext_score",
        "neg_log_wikitext_ppl",
        "hellaswag_acc",
        "blimp_overall_accuracy",
        "tinystories_score",
        "controlled_lang_s05_sa_score",
        "controlled_lang_inv_sa_score",
    )
    correlations = [
        {
            "metric": key,
            "spearman_vs_composite": _spearman(rows, key, "composite_score"),
            "spearman_vs_wikitext_score": _spearman(rows, key, "wikitext_score"),
        }
        for key in metric_keys
    ]
    top_core_rows = sorted(
        [row for row in rows if row.get("real_lm_core_score") is not None],
        key=lambda row: float(row["real_lm_core_score"]),
        reverse=True,
    )[:25]
    return {
        "coverage": {
            "rows": len(rows),
            "families": dict(
                sorted((k, len(v)) for k, v in _group_by_family(rows).items())
            ),
        },
        "correlations": correlations,
        "family_summary": _family_summary(rows),
        "top_core_rows": top_core_rows,
    }


def _group_by_family(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("research/reports/real_lm_quickcheck_audit.md"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("research/reports/real_lm_quickcheck_audit.json"),
    )
    args = parser.parse_args()

    report = build_report(args.db)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_markdown(report), encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
