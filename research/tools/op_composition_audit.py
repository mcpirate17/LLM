"""Read-only op-composition triage across the leaderboard.

Joins ``program_results`` (for ``wikitext_perplexity``) with
``program_graph_features`` (for ``motifs_json`` / ``template_name``).  Uses
:func:`research.eval.op_composition.classify_motifs` to assign multi-label op
flags per row.

Output:

* CSV at ``--out-csv`` — one row per (result_id, has_attention, has_ssm,
  has_routing, ..., template, ppl, composite) for queryable workflow.
* Markdown report at ``--out-md`` — PPL-band cross-tab + per-flag top-N + the
  "no-attention but learns" subset Tim asked for.

No GPU.  No DB writes.  No scoring/gate change.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from research.eval.op_composition import OP_LABELS, classify_motifs

logger = logging.getLogger(__name__)

_PPL_BANDS: tuple[tuple[str, float, float], ...] = (
    ("excellent (<50)", 0, 50),
    ("good [50,200)", 50, 200),
    ("ok [200,500)", 200, 500),
    ("weak [500,1000)", 500, 1000),
    ("broken [1000,+)", 1000, float("inf")),
)


def _load_rows(db: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            """
            SELECT pr.result_id, pr.wikitext_perplexity, pr.tinystories_score,
                   l.composite_score, l.tier, l.induction_intermediate_auc,
                   pgf.template_name, pgf.motifs_json, pgf.op_count
            FROM program_results pr
            LEFT JOIN leaderboard l ON l.result_id = pr.result_id
            LEFT JOIN program_graph_features pgf ON pgf.result_id = pr.result_id
            WHERE pr.wikitext_perplexity IS NOT NULL
              AND pgf.motifs_json IS NOT NULL
            """
        )
        rows = [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()
    for r in rows:
        r["op_flags"] = classify_motifs(r["motifs_json"])
    return rows


def _ppl_band(ppl: float | None) -> str:
    if ppl is None or not math.isfinite(float(ppl)):
        return "n/a"
    p = float(ppl)
    for name, lo, hi in _PPL_BANDS:
        if lo <= p < hi:
            return name
    return "n/a"


def _crosstab_band_by_flag(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Returns ``out[band][flag] = count`` and ``out[band]['_total'] = ...``."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        band = _ppl_band(r["wikitext_perplexity"])
        out[band]["_total"] += 1
        for label, on in r["op_flags"].items():
            if on:
                out[band][label] += 1
    return out


def _per_flag_top_n(
    rows: list[dict[str, Any]], n: int = 10
) -> dict[str, list[dict[str, Any]]]:
    """Top-N rows by lowest PPL within each (flag=True) subset."""
    out: dict[str, list[dict[str, Any]]] = {}
    for label in OP_LABELS:
        subset = [r for r in rows if r["op_flags"].get(label)]
        subset.sort(key=lambda r: float(r["wikitext_perplexity"]))
        out[label] = subset[:n]
    return out


def _no_attention_learners(
    rows: list[dict[str, Any]], ppl_max: float = 100.0
) -> list[dict[str, Any]]:
    """Subset Tim asked for: archs that learn (PPL < ppl_max) without attention."""
    return sorted(
        [
            r
            for r in rows
            if not r["op_flags"].get("attention")
            and float(r["wikitext_perplexity"]) < ppl_max
        ],
        key=lambda r: float(r["wikitext_perplexity"]),
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    flag_cols = [f"has_{label}" for label in OP_LABELS]
    fieldnames = [
        "result_id",
        "template_name",
        "tier",
        "wikitext_perplexity",
        "composite_score",
        "tinystories_score",
        "induction_intermediate_auc",
        "op_count",
        *flag_cols,
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {
                "result_id": r["result_id"],
                "template_name": r["template_name"],
                "tier": r["tier"],
                "wikitext_perplexity": r["wikitext_perplexity"],
                "composite_score": r["composite_score"],
                "tinystories_score": r["tinystories_score"],
                "induction_intermediate_auc": r["induction_intermediate_auc"],
                "op_count": r["op_count"],
            }
            for label in OP_LABELS:
                row[f"has_{label}"] = int(bool(r["op_flags"].get(label)))
            writer.writerow(row)


def _md_header(rows: list[dict[str, Any]], csv_path: Path) -> list[str]:
    return [
        "# Op-composition triage (read-only, no GPU)",
        "",
        f"Source: `program_results` × `program_graph_features` "
        f"({len(rows)} rows with both PPL and motifs).",
        "Multi-label classifier from `research.eval.op_composition.classify_motifs`. "
        "Codex's prior `_family()` was single-label by template name; this is "
        "per-motif so hybrid templates count toward each constituent label.",
        "",
        f"CSV with all rows + flags: `{csv_path}`",
        "",
    ]


def _md_crosstab(crosstab: dict[str, dict[str, int]]) -> list[str]:
    out = [
        "## PPL band × op-flag presence",
        "",
        "Cells = count of rows in the band that have that flag set. "
        "`_total` = rows in the band (at least one row may have multiple "
        "flags so flag columns can sum > _total).",
        "",
    ]
    header = ["band", "_total", *OP_LABELS]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join("---:" if c != "band" else "---" for c in header) + "|")
    band_order = [name for name, *_ in _PPL_BANDS] + ["n/a"]
    for band in band_order:
        cells = [band, str(crosstab.get(band, {}).get("_total", 0))]
        for label in OP_LABELS:
            cells.append(str(crosstab.get(band, {}).get(label, 0)))
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return out


def _md_per_flag_topn(
    rows: list[dict[str, Any]], top_n: dict[str, list[dict[str, Any]]]
) -> list[str]:
    out = ["## Per-flag top-10 by lowest WikiText PPL", ""]
    for label in OP_LABELS:
        subset = top_n[label]
        if not subset:
            continue
        ppls = [float(r["wikitext_perplexity"]) for r in subset]
        median_ppl = statistics.median(ppls) if ppls else float("nan")
        all_subset = [r for r in rows if r["op_flags"].get(label)]
        out.append(
            f"### `{label}` (n={len(all_subset)} rows, top-10 median PPL "
            f"{median_ppl:.1f})"
        )
        out.append("")
        out.append("| rid | template | ppl | composite | tier |")
        out.append("|---|---|---:|---:|---|")
        for r in subset:
            comp = r["composite_score"]
            comp_s = f"{comp:.0f}" if comp is not None else "—"
            tmpl = (r["template_name"] or "?")[:30]
            out.append(
                f"| `{r['result_id'][:14]}` | {tmpl} | "
                f"{float(r['wikitext_perplexity']):.1f} | {comp_s} | "
                f"{r['tier'] or '—'} |"
            )
        out.append("")
    return out


def _md_no_attn(no_attn: list[dict[str, Any]], ppl_max_no_attn: float) -> list[str]:
    out = [
        f"## No-attention archs that learn (PPL < {ppl_max_no_attn:.0f}) — "
        f"the subset Tim asked for",
        "",
        f"{len(no_attn)} rows match `has_attention=False` AND "
        f"`wikitext_perplexity < {ppl_max_no_attn:.0f}`. Top-30 below by PPL.",
        "",
        "| rid | template | ppl | composite | has_ssm | has_routing | has_conv | has_compress |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in no_attn[:30]:
        comp = r["composite_score"]
        comp_s = f"{comp:.0f}" if comp is not None else "—"
        f = r["op_flags"]
        out.append(
            f"| `{r['result_id'][:14]}` | {(r['template_name'] or '?')[:30]} | "
            f"{float(r['wikitext_perplexity']):.1f} | {comp_s} | "
            f"{int(f.get('ssm', 0))} | {int(f.get('routing', 0))} | "
            f"{int(f.get('conv', 0))} | {int(f.get('compress', 0))} |"
        )
    out.append("")
    return out


def _render_markdown(
    rows: list[dict[str, Any]],
    crosstab: dict[str, dict[str, int]],
    top_n: dict[str, list[dict[str, Any]]],
    no_attn: list[dict[str, Any]],
    *,
    ppl_max_no_attn: float,
    csv_path: Path,
) -> str:
    parts: list[str] = []
    parts.extend(_md_header(rows, csv_path))
    parts.extend(_md_crosstab(crosstab))
    parts.extend(_md_per_flag_topn(rows, top_n))
    parts.extend(_md_no_attn(no_attn, ppl_max_no_attn))
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=Path("research/runs.db"), type=Path)
    parser.add_argument(
        "--out-csv",
        default=Path("research/reports/op_composition_audit_2026-05-03.csv"),
        type=Path,
    )
    parser.add_argument(
        "--out-md",
        default=Path("research/reports/op_composition_audit_2026-05-03.md"),
        type=Path,
    )
    parser.add_argument(
        "--no-attn-ppl-max",
        type=float,
        default=100.0,
        help="Threshold for the 'no-attention learners' subset.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = _load_rows(args.db)
    logger.info("loaded %d rows with both PPL and motifs", len(rows))
    crosstab = _crosstab_band_by_flag(rows)
    top_n = _per_flag_top_n(rows, n=10)
    no_attn = _no_attention_learners(rows, ppl_max=args.no_attn_ppl_max)
    logger.info(
        "no-attention learners (ppl<%.0f): %d", args.no_attn_ppl_max, len(no_attn)
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.out_csv)
    md = _render_markdown(
        rows,
        crosstab,
        top_n,
        no_attn,
        ppl_max_no_attn=args.no_attn_ppl_max,
        csv_path=args.out_csv,
    )
    args.out_md.write_text(md)
    logger.info("wrote %s and %s", args.out_csv, args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
