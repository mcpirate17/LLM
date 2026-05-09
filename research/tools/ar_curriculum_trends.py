#!/usr/bin/env python
"""Trends/themes analyzer for ar_curriculum data.

Reads program_results rows that have ar_curriculum_auc_pair_final populated
and produces:

  - 2D scatter / quadrant counts of (auc_pair_final, s0_retention)
  - Per-template ranked breakdown (which templates_used patterns score high)
  - Per-paradigm breakdown (counts of motif primitives in graph_json:
    self_attention, ssm/mamba, rwkv/wkv, mlp/ffn, gating, etc.)
  - Spearman/Pearson correlations between upstream cheap signals and
    ar_curriculum_auc_pair_final / s0_retention. Used to prioritize the
    next wave of backfill candidates.

Output:
  research/runtime/ar_curriculum_experiment/trends_<run_id>.json
  research/runtime/ar_curriculum_experiment/trends_<run_id>.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.scientist.notebook import LabNotebook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO_ROOT / "research/runtime/ar_curriculum_experiment"
DEFAULT_DB = REPO_ROOT / "research/runs.db"

UPSTREAM_FEATURES: tuple[str, ...] = (
    "wikitext_perplexity",
    "induction_screening_auc",
    "binding_screening_auc",
    "ar_legacy_auc",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "fp_jacobian_erf_density",
    "fp_jacobian_erf_decay_slope",
    "validation_loss_ratio",
    "ar_intermediate_held_pair_lift",
    "ar_validation_held_pair_acc",
)

# Motifs to detect in graph_json by op-name substring matching.
MOTIF_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "attention",
        ("self_attention", "multihead", "softmax_attention", "flash_attention"),
    ),
    ("ssm", ("ssm", "mamba", "selective_scan", "state_space")),
    ("rwkv", ("rwkv", "wkv", "linear_attention")),
    ("conv", ("conv1d", "depthwise", "causal_conv")),
    ("mlp", ("ff_swiglu", "linear_proj", "gated_linear", "ffn")),
    ("gating", ("gated", "gate", "mixture_of_experts", "moe", "expert")),
    ("retrieval", ("retrieval", "cross_attention", "rag")),
    ("norm", ("layernorm", "rmsnorm", "groupnorm")),
)


def _ranks(xs: list[float]) -> list[float]:
    paired = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][1] == paired[i][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[paired[k][0]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return _pearson(_ranks(xs), _ranks(ys))


def _detect_motifs(graph_json_str: str | None) -> list[str]:
    if not graph_json_str:
        return []
    try:
        graph = json.loads(graph_json_str)
    except (TypeError, json.JSONDecodeError):
        return []
    op_names: list[str] = []
    for layer in graph.get("layers", []):
        for op in layer.get("ops", []):
            name = str(op.get("op_name") or op.get("name") or "")
            if name:
                op_names.append(name.lower())
    motifs: list[str] = []
    for tag, patterns in MOTIF_PATTERNS:
        if any(p in name for p in patterns for name in op_names):
            motifs.append(tag)
    return motifs


def _quadrant(
    auc: float, retention: float, *, auc_thr: float = 0.3, ret_thr: float = 0.5
) -> str:
    high_auc = auc >= auc_thr
    high_ret = retention >= ret_thr
    if high_auc and high_ret:
        return "Q1_learns_retains"
    if high_auc and not high_ret:
        return "Q2_learns_forgets"
    if not high_auc and high_ret:
        return "Q3_no_learn_retains"
    return "Q4_no_learn_no_retain"


def _fetch_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    feature_cols = ", ".join(f"pr.{f}" for f in UPSTREAM_FEATURES)
    sql = f"""
        SELECT
            pr.result_id,
            pr.graph_fingerprint,
            pr.graph_json,
            l.tier,
            l.composite_score,
            json_extract(pr.graph_json, '$.metadata.templates_used') AS templates_used,
            pr.ar_curriculum_auc_pair_final,
            pr.ar_curriculum_auc_class_final,
            pr.ar_curriculum_s0_held_pair_acc,
            pr.ar_curriculum_s0_retention,
            pr.ar_curriculum_max_passing_stage,
            pr.ar_curriculum_per_stage_held_pair_acc,
            {feature_cols}
        FROM program_results pr
        JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE pr.ar_curriculum_auc_pair_final IS NOT NULL
        ORDER BY pr.ar_curriculum_auc_pair_final DESC
    """
    rows = conn.execute(sql).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        d = (
            dict(row)
            if hasattr(row, "keys")
            else {k: row[i] for i, k in enumerate(row.keys())}
        )
        try:
            d["templates_list"] = (
                json.loads(d["templates_used"]) if d.get("templates_used") else []
            )
        except (TypeError, json.JSONDecodeError):
            d["templates_list"] = []
        d["motifs"] = _detect_motifs(d.get("graph_json"))
        try:
            d["per_stage"] = json.loads(
                d["ar_curriculum_per_stage_held_pair_acc"] or "[]"
            )
        except (TypeError, json.JSONDecodeError):
            d["per_stage"] = []
        out.append(d)
    return out


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    aucs = [r["ar_curriculum_auc_pair_final"] for r in rows]
    retentions = [r["ar_curriculum_s0_retention"] for r in rows]
    max_passes = [r["ar_curriculum_max_passing_stage"] for r in rows]

    def _summary(values: list[float]) -> dict[str, float]:
        return {
            "n": len(values),
            "mean": round(st.mean(values), 4),
            "median": round(st.median(values), 4),
            "std": round(st.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    quadrants: Counter[str] = Counter()
    for r in rows:
        quadrants[
            _quadrant(
                r["ar_curriculum_auc_pair_final"], r["ar_curriculum_s0_retention"]
            )
        ] += 1

    template_aucs: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for t in r["templates_list"]:
            template_aucs[str(t)].append(r["ar_curriculum_auc_pair_final"])
    template_summary = []
    for tpl, vals in template_aucs.items():
        if len(vals) >= 3:
            template_summary.append(
                {
                    "template": tpl,
                    "n": len(vals),
                    "mean_auc": round(st.mean(vals), 3),
                    "median_auc": round(st.median(vals), 3),
                    "std_auc": round(st.stdev(vals), 3) if len(vals) > 1 else 0.0,
                }
            )
    template_summary.sort(key=lambda d: d["mean_auc"], reverse=True)

    motif_aucs: dict[str, list[float]] = defaultdict(list)
    motif_retentions: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for m in r["motifs"]:
            motif_aucs[m].append(r["ar_curriculum_auc_pair_final"])
            motif_retentions[m].append(r["ar_curriculum_s0_retention"])
    motif_summary = []
    for m, vals in motif_aucs.items():
        if len(vals) >= 3:
            motif_summary.append(
                {
                    "motif": m,
                    "n": len(vals),
                    "mean_auc": round(st.mean(vals), 3),
                    "mean_retention": round(st.mean(motif_retentions[m]), 3),
                }
            )
    motif_summary.sort(key=lambda d: d["mean_auc"], reverse=True)

    correlations: list[dict[str, Any]] = []
    for feat in UPSTREAM_FEATURES:
        xs: list[float] = []
        ys_auc: list[float] = []
        ys_ret: list[float] = []
        for r in rows:
            v = r.get(feat)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            xs.append(fv)
            ys_auc.append(r["ar_curriculum_auc_pair_final"])
            ys_ret.append(r["ar_curriculum_s0_retention"])
        if len(xs) < 5:
            continue
        correlations.append(
            {
                "feature": feat,
                "n": len(xs),
                "spearman_vs_auc": round(_spearman(xs, ys_auc), 3),
                "pearson_vs_auc": round(_pearson(xs, ys_auc), 3),
                "spearman_vs_retention": round(_spearman(xs, ys_ret), 3),
            }
        )
    correlations.sort(key=lambda d: abs(d["spearman_vs_auc"]), reverse=True)

    top_archs = [
        {
            "fingerprint": r["graph_fingerprint"][:12],
            "tier": r["tier"],
            "composite": round(float(r["composite_score"] or 0), 1),
            "auc": round(r["ar_curriculum_auc_pair_final"], 3),
            "retention": round(r["ar_curriculum_s0_retention"], 3),
            "max_pass": r["ar_curriculum_max_passing_stage"],
            "templates": r["templates_list"][:2],
            "motifs": r["motifs"],
            "quadrant": _quadrant(
                r["ar_curriculum_auc_pair_final"], r["ar_curriculum_s0_retention"]
            ),
        }
        for r in rows[:30]
    ]

    return {
        "n": len(rows),
        "summary": {
            "auc_pair_final": _summary(aucs),
            "s0_retention": _summary(retentions),
            "max_passing_stage": _summary([float(m) for m in max_passes]),
        },
        "quadrants": dict(quadrants),
        "templates": template_summary,
        "motifs": motif_summary,
        "correlations": correlations,
        "top_archs": top_archs,
    }


def write_report(
    analysis: dict[str, Any], out_dir: Path, run_id: str
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"trends_{run_id}.json"
    md_path = out_dir / f"trends_{run_id}.md"
    json_path.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")

    n = analysis.get("n", 0)
    if n == 0:
        md_path.write_text("# AR curriculum trends — no data\n", encoding="utf-8")
        return json_path, md_path

    summ = analysis["summary"]
    lines: list[str] = [
        f"# AR curriculum trends — {run_id}",
        "",
        f"Analyzed n={n} archs with ar_curriculum data.",
        "",
        "## Distribution summary",
        "",
        "| metric | n | mean | median | std | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in [
        ("AUC pair final", "auc_pair_final"),
        ("S0 retention", "s0_retention"),
        ("Max passing stage", "max_passing_stage"),
    ]:
        s = summ[key]
        lines.append(
            f"| {label} | {s['n']} | {s['mean']:.3f} | {s['median']:.3f} | "
            f"{s['std']:.3f} | {s['min']:.3f} | {s['max']:.3f} |"
        )

    lines += [
        "",
        "## Quadrant distribution (auc≥0.3, retention≥0.5)",
        "",
        "| quadrant | description | n | % |",
        "|---|---|---:|---:|",
    ]
    descriptions = {
        "Q1_learns_retains": "high AUC + high retention (Mamba-class)",
        "Q2_learns_forgets": "high AUC + low retention (RWKV/dense-attn-class)",
        "Q3_no_learn_retains": "low AUC + high retention (under-trained or capacity-bound)",
        "Q4_no_learn_no_retain": "low AUC + low retention (broken)",
    }
    for q, desc in descriptions.items():
        c = analysis["quadrants"].get(q, 0)
        lines.append(f"| {q} | {desc} | {c} | {100 * c / n:.1f}% |")

    lines += [
        "",
        "## Top archs by AUC (top 30)",
        "",
        "| rank | fp | tier | composite | AUC | retention | max_pass | quadrant | motifs | template |",
        "|---:|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for i, r in enumerate(analysis["top_archs"], 1):
        tpl = r["templates"][0] if r["templates"] else "?"
        motifs = ",".join(r["motifs"]) if r["motifs"] else "—"
        lines.append(
            f"| {i} | {r['fingerprint']} | {r['tier']} | {r['composite']} | "
            f"{r['auc']:.3f} | {r['retention']:.3f} | {r['max_pass']} | "
            f"{r['quadrant'].replace('_', ' ')} | {motifs} | {tpl} |"
        )

    if analysis.get("templates"):
        lines += [
            "",
            "## Per-template breakdown (n≥3 archs/template)",
            "",
            "| template | n | mean AUC | median AUC | std |",
            "|---|---:|---:|---:|---:|",
        ]
        for t in analysis["templates"][:30]:
            lines.append(
                f"| {t['template']} | {t['n']} | {t['mean_auc']:.3f} | "
                f"{t['median_auc']:.3f} | {t['std_auc']:.3f} |"
            )

    if analysis.get("motifs"):
        lines += [
            "",
            "## Per-motif breakdown",
            "",
            "| motif | n | mean AUC | mean retention |",
            "|---|---:|---:|---:|",
        ]
        for m in analysis["motifs"]:
            lines.append(
                f"| {m['motif']} | {m['n']} | {m['mean_auc']:.3f} | "
                f"{m['mean_retention']:.3f} |"
            )

    if analysis.get("correlations"):
        lines += [
            "",
            "## Upstream-feature correlations",
            "",
            "Predictive value: high |spearman_vs_auc| means we can predict "
            "ar_curriculum from this cheap upstream signal.",
            "",
            "| feature | n | spearman vs AUC | pearson vs AUC | spearman vs retention |",
            "|---|---:|---:|---:|---:|",
        ]
        for c in analysis["correlations"]:
            lines.append(
                f"| {c['feature']} | {c['n']} | {c['spearman_vs_auc']:+.3f} | "
                f"{c['pearson_vs_auc']:+.3f} | {c['spearman_vs_retention']:+.3f} |"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    nb = LabNotebook(str(args.db), read_only=True)
    rows = _fetch_rows(nb.conn)
    nb.close()
    logger.info("Loaded %d archs with ar_curriculum data", len(rows))
    analysis = analyze(rows)
    json_path, md_path = write_report(analysis, RUNTIME_ROOT, run_id)
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
