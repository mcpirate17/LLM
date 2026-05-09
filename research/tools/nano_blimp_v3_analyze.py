"""Analyze a nano_blimp_v3 cohort audit JSON.

Reports:
  * Per-arch summary table (mean ± std) sorted by ho_score.
  * Spearman correlation of v3 ho_score and components against the
    existing leaderboard signals (composite, induction, binding, ppl).
  * SSM-class vs attention-class subcohort breakdown for the
    fairness check.
  * Hypothesis-match check: did the wider cohort reproduce the
    inverted ranking the 5-arch audit found?

Usage:
    python -m research.tools.nano_blimp_v3_analyze \
        research/reports/nano_blimp_v3_audit_top30_5seed_*.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

# Templates that contain explicit recurrent state-tracking lanes (Mamba,
# RWKV, latent-compress, SSM hybrids). Ranked by claim of "state-tracking"
# rather than "uses some attention".
_SSM_FAMILY_KEYWORDS = (
    "ssm",
    "latent_compress",
    "latent_attn_ssm",
    "mamba",
    "rwkv",
    "linear_attn",  # linear attn is rank-1 recurrent
)

_ATTN_FAMILY_KEYWORDS = (
    "attn_softmax",
    "local_attn",
    "attn_routing",
    "latent_attn_sparse",
    "latent_attn_conv",
)


def _classify(template: str | None) -> str:
    if not template:
        return "other"
    t = template.lower()
    for kw in _SSM_FAMILY_KEYWORDS:
        if kw in t:
            return "ssm_family"
    for kw in _ATTN_FAMILY_KEYWORDS:
        if kw in t:
            return "attn_family"
    return "other"


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Tie-aware Spearman ρ, no SciPy dep."""
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")

    def rank(vs: list[float]) -> list[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    n = len(rx)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _aggregate_per_arch(
    runs: list[dict[str, Any]], held_out_count: int
) -> list[dict[str, Any]]:
    by_arch: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        if r.get("status") != "ok" or r.get("held_out_count") != held_out_count:
            continue
        by_arch.setdefault(r["result_id"], []).append(r)
    out: list[dict[str, Any]] = []
    for rid, rows in by_arch.items():
        agg: dict[str, Any] = {
            "result_id": rid,
            "template": rows[0].get("template"),
            "n_seeds": len(rows),
        }
        for k in (
            "score",
            "held_out_score",
            "class_in_dist",
            "class_held_out",
            "binding_in_dist",
            "binding_held_out",
            "order",
        ):
            vals = [r[k] for r in rows]
            agg[f"{k}_mean"] = round(mean(vals), 4)
            agg[f"{k}_std"] = round(pstdev(vals), 4) if len(vals) > 1 else 0.0
        out.append(agg)
    return out


def _print_arch_table(
    per_arch: list[dict[str, Any]], summary: list[dict[str, Any]]
) -> None:
    # Join with leaderboard meta from summary (composite, induction, etc.).
    meta_by_rid: dict[str, dict[str, Any]] = {}
    for s in summary:
        meta_by_rid.setdefault(s["result_id"], s)

    rows = sorted(per_arch, key=lambda r: -r["held_out_score_mean"])
    header = (
        f"{'rank':>4s} {'result_id':30s} {'tmpl':28s} {'comp':>6s} "
        f"{'ind':>6s} {'bind':>6s} {'ppl':>7s} "
        f"{'cls_in':>11s} {'cls_ho':>11s} {'bnd_in':>11s} {'bnd_ho':>11s} "
        f"{'order':>11s} {'ho_score':>11s}"
    )
    print("\n=== Per-arch summary, sorted by ho_score (mean ± std over seeds) ===")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        m = meta_by_rid.get(r["result_id"], {})

        def fmt(name: str) -> str:
            return f"{r[f'{name}_mean']:.2f}±{r[f'{name}_std']:.2f}"

        print(
            f"{i:>4d} {r['result_id'][:30]:30s} {(r['template'] or '?')[:28]:28s} "
            f"{(m.get('composite') or 0):>6.0f} "
            f"{m.get('induction_screening_auc')!s:>6} "
            f"{m.get('binding_screening_auc')!s:>6} "
            f"{m.get('wikitext_perplexity')!s:>7} "
            f"{fmt('class_in_dist'):>11s} {fmt('class_held_out'):>11s} "
            f"{fmt('binding_in_dist'):>11s} {fmt('binding_held_out'):>11s} "
            f"{fmt('order'):>11s} {fmt('held_out_score'):>11s}"
        )


def _print_correlations(
    per_arch: list[dict[str, Any]], summary: list[dict[str, Any]]
) -> None:
    """Spearman ρ of v3 components vs leaderboard signals."""
    # Build aligned vectors (only archs that have both v3 result and leaderboard meta).
    meta_by_rid = {s["result_id"]: s for s in summary}
    paired = []
    for r in per_arch:
        m = meta_by_rid.get(r["result_id"])
        if not m:
            continue
        paired.append((r, m))

    def col_v3(name: str) -> list[float]:
        return [r[f"{name}_mean"] for r, _ in paired]

    def col_meta(key: str) -> list[float]:
        return [
            float(m[key]) if m.get(key) is not None else float("nan") for _, m in paired
        ]

    def safe_pairs(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
        out_x, out_y = [], []
        for x, y in zip(xs, ys):
            if math.isnan(x) or math.isnan(y):
                continue
            out_x.append(x)
            out_y.append(y)
        return out_x, out_y

    print("\n=== Spearman ρ: v3 components vs leaderboard signals ===")
    print(f"n = {len(paired)} archs with v3 results")
    print(
        f"{'':24s} {'composite':>10s} {'ind_auc':>10s} {'bind_auc':>10s} {'ppl':>10s}"
    )
    for v3_name in (
        "held_out_score",
        "class_held_out",
        "binding_held_out",
        "order",
        "score",
    ):
        v3_vals = col_v3(v3_name)
        line = f"{v3_name:24s}"
        for meta_key in (
            "composite",
            "induction_screening_auc",
            "binding_screening_auc",
            "wikitext_perplexity",
        ):
            x, y = safe_pairs(v3_vals, col_meta(meta_key))
            rho = _spearman(x, y) if len(x) >= 3 else float("nan")
            line += f" {rho:>10.3f}" if not math.isnan(rho) else f" {'n/a':>10s}"
        print(line)


def _print_subcohort(per_arch: list[dict[str, Any]]) -> None:
    print(
        "\n=== SSM-class vs attention-class vs other (mean ± std of per-arch ho_score) ==="
    )
    buckets: dict[str, list[float]] = {}
    by_template: dict[str, list[float]] = {}
    for r in per_arch:
        cls = _classify(r["template"])
        buckets.setdefault(cls, []).append(r["held_out_score_mean"])
        by_template.setdefault(r["template"] or "?", []).append(
            r["held_out_score_mean"]
        )

    for cls in ("ssm_family", "attn_family", "other"):
        vals = buckets.get(cls, [])
        if not vals:
            print(f"{cls:14s} n=0")
            continue
        m = mean(vals)
        s = pstdev(vals) if len(vals) > 1 else 0.0
        print(
            f"{cls:14s} n={len(vals):>2d}  ho_score={m:.3f} ± {s:.3f}  range=[{min(vals):.2f}, {max(vals):.2f}]"
        )

    print("\n=== Per-template mean ho_score (n>=2 only) ===")
    for tmpl, vals in sorted(by_template.items(), key=lambda kv: -mean(kv[1])):
        if len(vals) < 2:
            continue
        m = mean(vals)
        s = pstdev(vals)
        print(f"{tmpl:34s} n={len(vals):>2d}  ho_score={m:.3f} ± {s:.3f}")


def _print_5arch_check(
    per_arch: list[dict[str, Any]], target_ids: tuple[str, ...]
) -> None:
    """Did the wider cohort reproduce the original 5-arch ordering?"""
    by_rid = {r["result_id"]: r for r in per_arch}
    print("\n=== 5-arch hypothesis-match check ===")
    print("(original 3-seed ranking on these 5: 8d > 90 > 57 > f7 > ec)")
    rows = [(rid, by_rid.get(rid)) for rid in target_ids]
    rows = [(rid, r) for rid, r in rows if r is not None]
    if not rows:
        print("  none of the 5 archs found in the wider audit")
        return
    rows.sort(key=lambda kv: -kv[1]["held_out_score_mean"])
    for i, (rid, r) in enumerate(rows, 1):
        m = r["held_out_score_mean"]
        s = r["held_out_score_std"]
        print(
            f"  rank {i}: {rid:14s} {(r['template'] or '?')[:28]:28s} ho_score={m:.3f} ± {s:.3f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit_json", type=Path)
    ap.add_argument("--held-out", type=int, default=6)
    args = ap.parse_args()

    data = json.loads(args.audit_json.read_text())
    runs = data["runs"]
    summary = data["summary"]

    per_arch = _aggregate_per_arch(runs, held_out_count=args.held_out)
    if not per_arch:
        print(f"no rows with held_out_count={args.held_out} found")
        return 1

    _print_arch_table(per_arch, summary)
    _print_correlations(per_arch, summary)
    _print_subcohort(per_arch)
    _print_5arch_check(
        per_arch,
        (
            "ec7025d7-338",
            "574271ca-f37",
            "f70c17d0-d59",
            "8d087a16-692",
            "903157e5-219",
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
