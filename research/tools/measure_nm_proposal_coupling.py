"""O3 deliverable: measure the NM-op proposal-rate shift from the grammar coupling.

Before this session, `_build_compaction_block` / `_build_nmf_block` chose the
concrete NM op with a uniform `rng.choice(pool)`, so a strong S1 performer and a
weak one were proposed with identical probability regardless of measured success.
The fix routes both blocks through `weighted_op_choice`, which reads the same
`op_stats`-derived per-op weights (`graph.metadata["_op_weights"]`) that
`resolve_step` consumes.

This tool quantifies the effect: it reconstructs each NM op's grammar weight from
its ACTUAL `graph_runs` outcomes using the production `grammar_support` formula
(`_loss_quality_factor * _s1_quality_factor`, support-shrunk, clamped [0.25, 4.5]),
then measures each op's per-build proposal rate over N grammar seeds with the
coupling OFF (uniform, the old behavior) vs ON (weighted). Read-only on runs.db.

Two scenarios are reported:
  * `current` — real eval counts (1-8 today; most below the confidence prior of
    10, so weights sit near 1.0 and the near-term lift is modest — itself the
    finding that op_stats must accrue NM evals for the coupling to bite).
  * `matured` — the SAME per-op s1_rate / loss_ratio scaled to eval_count=40,
    showing the lift the coupling delivers once evidence accumulates.

CAVEAT surfaced by this measurement: `op_stats` currently holds ZERO rows for any
of the 18 NM ops, so in production the coupling is dormant (uniform fallback)
until `research/tools/backfill_stats.py` refreshes op_stats with NM outcomes.
That refresh is the second half of the O3 funnel fix; run it when runs.db is free
of a live campaign.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

from research.synthesis._templates_compaction import COMPACTION_OPS
from research.synthesis._templates_nmf import NMF_OPS
from research.synthesis.graph import ComputationGraph
from research.synthesis.grammar_support import (
    _bounded,
    _loss_quality_factor,
    _s1_quality_factor,
    _support_shrunk_multiplier,
)
from research.synthesis.templates import TEMPLATES

_DB = Path(__file__).resolve().parents[1] / "runs.db"
_NM_OPS = tuple(dict.fromkeys(COMPACTION_OPS + NMF_OPS))  # 18, order-stable, deduped
_OP_WEIGHT_CLAMP = (0.25, 4.5)  # grammar_support._build_db_op_weights
_CONFIDENCE_PRIOR = 10.0  # grammar_support prior used for op weights

_TEMPLATES = {
    "compaction_mixer_block": COMPACTION_OPS,
    "nmf_mixer_block": NMF_OPS,
}


def _fetch_op_stats(db: Path) -> dict[str, dict]:
    """Reconstruct (eval_count, s1_pass, mean loss_ratio) per NM op from
    graph_runs, mirroring what backfill_stats would fold into op_stats."""
    placeholders = ",".join("?" for _ in _NM_OPS)
    sql = f"""
        SELECT pgo.op_name,
               COUNT(DISTINCT gr.result_id) AS eval_count,
               SUM(gr.stage1_passed)        AS s1_pass_count,
               AVG(gr.loss_ratio)           AS mean_loss
        FROM program_graph_ops pgo
        JOIN graph_runs gr ON gr.result_id = pgo.result_id
        WHERE pgo.op_name IN ({placeholders})
          AND gr.loss_ratio IS NOT NULL
        GROUP BY pgo.op_name
    """
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(sql, _NM_OPS).fetchall()
    finally:
        conn.close()
    return {
        name: {
            "eval_count": int(ec or 0),
            "s1_pass_count": int(s1 or 0),
            "mean_loss": float(ml) if ml is not None else None,
        }
        for name, ec, s1, ml in rows
    }


def _op_weight(eval_count: int, s1_pass: int, mean_loss: float | None) -> float:
    """The production per-op grammar weight (capability_term omitted → 1.0, as it
    is for NM ops lacking capability-AUC columns)."""
    s1_rate = s1_pass / max(eval_count, 1)
    perf_term = _loss_quality_factor(mean_loss, k=2.0) * _s1_quality_factor(s1_rate)
    raw = _support_shrunk_multiplier(perf_term, eval_count, prior=_CONFIDENCE_PRIOR)
    lo, hi = _OP_WEIGHT_CLAMP
    return round(_bounded(raw, lo=lo, hi=hi), 4)


def _weights_for(
    stats: dict[str, dict], *, eval_override: int | None
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for op, s in stats.items():
        ec = eval_override if eval_override is not None else s["eval_count"]
        weights[op] = _op_weight(ec, s["s1_pass_count"], s["mean_loss"])
    return weights


def _proposal_rates(
    template_name: str, weights: dict[str, float] | None, n_builds: int, seed0: int
) -> dict[str, float]:
    """Fraction of builds whose graph contains each op."""
    pool = _TEMPLATES[template_name]
    counts = {op: 0 for op in pool}
    for i in range(n_builds):
        graph = ComputationGraph(model_dim=64)
        if weights is not None:
            graph.metadata["_op_weights"] = weights
        inp = graph.add_input()
        out = TEMPLATES[template_name](graph, inp, random.Random(seed0 + i), None)
        graph.set_output(out)
        present = {n.op_name for n in graph.nodes.values()}
        for op in pool:
            if op in present:
                counts[op] += 1
    return {op: counts[op] / n_builds for op in pool}


def _measure(stats: dict, n_builds: int) -> dict:
    report: dict = {}
    for scenario, override in (("current", None), ("matured", 40)):
        weights = _weights_for(stats, eval_override=override)
        per_template = {}
        for template_name, pool in _TEMPLATES.items():
            before = _proposal_rates(template_name, None, n_builds, seed0=0)
            after = _proposal_rates(template_name, weights, n_builds, seed0=10_000)
            rows = []
            for op in pool:
                rows.append(
                    {
                        "op": op,
                        "weight": weights.get(op, 1.0),
                        "rate_before": round(before[op], 4),
                        "rate_after": round(after[op], 4),
                        "shift": round(after[op] - before[op], 4),
                    }
                )
            rows.sort(key=lambda r: r["weight"], reverse=True)
            per_template[template_name] = rows
        report[scenario] = {"weights": weights, "per_template": per_template}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-builds", type=int, default=800, help=">=600 per plan")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reports"
        / "nm_proposal_coupling_2026-07-02.json",
    )
    args = ap.parse_args()

    stats = _fetch_op_stats(_DB)
    report = _measure(stats, args.n_builds)
    report["meta"] = {
        "n_builds": args.n_builds,
        "nm_ops_with_graph_runs": sorted(stats),
        "op_stats_nm_rows_today": 0,
        "note": (
            "op_stats holds 0 NM rows today; coupling is dormant (uniform "
            "fallback) in production until backfill_stats refreshes op_stats."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    for scenario in ("current", "matured"):
        print(f"\n=== scenario: {scenario} ===")
        for template_name, rows in report[scenario]["per_template"].items():
            top = rows[0]
            bot = rows[-1]
            print(f"  {template_name}: {args.n_builds} builds")
            print(
                f"    top-weight {top['op']} (w={top['weight']}): "
                f"{top['rate_before']} -> {top['rate_after']} "
                f"(shift {top['shift']:+.4f})"
            )
            print(
                f"    low-weight {bot['op']} (w={bot['weight']}): "
                f"{bot['rate_before']} -> {bot['rate_after']} "
                f"(shift {bot['shift']:+.4f})"
            )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
