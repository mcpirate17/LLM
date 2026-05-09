#!/usr/bin/env python
"""V4 — Mid-tier (composite 100-300) sweep at locked V3 config.

Validates discriminator behavior outside the top-by-composite cohort. The
top tier mostly maxed out (in_pair=1.0) so a real ranking spectrum needs
mid-tier where borderline cases live.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import statistics as st
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value

DB_PATH = REPO / "research/runs.db"
REPORTS_DIR = REPO / "research/reports"

SEEDS = (0, 1, 2)
SAMPLE_N = 20  # archs to sample from the mid-tier

CFG_KW = dict(
    wikitext_warmup_steps=2500,
    finetune_steps=400,
    n_pairs_per_noun=1,
    reps=10,
    n_distractors=480,
    n_adjectives=20,
    n_objects=25,
    timeout_s=600.0,
    from_s1=False,
)


def fetch_cohort() -> list[dict]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = conn.execute(
        """
        SELECT pr.result_id,
               COALESCE(lb.composite_score, 0) AS comp,
               COALESCE(pr.wikitext_perplexity, 0) AS ppl,
               json_extract(pr.graph_json, '$.metadata.templates_used') AS tpl,
               CASE WHEN pr.failure_op = 'nano_bind' THEN 1 ELSE 0 END AS nb_fail,
               pr.graph_json
        FROM program_results pr
        LEFT JOIN leaderboard lb ON lb.result_id = pr.result_id
        WHERE pr.graph_json IS NOT NULL
          AND lb.composite_score BETWEEN 100 AND 300
          AND pr.wikitext_perplexity IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (SAMPLE_N,),
    ).fetchall()
    out = []
    for r in rows:
        graph_json = resolve_graph_json_value(conn, DB_PATH, r[5])
        try:
            graph = json.loads(graph_json)
        except (TypeError, json.JSONDecodeError):
            graph = {}
        out.append(
            {
                "result_id": r[0],
                "composite_score": float(r[1]),
                "wikitext_perplexity": float(r[2]),
                "templates": (graph.get("metadata") or {}).get("templates_used") or [],
                "nano_bind_failer": bool(r[4]),
                "graph_json": graph_json,
            }
        )
    conn.close()
    return out


def main() -> None:
    from research.eval.ar_gate import ARGateConfig, ar_gate

    cohort = fetch_cohort()
    logger.info("Cohort: %d mid-tier archs", len(cohort))

    out_rows: list[dict] = []
    t0 = time.perf_counter()
    for i, arch in enumerate(cohort, start=1):
        rid = arch["result_id"][:12]
        in_pair = []
        in_class = []
        held_class = []
        for sd in SEEDS:
            cfg = ARGateConfig(seed=sd, **CFG_KW)
            r = ar_gate(graph_json=arch["graph_json"], device="cuda", cfg=cfg)
            in_pair.append(r.in_dist_pair_acc)
            in_class.append(r.in_dist_class_acc)
            held_class.append(r.held_class_acc)
        ip_mu = st.mean(in_pair)
        ip_sd = st.stdev(in_pair) if len(in_pair) > 1 else 0.0
        out_rows.append(
            {
                **{k: v for k, v in arch.items() if k != "graph_json"},
                "in_pair_mu": round(ip_mu, 3),
                "in_pair_sd": round(ip_sd, 3),
                "in_class_mu": round(st.mean(in_class), 3),
                "held_class_mu": round(st.mean(held_class), 3),
            }
        )
        logger.info(
            "[%d/%d] %s comp=%.0f ppl=%.0f nb_fail=%s in_pair=%.2f±%.2f held_class=%.2f",
            i,
            len(cohort),
            rid,
            arch["composite_score"],
            arch["wikitext_perplexity"],
            arch["nano_bind_failer"],
            ip_mu,
            ip_sd,
            st.mean(held_class),
        )

    elapsed = round(time.perf_counter() - t0, 1)
    out = {
        "n_archs": len(out_rows),
        "elapsed_s": elapsed,
        "config": CFG_KW,
        "rows": out_rows,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "ar_gate_midtier_3seed.json"
    md_path = REPORTS_DIR / "ar_gate_midtier_3seed.md"
    json_path.write_text(json.dumps(out, indent=2, default=str))
    rows_sorted = sorted(out_rows, key=lambda r: r["in_pair_mu"], reverse=True)
    lines = [
        "# AR gate-INV mid-tier (composite 100-300) 3-seed sweep",
        "",
        f"Wall: {elapsed}s | warmup={CFG_KW['wikitext_warmup_steps']} ft={CFG_KW['finetune_steps']} | seeds={SEEDS}",
        "",
        "| rid | template (first) | comp | ppl | nb_fail | in_pair μ±σ | in_class μ | held_class μ |",
        "|---|---|---:|---:|:---:|---:|---:|---:|",
    ]
    for r in rows_sorted:
        tpl = (r["templates"] or ["?"])[0]
        nb = "FAIL" if r["nano_bind_failer"] else " "
        lines.append(
            f"| {r['result_id'][:12]} | {tpl} | {r['composite_score']:.0f} | "
            f"{r['wikitext_perplexity']:.0f} | {nb} | "
            f"{r['in_pair_mu']:.2f} ± {r['in_pair_sd']:.2f} | "
            f"{r['in_class_mu']:.2f} | {r['held_class_mu']:.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", md_path)
    logger.info("Wrote %s", json_path)


if __name__ == "__main__":
    main()
