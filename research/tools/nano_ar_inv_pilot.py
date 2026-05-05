#!/usr/bin/env python
"""Pilot nano-AR-INV against the top-N leaderboard architectures.

Read-only against ``lab_notebook.db``. Writes a JSON + markdown summary to
``research/reports/`` — no DB writes, no scoring wiring, no continuous-mode
changes. Uses ``from_s1=False`` so each architecture trains fresh-init on the
nano-AR corpus alone (no wikitext warmup) — fast signal of whether the
architecture has the retrieval mechanism at all.

Usage::

    python -m research.tools.nano_ar_inv_pilot --top-n 30 --device cuda
    python -m research.tools.nano_ar_inv_pilot --top-n 10 --device cpu --train-steps 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DB_PATH = REPO / "research/lab_notebook.db"
REPORTS_DIR = REPO / "research/reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--top-n", type=int, default=30, help="N top arches by composite_score"
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--finetune-steps", type=int, default=400)
    p.add_argument("--wikitext-warmup-steps", type=int, default=500)
    p.add_argument("--n-pairs-per-noun", type=int, default=5)
    p.add_argument("--n-distractors", type=int, default=480)
    p.add_argument("--n-adjectives", type=int, default=20)
    p.add_argument("--n-objects", type=int, default=25)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--db", type=str, default=str(DB_PATH))
    p.add_argument(
        "--filter-template",
        default=None,
        help="Optional: restrict to archs whose templates_used contains this name",
    )
    p.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix (default: nano_ar_inv_pilot_<unix_ts>)",
    )
    return p.parse_args()


def fetch_top_archs(
    db_path: str, *, top_n: int, filter_template: str | None
) -> list[dict[str, Any]]:
    """Pull top-N rows joined to leaderboard composite_score."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        sql = """
            SELECT pr.result_id, pr.graph_json, lb.composite_score, lb.tier,
                   pr.wikitext_perplexity, pr.ar_auc, pr.induction_auc, pr.binding_auc
              FROM program_results pr
              JOIN leaderboard lb ON lb.result_id = pr.result_id
             WHERE pr.graph_json IS NOT NULL
               AND lb.composite_score IS NOT NULL
        """
        rows = conn.execute(sql + " ORDER BY lb.composite_score DESC").fetchall()
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for rid, gj, comp, tier, ppl, ar, ind, bnd in rows:
        try:
            graph = json.loads(gj)
        except (TypeError, json.JSONDecodeError):
            continue
        templates = graph.get("metadata", {}).get("templates_used") or []
        if filter_template and filter_template not in templates:
            continue
        out.append(
            {
                "result_id": str(rid),
                "graph_json": gj,
                "composite_score": float(comp),
                "tier": str(tier or ""),
                "wikitext_perplexity": float(ppl) if ppl is not None else None,
                "ar_auc_legacy": float(ar) if ar is not None else None,
                "induction_auc": float(ind) if ind is not None else None,
                "binding_auc": float(bnd) if bnd is not None else None,
                "templates": [str(t) for t in templates],
            }
        )
        if len(out) >= int(top_n):
            break
    return out


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    from research.eval.nano_ar_inv import NanoARInvConfig, nano_ar_inv

    archs = fetch_top_archs(
        args.db, top_n=args.top_n, filter_template=args.filter_template
    )
    logger.info("Pulled %d candidate arches", len(archs))

    cfg = NanoARInvConfig(
        seed=int(args.seed),
        finetune_steps=int(args.finetune_steps),
        wikitext_warmup_steps=int(args.wikitext_warmup_steps),
        n_pairs_per_noun=int(args.n_pairs_per_noun),
        n_distractors=int(args.n_distractors),
        n_adjectives=int(args.n_adjectives),
        n_objects=int(args.n_objects),
        reps=int(args.reps),
        timeout_s=float(args.timeout_s),
        from_s1=False,
    )

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, arch in enumerate(archs, start=1):
        rid = arch["result_id"][:12]
        logger.info(
            "[%d/%d] %s composite=%.1f templates=%s",
            i,
            len(archs),
            rid,
            arch["composite_score"],
            ",".join(arch["templates"][:3]),
        )
        result = nano_ar_inv(graph_json=arch["graph_json"], device=args.device, cfg=cfg)
        row = {
            **{k: v for k, v in arch.items() if k != "graph_json"},
            "nano_ar_inv": asdict(result),
        }
        rows.append(row)
        logger.info(
            "  → in_pair=%.3f in_class=%.3f held_pair=%.3f held_class=%.3f "
            "(%dms, status=%s)",
            result.in_dist_pair_match_acc,
            result.in_dist_class_acc,
            result.held_pair_match_acc,
            result.held_class_acc,
            int(result.elapsed_ms),
            result.status,
        )

    elapsed = round(time.perf_counter() - t0, 1)
    return {
        "n_archs": len(rows),
        "elapsed_s": elapsed,
        "config": asdict(cfg),
        "device": args.device,
        "rows": rows,
    }


def write_reports(report: dict[str, Any], out_prefix: str) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"{out_prefix}.json"
    md_path = REPORTS_DIR / f"{out_prefix}.md"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    rows = report["rows"]
    rows_by_score = sorted(
        rows,
        key=lambda r: r["nano_ar_inv"]["in_dist_pair_match_acc"],
        reverse=True,
    )
    cfg = report["config"]
    lines = [
        f"# nano-AR-INV pilot — {report['n_archs']} arches, {report['elapsed_s']}s wall",
        "",
        f"Config: warmup={cfg['wikitext_warmup_steps']} ft={cfg['finetune_steps']} "
        f"npairs={cfg['n_pairs_per_noun']} ndist={cfg['n_distractors']} "
        f"adj={cfg['n_adjectives']} obj={cfg['n_objects']} reps={cfg['reps']} "
        f"device={report['device']} seed={cfg['seed']}",
        "",
        "| rid | template (first) | composite | in_pair | in_class | held_pair | held_class | ar_legacy | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows_by_score:
        n = r["nano_ar_inv"]
        tpl = (r["templates"] or ["?"])[0]
        ar_legacy = r.get("ar_auc_legacy")
        ar_str = f"{ar_legacy:.3f}" if ar_legacy is not None else "—"
        lines.append(
            f"| {r['result_id'][:12]} | {tpl} | {r['composite_score']:.1f} | "
            f"{n['in_dist_pair_match_acc']:.3f} | {n['in_dist_class_acc']:.3f} | "
            f"{n['held_pair_match_acc']:.3f} | {n['held_class_acc']:.3f} | "
            f"{ar_str} | {n['status']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    args = parse_args()
    out_prefix = args.out_prefix or f"nano_ar_inv_pilot_{int(time.time())}"
    report = run_pilot(args)
    json_path, md_path = write_reports(report, out_prefix)
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
