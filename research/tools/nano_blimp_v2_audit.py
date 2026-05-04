"""Real-architecture cohort audit for nano_blimp_v2 held-out metrics.

Loads champion / frontier graphs from the lab notebook, briefly trains
each on wikitext-vocab random data (matching the screening regime),
then runs ``nano_blimp_score`` across multiple held_out_count settings
and seeds. Reports each held-out component separately so SSM-class
candidates that lag on positional ordering but learn the binding rule
are still visible.

Read-only: no DB writes, no leaderboard mutation. JSON output goes to
``research/reports/``.

Usage:
    python -m research.tools.nano_blimp_v2_audit \
        --targets ec7025d7-338 574271ca-f37 f70c17d0-d59 8d087a16-692 903157e5-219 \
        --seeds 1 2 3 \
        --held-out 2 3 5 \
        --vocab 80 --probe-steps 100 \
        --base-train-steps 750 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

from research.eval.nano_blimp_eval import nano_blimp_score
from research.tools._db_maintenance import connect_readonly
from research.tools.nano_blimp_tune import _train_base

logger = logging.getLogger(__name__)


def _load_arch(db_path: Path, result_id: str) -> dict[str, Any] | None:
    """Look up an architecture by ``result_id`` and pull the graph_json
    plus the leaderboard fields the audit reports against (induction,
    binding, ppl, tier). Different lookup key + wider column set than
    ``nano_blimp_tune._load_arch``, so kept separate."""
    conn = connect_readonly(db_path)
    try:
        row = conn.execute(
            """
            SELECT pr.result_id, pr.graph_json, pr.graph_fingerprint,
                   pgf.template_name,
                   l.entry_id, l.composite_score, l.tier,
                   l.induction_auc, l.binding_auc, l.wikitext_perplexity,
                   l.hellaswag_acc, l.blimp_overall_accuracy
            FROM program_results pr
            LEFT JOIN program_graph_features pgf ON pgf.result_id=pr.result_id
            LEFT JOIN leaderboard l ON l.result_id=pr.result_id
            WHERE pr.result_id=?
            """,
            (result_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _run_probe(
    model: torch.nn.Module,
    *,
    vocab: int,
    probe_steps: int,
    held_out_count: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    res = nano_blimp_score(
        model,
        active_vocab_size=vocab,
        n_train_steps=probe_steps,
        batch_size=32,
        lr=1e-3,
        device=device,
        seed=seed,
        held_out_count=held_out_count,
    )
    return {
        "status": res.status,
        "metric_version": res.metric_version,
        "vocab": vocab,
        "probe_steps": probe_steps,
        "held_out_count": held_out_count,
        "seed": seed,
        "score": res.score,
        "held_out_score": res.held_out_score,
        "class_in_dist": res.class_coherence_in_dist_acc,
        "class_held_out": res.class_coherence_held_out_acc,
        "binding_in_dist": res.binding_fidelity_in_dist_acc,
        "binding_held_out": res.binding_fidelity_held_out_acc,
        "order": res.order_grammaticality_acc,
        "n_in_dist_pairs": res.n_in_dist_pairs,
        "n_held_out_pairs": res.n_held_out_pairs,
        "n_train_steps_completed": res.n_train_steps,
        "elapsed_ms": res.elapsed_ms,
    }


def _stats(rows: list[dict[str, Any]], name: str) -> dict[str, float]:
    vals = [r[name] for r in rows if r["status"] == "ok"]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": round(mean(vals), 4),
        "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        name: _stats(rows, name)
        for name in (
            "score",
            "held_out_score",
            "class_in_dist",
            "class_held_out",
            "binding_in_dist",
            "binding_held_out",
            "order",
        )
    }


def _audit_one_seed(
    meta: dict[str, Any],
    *,
    seed: int,
    held_out_counts: list[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Train one base model for ``seed``, run the probe across all
    requested held_out_count values, return list of run rows."""
    torch.manual_seed(int(seed))
    try:
        model = _train_base(
            meta["graph_json"],
            base_steps=args.base_train_steps,
            device=args.device,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "base train failed for %s seed=%d: %s",
            meta["result_id"],
            seed,
            exc,
        )
        return []

    rows: list[dict[str, Any]] = []
    try:
        for ho in held_out_counts:
            t0 = time.perf_counter()
            try:
                row = _run_probe(
                    model,
                    vocab=args.vocab,
                    probe_steps=args.probe_steps,
                    held_out_count=ho,
                    seed=seed,
                    device=args.device,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "probe failed %s seed=%d ho=%d: %s",
                    meta["result_id"],
                    seed,
                    ho,
                    exc,
                )
                row = {
                    "status": f"failed:{type(exc).__name__}",
                    "error": str(exc),
                    "vocab": args.vocab,
                    "probe_steps": args.probe_steps,
                    "held_out_count": ho,
                    "seed": seed,
                }
            row["result_id"] = meta["result_id"]
            row["template"] = meta.get("template_name")
            row["wall_s"] = round(time.perf_counter() - t0, 2)
            rows.append(row)
            logger.info(
                "  seed=%d ho=%d status=%s "
                "cls_in=%.2f cls_ho=%.2f bnd_in=%.2f bnd_ho=%.2f order=%.2f "
                "ho_score=%.2f wall=%.1fs",
                seed,
                ho,
                row.get("status", "?"),
                row.get("class_in_dist", 0) or 0,
                row.get("class_held_out", 0) or 0,
                row.get("binding_in_dist", 0) or 0,
                row.get("binding_held_out", 0) or 0,
                row.get("order", 0) or 0,
                row.get("held_out_score", 0) or 0,
                row["wall_s"],
            )
    finally:
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    return rows


def _build_summary(
    runs: list[dict[str, Any]],
    arch_meta: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_arch_ho: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in runs:
        by_arch_ho.setdefault((r["result_id"], r["held_out_count"]), []).append(r)
    summary: list[dict[str, Any]] = []
    for (rid, ho), rows in by_arch_ho.items():
        meta = next((m for m in arch_meta if m["result_id"] == rid), {})
        summary.append(
            {
                "result_id": rid,
                "template": meta.get("template_name"),
                "tier": meta.get("tier"),
                "composite": meta.get("composite_score"),
                "induction_auc": meta.get("induction_auc"),
                "binding_auc": meta.get("binding_auc"),
                "wikitext_perplexity": meta.get("wikitext_perplexity"),
                "held_out_count": ho,
                "n_seeds": len(rows),
                **_aggregate(rows),
            }
        )
    return summary


def _print_summary_table(summary: list[dict[str, Any]]) -> None:
    print("\n=== nano_blimp_v2 cohort audit (mean ± std across seeds) ===")
    header = (
        f"{'result_id':14s} {'tmpl':28s} {'ho':>3s} "
        f"{'cls_in':>11s} {'cls_ho':>11s} {'bnd_in':>11s} {'bnd_ho':>11s} "
        f"{'order':>11s} {'ho_score':>11s}"
    )
    print(header)
    print("-" * len(header))
    for s in summary:

        def fmt(d: dict[str, float]) -> str:
            return f"{d['mean']:.2f}±{d['std']:.2f}"

        print(
            f"{s['result_id']:14s} {(s['template'] or '?')[:28]:28s} "
            f"{s['held_out_count']:>3d} "
            f"{fmt(s['class_in_dist']):>11s} {fmt(s['class_held_out']):>11s} "
            f"{fmt(s['binding_in_dist']):>11s} {fmt(s['binding_held_out']):>11s} "
            f"{fmt(s['order']):>11s} {fmt(s['held_out_score']):>11s}"
        )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="research/lab_notebook.db", type=Path)
    ap.add_argument("--targets", nargs="+", required=True, help="result_id list")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--held-out", nargs="+", type=int, default=[2, 3, 5])
    ap.add_argument("--vocab", type=int, default=80)
    ap.add_argument("--probe-steps", type=int, default=100)
    ap.add_argument("--base-train-steps", type=int, default=750)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(f"research/reports/nano_blimp_v2_audit_{int(time.time())}.json"),
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    arch_meta: list[dict[str, Any]] = []
    for rid in args.targets:
        meta = _load_arch(args.db, rid)
        if not meta or not meta.get("graph_json"):
            logger.warning("no graph_json for %s; skipping", rid)
            continue
        arch_meta.append(meta)
        logger.info(
            "=== %s tmpl=%s tier=%s comp=%s ind=%s bind=%s ppl=%s ===",
            rid,
            meta.get("template_name"),
            meta.get("tier"),
            meta.get("composite_score"),
            meta.get("induction_auc"),
            meta.get("binding_auc"),
            meta.get("wikitext_perplexity"),
        )
        for seed in args.seeds:
            runs.extend(
                _audit_one_seed(
                    meta, seed=seed, held_out_counts=args.held_out, args=args
                )
            )

    summary = _build_summary(runs, arch_meta)
    _print_summary_table(summary)

    args.out.write_text(
        json.dumps({"runs": runs, "summary": summary}, indent=2, default=str)
    )
    logger.info("wrote %d runs -> %s", len(runs), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
