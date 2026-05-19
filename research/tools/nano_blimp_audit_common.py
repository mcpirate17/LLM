"""Shared harness for Nano-BLIMP cohort audit tools."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

import torch

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.tools._db_maintenance import connect_readonly
from research.tools.nano_blimp_tune import _train_base

ProbeRunner = Callable[[torch.nn.Module, int, int, Any], dict[str, Any]]


def load_arch(db_path: Path, result_id: str) -> dict[str, Any] | None:
    """Look up an architecture by result_id with leaderboard metadata."""
    conn = connect_readonly(db_path)
    try:
        row = conn.execute(
            """
            SELECT pr.result_id, pr.graph_json, pr.graph_fingerprint,
                   pgf.template_name,
                   l.entry_id, l.composite_score, l.tier,
                   l.induction_screening_auc, l.binding_screening_auc, l.wikitext_perplexity,
                   l.hellaswag_acc, l.blimp_overall_accuracy
            FROM program_results_compat pr
            LEFT JOIN program_graph_features pgf ON pgf.result_id=pr.result_id
            LEFT JOIN leaderboard l ON l.result_id=pr.result_id
            WHERE pr.result_id=?
            """,
            (result_id,),
        ).fetchone()
        if not row:
            return None
        payload = dict(row)
        payload["graph_json"] = resolve_graph_json_value(
            conn,
            db_path,
            payload.get("graph_json"),
        )
        return payload
    finally:
        conn.close()


def stats(rows: list[dict[str, Any]], name: str) -> dict[str, float]:
    vals = [r[name] for r in rows if r["status"] == "ok"]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": round(mean(vals), 4),
        "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        name: stats(rows, name)
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


def audit_one_seed(
    meta: dict[str, Any],
    *,
    seed: int,
    held_out_counts: list[int],
    args: Any,
    run_probe: ProbeRunner,
    failure_fields: Callable[[Any], dict[str, Any]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Train one base model and run one Nano-BLIMP probe over held-out counts."""
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
                row = run_probe(model, seed, ho, args)
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
                    "held_out_count": ho,
                    "seed": seed,
                    **failure_fields(args),
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


def build_summary(
    runs: list[dict[str, Any]],
    arch_meta: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_arch_ho: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in runs:
        by_arch_ho.setdefault((row["result_id"], row["held_out_count"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for (rid, ho), rows in by_arch_ho.items():
        meta = next((m for m in arch_meta if m["result_id"] == rid), {})
        summary.append(
            {
                "result_id": rid,
                "template": meta.get("template_name"),
                "tier": meta.get("tier"),
                "composite": meta.get("composite_score"),
                "induction_screening_auc": meta.get("induction_screening_auc"),
                "binding_screening_auc": meta.get("binding_screening_auc"),
                "wikitext_perplexity": meta.get("wikitext_perplexity"),
                "held_out_count": ho,
                "n_seeds": len(rows),
                **aggregate(rows),
            }
        )
    return summary


def print_summary_table(summary: list[dict[str, Any]], title: str) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'result_id':14s} {'tmpl':28s} {'ho':>3s} "
        f"{'cls_in':>11s} {'cls_ho':>11s} {'bnd_in':>11s} {'bnd_ho':>11s} "
        f"{'order':>11s} {'ho_score':>11s}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:

        def fmt(data: dict[str, float]) -> str:
            return f"{data['mean']:.2f}+/-{data['std']:.2f}"

        print(
            f"{row['result_id']:14s} {(row['template'] or '?')[:28]:28s} "
            f"{row['held_out_count']:>3d} "
            f"{fmt(row['class_in_dist']):>11s} {fmt(row['class_held_out']):>11s} "
            f"{fmt(row['binding_in_dist']):>11s} {fmt(row['binding_held_out']):>11s} "
            f"{fmt(row['order']):>11s} {fmt(row['held_out_score']):>11s}"
        )


def run_audit(
    *,
    args: Any,
    run_probe: ProbeRunner,
    failure_fields: Callable[[Any], dict[str, Any]],
    summary_title: str,
    logger: logging.Logger,
) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    arch_meta: list[dict[str, Any]] = []
    for rid in args.targets:
        meta = load_arch(args.db, rid)
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
            meta.get("induction_screening_auc"),
            meta.get("binding_screening_auc"),
            meta.get("wikitext_perplexity"),
        )
        for seed in args.seeds:
            runs.extend(
                audit_one_seed(
                    meta,
                    seed=seed,
                    held_out_counts=args.held_out,
                    args=args,
                    run_probe=run_probe,
                    failure_fields=failure_fields,
                    logger=logger,
                )
            )

    summary = build_summary(runs, arch_meta)
    print_summary_table(summary, summary_title)
    args.out.write_text(
        json.dumps({"runs": runs, "summary": summary}, indent=2, default=str)
    )
    logger.info("wrote %d runs -> %s", len(runs), args.out)
    return 0
