"""Report-only permutation composition probe audit.

Reads top leaderboard rows, reconstructs each model from ``program_results`` and
runs ``permutation_composition_score`` without writing to SQLite. Outputs:

* JSONL rows with per-result metrics
* JSON summary with Spearman correlations against existing validation signals
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from research.eval.permutation_composition_probe import (
    PERMUTATION_COMPOSITION_METRIC_VERSION,
    permutation_composition_score,
)
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.scientist.native_runner import compile_model_native_first as compile_model
from research.synthesis.serializer import graph_from_json
from research.tools._db_maintenance import connect_readonly

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("research/runs.db")
DEFAULT_REPORT_DIR = Path("research/reports")


def _select_targets(
    db: Path, top_n: int, only_validation: bool
) -> list[dict[str, Any]]:
    where_tier = (
        "AND l.tier IN ('validation', 'breakthrough')" if only_validation else ""
    )
    conn = connect_readonly(db)
    try:
        rows = conn.execute(
            f"""
            SELECT l.entry_id, l.result_id, l.tier, l.composite_score,
                   l.wikitext_perplexity AS lb_wikitext_perplexity,
                   pr.graph_fingerprint, pr.graph_json,
                   pr.wikitext_perplexity,
                   pr.induction_intermediate_auc,
                   pr.binding_intermediate_auc,
                   pr.diagnostic_score,
                   pgf.template_name
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            LEFT JOIN program_graph_features pgf ON pgf.result_id = l.result_id
            WHERE l.composite_score IS NOT NULL
              AND TRIM(COALESCE(pr.graph_json, '')) <> ''
              AND pr.graph_json <> '{{}}'
              {where_tier}
            ORDER BY l.composite_score DESC
            LIMIT ?
            """,
            (int(top_n),),
        ).fetchall()
        payloads = []
        for row in rows:
            payload = dict(row)
            payload["graph_json"] = resolve_graph_json_value(
                conn, db, payload["graph_json"]
            )
            payloads.append(payload)
        return payloads
    finally:
        conn.close()


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for idx in order[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_dev = [x - x_mean for x in xs]
    y_dev = [y - y_mean for y in ys]
    denom = math.sqrt(sum(x * x for x in x_dev) * sum(y * y for y in y_dev))
    if denom <= 0:
        return None
    return sum(x * y for x, y in zip(x_dev, y_dev)) / denom


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_rank(xs), _rank(ys))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _correlations(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics = {
        "composite_score": "composite_score",
        "induction_intermediate_auc": "induction_intermediate_auc",
        "binding_intermediate_auc": "binding_intermediate_auc",
        "diagnostic_score": "diagnostic_score",
        "wikitext_perplexity_inv": "wikitext_perplexity_inv",
    }
    result: dict[str, dict[str, Any]] = {}
    for label, key in metrics.items():
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            score = _finite_float(row.get("permutation_composition_score"))
            target = _finite_float(row.get(key))
            if score is None or target is None:
                continue
            xs.append(score)
            ys.append(-target if key == "wikitext_perplexity_inv" else target)
        rho = _spearman(xs, ys)
        result[label] = {
            "n": len(xs),
            "spearman": round(rho, 4) if rho is not None else None,
        }
    return result


def _run_one(
    row: dict[str, Any],
    *,
    vocab_size: int,
    max_seq_len: int,
    n_items: int,
    train_chain_len: int,
    eval_chain_len: int,
    n_train_steps: int,
    n_eval_batches: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    graph = graph_from_json(str(row["graph_json"]))
    model = compile_model([graph], vocab_size=vocab_size, max_seq_len=max_seq_len).to(
        device
    )
    try:
        res = permutation_composition_score(
            model,
            n_items=n_items,
            train_chain_len=train_chain_len,
            eval_chain_len=eval_chain_len,
            n_train_steps=n_train_steps,
            n_eval_batches=n_eval_batches,
            batch_size=batch_size,
            device=device,
        )
        payload = res.to_dict()
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    ppl = row.get("wikitext_perplexity")
    if ppl is None:
        ppl = row.get("lb_wikitext_perplexity")
    return {
        "entry_id": row.get("entry_id"),
        "result_id": row.get("result_id"),
        "tier": row.get("tier"),
        "graph_fingerprint": row.get("graph_fingerprint"),
        "template_name": row.get("template_name"),
        "composite_score": row.get("composite_score"),
        "wikitext_perplexity_inv": ppl,
        "induction_intermediate_auc": row.get("induction_intermediate_auc"),
        "binding_intermediate_auc": row.get("binding_intermediate_auc"),
        "diagnostic_score": row.get("diagnostic_score"),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        **payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--only-validation", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument("--n-items", type=int, default=8)
    parser.add_argument("--train-chain-len", type=int, default=2)
    parser.add_argument("--eval-chain-len", type=int, default=4)
    parser.add_argument("--n-train-steps", type=int, default=80)
    parser.add_argument("--n-eval-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="default: research/reports/permutation_composition_report_<ts>.jsonl",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=None,
        help="default: same stem as JSONL with .summary.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ts = int(time.time())
    out_jsonl = args.out_jsonl or (
        DEFAULT_REPORT_DIR / f"permutation_composition_report_{ts}.jsonl"
    )
    out_summary = args.out_summary or out_jsonl.with_suffix(".summary.json")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    targets = _select_targets(args.db, args.top_n, args.only_validation)
    logger.info(
        "selected %d targets top_n=%d only_validation=%s device=%s",
        len(targets),
        args.top_n,
        args.only_validation,
        args.device,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for idx, target in enumerate(targets, start=1):
            try:
                result = _run_one(
                    target,
                    vocab_size=args.vocab_size,
                    max_seq_len=args.max_seq_len,
                    n_items=args.n_items,
                    train_chain_len=args.train_chain_len,
                    eval_chain_len=args.eval_chain_len,
                    n_train_steps=args.n_train_steps,
                    n_eval_batches=args.n_eval_batches,
                    batch_size=args.batch_size,
                    device=args.device,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%d/%d] %s failed: %s",
                    idx,
                    len(targets),
                    str(target.get("result_id"))[:12],
                    exc,
                )
                failures.append(
                    {
                        "result_id": target.get("result_id"),
                        "graph_fingerprint": target.get("graph_fingerprint"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            rows.append(result)
            fh.write(json.dumps(result, sort_keys=True) + "\n")
            fh.flush()
            logger.info(
                "[%d/%d] %s score=%s train=%s extrap=%s elapsed=%.2fs",
                idx,
                len(targets),
                str(target.get("result_id"))[:12],
                result.get("permutation_composition_score"),
                result.get("permutation_composition_train_chain_acc"),
                result.get("permutation_composition_extrapolation_acc"),
                result.get("elapsed_s") or 0.0,
            )

    summary = {
        "metric_version": PERMUTATION_COMPOSITION_METRIC_VERSION,
        "db": str(args.db),
        "top_n": args.top_n,
        "only_validation": bool(args.only_validation),
        "n_rows": len(rows),
        "n_failures": len(failures),
        "failures": failures,
        "config": {
            "vocab_size": args.vocab_size,
            "max_seq_len": args.max_seq_len,
            "n_items": args.n_items,
            "train_chain_len": args.train_chain_len,
            "eval_chain_len": args.eval_chain_len,
            "n_train_steps": args.n_train_steps,
            "n_eval_batches": args.n_eval_batches,
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "correlations": _correlations(rows),
        "jsonl": str(out_jsonl),
    }
    out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("wrote %s and %s", out_jsonl, out_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
