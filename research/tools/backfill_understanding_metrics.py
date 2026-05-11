"""Backfill diagnostic_score and cross_task_score on leaderboard fingerprints.

Both metrics live in v8/v10 understanding-tier scoring but are unpopulated
on a large fraction of leaderboard rows:

  * ``cross_task_score``: 0% populated across 17,399 program_results rows
    because the original HF dataset (codeparrot/github-code-clean) was
    retired by HF (script-based loaders no longer supported). The dataset
    swap to ``codeparrot/codeparrot-clean`` is in
    research/eval/cross_task_eval.py:_download_code_corpus.
  * ``diagnostic_score``: ~30% gap (1,349 of 4,430 leaderboard rows null).
    Synthetic eval (research/eval/diagnostic_tasks.run_diagnostic_suite),
    no download.

This tool walks leaderboard fingerprints sorted by composite_score DESC,
rebuilds each model from graph_json, runs both evals, and writes back
both score columns plus diagnostic_tasks_json.

Usage::

    # Smoke test (top 5 fps)
    python -m research.tools.backfill_understanding_metrics --workers 1 --limit 5

    # Production (all leaderboard rows missing either metric)
    python -m research.tools.backfill_understanding_metrics --workers 4
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.multiprocessing as mp

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "research" / "runs.db"
TOOL_NAME = "backfill_understanding_metrics"

DEFAULT_VOCAB = 100277  # cl100k_base
DEFAULT_DIAG_STEPS = 50
DEFAULT_CROSS_TASK_STEPS = 80
DEFAULT_CROSS_TASK_BATCH = 4
DEFAULT_CROSS_TASK_SEQ_LEN = 128

_UPDATE_COLUMNS = (
    "diagnostic_score",
    "diagnostic_tasks_json",
    "cross_task_score",
)


@dataclass(slots=True)
class FingerprintTask:
    fingerprint: str
    graph_json: str
    cost_score: int


@dataclass(slots=True)
class TaskResult:
    fingerprint: str
    payload: dict | None
    error: str | None
    elapsed_s: float


# ─── Worker process ────────────────────────────────────────────────────


def _worker_init(
    vocab_size: int,
    gpu_memory_fraction: float,
    diag_steps: int,
    ct_steps: int,
    ct_batch: int,
    ct_seq_len: int,
) -> None:
    """One-time setup per worker."""
    global _W_VOCAB, _W_DIAG_STEPS, _W_CT_STEPS, _W_CT_BATCH, _W_CT_SEQ_LEN
    import research.synthesis.compiler  # noqa: F401  populates op registry

    if torch.cuda.is_available():
        from research.tools._concurrency import cap_gpu_memory

        cap_gpu_memory(fraction=gpu_memory_fraction)
    _W_VOCAB = vocab_size
    _W_DIAG_STEPS = diag_steps
    _W_CT_STEPS = ct_steps
    _W_CT_BATCH = ct_batch
    _W_CT_SEQ_LEN = ct_seq_len


def _worker_measure(task: FingerprintTask) -> TaskResult:
    """Build model, run diagnostic + cross-task, return columns to write."""
    from research.synthesis import graph_from_json
    from research.synthesis.compiled_model import SynthesizedModel
    from research.eval.diagnostic_tasks import run_diagnostic_suite
    from research.eval.cross_task_eval import evaluate_cross_task_robustness

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload: dict = {}
    try:
        graph = graph_from_json(task.graph_json)
        model_dim = getattr(graph, "model_dim", None) or 256

        # Diagnostic synthetic suite (uses a fresh model from the factory).
        try:
            diag_model = SynthesizedModel(
                [graph], vocab_size=_W_VOCAB, model_dim=model_dim
            ).to(dev)
            diag = run_diagnostic_suite(
                diag_model, device=str(dev), n_steps=_W_DIAG_STEPS
            )
            payload["diagnostic_score"] = float(diag.diagnostic_score)
            payload["diagnostic_tasks_json"] = json.dumps(diag.to_dict())
            del diag_model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            print(
                f"[diag-fail] {task.fingerprint[:12]} {type(exc).__name__}: {str(exc)[:160]}",
                file=sys.stderr,
                flush=True,
            )

        # Cross-task: needs a model factory (rebuild fresh per domain).
        def _make_model():
            return SynthesizedModel([graph], vocab_size=_W_VOCAB, model_dim=model_dim)

        try:
            ct = evaluate_cross_task_robustness(
                make_model_fn=_make_model,
                vocab_size=_W_VOCAB,
                device=dev,
                n_train_steps=_W_CT_STEPS,
                batch_size=_W_CT_BATCH,
                seq_len=_W_CT_SEQ_LEN,
            )
            cts = ct.get("cross_task_score")
            if cts is not None:
                payload["cross_task_score"] = float(cts)
            elif ct.get("error"):
                print(
                    f"[ct-fail] {task.fingerprint[:12]} {ct.get('error')}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:
            print(
                f"[ct-fail] {task.fingerprint[:12]} {type(exc).__name__}: {str(exc)[:160]}",
                file=sys.stderr,
                flush=True,
            )

        if not payload:
            return TaskResult(
                task.fingerprint, None, "no_metrics_produced", time.time() - t0
            )
        return TaskResult(task.fingerprint, payload, None, time.time() - t0)
    except Exception as exc:
        return TaskResult(
            task.fingerprint,
            None,
            f"{type(exc).__name__}: {str(exc)[:160]}",
            time.time() - t0,
        )
    finally:
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# ─── Master ────────────────────────────────────────────────────────────


def _candidate_tasks(
    conn: sqlite3.Connection,
    limit: int | None,
    require_missing: tuple[str, ...],
) -> list[FingerprintTask]:
    missing_clauses = " OR ".join(f"pr.{c} IS NULL" for c in require_missing)
    sql = f"""
        SELECT pr.graph_fingerprint,
               MAX(lb.composite_score) AS comp,
               (SELECT graph_json FROM program_results_compat pr2
                 WHERE pr2.graph_fingerprint = pr.graph_fingerprint
                   AND pr2.graph_json IS NOT NULL
                   AND length(pr2.graph_json) > 10
                 ORDER BY length(pr2.graph_json) DESC, pr2.timestamp DESC LIMIT 1) AS graph_json
        FROM program_results_compat pr
        JOIN leaderboard lb ON lb.result_id = pr.result_id
        WHERE pr.graph_fingerprint IS NOT NULL
          AND ({missing_clauses})
        GROUP BY pr.graph_fingerprint
        HAVING graph_json IS NOT NULL
        ORDER BY comp DESC NULLS LAST
    """
    if limit and limit > 0:
        sql += f" LIMIT {int(limit)}"
    return [
        FingerprintTask(
            fingerprint=fp,
            graph_json=resolve_graph_json_value(conn, DB_PATH, gj),
            cost_score=1,
        )
        for fp, _comp, gj in conn.execute(sql)
        if gj
    ]


def _propagate(conn: sqlite3.Connection, fingerprint: str, payload: dict) -> int:
    cols = [c for c in _UPDATE_COLUMNS if c in payload]
    if not cols:
        return 0
    set_clause = ", ".join(f"{c} = ?" for c in cols)
    values = [payload.get(c) for c in cols]
    cur = conn.execute(
        f"UPDATE graph_runs SET {set_clause} WHERE graph_fingerprint = ?",
        (*values, fingerprint),
    )
    return cur.rowcount


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--diag-steps", type=int, default=DEFAULT_DIAG_STEPS)
    parser.add_argument("--ct-steps", type=int, default=DEFAULT_CROSS_TASK_STEPS)
    parser.add_argument("--ct-batch", type=int, default=DEFAULT_CROSS_TASK_BATCH)
    parser.add_argument("--ct-seq-len", type=int, default=DEFAULT_CROSS_TASK_SEQ_LEN)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--commit-every", type=int, default=20)
    parser.add_argument("--gpu-memory-fraction", type=float, default=None)
    parser.add_argument("--max-other-gpu-mib", type=int, default=4096)
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument(
        "--require-missing",
        default="diagnostic_score,cross_task_score",
        help="Comma-separated columns. Candidates are fps where ANY of these is NULL.",
    )
    args = parser.parse_args()

    import research.synthesis.compiler  # noqa: F401  warm registry
    from research.tools._concurrency import (
        acquire_gpu_lock,
        acquire_writer_lock,
        assert_gpu_quiet,
    )

    if torch.cuda.is_available():
        assert_gpu_quiet(
            max_other_used_mib=args.max_other_gpu_mib,
            tool_name=TOOL_NAME,
            sleep_until_quiet=args.wait_for_gpu,
        )

    if args.gpu_memory_fraction is None:
        args.gpu_memory_fraction = max(0.05, min(0.5, 0.85 / max(args.workers, 1)))

    require_missing = tuple(
        c.strip() for c in args.require_missing.split(",") if c.strip()
    )
    if not require_missing:
        parser.error("--require-missing must be non-empty")

    with (
        acquire_gpu_lock(tool_name=TOOL_NAME),
        acquire_writer_lock(tool_name=TOOL_NAME),
    ):
        conn = sqlite3.connect(str(DB_PATH))
        tasks = _candidate_tasks(conn, args.limit, require_missing)
        conn.close()
        total = len(tasks)
        print(
            f"[setup] understanding-metrics backfill scope: {total} unique fingerprints "
            f"(missing one of {require_missing})",
            flush=True,
        )
        print(
            f"[setup] starting {args.workers} workers; per-worker VRAM cap "
            f"{args.gpu_memory_fraction:.0%}",
            flush=True,
        )
        if total == 0:
            return

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(
                args.vocab_size,
                args.gpu_memory_fraction,
                args.diag_steps,
                args.ct_steps,
                args.ct_batch,
                args.ct_seq_len,
            ),
        )

        succeeded = 0
        failed = 0
        propagated = 0
        t_start = time.time()

        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            for idx, result in enumerate(
                pool.imap_unordered(_worker_measure, tasks, chunksize=1), start=1
            ):
                if result.payload is None:
                    failed += 1
                    if result.error:
                        print(
                            f"  [{idx}/{total}] {result.fingerprint[:12]} fail: {result.error}",
                            flush=True,
                        )
                    continue
                rowcount = _propagate(conn, result.fingerprint, result.payload)
                succeeded += 1
                propagated += rowcount
                if succeeded % args.commit_every == 0 or idx == total:
                    conn.commit()
                    elapsed = time.time() - t_start
                    rate = succeeded / max(elapsed, 1e-6)
                    eta_min = (total - idx) / max(rate, 1e-6) / 60.0
                    p = result.payload
                    print(
                        f"  [{idx}/{total}] ok={succeeded} fail={failed} rows={propagated} "
                        f"rate={rate:.2f}/s eta={eta_min:.0f}m | "
                        f"diag={p.get('diagnostic_score')} ct={p.get('cross_task_score')}",
                        flush=True,
                    )
        finally:
            pool.close()
            pool.join()
            conn.commit()
            conn.close()

        elapsed = time.time() - t_start
        rate = succeeded / max(elapsed, 1e-6)
        print(
            f"[done] ok={succeeded} fail={failed} rows_updated={propagated} "
            f"elapsed={elapsed / 60:.1f}m rate={rate:.2f}/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
