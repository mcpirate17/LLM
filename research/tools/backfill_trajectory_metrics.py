#!/usr/bin/env python
"""Backfill all 4 Gemini trajectory metrics + spec_norm via screening-budget training.

For each unique architecture this script:
  1. Rebuilds the model from ``program_results.graph_json``
  2. Trains it for 750 steps on WikiText103 (matching the live screening
     pipeline so backfilled values are comparable to live writes)
  3. Captures hidden-state participation-ratio snapshots at step 150 and
     step 750 (the only way to populate ``fp_id_collapse_rate``)
  4. Runs ``compute_trajectory_metrics`` with both snapshots so all four
     Gemini metrics + spec_norm + the ID-collapse rate get computed
  5. Writes ``fp_metric_phase = "screening_750"`` so ML training sees
     this row at the same lifecycle stage as live screening rows

Cost: ~30-40 s per unique fingerprint on CUDA (matches smoke timings).
For ~16k fingerprints this is ~5-8 hours of GPU. Acquires the GPU and
SQLite writer flocks so it can't race the dashboard or other tools.

Usage::

    python -m research.tools.backfill_trajectory_metrics
    python -m research.tools.backfill_trajectory_metrics --limit 50
    python -m research.tools.backfill_trajectory_metrics --wait-for-gpu
"""

from __future__ import annotations

import argparse
import sqlite3
import time

import torch

# Force-import compiler so OP_DISPATCH is populated before any CompiledOp.
import research.synthesis.compiler  # noqa: F401
from research.synthesis import graph_from_json
from research.synthesis.compiled_model import SynthesizedModel
from research.eval.trajectory_metrics import (
    capture_hidden_state_snapshot,
    compute_trajectory_metrics,
)
from research.tools._concurrency import (
    acquire_gpu_lock,
    acquire_writer_lock,
)
from research.tools._metric_backfill_common import (
    CORPUS_PATH,
    DB_PATH,
    TRAJECTORY_COLUMNS,
    add_gpu_safety_args,
    load_projected_corpus,
    prepare_cuda_if_requested,
    print_serial_commit_progress,
    sample_token_batch,
    train_next_token_step,
    update_graph_runs_columns,
)
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value

TOOL_NAME = "backfill_trajectory_metrics"

# Training schedule mirrors the live screening pipeline. Hidden-state
# snapshots at 150 and 750 give us the ID-collapse rate signal Gemini
# called out as the cleanest discriminator at small budgets.
SCREENING_STEPS = 750
ID_COLLAPSE_EARLY_STEP = 150
ID_COLLAPSE_LATE_STEP = 750
DEFAULT_SEQ_LEN = 64  # lighter than smoke's 128 to keep per-arch cost down
DEFAULT_BATCH_SIZE = 16
DEFAULT_VOCAB = 32000
LR = 3e-4


def _load_corpus(vocab_size: int, max_tokens: int = 4_000_000) -> torch.Tensor:
    return load_projected_corpus(
        CORPUS_PATH,
        vocab_size=vocab_size,
        max_tokens=max_tokens,
    )


def _sample_batch(
    tokens: torch.Tensor, batch_size: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    return sample_token_batch(tokens, batch_size, seq_len, device)


def _train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
) -> float:
    return train_next_token_step(model, optimizer, batch)


def _candidate_fingerprints(conn: sqlite3.Connection, limit: int | None) -> list[str]:
    """Return fingerprints that don't yet have a complete screening_750 measurement.

    We use ``fp_id_collapse_rate IS NULL`` as the canonical "didn't get a
    trained backfill" check since it's the only metric that strictly
    requires training-step snapshots. Rows already populated by the
    earlier at-init backfill (``fp_metric_phase='init'``) get re-measured
    here at the proper screening_750 phase, which is correct behavior.
    """
    sql = (
        "SELECT graph_fingerprint, MAX(timestamp) AS ts FROM program_results_compat "
        "WHERE graph_json IS NOT NULL AND length(graph_json) > 0 "
        "  AND graph_fingerprint IS NOT NULL "
        "  AND (fp_id_collapse_rate IS NULL "
        "       OR fp_metric_phase IS NULL "
        "       OR fp_metric_phase != 'screening_750') "
        "GROUP BY graph_fingerprint ORDER BY ts DESC"
    )
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"
    return [row[0] for row in conn.execute(sql)]


def _representative_graph_json(
    conn: sqlite3.Connection, fingerprint: str
) -> str | None:
    row = conn.execute(
        "SELECT graph_json FROM program_results_compat "
        "WHERE graph_fingerprint = ? AND graph_json IS NOT NULL "
        "  AND length(graph_json) > 0 "
        "ORDER BY timestamp DESC LIMIT 1",
        (fingerprint,),
    ).fetchone()
    return None if row is None else resolve_graph_json_value(conn, DB_PATH, row[0])


def _measure_screening_750(
    graph_json: str,
    *,
    tokens: torch.Tensor,
    device: str,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
) -> dict | None:
    """Train model to step 750, snapshot at 150 + 750, run all metrics."""
    dev = torch.device(device)
    graph = graph_from_json(graph_json)
    model = SynthesizedModel(
        [graph], vocab_size=vocab_size, model_dim=graph.model_dim
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    probe_ids = _sample_batch(tokens, 8, seq_len, dev)[:, :seq_len]

    early_snap = None
    late_snap = None
    try:
        # Train, capturing snapshots at the canonical step boundaries.
        for step in range(1, SCREENING_STEPS + 1):
            batch = _sample_batch(tokens, batch_size, seq_len, dev)
            _train_step(model, optimizer, batch)
            if step == ID_COLLAPSE_EARLY_STEP:
                early_snap = capture_hidden_state_snapshot(
                    model, probe_ids, step=step, device=str(dev)
                )
                model.train()
            if step == ID_COLLAPSE_LATE_STEP:
                late_snap = capture_hidden_state_snapshot(
                    model, probe_ids, step=step, device=str(dev)
                )
                model.train()

        result = compute_trajectory_metrics(
            model,
            metric_phase="screening_750",
            device=str(dev),
            spec_norm_vocab_size=vocab_size,
            id_collapse_early=early_snap,
            id_collapse_late=late_snap,
        )
    finally:
        del model, optimizer
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    payload = result.to_column_dict()
    if payload.get("fp_jacobian_erf_status") in (None, "init"):
        return None
    return {k: payload.get(k) for k in TRAJECTORY_COLUMNS}


def _propagate(conn: sqlite3.Connection, fingerprint: str, payload: dict) -> int:
    return update_graph_runs_columns(
        conn,
        fingerprint,
        payload,
        TRAJECTORY_COLUMNS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--commit-every", type=int, default=10)
    add_gpu_safety_args(parser)
    args = parser.parse_args()

    prepare_cuda_if_requested(
        device=args.device,
        max_other_gpu_mib=args.max_other_gpu_mib,
        tool_name=TOOL_NAME,
        wait_for_gpu=args.wait_for_gpu,
        gpu_memory_fraction=args.gpu_memory_fraction,
    )

    with (
        acquire_gpu_lock(tool_name=TOOL_NAME),
        acquire_writer_lock(tool_name=TOOL_NAME),
    ):
        _run_backfill(args)


def _run_backfill(args: argparse.Namespace) -> None:
    print(f"[setup] loading WikiText103 corpus (vocab={args.vocab_size})", flush=True)
    tokens = _load_corpus(args.vocab_size)
    print(f"[setup] {tokens.numel():,} tokens", flush=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    fingerprints = _candidate_fingerprints(conn, args.limit)
    total = len(fingerprints)
    print(f"[setup] backfill scope: {total} unique fingerprints", flush=True)
    if total == 0:
        return

    succeeded = 0
    failed = 0
    propagated = 0
    t_start = time.time()

    for idx, fingerprint in enumerate(fingerprints, start=1):
        graph_json = _representative_graph_json(conn, fingerprint)
        if graph_json is None:
            failed += 1
            continue
        try:
            payload = _measure_screening_750(
                graph_json,
                tokens=tokens,
                device=args.device,
                vocab_size=args.vocab_size,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
            )
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            failed += 1
            print(
                f"  [{idx}/{total}] {fingerprint[:12]} measure-failed "
                f"({type(exc).__name__}): {str(exc)[:160]}",
                flush=True,
            )
            continue
        if payload is None:
            failed += 1
            continue
        rowcount = _propagate(conn, fingerprint, payload)
        succeeded += 1
        propagated += rowcount

        if succeeded % args.commit_every == 0:
            print_serial_commit_progress(
                conn,
                idx=idx,
                total=total,
                succeeded=succeeded,
                failed=failed,
                propagated=propagated,
                t_start=t_start,
                eta_decimals=0,
                flush=True,
            )

    conn.commit()
    elapsed = time.time() - t_start
    print(
        f"[done] ok={succeeded} fail={failed} rows_updated={propagated} "
        f"elapsed={elapsed / 60:.1f}m",
        flush=True,
    )
    conn.close()


if __name__ == "__main__":
    main()
