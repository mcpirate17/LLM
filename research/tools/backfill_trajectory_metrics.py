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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    assert_gpu_quiet,
    cap_gpu_memory,
)
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "research" / "runs.db"
CORPUS_PATH = ROOT / "research" / "corpus" / "wikitext103_train.npy"
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

_TRAJECTORY_COLUMNS = (
    "fp_jacobian_spectral_norm",
    "fp_jacobian_effective_rank",
    "fp_sensitivity_uniformity",
    "fp_spec_norm_status",
    "fp_metric_phase",
    "fp_jacobian_erf_density",
    "fp_jacobian_erf_variance",
    "fp_jacobian_erf_decay_slope",
    "fp_jacobian_erf_last_norm",
    "fp_jacobian_erf_first_norm",
    "fp_jacobian_erf_status",
    "fp_jacobian_erf_elapsed_ms",
    "fp_icld_velocity",
    "fp_icld_early_loss",
    "fp_icld_late_loss",
    "fp_icld_delta_loss",
    "fp_icld_seq_len",
    "fp_icld_status",
    "fp_icld_elapsed_ms",
    "fp_id_pr_early",
    "fp_id_pr_late",
    "fp_id_norm_early",
    "fp_id_norm_late",
    "fp_id_step_early",
    "fp_id_step_late",
    "fp_id_collapse_rate",
    "fp_id_collapse_rate_normalized",
    "fp_id_collapse_status",
    "fp_id_collapse_elapsed_ms",
    "fp_logit_margin_velocity",
    "fp_logit_margin_initial",
    "fp_logit_margin_final",
    "fp_logit_margin_delta",
    "fp_logit_margin_n_steps",
    "fp_logit_margin_status",
    "fp_logit_margin_elapsed_ms",
)


def _load_corpus(vocab_size: int, max_tokens: int = 4_000_000) -> torch.Tensor:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"corpus npy not found: {CORPUS_PATH}")
    arr = np.load(CORPUS_PATH, mmap_mode="r")
    if arr.size > max_tokens:
        arr = arr[:max_tokens]
    tokens = torch.as_tensor(np.asarray(arr), dtype=torch.long)
    return tokens % vocab_size


def _sample_batch(
    tokens: torch.Tensor, batch_size: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    n = tokens.numel() - seq_len - 1
    if n <= 0:
        raise ValueError("corpus too small")
    starts = torch.randint(0, n, (batch_size,))
    return torch.stack(
        [tokens[s : s + seq_len + 1] for s in starts.tolist()], dim=0
    ).to(device)


def _train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
) -> float:
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
    if not torch.isfinite(loss):
        return float("nan")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


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
    return {k: payload.get(k) for k in _TRAJECTORY_COLUMNS}


def _propagate(conn: sqlite3.Connection, fingerprint: str, payload: dict) -> int:
    set_clause = ", ".join(f"{c} = ?" for c in _TRAJECTORY_COLUMNS)
    values = [payload.get(c) for c in _TRAJECTORY_COLUMNS]
    cursor = conn.execute(
        f"UPDATE graph_runs SET {set_clause} WHERE graph_fingerprint = ?",
        (*values, fingerprint),
    )
    return cursor.rowcount


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
    parser.add_argument("--max-other-gpu-mib", type=int, default=4096)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.5)
    parser.add_argument("--wait-for-gpu", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        assert_gpu_quiet(
            max_other_used_mib=args.max_other_gpu_mib,
            tool_name=TOOL_NAME,
            sleep_until_quiet=args.wait_for_gpu,
        )
        cap_gpu_memory(fraction=args.gpu_memory_fraction)

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
            conn.commit()
            elapsed = time.time() - t_start
            rate = succeeded / max(elapsed, 1e-6)
            eta_min = (total - idx) / max(rate, 1e-6) / 60.0
            print(
                f"  [{idx}/{total}] ok={succeeded} fail={failed} "
                f"rows={propagated} rate={rate:.2f}/s eta={eta_min:.0f}m",
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
