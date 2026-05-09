#!/usr/bin/env python
"""Backfill jacobian spectral norm + effective rank + sensitivity uniformity.

Rebuilds models from program_results.graph_json, runs analyze_sensitivity with
the autograd-correct kernel gates (see compiler_op_utils._t / _c), and writes
real values into program_results. Dedupes by graph_fingerprint — one model per
unique architecture, results propagated to all rows sharing the fingerprint.

Usage:
    python -m research.tools.backfill_spec_norm --device cuda
    python -m research.tools.backfill_spec_norm --device cuda --limit 200
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import torch

# Force-import compiler so OP_DISPATCH is populated before any CompiledOp is built.
import research.synthesis.compiler  # noqa: F401
from research.synthesis import graph_from_json
from research.synthesis.compiled_model import SynthesizedModel
from research.eval.fingerprint_sensitivity import analyze_sensitivity
from research.tools._concurrency import (
    acquire_gpu_lock,
    acquire_writer_lock,
    assert_gpu_quiet,
    cap_gpu_memory,
)
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value

DB_PATH = Path(__file__).resolve().parents[1] / "runs.db"
TOOL_NAME = "backfill_spec_norm"


def _candidate_fingerprints(conn: sqlite3.Connection, limit: int | None) -> list[str]:
    sql = (
        "SELECT graph_fingerprint, MAX(timestamp) AS ts FROM program_results "
        "WHERE graph_json IS NOT NULL AND length(graph_json) > 0 "
        "AND graph_fingerprint IS NOT NULL "
        "AND fp_jacobian_spectral_norm IS NULL "
        "GROUP BY graph_fingerprint ORDER BY ts DESC"
    )
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"
    return [row[0] for row in conn.execute(sql)]


def _representative_graph_json(
    conn: sqlite3.Connection, fingerprint: str
) -> tuple[str, int] | None:
    row = conn.execute(
        "SELECT graph_json, COALESCE(graph_n_ops, 0) FROM program_results "
        "WHERE graph_fingerprint = ? AND graph_json IS NOT NULL "
        "AND length(graph_json) > 0 "
        "ORDER BY timestamp DESC LIMIT 1",
        (fingerprint,),
    ).fetchone()
    return (
        None
        if row is None
        else (resolve_graph_json_value(conn, DB_PATH, row[0]), row[1])
    )


def _measure(
    graph_json: str, device: str, seq_len: int, vocab_size: int
) -> dict | None:
    graph = graph_from_json(graph_json)
    model = SynthesizedModel(
        [graph],
        vocab_size=vocab_size,
        model_dim=graph.model_dim,
    ).to(device)
    try:
        result = analyze_sensitivity(
            model,
            torch.device(device),
            seq_len=seq_len,
            vocab_size=vocab_size,
        )
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if not result.get("_succeeded"):
        return None
    return {
        "spectral_norm": float(result["spectral_norm"]),
        "effective_rank": float(result["effective_rank"]),
        "uniformity": float(result["uniformity"]),
    }


def _propagate(conn: sqlite3.Connection, fingerprint: str, payload: dict) -> int:
    cursor = conn.execute(
        "UPDATE program_results SET "
        "fp_jacobian_spectral_norm = ?, "
        "fp_jacobian_effective_rank = ?, "
        "fp_sensitivity_uniformity = ? "
        "WHERE graph_fingerprint = ? AND fp_jacobian_spectral_norm IS NULL",
        (
            payload["spectral_norm"],
            payload["effective_rank"],
            payload["uniformity"],
            fingerprint,
        ),
    )
    return cursor.rowcount


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N unique fingerprints (default: all)",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=25,
        help="Commit DB writes every N successful fingerprints",
    )
    parser.add_argument(
        "--max-other-gpu-mib",
        type=int,
        default=4096,
        help=(
            "Refuse to start if any other GPU process is using more than this "
            "many MiB of VRAM. Set to a large value to override (e.g. 30000)."
        ),
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.5,
        help="Cap our process's CUDA memory at this fraction of the card.",
    )
    parser.add_argument(
        "--wait-for-gpu",
        action="store_true",
        help="Sleep-poll until the GPU is quiet instead of exiting on busy.",
    )
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
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    fingerprints = _candidate_fingerprints(conn, args.limit)
    total = len(fingerprints)
    print(f"backfill scope: {total} unique fingerprints")
    if total == 0:
        return

    succeeded = 0
    failed = 0
    propagated = 0
    t_start = time.time()

    for idx, fingerprint in enumerate(fingerprints, start=1):
        graph_payload = _representative_graph_json(conn, fingerprint)
        if graph_payload is None:
            failed += 1
            continue
        graph_json, n_ops = graph_payload
        try:
            payload = _measure(
                graph_json,
                device=args.device,
                seq_len=args.seq_len,
                vocab_size=args.vocab_size,
            )
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            # KeyError: stored graph_json missing required field
            #   (e.g. 'model_dim' from very old rows).
            # TypeError: malformed graph_json structure.
            # RuntimeError: forward-pass shape mismatch / OOM.
            # ValueError: graph reconstruction issue.
            failed += 1
            print(
                f"  [{idx}/{total}] {fingerprint[:12]} measure-failed "
                f"({type(exc).__name__}): {exc}"
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
                f"  [{idx}/{total}] ok={succeeded} fail={failed} rows={propagated} "
                f"rate={rate:.2f}/s eta={eta_min:.1f}m"
            )

    conn.commit()
    elapsed = time.time() - t_start
    print(
        f"done: ok={succeeded} fail={failed} rows_updated={propagated} "
        f"elapsed={elapsed / 60:.1f}m"
    )
    conn.close()


if __name__ == "__main__":
    main()
