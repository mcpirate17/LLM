#!/usr/bin/env python
"""Run AR Validation champion probes over top-scoring leaderboard fingerprints.

This is a read-only sweep harness. It selects one representative leaderboard row
per graph fingerprint, compiles that graph at a fixed layer count, runs the
current AR Validation champion probe, and appends every result to a CSV that can be
loaded into the notebook DB later.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from research.eval.ar_validation import ARValidationConfig, run_ar_validation
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.scientist.native_runner import compile_model_native_first as compile_model
from research.synthesis.serializer import graph_from_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "research/runs.db"
DEFAULT_CORPUS = PROJECT_ROOT / "research/corpus/wikitext103_train.npy"
DEFAULT_OUT_DIR = PROJECT_ROOT / "research/runtime/ar_validation_fingerprint_sweep"
CSV_FIELDS = [
    "run_id",
    "created_unix",
    "rank",
    "selection_offset",
    "result_id",
    "experiment_id",
    "graph_fingerprint",
    "model_source",
    "tier",
    "is_reference",
    "reference_name",
    "composite_score",
    "validation_loss_ratio",
    "loss_ratio",
    "graph_model_dim",
    "compiled_layers",
    "compiled_vocab_size",
    "pretrain_corpus_path",
    "pretrain_steps",
    "pretrain_batch_size",
    "pretrain_seq_len",
    "pretrain_lr",
    "pretrain_final_loss",
    "pretrain_elapsed_ms",
    "checkpoint_path",
    "config_json",
    "ar_validation_metric_version",
    "ar_validation_status",
    "ar_validation_final_acc",
    "ar_validation_held_pair_acc",
    "ar_validation_held_class_acc",
    "ar_validation_steps_to_floor",
    "ar_validation_rank_score",
    "ar_validation_elapsed_ms",
    "wall_seconds",
    "learning_curve_json",
    "error",
]


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _query_top_fingerprints(
    conn: sqlite3.Connection,
    *,
    db_path: Path | None = None,
    limit: int,
    offset: int,
    include_references: bool,
) -> list[dict[str, Any]]:
    ref_clause = "" if include_references else "AND COALESCE(l.is_reference, 0) = 0"
    query = f"""
        WITH ranked AS (
            SELECT
                l.result_id,
                pr.experiment_id,
                COALESCE(l.graph_fingerprint, pr.graph_fingerprint) AS graph_fingerprint,
                pr.graph_json,
                l.model_source,
                l.tier,
                COALESCE(l.is_reference, 0) AS is_reference,
                l.reference_name,
                l.composite_score,
                l.validation_loss_ratio,
                pr.loss_ratio,
                ROW_NUMBER() OVER (
                    PARTITION BY COALESCE(l.graph_fingerprint, pr.graph_fingerprint)
                    ORDER BY l.composite_score DESC NULLS LAST, l.timestamp DESC
                ) AS fp_rank
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            WHERE COALESCE(l.graph_fingerprint, pr.graph_fingerprint, '') <> ''
              AND COALESCE(pr.graph_json, '') NOT IN ('', '{{}}')
              AND l.composite_score IS NOT NULL
              {ref_clause}
        )
        SELECT *
        FROM ranked
        WHERE fp_rank = 1
        ORDER BY composite_score DESC NULLS LAST
        LIMIT ? OFFSET ?
    """
    rows = [
        dict(row) for row in conn.execute(query, (int(limit), int(offset))).fetchall()
    ]
    for row in rows:
        row["graph_json"] = resolve_graph_json_value(
            conn,
            db_path or DEFAULT_DB,
            row.get("graph_json"),
        )
    return rows


def _load_done_fingerprints(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="") as handle:
        return {
            str(row.get("graph_fingerprint") or "")
            for row in csv.DictReader(handle)
            if row.get("ar_validation_status") == "ok"
        }


def _load_projected_corpus(
    corpus_path: Path,
    vocab_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    arr = np.load(str(corpus_path), mmap_mode="r")
    tokens_np = np.array(arr, dtype=np.int64, copy=True)
    if int(vocab_size) > 0:
        np.remainder(tokens_np, int(vocab_size), out=tokens_np)
    tokens = torch.from_numpy(tokens_np)
    return tokens.contiguous().to(device=device, non_blocking=True)


def _sample_lm_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    max_start = int(tokens.numel()) - int(seq_len) - 1
    if max_start <= 0:
        raise ValueError("corpus is too small for requested pretrain_seq_len")
    starts = torch.randint(
        0,
        max_start,
        (int(batch_size),),
        generator=generator,
        device=device,
    )
    positions = torch.arange(int(seq_len) + 1, device=device)
    return tokens[starts.unsqueeze(1) + positions.unsqueeze(0)]


def _pretrain_lm(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    seed: int,
    progress_every: int,
    progress_label: dict[str, Any],
) -> tuple[float | None, float]:
    if int(steps) <= 0:
        return None, 0.0
    model.train()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    try:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            fused=(device.type == "cuda"),
        )
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
    t0 = time.perf_counter()
    final_loss: float | None = None
    progress_every = max(0, int(progress_every))
    for step in range(1, int(steps) + 1):
        batch = _sample_lm_batch(
            tokens,
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            device=device,
            generator=gen,
        )
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch[:, 1:].reshape(-1),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non_finite_pretrain_loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_loss = float(loss.item())
        if progress_every and (step % progress_every == 0 or step == int(steps)):
            elapsed = time.perf_counter() - t0
            print(
                json.dumps(
                    {
                        **progress_label,
                        "event": "pretrain_progress",
                        "step": step,
                        "steps": int(steps),
                        "loss": round(final_loss, 6),
                        "elapsed_s": round(elapsed, 3),
                        "steps_per_s": round(step / max(elapsed, 1e-9), 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return final_loss, (time.perf_counter() - t0) * 1000.0


def _append_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        handle.flush()


def _result_error_row(
    base: dict[str, Any],
    *,
    status: str,
    error: str,
    wall_seconds: float,
) -> dict[str, Any]:
    return {
        **base,
        "ar_validation_status": status,
        "wall_seconds": round(float(wall_seconds), 3),
        "error": error,
    }


def _run_one(
    row: sqlite3.Row,
    *,
    run_id: str,
    rank: int,
    offset: int,
    cfg: ARValidationConfig,
    layers: int,
    vocab_size: int,
    device: str,
    init_seed: int,
    corpus_tokens: torch.Tensor,
    corpus_path: Path,
    pretrain_steps: int,
    pretrain_batch_size: int,
    pretrain_seq_len: int,
    pretrain_lr: float,
    progress_every: int,
    checkpoint_dir: Path,
    save_checkpoint: bool,
) -> dict[str, Any]:
    created = round(time.time(), 3)
    graph = graph_from_json(str(row["graph_json"]))
    graph_dim = int(getattr(graph, "model_dim", 0) or 0)
    base: dict[str, Any] = {
        "run_id": run_id,
        "created_unix": created,
        "rank": int(rank),
        "selection_offset": int(offset),
        "result_id": str(row["result_id"]),
        "experiment_id": str(row["experiment_id"]),
        "graph_fingerprint": str(row["graph_fingerprint"]),
        "model_source": str(row["model_source"] or ""),
        "tier": str(row["tier"] or ""),
        "is_reference": int(row["is_reference"] or 0),
        "reference_name": str(row["reference_name"] or ""),
        "composite_score": row["composite_score"],
        "validation_loss_ratio": row["validation_loss_ratio"],
        "loss_ratio": row["loss_ratio"],
        "graph_model_dim": graph_dim,
        "compiled_layers": int(layers),
        "compiled_vocab_size": int(vocab_size),
        "pretrain_corpus_path": str(corpus_path),
        "pretrain_steps": int(pretrain_steps),
        "pretrain_batch_size": int(pretrain_batch_size),
        "pretrain_seq_len": int(pretrain_seq_len),
        "pretrain_lr": float(pretrain_lr),
        "config_json": json.dumps(asdict(cfg), sort_keys=True),
    }
    t0 = time.perf_counter()
    try:
        torch.manual_seed(int(init_seed) + int(rank))
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(init_seed) + int(rank))
        model = compile_model(
            [graph] * int(layers),
            vocab_size=int(vocab_size),
            max_seq_len=max(
                512, int(pretrain_seq_len), 3 * int(cfg.pairs_per_example) + 4
            ),
        )
        model.to(device)
        pre_loss, pre_ms = _pretrain_lm(
            model,
            corpus_tokens,
            device=torch.device(device),
            steps=int(pretrain_steps),
            batch_size=int(pretrain_batch_size),
            seq_len=int(pretrain_seq_len),
            lr=float(pretrain_lr),
            seed=int(init_seed) + int(rank) * 1009,
            progress_every=int(progress_every),
            progress_label={
                "rank": int(rank),
                "result_id": str(row["result_id"]),
                "graph_fingerprint": str(row["graph_fingerprint"]),
            },
        )
        checkpoint_path = ""
        if save_checkpoint:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = str(
                checkpoint_dir
                / f"rank{int(rank):04d}_{str(row['result_id'])}_{str(row['graph_fingerprint'])[:12]}_pretrain{int(pretrain_steps)}.pt"
            )
            torch.save(
                {
                    "artifact_kind": "ar_validation_fingerprint_sweep_pretrained_model",
                    "run_id": run_id,
                    "created_unix": created,
                    "result_id": str(row["result_id"]),
                    "experiment_id": str(row["experiment_id"]),
                    "graph_fingerprint": str(row["graph_fingerprint"]),
                    "graph_json": str(row["graph_json"]),
                    "compiled_layers": int(layers),
                    "compiled_vocab_size": int(vocab_size),
                    "pretrain_steps": int(pretrain_steps),
                    "pretrain_batch_size": int(pretrain_batch_size),
                    "pretrain_seq_len": int(pretrain_seq_len),
                    "pretrain_lr": float(pretrain_lr),
                    "pretrain_corpus_path": str(corpus_path),
                    "model_state_dict": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                },
                checkpoint_path,
            )
        result = run_ar_validation(model, cfg=cfg, device=device)
        payload = result.to_dict()
        return {
            **base,
            **payload,
            "pretrain_final_loss": pre_loss,
            "pretrain_elapsed_ms": round(pre_ms, 1),
            "checkpoint_path": checkpoint_path,
            "learning_curve_json": payload.get(
                "ar_validation_learning_curve_json",
                "",
            ),
            "wall_seconds": round(time.perf_counter() - t0, 3),
            "error": result.error or "",
        }
    except Exception as exc:  # noqa: BLE001
        return _result_error_row(
            base,
            status="exception",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=12)}",
            wall_seconds=time.perf_counter() - t0,
        )
    finally:
        if "model" in locals():
            del model
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--pretrain-steps", type=int, default=5_000)
    parser.add_argument("--pretrain-batch-size", type=int, default=8)
    parser.add_argument("--pretrain-seq-len", type=int, default=128)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument("--include-references", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested_device = torch.device(args.device)
    if requested_device.type != "cuda":
        raise SystemExit("ar_validation_fingerprint_sweep_requires_cuda")
    if not torch.cuda.is_available():
        raise SystemExit("cuda_unavailable")
    run_id = time.strftime("ar_validation_fp_sweep_%Y%m%dT%H%M%S")
    out = args.out or (
        DEFAULT_OUT_DIR
        / f"{run_id}_offset{int(args.offset):04d}_limit{int(args.limit):04d}.csv"
    )
    checkpoint_dir = args.checkpoint_dir or (DEFAULT_OUT_DIR / "checkpoints" / run_id)
    cfg_kwargs: dict[str, Any] = {
        "timeout_s": float(args.timeout_s),
        "copy_model": False,
    }
    if args.train_steps is not None:
        cfg_kwargs["train_steps"] = int(args.train_steps)
    cfg = ARValidationConfig(**cfg_kwargs)

    conn = _connect_ro(args.db)
    selected = _query_top_fingerprints(
        conn,
        db_path=args.db,
        limit=int(args.limit),
        offset=int(args.offset),
        include_references=bool(args.include_references),
    )
    done = set() if args.force else _load_done_fingerprints(out)

    print(
        json.dumps(
            {
                "event": "selected",
                "run_id": run_id,
                "db": str(args.db),
                "out": str(out),
                "limit": int(args.limit),
                "offset": int(args.offset),
                "n_selected": len(selected),
                "layers": int(args.layers),
                "vocab_size": int(args.vocab_size),
                "pretrain_steps": int(args.pretrain_steps),
                "pretrain_batch_size": int(args.pretrain_batch_size),
                "pretrain_seq_len": int(args.pretrain_seq_len),
                "pretrain_lr": float(args.pretrain_lr),
                "progress_every": int(args.progress_every),
                "corpus_path": str(args.corpus_path),
                "checkpoint_dir": str(checkpoint_dir),
                "save_checkpoints": not bool(args.no_save_checkpoints),
                "device": args.device,
                "metric_version": "ar_validation_v2_easy25",
                "config": asdict(cfg),
                "dry_run": bool(args.dry_run),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    corpus_tokens: torch.Tensor | None = None
    if not args.dry_run:
        print(
            json.dumps(
                {
                    "event": "load_corpus",
                    "corpus_path": str(args.corpus_path),
                    "vocab_size": int(args.vocab_size),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        corpus_tokens = _load_projected_corpus(
            args.corpus_path,
            int(args.vocab_size),
            device=requested_device,
        )
        print(
            json.dumps(
                {
                    "event": "corpus_loaded",
                    "n_tokens": int(corpus_tokens.numel()),
                    "dtype": str(corpus_tokens.dtype),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for idx, row in enumerate(selected, start=1):
        rank = int(args.offset) + idx
        fp = str(row["graph_fingerprint"])
        if fp in done:
            print(
                json.dumps(
                    {
                        "event": "skip_done",
                        "rank": rank,
                        "graph_fingerprint": fp,
                        "result_id": row["result_id"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "event": "candidate",
                        "rank": rank,
                        "result_id": row["result_id"],
                        "graph_fingerprint": fp,
                        "composite_score": row["composite_score"],
                        "tier": row["tier"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        print(
            json.dumps(
                {
                    "event": "start",
                    "rank": rank,
                    "result_id": row["result_id"],
                    "graph_fingerprint": fp,
                    "composite_score": row["composite_score"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        result_row = _run_one(
            row,
            run_id=run_id,
            rank=rank,
            offset=int(args.offset),
            cfg=cfg,
            layers=int(args.layers),
            vocab_size=int(args.vocab_size),
            device=str(args.device),
            init_seed=int(args.seed),
            corpus_tokens=corpus_tokens,
            corpus_path=args.corpus_path,
            pretrain_steps=int(args.pretrain_steps),
            pretrain_batch_size=int(args.pretrain_batch_size),
            pretrain_seq_len=int(args.pretrain_seq_len),
            pretrain_lr=float(args.pretrain_lr),
            progress_every=int(args.progress_every),
            checkpoint_dir=checkpoint_dir,
            save_checkpoint=not bool(args.no_save_checkpoints),
        )
        _append_row(out, result_row)
        print(
            json.dumps(
                {
                    "event": "done",
                    "rank": rank,
                    "result_id": row["result_id"],
                    "graph_fingerprint": fp,
                    "status": result_row.get("ar_validation_status"),
                    "score": result_row.get("ar_validation_rank_score"),
                    "held_pair": result_row.get(
                        "ar_validation_held_pair_acc",
                    ),
                    "held_class": result_row.get("ar_validation_held_class_acc"),
                    "wall_seconds": result_row.get("wall_seconds"),
                    "out": str(out),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
