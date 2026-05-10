#!/usr/bin/env python
"""Run AR Validation/easy25 for DB rows missing first-class AR Validation metrics.

Default mode is a dry-run that only prints the selected cohort. Use ``--write``
to run the CUDA probe and persist ``ar_validation_*`` fields on existing
``program_results`` rows. Re-running resumes naturally because rows with any
AR Validation value are skipped unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

import torch

from research.eval.ar_validation import ARValidationConfig
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.tools.check_backup_freshness import main as check_backup_freshness_main
from research.tools.import_ar_validation_fingerprint_sweep import (
    AR_VALIDATION_COLUMNS,
    _merge_provenance,
    _program_results_read_table,
    _table_columns,
    ensure_ar_validation_columns,
)
from research.tools.run_ar_validation_fingerprint_sweep import (
    DEFAULT_CORPUS,
    DEFAULT_DB,
    DEFAULT_OUT_DIR,
    _append_row,
    _load_projected_corpus,
    _run_one,
)


DEFAULT_BACKFILL_OUT_DIR = DEFAULT_OUT_DIR / "db_backfill"
DEFAULT_TIERS = ("validation",)


def _missing_ar_validation_clause() -> str:
    return " AND ".join(f"pr.{column} IS NULL" for column in AR_VALIDATION_COLUMNS)


def has_existing_ar_validation_result(
    conn: sqlite3.Connection,
    result_id: str,
    *,
    columns: dict[str, str] | None = None,
) -> bool:
    columns = columns or _table_columns(conn, "program_results")
    ar_validation_columns = [name for name in AR_VALIDATION_COLUMNS if name in columns]
    if not ar_validation_columns:
        return False
    table = _program_results_read_table(conn)
    row = conn.execute(
        f"SELECT {', '.join(ar_validation_columns)} FROM {table} WHERE result_id = ?",
        (result_id,),
    ).fetchone()
    if row is None:
        return False
    return any(row[name] is not None for name in ar_validation_columns)


def select_backfill_rows(
    conn: sqlite3.Connection,
    *,
    tiers: tuple[str, ...],
    result_ids: tuple[str, ...],
    fingerprints: tuple[str, ...],
    limit: int,
    offset: int,
    overwrite: bool,
) -> list[sqlite3.Row]:
    where = [
        "COALESCE(pr.graph_json, '') NOT IN ('', '{}')",
        "COALESCE(pr.graph_fingerprint, l.graph_fingerprint, '') <> ''",
    ]
    params: list[Any] = []
    order_params: list[Any] = []
    order_clause = "l.composite_score DESC NULLS LAST, pr.timestamp DESC"
    if result_ids:
        where.append(f"pr.result_id IN ({', '.join('?' for _ in result_ids)})")
        params.extend(result_ids)
        order_parts = []
        for idx, result_id in enumerate(result_ids):
            order_parts.append("WHEN ? THEN ?")
            order_params.extend([result_id, idx])
        order_params.append(len(result_ids))
        order_clause = (
            f"CASE pr.result_id {' '.join(order_parts)} ELSE ? END, "
            "l.composite_score DESC NULLS LAST, pr.timestamp DESC"
        )
    else:
        if tiers:
            where.append(f"COALESCE(l.tier, '') IN ({', '.join('?' for _ in tiers)})")
            params.extend(tiers)
        if fingerprints:
            where.append(
                f"COALESCE(l.graph_fingerprint, pr.graph_fingerprint) IN ({', '.join('?' for _ in fingerprints)})",
            )
            params.extend(fingerprints)
    if not overwrite:
        where.append(_missing_ar_validation_clause())
    table = _program_results_read_table(conn)
    query = f"""
        SELECT
            pr.result_id,
            pr.experiment_id,
            COALESCE(l.graph_fingerprint, pr.graph_fingerprint) AS graph_fingerprint,
            pr.graph_json,
            COALESCE(l.model_source, pr.model_source, '') AS model_source,
            COALESCE(l.tier, '') AS tier,
            COALESCE(l.is_reference, 0) AS is_reference,
            COALESCE(l.reference_name, '') AS reference_name,
            l.composite_score,
            l.validation_loss_ratio,
            pr.loss_ratio
        FROM {table} pr
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE {" AND ".join(where)}
        ORDER BY {order_clause}
        LIMIT ? OFFSET ?
    """
    params.extend(order_params)
    params.extend([int(limit), int(offset)])
    return list(conn.execute(query, params).fetchall())


def _ar_validation_values_from_result_row(result_row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result_row[key]
        for key in AR_VALIDATION_COLUMNS
        if key in result_row and result_row[key] is not None and result_row[key] != ""
    }


def persist_ar_validation_result(
    conn: sqlite3.Connection,
    *,
    result_id: str,
    values: dict[str, Any],
    provenance: dict[str, Any],
    overwrite: bool,
    columns: dict[str, str] | None = None,
) -> bool:
    columns = columns or _table_columns(conn, "program_results")
    items = [(key, value) for key, value in values.items() if key in columns]
    if not items:
        return False
    if "data_provenance_json" in columns:
        table = _program_results_read_table(conn)
        row = conn.execute(
            f"SELECT data_provenance_json FROM {table} WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        raw = row["data_provenance_json"] if row else None
        items.append(("data_provenance_json", _merge_provenance(raw, provenance)))
    set_clause = ", ".join(f"{key} = ?" for key, _value in items)
    params = [value for _key, value in items]
    params.append(result_id)
    where = "result_id = ?"
    if not overwrite:
        missing_clause = " AND ".join(
            f"{name} IS NULL" for name in AR_VALIDATION_COLUMNS if name in columns
        )
        if missing_clause:
            where = f"{where} AND {missing_clause}"
    cursor = conn.execute(
        f"UPDATE program_results SET {set_clause} WHERE {where}", params
    )
    conn.commit()
    return cursor.rowcount > 0


def run(args: argparse.Namespace, out: TextIO = sys.stdout) -> int:
    tiers = tuple(args.tier or DEFAULT_TIERS)
    result_ids = tuple(args.result_id or ())
    fingerprints = tuple(args.fingerprint or ())
    if args.write:
        rc = check_backup_freshness_main([])
        if rc != 0:
            return rc
        if torch.device(args.device).type != "cuda":
            raise SystemExit("ar_validation_db_backfill_requires_cuda")
        if not torch.cuda.is_available():
            raise SystemExit("cuda_unavailable")
        conn = sqlite3.connect(str(args.db), timeout=30.0)
    else:
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    run_id = time.strftime("ar_validation_db_backfill_%Y%m%dT%H%M%S")
    out_csv = args.out or (DEFAULT_BACKFILL_OUT_DIR / f"{run_id}.csv")
    cfg_kwargs: dict[str, Any] = {
        "timeout_s": float(args.timeout_s),
        "copy_model": False,
    }
    if args.train_steps is not None:
        cfg_kwargs["train_steps"] = int(args.train_steps)
    cfg = ARValidationConfig(**cfg_kwargs)
    try:
        if args.write:
            ensure_ar_validation_columns(conn)
            program_columns = _table_columns(conn, "program_results")
        else:
            program_columns = _table_columns(conn, "program_results")
            missing = [
                name for name in AR_VALIDATION_COLUMNS if name not in program_columns
            ]
            if missing:
                print(f"missing_ar_validation_columns={','.join(missing)}", file=out)
                return 1
        rows = [
            dict(row)
            for row in select_backfill_rows(
                conn,
                tiers=tiers,
                result_ids=result_ids,
                fingerprints=fingerprints,
                limit=int(args.limit),
                offset=int(args.offset),
                overwrite=bool(args.overwrite),
            )
        ]
        for row in rows:
            row["graph_json"] = resolve_graph_json_value(
                conn,
                args.db,
                row.get("graph_json"),
            )
        print(
            json.dumps(
                {
                    "mode": "WRITE" if args.write else "DRY-RUN",
                    "selected": len(rows),
                    "db": str(args.db),
                    "out": str(out_csv),
                    "tiers": tiers,
                    "result_ids": result_ids,
                    "fingerprints": fingerprints,
                    "limit": int(args.limit),
                    "offset": int(args.offset),
                    "overwrite": bool(args.overwrite),
                    "device": args.device,
                    "metric_version": "ar_validation_v2_easy25",
                    "config": asdict(cfg),
                },
                sort_keys=True,
            ),
            file=out,
        )
        for idx, row in enumerate(rows[:25], start=1):
            print(
                json.dumps(
                    {
                        "candidate": idx,
                        "result_id": row["result_id"],
                        "graph_fingerprint": row["graph_fingerprint"],
                        "tier": row["tier"],
                        "composite_score": row["composite_score"],
                    },
                    sort_keys=True,
                ),
                file=out,
            )
        if len(rows) > 25:
            print(f"omitted_candidates={len(rows) - 25}", file=out)
        if not args.write or not rows:
            return 0

        corpus_tokens = _load_projected_corpus(
            args.corpus_path,
            int(args.vocab_size),
            device=torch.device(args.device),
        )
        updated = 0
        for idx, row in enumerate(rows, start=1):
            rank = int(args.offset) + idx
            if not args.overwrite and has_existing_ar_validation_result(
                conn,
                str(row["result_id"]),
                columns=program_columns,
            ):
                print(
                    json.dumps(
                        {
                            "event": "skipped_existing_ar_validation",
                            "rank": rank,
                            "result_id": row["result_id"],
                            "graph_fingerprint": row["graph_fingerprint"],
                        },
                        sort_keys=True,
                    ),
                    file=out,
                    flush=True,
                )
                continue
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
                checkpoint_dir=args.checkpoint_dir
                or (DEFAULT_BACKFILL_OUT_DIR / "checkpoints" / run_id),
                save_checkpoint=not bool(args.no_save_checkpoints),
            )
            _append_row(out_csv, result_row)
            values = _ar_validation_values_from_result_row(result_row)
            persisted = persist_ar_validation_result(
                conn,
                result_id=str(row["result_id"]),
                values=values,
                provenance={
                    "source": "ar_validation_db_backfill",
                    "run_id": run_id,
                    "out_csv": str(out_csv),
                    "rank": rank,
                    "overwrite": bool(args.overwrite),
                    "updated_at_unix": round(time.time(), 3),
                },
                overwrite=bool(args.overwrite),
                columns=program_columns,
            )
            if persisted:
                updated += 1
            print(
                json.dumps(
                    {
                        "event": "persisted",
                        "rank": rank,
                        "result_id": row["result_id"],
                        "graph_fingerprint": row["graph_fingerprint"],
                        "status": result_row.get("ar_validation_status"),
                        "score": result_row.get("ar_validation_rank_score"),
                        "updated": persisted,
                    },
                    sort_keys=True,
                ),
                file=out,
                flush=True,
            )
        print(
            json.dumps({"updated": updated, "out": str(out_csv)}, sort_keys=True),
            file=out,
        )
        return 0
    finally:
        conn.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--write", action="store_true", help="Run probes and persist DB updates."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun rows with existing AR Validation values.",
    )
    parser.add_argument(
        "--tier",
        action="append",
        help="Leaderboard tier to include. Defaults to validation.",
    )
    parser.add_argument(
        "--result-id", action="append", help="Specific result_id to backfill."
    )
    parser.add_argument(
        "--fingerprint", action="append", help="Specific graph_fingerprint to backfill."
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
