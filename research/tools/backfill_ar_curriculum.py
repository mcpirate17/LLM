#!/usr/bin/env python
"""Backfill `ar_curriculum_*` columns into program_results / leaderboard.

Reconstructs each candidate's compiled model and runs `ar_curriculum_probe`
in cumulative mode. Persists the full result dict (per-stage curve, AUC,
retention, lift, Z-scores, learning curve JSON) via the standard
`store_probe_results` path so the data is queryable for ML predictor training
and dashboard surfacing.

Tiered strategy (recommended for ML training data — see docstring at end of
file). Default selects top-N validation/investigation candidates ordered by
composite_score, skipping rows that already have the metric. Pass `--force`
to overwrite.

Usage:
    python -m research.tools.backfill_ar_curriculum \
        --tiers validation,investigation --top-per-tier 200 --device cuda

    # Smaller pilot batch to verify wiring before a long run:
    python -m research.tools.backfill_ar_curriculum \
        --tiers validation --top-per-tier 5 --device cuda --dry-run

    # Sharded across N workers (reproducible disjoint coverage):
    python -m research.tools.backfill_ar_curriculum \
        --tiers investigation --top-per-tier 1000 --shard 0/4 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from research.eval.ar_curriculum_probe import (
    AR_CURRICULUM_METRIC_VERSION,
    ARCurriculumConfig,
    ar_curriculum_probe,
    required_vocab_size,
)
from research.scientist.notebook import LabNotebook
from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_from_json
from research.tools.backfill import query_candidates, store_probe_results
from research.tools._db_maintenance import connect_writer, table_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "research/runs.db"

AR_CURRICULUM_COLUMNS: tuple[tuple[str, str], ...] = (
    ("ar_curriculum_metric_version", "TEXT"),
    ("ar_curriculum_auc_pair_final", "REAL"),
    ("ar_curriculum_auc_class_final", "REAL"),
    ("ar_curriculum_s0_held_pair_acc", "REAL"),
    ("ar_curriculum_s0_retention", "REAL"),
    ("ar_curriculum_max_passing_stage", "INTEGER"),
    ("ar_curriculum_per_stage_held_pair_acc", "TEXT"),
    ("ar_curriculum_per_stage_held_class_acc", "TEXT"),
    ("ar_curriculum_per_stage_lift_pair", "TEXT"),
    ("ar_curriculum_per_stage_z_score_pair", "TEXT"),
    ("ar_curriculum_per_stage_chance_pair", "TEXT"),
    ("ar_curriculum_learning_curve_json", "TEXT"),
    ("ar_curriculum_steps_trained", "INTEGER"),
    ("ar_curriculum_n_eval_examples", "INTEGER"),
    ("ar_curriculum_mode", "TEXT"),
    ("ar_curriculum_elapsed_ms", "REAL"),
    ("ar_curriculum_status", "TEXT"),
    ("ar_curriculum_error", "TEXT"),
)


def ensure_ar_curriculum_columns(db_path: Path) -> bool:
    """Idempotently add ar_curriculum_* columns to program_results.

    Returns True if any column was added (i.e. first-time schema evolution).
    Acquires the aria-db writer manager via ``connect_writer``; the caller
    must not hold the writer lock when this runs.
    """
    conn = connect_writer(db_path)
    try:
        existing = set(table_columns(conn, "program_results"))
        added: list[str] = []
        for name, decl in AR_CURRICULUM_COLUMNS:
            if name not in existing:
                conn.execute(f"ALTER TABLE program_results ADD COLUMN {name} {decl}")
                added.append(name)
        if added:
            conn.commit()
            logger.info("Added %d ar_curriculum columns: %s", len(added), added)
        return bool(added)
    finally:
        conn.close()


def _parse_shard(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    try:
        i, n = spec.split("/")
        return int(i), int(n)
    except (ValueError, AttributeError) as exc:
        raise SystemExit(f"--shard must be 'i/N' (got {spec!r}): {exc}")


def _load_priority_candidates(nb, jsonl_path: Path, *, force: bool):
    from research.tools.backfill import Candidate

    result_ids: list[str] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(entry.get("result_id") or "")
            if rid:
                result_ids.append(rid)
    if not result_ids:
        return []
    placeholders = ",".join("?" for _ in result_ids)
    null_filter = "" if force else "AND pr.ar_curriculum_auc_pair_final IS NULL"
    rows = nb.conn.execute(
        f"""
        SELECT
            l.entry_id, l.result_id, l.tier, l.composite_score,
            COALESCE(l.is_reference, 0) AS is_reference,
            COALESCE(l.model_source, '') AS model_source,
            pr.graph_json, pr.graph_fingerprint
        FROM program_results pr
        JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE pr.result_id IN ({placeholders})
          AND pr.graph_json IS NOT NULL
          {null_filter}
        """,
        tuple(result_ids),
    ).fetchall()
    by_rid = {str(r["result_id"]): r for r in rows}
    out = []
    for rid in result_ids:
        r = by_rid.get(rid)
        if r is None:
            continue
        out.append(
            Candidate(
                entry_id=str(r["entry_id"]),
                result_id=str(r["result_id"]),
                tier=str(r["tier"] or ""),
                composite_score=float(r["composite_score"] or 0.0),
                is_reference=bool(r["is_reference"]),
                model_source=str(r["model_source"] or ""),
                graph_json=r["graph_json"],
                graph_fingerprint=str(r["graph_fingerprint"] or ""),
            )
        )
    return out


def _build_model(graph_json_str: str, device: str, vocab_size: int) -> torch.nn.Module:
    """Compile the candidate's graph at the curriculum-required vocab.

    Note: this overrides the canonical large vocab with the smaller one needed
    for the AR curriculum probe — fine because the probe uses only token IDs
    inside the disjoint AR ranges, and the embedding/lm_head tables are
    re-initialized per probe call so no learned weights are wasted.
    """
    graph = graph_from_json(graph_json_str)
    model = compile_model([graph], vocab_size=vocab_size)
    return model.to(device).eval()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument(
        "--tiers",
        default="validation,investigation",
        help="Comma-separated tier names to backfill.",
    )
    p.add_argument("--top-per-tier", type=int, default=200)
    p.add_argument(
        "--shard",
        default=None,
        help="Worker shard 'i/N' (e.g. '0/4'). Disjoint per-tier coverage.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps-per-stage", type=int, default=1000)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite rows that already have ar_curriculum_auc_pair_final.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--priority-jsonl",
        type=Path,
        default=None,
        help="JSONL of result_ids (output of ar_curriculum_priority) to process in order. Bypasses --tiers/--top-per-tier.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Hard cap on total candidates regardless of --top-per-tier.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tiers = tuple(t.strip() for t in str(args.tiers).split(",") if t.strip())
    shard = _parse_shard(args.shard)
    null_col = None if args.force else "ar_curriculum_auc_pair_final"
    vocab_size = max(required_vocab_size(), 2048)
    run_id = datetime.now(timezone.utc).strftime("ar_curriculum_backfill_%Y%m%d_%H%M%S")

    logger.info(
        "%s tiers=%s top_per_tier=%d shard=%s device=%s force=%s dry_run=%s",
        run_id,
        tiers,
        args.top_per_tier,
        shard,
        args.device,
        args.force,
        args.dry_run,
    )

    if not args.dry_run:
        ensure_ar_curriculum_columns(args.db)
    nb = LabNotebook(str(args.db), read_only=bool(args.dry_run))
    pr_cols = set(table_columns(nb.conn, "program_results"))
    if null_col and null_col not in pr_cols:
        logger.info(
            "Column %s missing on dry-run; treating all rows as candidates", null_col
        )
        null_col = None
    if args.priority_jsonl:
        candidates = _load_priority_candidates(
            nb, args.priority_jsonl, force=bool(args.force)
        )
        logger.info("Loaded %d candidates from priority list", len(candidates))
    else:
        candidates = query_candidates(
            nb,
            tiers=tiers,
            top_per_tier=int(args.top_per_tier),
            null_column=null_col,
            force=bool(args.force),
            shard=shard,
        )
    if args.limit:
        candidates = candidates[: int(args.limit)]
    logger.info("Selected %d candidates", len(candidates))
    if args.dry_run:
        for c in candidates[:20]:
            logger.info(
                "  %s tier=%s composite=%s",
                c.graph_fingerprint[:12],
                c.tier,
                c.composite_score,
            )
        if len(candidates) > 20:
            logger.info("  ... %d more", len(candidates) - 20)
        return 0
    if not candidates:
        return 0

    cfg = ARCurriculumConfig(
        seed=int(args.seed),
        steps_per_stage=int(args.steps_per_stage),
        batch_size=int(args.batch_size),
        eval_batches=int(args.eval_batches),
        mode="cumulative",
    )

    wrote = 0
    failed = 0
    skipped = 0
    t_start = time.perf_counter()
    # LabNotebook(read_only=False) already holds aria-db's writer manager.
    # Adding a second flock from the same process deadlocks; rely on aria-db.
    for i, cand in enumerate(candidates, 1):
        t0 = time.perf_counter()
        try:
            if not cand.graph_json:
                skipped += 1
                continue
            model = _build_model(cand.graph_json, args.device, vocab_size)
            result = ar_curriculum_probe(model, cfg=cfg, device=args.device)
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.warning(
                "[%d/%d] %s FAILED: %s",
                i,
                len(candidates),
                cand.graph_fingerprint[:12],
                exc,
            )
            continue

        if result.status != "ok":
            failed += 1
            logger.warning(
                "[%d/%d] %s status=%s error=%s",
                i,
                len(candidates),
                cand.graph_fingerprint[:12],
                result.status,
                result.error,
            )
            continue

        updates = result.to_dict()
        store_probe_results(
            nb,
            cand.result_id,
            updates,
            write_leaderboard=False,
            provenance_context={
                "source": "ar_curriculum_backfill",
                "metric_version": AR_CURRICULUM_METRIC_VERSION,
                "run_id": run_id,
                "imported_at_unix": round(time.time(), 3),
            },
        )
        wrote += 1
        elapsed = time.perf_counter() - t0
        logger.info(
            "[%d/%d] %s tier=%s auc=%.3f s0=%.3f max_pass=%d wall=%.1fs",
            i,
            len(candidates),
            cand.graph_fingerprint[:12],
            cand.tier,
            result.auc_pair_final,
            result.s0_held_pair_acc,
            result.max_passing_stage,
            elapsed,
        )
        if i % 10 == 0:
            nb.conn.commit()
    nb.conn.commit()
    nb.close()

    total = time.perf_counter() - t_start
    logger.info(
        "Done. wrote=%d failed=%d skipped=%d total_wall=%.1fs avg=%.1fs/candidate",
        wrote,
        failed,
        skipped,
        total,
        total / max(wrote + failed, 1),
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
