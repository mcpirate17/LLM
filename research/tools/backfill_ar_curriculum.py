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
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from research.eval.ar_curriculum_probe import (
    AR_CURRICULUM_METRIC_VERSION,
    ARCurriculumConfig,
    ARCurriculumResult,
    ar_curriculum_probe,
    required_vocab_size,
)
from research.scientist.notebook import LabNotebook
from research.scientist.native_runner import compile_model_native_first as compile_model
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
                # name/decl come from hardcoded AR_CURRICULUM_COLUMNS, not user input
                ddl = f"ALTER TABLE program_results ADD COLUMN {name} {decl}"  # nosemgrep: python-sql-string-formatting
                conn.execute(ddl)
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
    # LEFT JOIN so we can backfill program_results rows that have no leaderboard
    # entry (e.g., investigation experiments whose row never got promoted).
    sql = (
        "SELECT COALESCE(l.entry_id, '') AS entry_id, pr.result_id, "
        "COALESCE(l.tier, '') AS tier, "
        "COALESCE(l.composite_score, 0.0) AS composite_score, "
        "COALESCE(l.is_reference, 0) AS is_reference, "
        "COALESCE(l.model_source, '') AS model_source, "
        "pr.graph_json, pr.graph_fingerprint "
        "FROM program_results_compat pr "
        "LEFT JOIN leaderboard l ON l.result_id = pr.result_id "
        f"WHERE pr.result_id IN ({placeholders}) AND pr.graph_json IS NOT NULL {null_filter}"  # nosec B608 - '?' placeholders; null_filter is an internal constant
    )
    rows = nb.conn.execute(sql, tuple(result_ids)).fetchall()
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
    p.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds (e.g. '0,1,2'). When given, each candidate is "
        "probed once per seed and the MEDIAN-AUC seed's full result is persisted "
        "(curriculum + nb probes are very seed-sensitive — see "
        "feedback_probe_hierarchy). Overrides --seed. Default: single --seed.",
    )
    p.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Optional path to dump per-candidate per-seed AUCs + median for "
        "offline template ranking.",
    )
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
    p.add_argument(
        "--no-ar-gate",
        action="store_true",
        help="Disable the AR-gate pre-filter. By default a candidate flagged "
        "ar_gate_no_go=1 is skipped, and a candidate with no ar_gate_score is "
        "run through ar_gate(from_s1=False) first — only AR-gate passers proceed "
        "to the (expensive) AR-curriculum probe.",
    )
    return p.parse_args()


def _select_candidates(args: argparse.Namespace, nb: LabNotebook, null_col: str | None):
    tiers = tuple(t.strip() for t in str(args.tiers).split(",") if t.strip())
    shard = _parse_shard(args.shard)
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
    return candidates, tiers, shard


def _log_dry_run_candidates(candidates) -> None:
    for candidate in candidates[:20]:
        logger.info(
            "  %s tier=%s composite=%s",
            candidate.graph_fingerprint[:12],
            candidate.tier,
            candidate.composite_score,
        )
    if len(candidates) > 20:
        logger.info("  ... %d more", len(candidates) - 20)


def _ar_gate_allows(
    nb: LabNotebook, cand, *, device: str, run_id: str
) -> tuple[bool, str]:
    """Funnel gate: only AR-gate passers reach the AR-curriculum probe.

    Uses a cached ``ar_gate_no_go`` / ``ar_gate_score`` when present; otherwise
    runs ``ar_gate(from_s1=False)`` once and persists the verdict (same column
    set + no-go recipe as the runner, via ``eval.ar_gate``) so it is not
    recomputed. Transient (non-``ok``) AR-gate runs are not treated as a no-go.
    """
    from research.eval.ar_gate import (
        ARGateConfig,
        ar_gate,
        ar_gate_is_no_go,
        ar_gate_score,
    )

    row = nb.conn.execute(
        "SELECT ar_gate_no_go, ar_gate_score FROM program_results_compat "
        "WHERE result_id = ?",
        (cand.result_id,),
    ).fetchone()
    if row is not None:
        if row["ar_gate_no_go"] is not None and int(row["ar_gate_no_go"]) == 1:
            return False, "ar_gate_no_go_cached"
        if row["ar_gate_score"] is not None:
            return True, "ar_gate_pass_cached"

    res = ar_gate(
        graph_json=cand.graph_json, device=device, cfg=ARGateConfig(from_s1=False)
    )
    if res.status != "ok":
        return True, f"ar_gate_{res.status}_proceed"

    no_go = ar_gate_is_no_go(res)
    store_probe_results(
        nb,
        cand.result_id,
        {
            "ar_gate_metric_version": res.metric_version,
            "ar_gate_in_dist_pair_acc": res.in_dist_pair_acc,
            "ar_gate_in_dist_class_acc": res.in_dist_class_acc,
            "ar_gate_held_pair_acc": res.held_pair_acc,
            "ar_gate_held_class_acc": res.held_class_acc,
            "ar_gate_score": ar_gate_score(res),
            "ar_gate_status": res.status,
            "ar_gate_elapsed_ms": res.elapsed_ms,
            "ar_gate_train_steps_done": res.finetune_steps_done,
            "ar_gate_no_go": int(no_go),
        },
        write_leaderboard=False,
        provenance_context={
            "source": "ar_curriculum_backfill_ar_gate",
            "run_id": run_id,
            "imported_at_unix": round(time.time(), 3),
        },
    )
    return (not no_go), ("ar_gate_no_go_fresh" if no_go else "ar_gate_pass_fresh")


def _probe_seeds(
    model: torch.nn.Module, base_cfg: ARCurriculumConfig, seeds: list[int], device: str
):
    """Run the curriculum probe once per seed; return (median_result, ok_results).

    The probe deepcopies the model (``copy_model=True``) so weight init is held
    constant across seeds — only data sampling / minibatch order vary. The
    MEDIAN-AUC seed's full result is returned so the persisted per-stage arrays
    stay internally consistent (a real observed run, not a synthetic blend).
    """
    ok: list[ARCurriculumResult] = []
    for s in seeds:
        cfg = replace(base_cfg, seed=int(s))
        res = ar_curriculum_probe(model, cfg=cfg, device=device)
        if res.status == "ok":
            ok.append(res)
    if not ok:
        return None, ok
    ok_sorted = sorted(ok, key=lambda r: r.auc_pair_final)
    median = ok_sorted[(len(ok_sorted) - 1) // 2]  # lower-median element
    return median, ok


def _persist_median(
    nb: LabNotebook,
    cand,
    result: ARCurriculumResult,
    ok_results: list[ARCurriculumResult],
    seeds: list[int],
    run_id: str,
    results_sink: list | None,
) -> list[float]:
    """Persist the median-seed result + record per-seed AUCs. Returns per-seed list."""
    per_seed = [round(r.auc_pair_final, 4) for r in ok_results]
    store_probe_results(
        nb,
        cand.result_id,
        result.to_dict(),
        write_leaderboard=False,
        provenance_context={
            "source": "ar_curriculum_backfill",
            "metric_version": AR_CURRICULUM_METRIC_VERSION,
            "run_id": run_id,
            "n_seeds": len(ok_results),
            "seeds": list(seeds),
            "per_seed_auc_pair_final": per_seed,
            "imported_at_unix": round(time.time(), 3),
        },
    )
    if results_sink is not None:
        results_sink.append(
            {
                "result_id": cand.result_id,
                "fp": cand.graph_fingerprint[:12],
                "tier": cand.tier,
                "n_seeds": len(ok_results),
                "per_seed_auc_pair_final": per_seed,
                "median_auc_pair_final": round(result.auc_pair_final, 4),
                "median_max_passing_stage": result.max_passing_stage,
                "median_s0_held_pair_acc": round(result.s0_held_pair_acc, 4),
            }
        )
    return per_seed


def _run_backfill_candidates(
    *,
    args: argparse.Namespace,
    nb: LabNotebook,
    candidates,
    vocab_size: int,
    run_id: str,
    seeds: list[int],
    results_sink: list | None,
) -> tuple[int, int, int, float]:
    base_cfg = ARCurriculumConfig(
        steps_per_stage=int(args.steps_per_stage),
        batch_size=int(args.batch_size),
        eval_batches=int(args.eval_batches),
        mode="cumulative",
    )

    wrote = 0
    failed = 0
    skipped = 0
    t_start = time.perf_counter()
    for i, cand in enumerate(candidates, 1):
        t0 = time.perf_counter()
        try:
            if not cand.graph_json:
                skipped += 1
                continue
            if not args.no_ar_gate:
                allowed, reason = _ar_gate_allows(
                    nb, cand, device=args.device, run_id=run_id
                )
                if not allowed:
                    skipped += 1
                    logger.info(
                        "[%d/%d] %s SKIP (%s) — no AR-curriculum run",
                        i,
                        len(candidates),
                        cand.graph_fingerprint[:12],
                        reason,
                    )
                    continue
            model = _build_model(cand.graph_json, args.device, vocab_size)
            result, ok_results = _probe_seeds(model, base_cfg, seeds, args.device)
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

        if result is None:
            failed += 1
            logger.warning(
                "[%d/%d] %s all seeds non-ok",
                i,
                len(candidates),
                cand.graph_fingerprint[:12],
            )
            continue

        per_seed = _persist_median(
            nb, cand, result, ok_results, seeds, run_id, results_sink
        )
        wrote += 1
        elapsed = time.perf_counter() - t0
        logger.info(
            "[%d/%d] %s tier=%s median_auc=%.3f per_seed=%s s0=%.3f max_pass=%d wall=%.1fs",
            i,
            len(candidates),
            cand.graph_fingerprint[:12],
            cand.tier,
            result.auc_pair_final,
            per_seed,
            result.s0_held_pair_acc,
            result.max_passing_stage,
            elapsed,
        )
        if i % 10 == 0:
            nb.conn.commit()
    return wrote, failed, skipped, time.perf_counter() - t_start


def main() -> int:
    args = parse_args()
    if args.seeds:
        seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    else:
        seeds = [int(args.seed)]
    null_col = None if args.force else "ar_curriculum_auc_pair_final"
    vocab_size = max(required_vocab_size(), 2048)
    run_id = datetime.now(timezone.utc).strftime("ar_curriculum_backfill_%Y%m%d_%H%M%S")

    if not args.dry_run:
        ensure_ar_curriculum_columns(args.db)
    nb = LabNotebook(str(args.db), read_only=bool(args.dry_run))
    candidates, tiers, shard = _select_candidates(args, nb, null_col)
    logger.info(
        "%s tiers=%s top_per_tier=%d selected=%d shard=%s device=%s seeds=%s force=%s dry_run=%s",
        run_id,
        tiers,
        args.top_per_tier,
        len(candidates),
        shard,
        args.device,
        seeds,
        args.force,
        args.dry_run,
    )
    if args.dry_run:
        _log_dry_run_candidates(candidates)
        nb.close()
        return 0
    if not candidates:
        nb.close()
        return 0

    results_sink: list | None = [] if args.results_json else None
    wrote, failed, skipped, total = _run_backfill_candidates(
        args=args,
        nb=nb,
        candidates=candidates,
        vocab_size=vocab_size,
        run_id=run_id,
        seeds=seeds,
        results_sink=results_sink,
    )
    nb.conn.commit()
    nb.close()
    if results_sink is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(results_sink, indent=1))
        logger.info(
            "Wrote %d per-candidate records -> %s", len(results_sink), args.results_json
        )
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
