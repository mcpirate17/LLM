"""Massive backfill: run the language-control probe ladder (S0.5/S1.0/Inv)
on leaderboard rows that lack data, then write 9 columns + version per row.

Order: top-N by composite_score descending. Resumable — skips rows that
already have all three tier sa_scores populated.

Cost per fingerprint: ~5s base train + 3 × ~5s probe = ~20s.
Top-200 ≈ 67 min wall.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path

import torch

from research.eval.language_control_probe import (
    LANGUAGE_CONTROL_METRIC_VERSION,
    language_control_probe,
)
from research.eval.utils import micro_train_loop
from research.scientist.language_control_gates import (
    LANGUAGE_CONTROL_NB_GATES,
    S05_SA_FAILURE_OP,
    S05_SA_SCREENING_FAILURE_THRESHOLD,
    S05_NB_SCREENING_FAILURE_THRESHOLD,
    S10_NB_SA_FAILURE_OP,
    S10_NB_SA_NB_SCREENING_FAILURE_THRESHOLD,
    S10_NB_SA_SA_SCREENING_FAILURE_THRESHOLD,
    apply_language_control_nb_screening_failure,
    apply_s05_sa_screening_failure,
    apply_s10_nb_sa_screening_failure,
    allows_language_control_advanced_tiers,
    is_language_control_nb_screening_failure,
    is_s05_sa_screening_failure,
    is_s10_nb_sa_screening_failure,
)
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_from_json
from research.tools._db_maintenance import connect_readonly

logger = logging.getLogger(__name__)

VOCAB_SIZE = 50257
BASE_TRAIN_STEPS = 750  # match screening default

# Tier configs — must match what _V14_CONFIG anchors were calibrated for.
TIERS = (
    ("s05", {"active_vocab_size": 120, "n_train_steps": 40}),
    (
        "s10",
        {
            "active_vocab_size": 240,
            "n_train_steps": 2000,
            "checkpoints": (500, 1000, 2000),
            "timeout_s": 240.0,
        },
    ),
    (
        "inv",
        {
            "active_vocab_size": 360,
            "n_train_steps": 2000,
            "checkpoints": (500, 1000, 2000),
            "timeout_s": 300.0,
        },
    ),
)

_CHECKPOINT_COLUMNS = {
    "language_control_s10_checkpoints_json": "TEXT",
    "language_control_investigation_checkpoints_json": "TEXT",
}


def _select_targets(
    db: Path,
    top_n: int,
    force: bool,
    required_tiers: tuple[str, ...],
    *,
    target_cohorts: tuple[str, ...] = (),
    missing_before_limit: bool = False,
    require_s05_nb_pass: bool = False,
) -> list[dict]:
    """Top-N leaderboard rows; skip rows that already have the required
    tier sa_scores populated (idempotent resume)."""
    conn = connect_readonly(db)
    try:
        where_extra = ""
        params: list[object] = []
        if require_s05_nb_pass:
            where_extra += (
                f" AND pr.language_control_s05_binding_score >= "
                f"{S05_NB_SCREENING_FAILURE_THRESHOLD}"
            )
        if target_cohorts:
            cohort_clauses = []
            for cohort in target_cohorts:
                if cohort == "reference":
                    cohort_clauses.append("COALESCE(l.is_reference, 0) = 1")
                elif cohort == "validation_pending":
                    cohort_clauses.append(
                        "(COALESCE(l.is_reference, 0) = 0 AND l.tier = 'validation' "
                        "AND COALESCE(l.validation_passed, 0) = 0)"
                    )
                elif cohort == "validation":
                    cohort_clauses.append(
                        "(COALESCE(l.is_reference, 0) = 0 AND l.tier = 'validation' "
                        "AND COALESCE(l.validation_passed, 0) = 1)"
                    )
                elif cohort in {"screening", "investigation", "investigation_failed"}:
                    cohort_clauses.append(
                        f"(COALESCE(l.is_reference, 0) = 0 AND l.tier = '{cohort}')"
                    )
                elif cohort == "breakthrough":
                    cohort_clauses.append(
                        "(COALESCE(l.is_reference, 0) = 0 AND l.tier = 'breakthrough')"
                    )
                else:
                    raise ValueError(f"unknown target cohort: {cohort}")
            where_extra = " AND (" + " OR ".join(cohort_clauses) + ")"
        if missing_before_limit and not force:
            missing_clauses = []
            missing_clauses.append(
                "pr.language_control_s05_binding_score < "
                f"{S05_NB_SCREENING_FAILURE_THRESHOLD}"
            )
            missing_clauses.append(
                "("
                "pr.language_control_s05_sentence_assoc_score < "
                f"{S05_SA_SCREENING_FAILURE_THRESHOLD} "
                "AND NOT ("
                "COALESCE(pr.fp_jacobian_erf_density, -1.0) >= 0.0625 "
                "AND COALESCE(pr.fp_jacobian_erf_decay_slope, 1.0) <= -0.103282"
                ") "
                "AND COALESCE(pr.graph_category_histogram, '') NOT LIKE '%\"mixing\"%'"
                ")"
            )
            if "s05" in required_tiers:
                missing_clauses.append(
                    "pr.language_control_s05_sentence_assoc_score IS NULL"
                )
            if "s10" in required_tiers:
                missing_clauses.append(
                    "pr.language_control_s10_sentence_assoc_score IS NULL "
                    "OR pr.language_control_s10_binding_score < "
                    f"{S05_NB_SCREENING_FAILURE_THRESHOLD} "
                    "OR ("
                    "pr.language_control_s10_binding_score < "
                    f"{S10_NB_SA_NB_SCREENING_FAILURE_THRESHOLD} "
                    "AND pr.language_control_s10_sentence_assoc_score < "
                    f"{S10_NB_SA_SA_SCREENING_FAILURE_THRESHOLD}"
                    ") "
                    "OR pr.language_control_s10_checkpoints_json IS NULL "
                    f"OR COALESCE(pr.language_control_metric_version, '') != "
                    f"'{LANGUAGE_CONTROL_METRIC_VERSION}'"
                )
            if "inv" in required_tiers:
                missing_clauses.append(
                    "pr.language_control_investigation_sentence_assoc_score IS NULL "
                    "OR pr.language_control_investigation_binding_score < "
                    f"{S05_NB_SCREENING_FAILURE_THRESHOLD} "
                    "OR pr.language_control_investigation_checkpoints_json IS NULL "
                    f"OR COALESCE(pr.language_control_metric_version, '') != "
                    f"'{LANGUAGE_CONTROL_METRIC_VERSION}'"
                )
            if missing_clauses:
                where_extra += " AND (" + " OR ".join(missing_clauses) + ")"
        params.append(top_n)
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT l.entry_id, l.composite_score, l.tier, pr.result_id,
                       pr.graph_fingerprint, pr.graph_json,
                       pr.language_control_metric_version AS metric_version,
                       pr.language_control_s05_sentence_assoc_score AS s05,
                       pr.language_control_s10_sentence_assoc_score AS s10,
                       pr.language_control_investigation_sentence_assoc_score AS inv,
                       pr.language_control_s05_binding_order_acc AS s05_nb_order,
                       pr.language_control_s05_binding_score AS s05_nb,
                       pr.language_control_s10_binding_order_acc AS s10_nb_order,
                       pr.language_control_s10_binding_score AS s10_nb,
                       pr.language_control_investigation_binding_order_acc AS inv_nb_order,
                       pr.language_control_investigation_binding_score AS inv_nb,
                       pr.language_control_s10_checkpoints_json AS s10_checkpoints_json,
                       pr.language_control_investigation_checkpoints_json AS inv_checkpoints_json,
                       pr.fp_jacobian_erf_density AS erf_density,
                       pr.fp_jacobian_erf_decay_slope AS erf_decay_slope,
                       pr.graph_category_histogram AS graph_category_histogram,
                       pgf.template_name
                FROM leaderboard l
                JOIN program_results pr ON pr.result_id=l.result_id
                LEFT JOIN program_graph_features pgf ON pgf.result_id=l.result_id
                WHERE l.composite_score IS NOT NULL
                  AND COALESCE(l.tier, '') NOT IN ('screened_out', 'retired')
                  AND pr.graph_json IS NOT NULL AND pr.graph_json != '{{}}'
                  {where_extra}
                ORDER BY l.composite_score DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        ]
        for row in rows:
            row["graph_json"] = resolve_graph_json_value(
                conn,
                db,
                row.get("graph_json"),
            )
    finally:
        conn.close()
    out = []
    skipped = 0
    skipped_s05_no_go = 0
    for d in rows:
        advanced_requested = any(t in {"s10", "inv"} for t in required_tiers)
        existing_gate_tier = _existing_gate_tier(d, required_tiers, force=force)
        if existing_gate_tier:
            d["_gate_only_tier"] = existing_gate_tier
            d["_s05_gate_only"] = existing_gate_tier.startswith("s05_")
            out.append(d)
            skipped_s05_no_go += 1
            continue
        stale_advanced_metric = advanced_requested and (
            d.get("metric_version") != LANGUAGE_CONTROL_METRIC_VERSION
            or ("s10" in required_tiers and not d.get("s10_checkpoints_json"))
            or ("inv" in required_tiers and not d.get("inv_checkpoints_json"))
        )
        if (
            not force
            and not stale_advanced_metric
            and all(d.get(t) is not None for t in required_tiers)
        ):
            skipped += 1
            continue
        s05_requested = "s05" in required_tiers
        if advanced_requested:
            if not s05_requested:
                if not allows_language_control_advanced_tiers(
                    d.get("s05_nb"),
                    sa_score=d.get("s05"),
                    erf_density=d.get("erf_density"),
                    erf_decay_slope=d.get("erf_decay_slope"),
                    graph_category_histogram=d.get("graph_category_histogram"),
                ):
                    skipped_s05_no_go += 1
                    continue
        out.append(d)
    if skipped:
        logger.info("skipping %d rows already fully populated", skipped)
    if skipped_s05_no_go:
        logger.info(
            "found %d rows blocked by language-control no-go",
            skipped_s05_no_go,
        )
    return out


def _existing_gate_tier(
    row: dict,
    required_tiers: tuple[str, ...],
    *,
    force: bool,
) -> str | None:
    if force:
        return None
    if is_language_control_nb_screening_failure(row.get("s05_nb")):
        return "s05_nb"
    if is_s05_sa_screening_failure(
        row.get("s05"),
        erf_density=row.get("erf_density"),
        erf_decay_slope=row.get("erf_decay_slope"),
        graph_category_histogram=row.get("graph_category_histogram"),
    ):
        return "s05_sa"
    for tier in required_tiers:
        if tier == "s10":
            if is_language_control_nb_screening_failure(row.get("s10_nb")):
                return "s10_nb"
            if is_s10_nb_sa_screening_failure(
                nb_score=row.get("s10_nb"),
                sa_score=row.get("s10"),
            ):
                return "s10_nb_sa"
        elif tier == "inv" and is_language_control_nb_screening_failure(
            row.get("inv_nb")
        ):
            return "inv_nb"
    return None


def _gate_only_updates(fp: dict, gate_label: str) -> dict:
    if gate_label == "s05_sa":
        return {
            "language_control_s05_sentence_assoc_score": fp.get("s05"),
        }
    if gate_label == "s10_nb_sa":
        return {
            "language_control_s10_sentence_assoc_score": fp.get("s10"),
            "language_control_s10_binding_score": fp.get("s10_nb"),
        }
    tier = gate_label.removesuffix("_nb")
    key = str(LANGUAGE_CONTROL_NB_GATES[tier]["score_key"])
    return {key: fp.get(f"{tier}_nb")}


def _apply_first_language_control_failure(
    con: sqlite3.Connection,
    *,
    result_id: str,
    updates: dict,
    context: dict | None = None,
    source: str,
) -> str | None:
    context = context or {}
    s05_nb_gate = LANGUAGE_CONTROL_NB_GATES["s05"]
    s05_nb_score = updates.get(s05_nb_gate["score_key"])
    if is_language_control_nb_screening_failure(s05_nb_score):
        if apply_language_control_nb_screening_failure(
            con,
            result_id=result_id,
            tier="s05",
            score=s05_nb_score,
            source=source,
        ):
            return str(s05_nb_gate["failure_op"])
        return None
    s05_sa_score = updates.get("language_control_s05_sentence_assoc_score")
    if apply_s05_sa_screening_failure(
        con,
        result_id=result_id,
        score=s05_sa_score,
        erf_density=context.get("erf_density"),
        erf_decay_slope=context.get("erf_decay_slope"),
        graph_category_histogram=context.get("graph_category_histogram"),
        source=source,
    ):
        return S05_SA_FAILURE_OP
    s10_gate = LANGUAGE_CONTROL_NB_GATES["s10"]
    s10_nb_score = updates.get(s10_gate["score_key"])
    if is_language_control_nb_screening_failure(s10_nb_score):
        if apply_language_control_nb_screening_failure(
            con,
            result_id=result_id,
            tier="s10",
            score=s10_nb_score,
            source=source,
        ):
            return str(s10_gate["failure_op"])
        return None
    if apply_s10_nb_sa_screening_failure(
        con,
        result_id=result_id,
        nb_score=s10_nb_score,
        sa_score=updates.get("language_control_s10_sentence_assoc_score"),
        source=source,
    ):
        return S10_NB_SA_FAILURE_OP
    inv_gate = LANGUAGE_CONTROL_NB_GATES["inv"]
    inv_nb_score = updates.get(inv_gate["score_key"])
    if is_language_control_nb_screening_failure(inv_nb_score):
        if apply_language_control_nb_screening_failure(
            con,
            result_id=result_id,
            tier="inv",
            score=inv_nb_score,
            source=source,
        ):
            return str(inv_gate["failure_op"])
        return None
    return None


def _train_base(graph_json_str: str, *, device: str) -> torch.nn.Module:
    graph = graph_from_json(graph_json_str)
    model = compile_model([graph]).to(device)
    batches = [torch.randint(0, VOCAB_SIZE, (4, 128), device=device) for _ in range(8)]
    micro_train_loop(
        model, batches, vocab_size=VOCAB_SIZE, n_steps=BASE_TRAIN_STEPS, lr=3e-4
    )
    return model


def _run_one(fp: dict, *, device: str, tier_names: tuple[str, ...]) -> dict | None:
    """Train base, run requested tiers. Returns dict of column → value to write."""
    try:
        model = _train_base(fp["graph_json"], device=device)
    except Exception as exc:  # noqa: BLE001
        logger.error("  %s base train failed: %s", fp["entry_id"], exc)
        return None

    out: dict = {"language_control_metric_version": LANGUAGE_CONTROL_METRIC_VERSION}
    tiers_by_name = dict(TIERS)
    for tier_name in tier_names:
        cfg = tiers_by_name.get(tier_name)
        if cfg is None:
            logger.warning("  unknown tier %s; skipping", tier_name)
            continue
        try:
            res = language_control_probe(
                model,
                active_vocab_size=cfg["active_vocab_size"],
                n_train_steps=cfg["n_train_steps"],
                checkpoint_steps=cfg.get("checkpoints"),
                timeout_s=float(cfg.get("timeout_s", 60.0)),
                device=device,
                preserve_state=len(tier_names) > 1,
            )
            payload = res.to_dict()
            sa = (res.synthetic_association or {}).get("synthetic_association_score")
            nb_order = (res.nano_blimp or {}).get("nano_blimp_order_grammaticality_acc")
            nb_score = (res.nano_blimp or {}).get("nano_blimp_score")
            out[f"language_control_{tier_name}_sa_score"] = sa
            out[f"language_control_{tier_name}_nb_order_acc"] = nb_order
            out[f"language_control_{tier_name}_nb_score"] = nb_score
            checkpoints = payload.get("language_control_checkpoints")
            if checkpoints and tier_name in {"s10", "inv"}:
                out[f"language_control_{tier_name}_checkpoints_json"] = json.dumps(
                    checkpoints, sort_keys=True, separators=(",", ":")
                )
            failed_gate = _first_failed_gate_in_updates(out, tier_name, context=fp)
            if failed_gate is not None:
                logger.info(
                    "  %s %s %.4f below no-go threshold; skipping later tiers",
                    fp["entry_id"],
                    failed_gate["label"],
                    float(out[failed_gate["score_key"]]),
                )
                break
        except Exception as exc:  # noqa: BLE001
            logger.warning("  %s tier %s failed: %s", fp["entry_id"], tier_name, exc)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _first_failed_gate_in_updates(
    updates: dict,
    tier_name: str,
    *,
    context: dict | None = None,
) -> dict[str, str] | None:
    context = context or {}
    gate = LANGUAGE_CONTROL_NB_GATES.get(tier_name)
    if gate and is_language_control_nb_screening_failure(
        updates.get(gate["score_key"])
    ):
        return gate
    if tier_name == "s10" and is_s10_nb_sa_screening_failure(
        nb_score=updates.get("language_control_s10_binding_score"),
        sa_score=updates.get("language_control_s10_sentence_assoc_score"),
    ):
        return {
            "failure_op": S10_NB_SA_FAILURE_OP,
            "score_key": "language_control_s10_binding_score",
            "label": "language_control_s10_nb_sa",
        }
    if tier_name == "s05" and is_s05_sa_screening_failure(
        updates.get("language_control_s05_sentence_assoc_score"),
        erf_density=context.get("erf_density"),
        erf_decay_slope=context.get("erf_decay_slope"),
        graph_category_histogram=context.get("graph_category_histogram"),
    ):
        return {
            "failure_op": S05_SA_FAILURE_OP,
            "score_key": "language_control_s05_sentence_assoc_score",
            "label": "language_control_s05_sa",
        }
    return None


def _ensure_backfill_columns(con: sqlite3.Connection) -> None:
    existing = {
        row[1] for row in con.execute("PRAGMA table_info(program_results)").fetchall()
    }
    for col_name, col_type in _CHECKPOINT_COLUMNS.items():
        if col_name not in existing:
            con.execute(f"ALTER TABLE program_results ADD COLUMN {col_name} {col_type}")


def _write_row(con: sqlite3.Connection, result_id: str, updates: dict) -> int:
    set_clauses = []
    vals = []
    for k, v in updates.items():
        set_clauses.append(f"{k}=?")
        vals.append(v)
    if not set_clauses:
        return 0
    vals.append(result_id)
    con.execute(
        f"UPDATE program_results SET {', '.join(set_clauses)} WHERE result_id=?",
        vals,
    )
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="research/runs.db", type=Path)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="re-probe even if data exists")
    ap.add_argument(
        "--tiers",
        default="s05,s10,inv",
        help="comma-separated tier names (s05/s10/inv); defaults to all three",
    )
    ap.add_argument(
        "--target-cohorts",
        default="",
        help=(
            "optional comma-separated cohort filter: reference, screening, "
            "investigation, investigation_failed, validation_pending, "
            "validation, breakthrough"
        ),
    )
    ap.add_argument(
        "--missing-before-limit",
        action="store_true",
        help=(
            "apply the required-tier missing/stale filter in SQL before LIMIT; "
            "use this for full next-N chunks when earlier ranked rows are done"
        ),
    )
    ap.add_argument(
        "--require-s05-nb-pass",
        action="store_true",
        help=(
            "only select rows whose language_control_s05_binding_score is at or above "
            "the s05 nano-bind screening pass threshold "
            f"({S05_NB_SCREENING_FAILURE_THRESHOLD}); use when promoting screening-tier "
            "rows to s10 so we don't burn GPU on graphs that already failed s05 nano-bind"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            f"research/reports/language_control_backfill_{int(time.time())}.jsonl"
        ),
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    requested_tiers = {t.strip() for t in args.tiers.split(",") if t.strip()}
    tier_names = tuple(t for t, _cfg in TIERS if t in requested_tiers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    target_cohorts = tuple(
        c.strip() for c in args.target_cohorts.split(",") if c.strip()
    )
    targets = _select_targets(
        args.db,
        args.top_n,
        args.force,
        tier_names,
        target_cohorts=target_cohorts,
        missing_before_limit=bool(args.missing_before_limit),
        require_s05_nb_pass=bool(args.require_s05_nb_pass),
    )
    logger.info(
        "selected %d targets (top-%d, tiers=%s, %s)",
        len(targets),
        args.top_n,
        ",".join(tier_names),
        args.device,
    )

    con = sqlite3.connect(str(args.db), timeout=30.0)
    con.execute("PRAGMA journal_mode=DELETE")
    con.execute("PRAGMA busy_timeout=15000")
    _ensure_backfill_columns(con)

    t_start = time.perf_counter()
    written = 0
    failed = 0
    with args.out.open("w") as out_fh:
        for idx, fp in enumerate(targets, 1):
            ent = fp["entry_id"]
            t0 = time.perf_counter()
            gate_only_tier = fp.get("_gate_only_tier")
            gate_only = bool(gate_only_tier)
            updates = (
                _gate_only_updates(fp, str(gate_only_tier))
                if gate_only
                else _run_one(fp, device=args.device, tier_names=tier_names)
            )
            elapsed = time.perf_counter() - t0
            if updates:
                if not gate_only:
                    _write_row(con, fp["result_id"], updates)
                gate_failure_op = _apply_first_language_control_failure(
                    con,
                    result_id=fp["result_id"],
                    updates=updates,
                    context=fp,
                    source="language_control_backfill",
                )
                con.commit()
                written += 1
                row = {
                    "entry_id": ent,
                    "result_id": fp["result_id"],
                    "fingerprint": fp.get("graph_fingerprint"),
                    "template": fp.get("template_name"),
                    "composite": fp.get("composite_score"),
                    "elapsed_s": round(elapsed, 1),
                    "language_control_gate_only_label": gate_only_tier,
                    "s05_gate_only": str(gate_only_tier or "").startswith("s05_"),
                    "language_control_screening_failure_op": gate_failure_op,
                    "language_control_nb_screening_failure_op": (
                        gate_failure_op
                        if str(gate_failure_op or "").endswith("_nb")
                        else None
                    ),
                    "s05_nb_screening_failure_applied": gate_failure_op
                    == "language_control_s05_nb",
                    "s05_sa_screening_failure_applied": gate_failure_op
                    == S05_SA_FAILURE_OP,
                    "s10_nb_sa_screening_failure_applied": gate_failure_op
                    == S10_NB_SA_FAILURE_OP,
                    **updates,
                }
                out_fh.write(json.dumps(row) + "\n")
                out_fh.flush()
                logger.info(
                    (
                        "[%d/%d] %s: s05_sa=%s s05_nb=%s "
                        "s10_sa=%s s10_nb=%s inv_sa=%s inv_nb=%s "
                        "gate=%s (%.1fs)"
                    ),
                    idx,
                    len(targets),
                    ent,
                    updates.get("language_control_s05_sentence_assoc_score"),
                    updates.get("language_control_s05_binding_score"),
                    updates.get("language_control_s10_sentence_assoc_score"),
                    updates.get("language_control_s10_binding_score"),
                    updates.get("language_control_investigation_sentence_assoc_score"),
                    updates.get("language_control_investigation_binding_score"),
                    gate_failure_op or "ok",
                    elapsed,
                )
            else:
                failed += 1

    con.close()
    total = time.perf_counter() - t_start
    logger.info(
        "backfill done: %d written, %d failed in %.1fs (%.1fmin)",
        written,
        failed,
        total,
        total / 60,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
