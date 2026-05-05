"""Massive backfill: run the controlled-language probe ladder (S0.5/S1.0/Inv)
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

from research.eval.controlled_lang_probe import (
    CONTROLLED_LANG_METRIC_VERSION,
    controlled_lang_probe,
)
from research.eval.utils import micro_train_loop
from research.scientist.controlled_lang_gates import (
    CONTROLLED_LANG_SCORE_GATES,
    S05_NB_SCREENING_FAILURE_THRESHOLD,
    apply_controlled_lang_screening_failure,
    allows_controlled_lang_advanced_tiers,
    is_controlled_lang_screening_failure,
)
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
    "controlled_lang_s10_checkpoints_json": "TEXT",
    "controlled_lang_inv_checkpoints_json": "TEXT",
}


def _select_targets(
    db: Path,
    top_n: int,
    force: bool,
    required_tiers: tuple[str, ...],
    *,
    target_cohorts: tuple[str, ...] = (),
    missing_before_limit: bool = False,
) -> list[dict]:
    """Top-N leaderboard rows; skip rows that already have the required
    tier sa_scores populated (idempotent resume)."""
    conn = connect_readonly(db)
    try:
        where_extra = ""
        params: list[object] = []
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
                "pr.controlled_lang_s05_sa_score < "
                f"{S05_NB_SCREENING_FAILURE_THRESHOLD}"
            )
            missing_clauses.append(
                "pr.controlled_lang_s05_nb_order_acc < "
                f"{S05_NB_SCREENING_FAILURE_THRESHOLD}"
            )
            missing_clauses.append(
                "pr.controlled_lang_s05_nb_score < "
                f"{S05_NB_SCREENING_FAILURE_THRESHOLD}"
            )
            if "s05" in required_tiers:
                missing_clauses.append("pr.controlled_lang_s05_sa_score IS NULL")
            if "s10" in required_tiers:
                missing_clauses.append(
                    "pr.controlled_lang_s10_sa_score IS NULL "
                    "OR pr.controlled_lang_s10_sa_score < "
                    f"{S05_NB_SCREENING_FAILURE_THRESHOLD} "
                    "OR pr.controlled_lang_s10_nb_order_acc < "
                    f"{S05_NB_SCREENING_FAILURE_THRESHOLD} "
                    "OR pr.controlled_lang_s10_nb_score < "
                    f"{S05_NB_SCREENING_FAILURE_THRESHOLD} "
                    "OR pr.controlled_lang_s10_checkpoints_json IS NULL "
                    f"OR COALESCE(pr.controlled_lang_metric_version, '') != "
                    f"'{CONTROLLED_LANG_METRIC_VERSION}'"
                )
            if "inv" in required_tiers:
                missing_clauses.append(
                    "pr.controlled_lang_inv_sa_score IS NULL "
                    "OR pr.controlled_lang_inv_sa_score < "
                    f"{S05_NB_SCREENING_FAILURE_THRESHOLD} "
                    "OR pr.controlled_lang_inv_nb_order_acc < "
                    f"{S05_NB_SCREENING_FAILURE_THRESHOLD} "
                    "OR pr.controlled_lang_inv_nb_score < "
                    f"{S05_NB_SCREENING_FAILURE_THRESHOLD} "
                    "OR pr.controlled_lang_inv_checkpoints_json IS NULL "
                    f"OR COALESCE(pr.controlled_lang_metric_version, '') != "
                    f"'{CONTROLLED_LANG_METRIC_VERSION}'"
                )
            if missing_clauses:
                where_extra += " AND (" + " OR ".join(missing_clauses) + ")"
        params.append(top_n)
        rows = conn.execute(
            f"""
            SELECT l.entry_id, l.composite_score, l.tier, pr.result_id,
                   pr.graph_fingerprint, pr.graph_json,
                   pr.controlled_lang_metric_version AS metric_version,
                   pr.controlled_lang_s05_sa_score AS s05,
                   pr.controlled_lang_s10_sa_score AS s10,
                   pr.controlled_lang_inv_sa_score AS inv,
                   pr.controlled_lang_s05_nb_order_acc AS s05_nb_order,
                   pr.controlled_lang_s05_nb_score AS s05_nb,
                   pr.controlled_lang_s10_nb_order_acc AS s10_nb_order,
                   pr.controlled_lang_s10_nb_score AS s10_nb,
                   pr.controlled_lang_inv_nb_order_acc AS inv_nb_order,
                   pr.controlled_lang_inv_nb_score AS inv_nb,
                   pr.controlled_lang_s10_checkpoints_json AS s10_checkpoints_json,
                   pr.controlled_lang_inv_checkpoints_json AS inv_checkpoints_json,
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
    finally:
        conn.close()
    out = []
    skipped = 0
    skipped_s05_no_go = 0
    for r in rows:
        d = dict(r)
        advanced_requested = any(t in {"s10", "inv"} for t in required_tiers)
        existing_gate_label = _existing_gate_label(d, required_tiers, force=force)
        if existing_gate_label:
            d["_gate_only_label"] = existing_gate_label
            d["_s05_gate_only"] = existing_gate_label.startswith("controlled_lang_s05_")
            out.append(d)
            skipped_s05_no_go += 1
            continue
        stale_advanced_metric = advanced_requested and (
            d.get("metric_version") != CONTROLLED_LANG_METRIC_VERSION
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
                if not allows_controlled_lang_advanced_tiers(d.get("s05_nb")):
                    skipped_s05_no_go += 1
                    continue
        out.append(d)
    if skipped:
        logger.info("skipping %d rows already fully populated", skipped)
    if skipped_s05_no_go:
        logger.info(
            "found %d rows blocked by controlled-language score no-go",
            skipped_s05_no_go,
        )
    return out


def _existing_gate_label(
    row: dict,
    required_tiers: tuple[str, ...],
    *,
    force: bool,
) -> str | None:
    if force:
        return None
    tiers_to_check = ["s05"]
    tiers_to_check.extend(tier for tier in required_tiers if tier in {"s10", "inv"})
    seen: set[str] = set()
    for tier in tiers_to_check:
        if tier in seen:
            continue
        seen.add(tier)
        for gate in _gates_for_tier(tier):
            if is_controlled_lang_screening_failure(row.get(_row_alias_for_gate(gate))):
                return str(gate["label"])
    return None


def _gates_for_tier(tier: str) -> tuple[dict[str, str], ...]:
    prefix = f"controlled_lang_{tier}_"
    return tuple(
        gate
        for gate in CONTROLLED_LANG_SCORE_GATES
        if gate["score_key"].startswith(prefix)
    )


def _row_alias_for_gate(gate: dict[str, str]) -> str:
    key = str(gate["score_key"])
    if key.endswith("_sa_score"):
        return key.removeprefix("controlled_lang_").removesuffix("_sa_score")
    return (
        key.removeprefix("controlled_lang_").removesuffix("_score").removesuffix("_acc")
    )


def _gate_only_updates(fp: dict, gate_label: str) -> dict:
    for gate in CONTROLLED_LANG_SCORE_GATES:
        if gate["label"] == gate_label:
            return {str(gate["score_key"]): fp.get(_row_alias_for_gate(gate))}
    raise ValueError(f"unknown controlled-language gate: {gate_label}")


def _apply_first_controlled_lang_failure(
    con: sqlite3.Connection,
    *,
    result_id: str,
    updates: dict,
    source: str,
) -> str | None:
    for gate in CONTROLLED_LANG_SCORE_GATES:
        score = updates.get(gate["score_key"])
        if not is_controlled_lang_screening_failure(score):
            continue
        if apply_controlled_lang_screening_failure(
            con,
            result_id=result_id,
            gate=gate,
            score=score,
            source=source,
        ):
            return str(gate["failure_op"])
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

    out: dict = {"controlled_lang_metric_version": CONTROLLED_LANG_METRIC_VERSION}
    tiers_by_name = dict(TIERS)
    for tier_name in tier_names:
        cfg = tiers_by_name.get(tier_name)
        if cfg is None:
            logger.warning("  unknown tier %s; skipping", tier_name)
            continue
        try:
            res = controlled_lang_probe(
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
            out[f"controlled_lang_{tier_name}_sa_score"] = sa
            out[f"controlled_lang_{tier_name}_nb_order_acc"] = nb_order
            out[f"controlled_lang_{tier_name}_nb_score"] = nb_score
            checkpoints = payload.get("controlled_lang_checkpoints")
            if checkpoints and tier_name in {"s10", "inv"}:
                out[f"controlled_lang_{tier_name}_checkpoints_json"] = json.dumps(
                    checkpoints, sort_keys=True, separators=(",", ":")
                )
            failed_gate = _first_failed_gate_in_updates(out, tier_name)
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
) -> dict[str, str] | None:
    for gate in _gates_for_tier(tier_name):
        if is_controlled_lang_screening_failure(updates.get(gate["score_key"])):
            return gate
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
    ap.add_argument("--db", default="research/lab_notebook.db", type=Path)
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
        "--out",
        type=Path,
        default=Path(
            f"research/reports/controlled_lang_backfill_{int(time.time())}.jsonl"
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
            gate_only_label = fp.get("_gate_only_label")
            gate_only = bool(gate_only_label)
            updates = (
                _gate_only_updates(fp, str(gate_only_label))
                if gate_only
                else _run_one(fp, device=args.device, tier_names=tier_names)
            )
            elapsed = time.perf_counter() - t0
            if updates:
                if not gate_only:
                    _write_row(con, fp["result_id"], updates)
                gate_failure_op = _apply_first_controlled_lang_failure(
                    con,
                    result_id=fp["result_id"],
                    updates=updates,
                    source="controlled_lang_backfill",
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
                    "controlled_lang_gate_only_label": gate_only_label,
                    "s05_gate_only": str(gate_only_label or "").startswith(
                        "controlled_lang_s05_"
                    ),
                    "controlled_lang_screening_failure_op": gate_failure_op,
                    "controlled_lang_nb_screening_failure_op": gate_failure_op,
                    "s05_nb_screening_failure_applied": gate_failure_op
                    == "controlled_lang_s05_nb",
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
                    updates.get("controlled_lang_s05_sa_score"),
                    updates.get("controlled_lang_s05_nb_score"),
                    updates.get("controlled_lang_s10_sa_score"),
                    updates.get("controlled_lang_s10_nb_score"),
                    updates.get("controlled_lang_inv_sa_score"),
                    updates.get("controlled_lang_inv_nb_score"),
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
