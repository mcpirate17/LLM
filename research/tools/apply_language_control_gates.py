"""Apply language-control cascade gates to already-backfilled leaderboard rows."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from research.defaults import RUNS_DB
from research.scientist.language_control_gates import (
    LANGUAGE_CONTROL_NB_GATES,
    S05_SA_SCREENING_FAILURE_THRESHOLD,
    S05_NB_SCREENING_FAILURE_THRESHOLD,
    S10_NB_SA_NB_SCREENING_FAILURE_THRESHOLD,
    S10_NB_SA_SA_SCREENING_FAILURE_THRESHOLD,
    apply_language_control_nb_screening_failure,
    apply_s05_sa_screening_failure,
    apply_s10_nb_sa_screening_failure,
    language_control_gate_manual_override,
    is_s05_sa_screening_failure,
    is_s10_nb_sa_screening_failure,
)
from research.tools.db_health import backup_sqlite_db


def _candidate_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """
        SELECT l.entry_id,
               l.result_id,
               l.tier,
               pr.graph_fingerprint,
               pr.language_control_s05_sentence_assoc_score,
               pr.language_control_s05_binding_score,
               pr.language_control_s10_sentence_assoc_score,
               pr.language_control_s10_binding_score,
               pr.language_control_investigation_binding_score,
               pr.fp_jacobian_erf_density,
               pr.fp_jacobian_erf_decay_slope,
               pr.graph_category_histogram
        FROM leaderboard l
        JOIN program_results_compat pr ON pr.result_id = l.result_id
        WHERE (
              pr.language_control_s05_binding_score < ?
           OR pr.language_control_s10_binding_score < ?
           OR pr.language_control_investigation_binding_score < ?
           OR (
                 pr.language_control_s10_binding_score < ?
             AND pr.language_control_s10_sentence_assoc_score < ?
           )
           OR (
                 pr.language_control_s05_sentence_assoc_score < ?
             AND NOT (
                    COALESCE(pr.fp_jacobian_erf_density, -1.0) >= 0.0625
                AND COALESCE(pr.fp_jacobian_erf_decay_slope, 1.0) <= -0.103282
             )
             AND COALESCE(pr.graph_category_histogram, '') NOT LIKE '%"mixing"%'
           )
        )
          AND COALESCE(l.is_reference, 0) = 0
          AND COALESCE(l.tier, '') NOT IN ('screened_out', 'retired')
        ORDER BY l.tier, l.composite_score DESC
        """,
        (
            S05_NB_SCREENING_FAILURE_THRESHOLD,
            S05_NB_SCREENING_FAILURE_THRESHOLD,
            S05_NB_SCREENING_FAILURE_THRESHOLD,
            S10_NB_SA_NB_SCREENING_FAILURE_THRESHOLD,
            S10_NB_SA_SA_SCREENING_FAILURE_THRESHOLD,
            S05_SA_SCREENING_FAILURE_THRESHOLD,
        ),
    ).fetchall()


def _first_failed_tier(row: sqlite3.Row) -> str | None:
    gate = LANGUAGE_CONTROL_NB_GATES["s05"]
    score = row[gate["score_key"]]
    if score is not None and float(score) < S05_NB_SCREENING_FAILURE_THRESHOLD:
        return "s05_nb"
    if is_s05_sa_screening_failure(
        row["language_control_s05_sentence_assoc_score"],
        erf_density=row["fp_jacobian_erf_density"],
        erf_decay_slope=row["fp_jacobian_erf_decay_slope"],
        graph_category_histogram=row["graph_category_histogram"],
    ):
        return "s05_sa"
    gate = LANGUAGE_CONTROL_NB_GATES["s10"]
    score = row[gate["score_key"]]
    if score is not None and float(score) < S05_NB_SCREENING_FAILURE_THRESHOLD:
        return "s10_nb"
    if is_s10_nb_sa_screening_failure(
        nb_score=row["language_control_s10_binding_score"],
        sa_score=row["language_control_s10_sentence_assoc_score"],
    ):
        return "s10_nb_sa"
    gate = LANGUAGE_CONTROL_NB_GATES["inv"]
    score = row[gate["score_key"]]
    if score is not None and float(score) < S05_NB_SCREENING_FAILURE_THRESHOLD:
        return "inv_nb"
    return None


def _failure_op_for_label(failure_label: str) -> str:
    if failure_label == "s05_sa":
        return "language_control_s05_sa"
    if failure_label == "s10_nb_sa":
        return "language_control_s10_nb_sa"
    tier = failure_label.removesuffix("_nb")
    return str(LANGUAGE_CONTROL_NB_GATES[tier]["failure_op"])


def _classify_row(row: sqlite3.Row) -> tuple[str | None, dict | None]:
    failure_label = _first_failed_tier(row)
    if failure_label is None:
        return None, None
    override = language_control_gate_manual_override(
        entry_id=row["entry_id"],
        result_id=row["result_id"],
        failure_op=_failure_op_for_label(failure_label),
    )
    return failure_label, override


def _manual_override_for_row(row: sqlite3.Row) -> dict | None:
    _failure_label, override = _classify_row(row)
    return override


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(RUNS_DB))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a SQLite backup before applying changes.",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(str(args.db), timeout=30.0)
    candidate_rows = _candidate_rows(conn)
    classified = [(row, *_classify_row(row)) for row in candidate_rows]
    manual_override_rows = [
        (row, label, override)
        for row, label, override in classified
        if override is not None
    ]
    rows = [
        (row, label)
        for row, label, override in classified
        if override is None and label is not None
    ]
    by_tier: dict[str, int] = {}
    for row, _ in rows:
        by_tier[str(row["tier"] or "")] = by_tier.get(str(row["tier"] or ""), 0) + 1

    print(
        f"candidates={len(rows)} nb_threshold={S05_NB_SCREENING_FAILURE_THRESHOLD:.2f} "
        f"s05_sa_threshold={S05_SA_SCREENING_FAILURE_THRESHOLD:.2f} "
        f"s10_nb_sa_thresholds={S10_NB_SA_NB_SCREENING_FAILURE_THRESHOLD:.2f}/"
        f"{S10_NB_SA_SA_SCREENING_FAILURE_THRESHOLD:.2f} "
        f"manual_overrides={len(manual_override_rows)}"
    )
    for tier, count in sorted(by_tier.items(), key=lambda item: (-item[1], item[0])):
        print(f"{tier or 'unknown'}={count}")
    for row, failure_label, override in manual_override_rows:
        override = override or {}
        print(
            "manual_override "
            f"entry_id={row['entry_id']} result_id={row['result_id']} "
            f"failure={_failure_op_for_label(failure_label) if failure_label else 'unknown'} "
            f"reason={override.get('reason', '')}"
        )

    if args.dry_run or not rows:
        conn.close()
        return 0

    if not args.no_backup:
        backup_path = backup_sqlite_db(args.db, suffix="pre_language_control_gates")
        print(f"backup={backup_path}")

    applied = 0
    for row, failure_label in rows:
        if failure_label == "s05_sa":
            if apply_s05_sa_screening_failure(
                conn,
                result_id=str(row["result_id"]),
                score=row["language_control_s05_sentence_assoc_score"],
                erf_density=row["fp_jacobian_erf_density"],
                erf_decay_slope=row["fp_jacobian_erf_decay_slope"],
                graph_category_histogram=row["graph_category_histogram"],
                source="apply_language_control_gates",
            ):
                applied += 1
            continue
        if failure_label == "s10_nb_sa":
            if apply_s10_nb_sa_screening_failure(
                conn,
                result_id=str(row["result_id"]),
                nb_score=row["language_control_s10_binding_score"],
                sa_score=row["language_control_s10_sentence_assoc_score"],
                source="apply_language_control_gates",
            ):
                applied += 1
            continue
        tier = failure_label.removesuffix("_nb")
        gate = LANGUAGE_CONTROL_NB_GATES[tier]
        if apply_language_control_nb_screening_failure(
            conn,
            result_id=str(row["result_id"]),
            tier=tier,
            score=row[gate["score_key"]],
            source="apply_language_control_gates",
        ):
            applied += 1
    conn.commit()
    conn.close()
    print(f"applied={applied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
