"""Controlled-language cascade gates."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


CONTROLLED_LANG_SCREENING_FAILURE_THRESHOLD = 0.65
CONTROLLED_LANG_NB_SCREENING_FAILURE_THRESHOLD = (
    CONTROLLED_LANG_SCREENING_FAILURE_THRESHOLD
)
S05_NB_SCREENING_FAILURE_THRESHOLD = CONTROLLED_LANG_SCREENING_FAILURE_THRESHOLD
S05_NB_FAILURE_OP = "controlled_lang_s05_nb"

CONTROLLED_LANG_NB_GATES = {
    "s05": {
        "failure_op": S05_NB_FAILURE_OP,
        "stage": "s0.5",
        "score_key": "controlled_lang_s05_nb_score",
        "label": "controlled_lang_s05_nb",
    },
    "s10": {
        "failure_op": "controlled_lang_s10_nb",
        "stage": "s1.0",
        "score_key": "controlled_lang_s10_nb_score",
        "label": "controlled_lang_s10_nb",
    },
    "inv": {
        "failure_op": "controlled_lang_inv_nb",
        "stage": "investigation",
        "score_key": "controlled_lang_inv_nb_score",
        "label": "controlled_lang_inv_nb",
    },
}

CONTROLLED_LANG_SCORE_GATES = (
    {
        "failure_op": "controlled_lang_s05_sa",
        "stage": "s0.5",
        "score_key": "controlled_lang_s05_sa_score",
        "label": "controlled_lang_s05_sa",
    },
    {
        "failure_op": "controlled_lang_s05_nb_order",
        "stage": "s0.5",
        "score_key": "controlled_lang_s05_nb_order_acc",
        "label": "controlled_lang_s05_nb_order",
    },
    CONTROLLED_LANG_NB_GATES["s05"],
    {
        "failure_op": "controlled_lang_s10_sa",
        "stage": "s1.0",
        "score_key": "controlled_lang_s10_sa_score",
        "label": "controlled_lang_s10_sa",
    },
    {
        "failure_op": "controlled_lang_s10_nb_order",
        "stage": "s1.0",
        "score_key": "controlled_lang_s10_nb_order_acc",
        "label": "controlled_lang_s10_nb_order",
    },
    CONTROLLED_LANG_NB_GATES["s10"],
    {
        "failure_op": "controlled_lang_inv_sa",
        "stage": "investigation",
        "score_key": "controlled_lang_inv_sa_score",
        "label": "controlled_lang_inv_sa",
    },
    {
        "failure_op": "controlled_lang_inv_nb_order",
        "stage": "investigation",
        "score_key": "controlled_lang_inv_nb_order_acc",
        "label": "controlled_lang_inv_nb_order",
    },
    CONTROLLED_LANG_NB_GATES["inv"],
)


def is_controlled_lang_screening_failure(score: Any) -> bool:
    """Return true when a controlled-language accuracy score is a hard no-go."""
    if score is None:
        return False
    try:
        value = float(score)
    except (TypeError, ValueError):
        return False
    return value < CONTROLLED_LANG_SCREENING_FAILURE_THRESHOLD


def is_controlled_lang_nb_screening_failure(score: Any) -> bool:
    """Return true when a controlled-language NanoBind/NanoBLiMP score is a hard no-go."""
    return is_controlled_lang_screening_failure(score)


def is_s05_nb_screening_failure(score: Any) -> bool:
    """Return true when the S0.5 NanoBind/NanoBLiMP score is a hard no-go."""
    return is_controlled_lang_nb_screening_failure(score)


def allows_controlled_lang_advanced_tiers(score: Any) -> bool:
    """Return true when S1.0/INV controlled-language probes may run."""
    if score is None:
        return False
    return not is_s05_nb_screening_failure(score)


def controlled_lang_failure_details(
    gate: dict[str, str],
    score: Any,
    *,
    source: str,
) -> str:
    score_key = str(gate["score_key"])
    payload = {
        "failure_op": gate["failure_op"],
        "reason": f"{gate['label']}_below_threshold",
        "stage": gate["stage"],
        score_key: float(score),
        "threshold": CONTROLLED_LANG_SCREENING_FAILURE_THRESHOLD,
        "source": source,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def controlled_lang_nb_failure_details(
    tier: str,
    score: Any,
    *,
    source: str,
) -> str:
    return controlled_lang_failure_details(
        CONTROLLED_LANG_NB_GATES[tier],
        score,
        source=source,
    )


def s05_nb_failure_details(score: Any, *, source: str) -> str:
    return controlled_lang_nb_failure_details("s05", score, source=source)


def apply_controlled_lang_screening_failure(
    conn: sqlite3.Connection,
    *,
    result_id: str,
    gate: dict[str, str],
    score: Any,
    source: str,
) -> bool:
    """Downgrade a non-reference leaderboard row that fails a controlled-language gate."""
    if not is_controlled_lang_screening_failure(score):
        return False

    details_json = controlled_lang_failure_details(gate, score, source=source)
    cur = conn.execute(
        """
        UPDATE leaderboard
        SET tier = 'screened_out',
            validation_passed = 0,
            notes = TRIM(
                COALESCE(notes, '') ||
                CASE WHEN COALESCE(notes, '') = '' THEN '' ELSE ' | ' END ||
                ? ||
                ': score ' ||
                printf('%.4f', ?) ||
                ' below screening threshold ' ||
                printf('%.2f', ?)
            )
        WHERE result_id = ?
          AND COALESCE(is_reference, 0) = 0
          AND COALESCE(tier, '') NOT IN ('screened_out', 'retired')
        """,
        (
            gate["label"],
            float(score),
            CONTROLLED_LANG_SCREENING_FAILURE_THRESHOLD,
            result_id,
        ),
    )
    if cur.rowcount <= 0:
        return False

    conn.execute(
        """
        UPDATE program_results
        SET failure_op = ?,
            failure_details_json = ?
        WHERE result_id = ?
        """,
        (gate["failure_op"], details_json, result_id),
    )
    return True


def apply_controlled_lang_nb_screening_failure(
    conn: sqlite3.Connection,
    *,
    result_id: str,
    tier: str,
    score: Any,
    source: str,
) -> bool:
    """Downgrade a non-reference leaderboard row that fails a controlled-language NB gate."""
    if tier not in CONTROLLED_LANG_NB_GATES:
        raise ValueError(f"unknown controlled-language NB gate tier: {tier}")
    return apply_controlled_lang_screening_failure(
        conn,
        result_id=result_id,
        gate=CONTROLLED_LANG_NB_GATES[tier],
        score=score,
        source=source,
    )


def apply_s05_nb_screening_failure(
    conn: sqlite3.Connection,
    *,
    result_id: str,
    score: Any,
    source: str,
) -> bool:
    """Downgrade a non-reference leaderboard row that fails the S0.5 NB gate."""
    return apply_controlled_lang_nb_screening_failure(
        conn,
        result_id=result_id,
        tier="s05",
        score=score,
        source=source,
    )
