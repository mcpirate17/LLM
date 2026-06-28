"""Language-control cascade gates."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .shared_utils import coerce_finite_float


LANGUAGE_CONTROL_NB_SCREENING_FAILURE_THRESHOLD = 0.65
S05_NB_SCREENING_FAILURE_THRESHOLD = LANGUAGE_CONTROL_NB_SCREENING_FAILURE_THRESHOLD
S05_NB_FAILURE_OP = "language_control_s05_nb"
S05_SA_SCREENING_FAILURE_THRESHOLD = 0.65
S05_SA_ERF_DENSITY_ESCAPE_THRESHOLD = 0.0625
S05_SA_ERF_DECAY_ESCAPE_THRESHOLD = -0.103282
S05_SA_FAILURE_OP = "language_control_s05_sa"
S10_NB_SA_NB_SCREENING_FAILURE_THRESHOLD = 0.80
S10_NB_SA_SA_SCREENING_FAILURE_THRESHOLD = 0.65
S10_NB_SA_FAILURE_OP = "language_control_s10_nb_sa"

LANGUAGE_CONTROL_GATE_MANUAL_OVERRIDES = {
    "1b70ab74-c98": {
        "entry_id": "c03cdadf-af7",
        "failure_ops": (S05_SA_FAILURE_OP,),
        "reason": "manual_pass_harder_ar_gate_good",
        "reviewer": "tim",
        "reviewed_on": "2026-05-06",
    },
}

LANGUAGE_CONTROL_NB_GATES = {
    "s05": {
        "failure_op": S05_NB_FAILURE_OP,
        "stage": "s0.5",
        "score_key": "language_control_s05_binding_score",
        "label": "language_control_s05_nb",
    },
    "s10": {
        "failure_op": "language_control_s10_nb",
        "stage": "s1.0",
        "score_key": "language_control_s10_binding_score",
        "label": "language_control_s10_nb",
    },
    "inv": {
        "failure_op": "language_control_investigation_binding",
        "stage": "investigation",
        "score_key": "language_control_investigation_binding_score",
        "label": "language_control_investigation_binding",
    },
}


def language_control_gate_manual_override(
    *,
    result_id: Any,
    failure_op: str,
    entry_id: Any = None,
) -> dict[str, Any] | None:
    """Return manual override metadata for a specific language-control gate."""
    if result_id is None:
        return None
    payload = LANGUAGE_CONTROL_GATE_MANUAL_OVERRIDES.get(str(result_id))
    if payload is None:
        return None
    expected_entry_id = payload.get("entry_id")
    if (
        entry_id is not None
        and expected_entry_id
        and str(entry_id) != expected_entry_id
    ):
        return None
    failure_ops = tuple(str(op) for op in payload.get("failure_ops", ()))
    if str(failure_op) not in failure_ops:
        return None
    return dict(payload)


def is_language_control_nb_screening_failure(score: Any) -> bool:
    """Return true when a language-control NanoBind/NanoBLiMP score is a hard no-go."""
    if score is None:
        return False
    try:
        value = float(score)
    except (TypeError, ValueError):
        return False
    return value < LANGUAGE_CONTROL_NB_SCREENING_FAILURE_THRESHOLD


def is_s05_nb_screening_failure(score: Any) -> bool:
    """Return true when the S0.5 NanoBind/NanoBLiMP score is a hard no-go."""
    return is_language_control_nb_screening_failure(score)


def graph_category_has_mixing(category_histogram: Any) -> bool:
    """Return true when graph category metadata includes a mixing op."""
    if category_histogram is None:
        return False
    if isinstance(category_histogram, dict):
        return bool(category_histogram.get("mixing", 0))
    if not isinstance(category_histogram, str):
        return False
    try:
        parsed = json.loads(category_histogram)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and bool(parsed.get("mixing", 0))


def has_s05_sa_escape(
    *,
    erf_density: Any,
    erf_decay_slope: Any,
    graph_category_histogram: Any,
) -> bool:
    """Return true when a low S0.5 SA row has a rescue signal."""
    density = coerce_finite_float(erf_density)
    decay = coerce_finite_float(erf_decay_slope)
    has_erf_pair = (
        density is not None
        and density >= S05_SA_ERF_DENSITY_ESCAPE_THRESHOLD
        and decay is not None
        and decay <= S05_SA_ERF_DECAY_ESCAPE_THRESHOLD
    )
    return has_erf_pair or graph_category_has_mixing(graph_category_histogram)


def is_s05_sa_screening_failure(
    score: Any,
    *,
    erf_density: Any,
    erf_decay_slope: Any,
    graph_category_histogram: Any,
) -> bool:
    """Return true when S0.5 exact-answer accuracy is too low without escape."""
    value = coerce_finite_float(score)
    if value is None or value >= S05_SA_SCREENING_FAILURE_THRESHOLD:
        return False
    return not has_s05_sa_escape(
        erf_density=erf_density,
        erf_decay_slope=erf_decay_slope,
        graph_category_histogram=graph_category_histogram,
    )


def is_s10_nb_sa_screening_failure(*, nb_score: Any, sa_score: Any) -> bool:
    """Return true when S1.0 NB is weak and S1.0 exact-answer accuracy is also weak."""
    nb_value = coerce_finite_float(nb_score)
    sa_value = coerce_finite_float(sa_score)
    if nb_value is None or sa_value is None:
        return False
    return (
        nb_value < S10_NB_SA_NB_SCREENING_FAILURE_THRESHOLD
        and sa_value < S10_NB_SA_SA_SCREENING_FAILURE_THRESHOLD
    )


def allows_language_control_advanced_tiers(
    nb_score: Any,
    *,
    sa_score: Any = None,
    erf_density: Any = None,
    erf_decay_slope: Any = None,
    graph_category_histogram: Any = None,
) -> bool:
    """Return true when S1.0/INV language-control probes may run."""
    if nb_score is None:
        return False
    if is_s05_nb_screening_failure(nb_score):
        return False
    return not is_s05_sa_screening_failure(
        sa_score,
        erf_density=erf_density,
        erf_decay_slope=erf_decay_slope,
        graph_category_histogram=graph_category_histogram,
    )


def language_control_nb_failure_details(
    tier: str,
    score: Any,
    *,
    source: str,
) -> str:
    gate = LANGUAGE_CONTROL_NB_GATES[tier]
    score_key = str(gate["score_key"])
    payload = {
        "failure_op": gate["failure_op"],
        "reason": f"{gate['label']}_below_threshold",
        "stage": gate["stage"],
        score_key: float(score),
        "threshold": LANGUAGE_CONTROL_NB_SCREENING_FAILURE_THRESHOLD,
        "source": source,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def s05_sa_failure_details(
    score: Any,
    *,
    erf_density: Any,
    erf_decay_slope: Any,
    graph_category_histogram: Any,
    source: str,
) -> str:
    payload = {
        "failure_op": S05_SA_FAILURE_OP,
        "reason": "language_control_s05_sa_below_threshold_without_escape",
        "stage": "s0.5",
        "language_control_s05_sentence_assoc_score": float(score),
        "threshold": S05_SA_SCREENING_FAILURE_THRESHOLD,
        "erf_density": coerce_finite_float(erf_density),
        "erf_density_escape_threshold": S05_SA_ERF_DENSITY_ESCAPE_THRESHOLD,
        "erf_decay_slope": coerce_finite_float(erf_decay_slope),
        "erf_decay_escape_threshold": S05_SA_ERF_DECAY_ESCAPE_THRESHOLD,
        "graph_has_mixing": graph_category_has_mixing(graph_category_histogram),
        "source": source,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def s10_nb_sa_failure_details(
    *,
    nb_score: Any,
    sa_score: Any,
    source: str,
) -> str:
    payload = {
        "failure_op": S10_NB_SA_FAILURE_OP,
        "reason": "language_control_s10_nb_sa_below_threshold",
        "stage": "s1.0",
        "language_control_s10_binding_score": float(nb_score),
        "language_control_s10_nb_threshold": S10_NB_SA_NB_SCREENING_FAILURE_THRESHOLD,
        "language_control_s10_sentence_assoc_score": float(sa_score),
        "language_control_s10_sa_threshold": S10_NB_SA_SA_SCREENING_FAILURE_THRESHOLD,
        "source": source,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def apply_language_control_nb_screening_failure(
    conn: sqlite3.Connection,
    *,
    result_id: str,
    tier: str,
    score: Any,
    source: str,
) -> bool:
    """Downgrade a non-reference leaderboard row that fails a language-control NB gate."""
    if tier not in LANGUAGE_CONTROL_NB_GATES:
        raise ValueError(f"unknown language-control NB gate tier: {tier}")
    if not is_language_control_nb_screening_failure(score):
        return False

    gate = LANGUAGE_CONTROL_NB_GATES[tier]
    if language_control_gate_manual_override(
        result_id=result_id,
        failure_op=str(gate["failure_op"]),
    ):
        return False
    details_json = language_control_nb_failure_details(tier, score, source=source)
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
            LANGUAGE_CONTROL_NB_SCREENING_FAILURE_THRESHOLD,
            result_id,
        ),
    )
    if cur.rowcount <= 0:
        return False

    conn.execute(
        """
        UPDATE graph_runs
        SET failure_op = ?,
            failure_details_json = ?
        WHERE result_id = ?
        """,
        (gate["failure_op"], details_json, result_id),
    )
    return True


def apply_s10_nb_sa_screening_failure(
    conn: sqlite3.Connection,
    *,
    result_id: str,
    nb_score: Any,
    sa_score: Any,
    source: str,
) -> bool:
    """Downgrade a non-reference row that fails the combined S1.0 NB/SA gate."""
    if not is_s10_nb_sa_screening_failure(nb_score=nb_score, sa_score=sa_score):
        return False
    if language_control_gate_manual_override(
        result_id=result_id,
        failure_op=S10_NB_SA_FAILURE_OP,
    ):
        return False

    details_json = s10_nb_sa_failure_details(
        nb_score=nb_score,
        sa_score=sa_score,
        source=source,
    )
    cur = conn.execute(
        """
        UPDATE leaderboard
        SET tier = 'screened_out',
            validation_passed = 0,
            notes = TRIM(
                COALESCE(notes, '') ||
                CASE WHEN COALESCE(notes, '') = '' THEN '' ELSE ' | ' END ||
                'language_control_s10_nb_sa: nb ' ||
                printf('%.4f', ?) ||
                ' below ' ||
                printf('%.2f', ?) ||
                ' and sa ' ||
                printf('%.4f', ?) ||
                ' below ' ||
                printf('%.2f', ?)
            )
        WHERE result_id = ?
          AND COALESCE(is_reference, 0) = 0
          AND COALESCE(tier, '') NOT IN ('screened_out', 'retired')
        """,
        (
            float(nb_score),
            S10_NB_SA_NB_SCREENING_FAILURE_THRESHOLD,
            float(sa_score),
            S10_NB_SA_SA_SCREENING_FAILURE_THRESHOLD,
            result_id,
        ),
    )
    if cur.rowcount <= 0:
        return False

    conn.execute(
        """
        UPDATE graph_runs
        SET failure_op = ?,
            failure_details_json = ?
        WHERE result_id = ?
        """,
        (S10_NB_SA_FAILURE_OP, details_json, result_id),
    )
    return True


def apply_s05_sa_screening_failure(
    conn: sqlite3.Connection,
    *,
    result_id: str,
    score: Any,
    erf_density: Any,
    erf_decay_slope: Any,
    graph_category_histogram: Any,
    source: str,
) -> bool:
    """Downgrade a non-reference leaderboard row that fails the S0.5 SA gate."""
    if not is_s05_sa_screening_failure(
        score,
        erf_density=erf_density,
        erf_decay_slope=erf_decay_slope,
        graph_category_histogram=graph_category_histogram,
    ):
        return False
    if language_control_gate_manual_override(
        result_id=result_id,
        failure_op=S05_SA_FAILURE_OP,
    ):
        return False

    details_json = s05_sa_failure_details(
        score,
        erf_density=erf_density,
        erf_decay_slope=erf_decay_slope,
        graph_category_histogram=graph_category_histogram,
        source=source,
    )
    cur = conn.execute(
        """
        UPDATE leaderboard
        SET tier = 'screened_out',
            validation_passed = 0,
            notes = TRIM(
                COALESCE(notes, '') ||
                CASE WHEN COALESCE(notes, '') = '' THEN '' ELSE ' | ' END ||
                'language_control_s05_sa: score ' ||
                printf('%.4f', ?) ||
                ' below screening threshold ' ||
                printf('%.2f', ?) ||
                ' without ERF/mixing escape'
            )
        WHERE result_id = ?
          AND COALESCE(is_reference, 0) = 0
          AND COALESCE(tier, '') NOT IN ('screened_out', 'retired')
        """,
        (float(score), S05_SA_SCREENING_FAILURE_THRESHOLD, result_id),
    )
    if cur.rowcount <= 0:
        return False

    conn.execute(
        """
        UPDATE graph_runs
        SET failure_op = ?,
            failure_details_json = ?
        WHERE result_id = ?
        """,
        (S05_SA_FAILURE_OP, details_json, result_id),
    )
    return True


def apply_s05_nb_screening_failure(
    conn: sqlite3.Connection,
    *,
    result_id: str,
    score: Any,
    source: str,
) -> bool:
    """Downgrade a non-reference leaderboard row that fails the S0.5 NB gate."""
    return apply_language_control_nb_screening_failure(
        conn,
        result_id=result_id,
        tier="s05",
        score=score,
        source=source,
    )
