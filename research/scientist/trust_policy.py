"""Shared trust-tier policy for candidate-facing views and promotion gates."""

from __future__ import annotations

from typing import Any, Mapping

TRUSTED_TRUST_LABELS = (
    "candidate_screening",
    "candidate_grade",
    "reference",
)

TRUSTED_COMPARABILITY_LABELS = (
    "screening_only",
    "candidate_comparable",
    "reference_comparable",
)

PROMOTABLE_TRUST_LABELS = (
    "candidate_grade",
    "reference",
)

PROMOTABLE_COMPARABILITY_LABELS = (
    "candidate_comparable",
    "reference_comparable",
)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def is_trusted_entry(entry: Mapping[str, Any] | None) -> bool:
    if not entry:
        return False
    return (
        _norm(entry.get("trust_label")) in TRUSTED_TRUST_LABELS
        and _norm(entry.get("comparability_label")) in TRUSTED_COMPARABILITY_LABELS
    )


def is_promotable_entry(entry: Mapping[str, Any] | None) -> bool:
    if not entry:
        return False
    return (
        _norm(entry.get("trust_label")) in PROMOTABLE_TRUST_LABELS
        and _norm(entry.get("comparability_label")) in PROMOTABLE_COMPARABILITY_LABELS
    )


def sql_trusted_clause(*, table_alias: str | None = None) -> str:
    prefix = f"{table_alias}." if table_alias else ""
    trust_values = ", ".join(f"'{value}'" for value in TRUSTED_TRUST_LABELS)
    comparability_values = ", ".join(
        f"'{value}'" for value in TRUSTED_COMPARABILITY_LABELS
    )
    return (
        f"COALESCE({prefix}trust_label, '') IN ({trust_values})"
        f" AND COALESCE({prefix}comparability_label, '') IN ({comparability_values})"
    )
