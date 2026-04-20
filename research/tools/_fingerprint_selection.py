from __future__ import annotations

from typing import Any, Iterable, List, TypeVar

T = TypeVar("T")


def _text_field(record: Any, key: str) -> str:
    value = record[key]
    return str(value or "").strip()


def dedupe_records_by_fingerprint(
    records: Iterable[T],
    *,
    fingerprint_key: str = "graph_fingerprint",
    result_id_key: str | None = None,
) -> List[T]:
    """Keep the first record for each fingerprint after caller-defined ordering.

    The caller owns row ordering. This helper only removes repeated
    ``graph_fingerprint`` values, and optionally repeated ``result_id`` values,
    while preserving blank fingerprints as distinct rows.
    """

    deduped: List[T] = []
    seen_fingerprints: set[str] = set()
    seen_result_ids: set[str] = set()

    for record in records:
        if result_id_key is not None:
            result_id = _text_field(record, result_id_key)
            if not result_id or result_id in seen_result_ids:
                continue
            seen_result_ids.add(result_id)

        fingerprint = _text_field(record, fingerprint_key)
        if fingerprint and fingerprint in seen_fingerprints:
            continue
        if fingerprint:
            seen_fingerprints.add(fingerprint)
        deduped.append(record)

    return deduped
