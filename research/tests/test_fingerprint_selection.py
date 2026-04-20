from __future__ import annotations

from research.tools._fingerprint_selection import dedupe_records_by_fingerprint


def test_dedupe_records_by_fingerprint_preserves_first_sorted_row() -> None:
    rows = [
        {"result_id": "latest_fp1", "graph_fingerprint": "fp1"},
        {"result_id": "older_fp1", "graph_fingerprint": "fp1"},
        {"result_id": "only_fp2", "graph_fingerprint": "fp2"},
    ]

    deduped = dedupe_records_by_fingerprint(rows)

    assert [row["result_id"] for row in deduped] == ["latest_fp1", "only_fp2"]


def test_dedupe_records_by_fingerprint_can_also_dedupe_result_ids() -> None:
    rows = [
        {"result_id": "r1", "graph_fingerprint": "fp1"},
        {"result_id": "r2", "graph_fingerprint": "fp1"},
        {"result_id": "r1", "graph_fingerprint": "fp2"},
        {"result_id": "r3", "graph_fingerprint": ""},
        {"result_id": "r4", "graph_fingerprint": ""},
    ]

    deduped = dedupe_records_by_fingerprint(rows, result_id_key="result_id")

    assert [row["result_id"] for row in deduped] == ["r1", "r3", "r4"]
