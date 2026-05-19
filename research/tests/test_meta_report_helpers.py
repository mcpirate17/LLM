from __future__ import annotations

import csv

from research.tools.meta_report_helpers import markdown_table, write_csv


def test_write_csv_preserves_first_row_field_order(tmp_path):
    path = tmp_path / "rows.csv"

    write_csv(path, [{"b": 2, "a": 1}, {"b": 4, "a": 3}])

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"b": "2", "a": "1"}, {"b": "4", "a": "3"}]
    assert path.read_text().splitlines()[0] == "b,a"


def test_write_csv_empty_rows_writes_empty_file(tmp_path):
    path = tmp_path / "empty.csv"

    write_csv(path, [])

    assert path.read_text() == ""


def test_markdown_table_limits_rows_and_fills_missing_fields():
    lines = markdown_table(
        [{"name": "a", "score": 1}, {"name": "b"}],
        ["name", "score"],
        limit=1,
    )

    assert lines == [
        "| name | score |",
        "| --- | --- |",
        "| a | 1 |",
        "",
    ]


def test_markdown_table_empty_rows():
    assert markdown_table([], ["name"], limit=5) == ["_No rows._", ""]
