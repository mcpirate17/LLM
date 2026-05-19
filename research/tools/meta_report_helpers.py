"""Shared output helpers for read-only meta-analysis tools."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(
    rows: list[dict[str, Any]], fields: list[str], *, limit: int
) -> list[str]:
    if not rows:
        return ["_No rows._", ""]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows[:limit]:
        lines.append(
            "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        )
    lines.append("")
    return lines
