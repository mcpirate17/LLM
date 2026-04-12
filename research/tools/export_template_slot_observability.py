#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from research.scientist.notebook import LabNotebook

DEFAULT_DB = "research/lab_notebook.db"
DEFAULT_OUT_DIR = "research/reports/template_slot_observability"
DEFAULT_STRENGTH_REPORT = "research/reports/model_strength/model_strength_report.json"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_list(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return "" if value is None else str(value)


def _load_strength_maps(
    strength_report_path: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    path = Path(strength_report_path)
    if not path.exists():
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    template_rows = payload.get("rankings", {}).get("best_templates_overall", []) or []
    slot_rows = (
        payload.get("rankings", {}).get("best_slot_components_overall", []) or []
    )
    return (
        {str(row.get("name")): row for row in template_rows if row.get("name")},
        {str(row.get("name")): row for row in slot_rows if row.get("name")},
    )


def _template_alignment(
    row: dict[str, Any], strength_row: dict[str, Any] | None
) -> str:
    if not strength_row:
        return "none"
    tier = str(strength_row.get("confidence_tier") or "")
    effect = float(strength_row.get("adjusted_effect") or 0.0)
    if effect > 0 and tier in {"high", "medium"}:
        return "aligned_positive_signal"
    if effect < 0 and tier in {"high", "medium"}:
        return "aligned_negative_signal"
    return "watchlist_only"


def _slot_alignment(strength_row: dict[str, Any] | None) -> str:
    if not strength_row:
        return "none"
    effect = float(strength_row.get("adjusted_effect") or 0.0)
    if effect < 0:
        return "likely_negative_pattern"
    if effect > 0:
        return "aligned_positive_signal"
    return "watchlist_only"


def build_template_export_rows(
    observability: dict[str, Any],
    template_strength_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in observability.get("all_templates", []) or []:
        strength_row = template_strength_map.get(str(row.get("name")))
        out = dict(row)
        out["actions"] = _fmt_list(out.get("actions"))
        out["diagnosis"] = _fmt_list(out.get("diagnosis"))
        out["failure_reasons"] = _fmt_list(out.get("failure_reasons"))
        out["strength_alignment"] = _template_alignment(row, strength_row)
        out["strength_template_effect"] = (
            strength_row.get("adjusted_effect") if strength_row else None
        )
        out["strength_template_confidence_tier"] = (
            strength_row.get("confidence_tier") if strength_row else None
        )
        out["strength_template_support_graphs"] = (
            strength_row.get("support_graphs") if strength_row else None
        )
        out["strength_template_matched_controls"] = (
            strength_row.get("matched_template_controls") if strength_row else None
        )
        out["strength_template_artifact_flags"] = _fmt_list(
            strength_row.get("artifact_flags") if strength_row else None
        )
        rows.append(out)
    return rows


def build_slot_export_rows(
    observability: dict[str, Any],
    slot_strength_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in observability.get("all_slots", []) or []:
        slot_component_key = None
        if row.get("top_selected_motif"):
            slot_component_key = (
                f"{row.get('slot_key')}:{row.get('top_selected_motif')}"
            )
        strength_row = (
            slot_strength_map.get(str(slot_component_key))
            if slot_component_key
            else None
        )
        out = dict(row)
        out["slot_classes"] = _fmt_list(out.get("slot_classes"))
        out["strength_slot_component_key"] = slot_component_key
        out["strength_alignment"] = _slot_alignment(strength_row)
        out["strength_slot_effect"] = (
            strength_row.get("adjusted_effect") if strength_row else None
        )
        out["strength_slot_confidence_tier"] = (
            strength_row.get("confidence_tier") if strength_row else None
        )
        out["strength_slot_support_graphs"] = (
            strength_row.get("support_graphs") if strength_row else None
        )
        out["strength_slot_artifact_flags"] = _fmt_list(
            strength_row.get("artifact_flags") if strength_row else None
        )
        rows.append(out)
    return rows


def export_template_slot_observability(
    *,
    db_path: str | Path,
    out_dir: str | Path,
    strength_report_path: str | Path,
) -> dict[str, Path]:
    nb = LabNotebook(str(db_path))
    try:
        observability = nb.get_template_slot_observability(limit=8)
    finally:
        nb.close()
    template_strength_map, slot_strength_map = _load_strength_maps(strength_report_path)
    template_rows = build_template_export_rows(observability, template_strength_map)
    slot_rows = build_slot_export_rows(observability, slot_strength_map)
    out_base = Path(out_dir)
    template_path = out_base / "template_observability.csv"
    slot_path = out_base / "slot_observability.csv"
    _write_csv(template_path, template_rows)
    _write_csv(slot_path, slot_rows)
    return {"template_csv": template_path, "slot_csv": slot_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export template and slot observability to CSV"
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--strength-report", default=DEFAULT_STRENGTH_REPORT)
    args = parser.parse_args()
    paths = export_template_slot_observability(
        db_path=args.db,
        out_dir=args.out_dir,
        strength_report_path=args.strength_report,
    )
    print(paths["template_csv"])
    print(paths["slot_csv"])


if __name__ == "__main__":
    main()
