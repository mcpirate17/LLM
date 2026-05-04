"""Phase 1.1 — Slot inventory.

Builds research/reports/slot_inventory.json from:
  1. The TEMPLATES registry in research/synthesis/templates.py (175 names).
  2. Empirical slot usage in program_graph_features.slot_usage_json.

Cohort filter: rows with controlled_lang_s05_sa_score, non-reference. Includes
screened_out tier — we want the template signature, not just non-failed graphs.

Note: notebook_observability.py also aggregates slot_usage but bound to
LabNotebook's stage1_passed signal and row-iteration loop. Not reusable here
(different cohort filter, different pass signal, different output shape).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.synthesis.templates import (  # noqa: E402
    DEFAULT_TEMPLATE_WEIGHTS,
    TEMPLATES,
)

DB = f"file:{REPO / 'research/lab_notebook.db'}?mode=ro&immutable=0"
OUT = REPO / "research/reports/slot_inventory.json"

SlotKey = tuple[str, int]


class _Accumulator:
    """Per-slot accumulators keyed by (template_name, slot_index)."""

    __slots__ = (
        "classes",
        "candidates",
        "motifs",
        "motif_classes",
        "n",
        "row_count_per_template",
        "templates_observed",
    )

    def __init__(self) -> None:
        self.classes: dict[SlotKey, set[str]] = defaultdict(set)
        self.candidates: dict[SlotKey, Counter[int]] = defaultdict(Counter)
        self.motifs: dict[SlotKey, Counter[str]] = defaultdict(Counter)
        self.motif_classes: dict[SlotKey, Counter[str]] = defaultdict(Counter)
        self.n: Counter[SlotKey] = Counter()
        self.row_count_per_template: Counter[str] = Counter()
        self.templates_observed: set[str] = set()


def load_slot_rows() -> list[tuple[str, str]]:
    """Return (row_template_name, slot_usage_json) over the cohort."""
    conn = sqlite3.connect(DB, uri=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pgf.template_name, pgf.slot_usage_json
        FROM program_graph_features pgf
        JOIN program_results pr ON pr.result_id = pgf.result_id
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE pr.controlled_lang_s05_sa_score IS NOT NULL
          AND COALESCE(l.is_reference, 0) = 0
          AND pgf.slot_usage_json IS NOT NULL
          AND pgf.slot_usage_json NOT IN ('', '[]', 'null', '{}')
        """
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def _ingest_entry(entry: dict, row_template: str, acc: _Accumulator) -> None:
    tpl = entry.get("template_name") or row_template
    if not tpl:
        return
    slot_idx = entry.get("slot_index")
    if slot_idx is None:
        return
    key: SlotKey = (tpl, int(slot_idx))
    acc.templates_observed.add(tpl)
    classes = entry.get("slot_classes") or []
    if isinstance(classes, list):
        acc.classes[key].update(str(c) for c in classes)
    cand = entry.get("candidate_count")
    if isinstance(cand, int):
        acc.candidates[key][cand] += 1
    motif = entry.get("selected_motif")
    if motif:
        acc.motifs[key][str(motif)] += 1
    motif_class = entry.get("selected_motif_class")
    if motif_class:
        acc.motif_classes[key][str(motif_class)] += 1
    acc.n[key] += 1


def aggregate(rows: list[tuple[str, str]]) -> _Accumulator:
    acc = _Accumulator()
    for row_template, raw in rows:
        try:
            entries = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(entries, list):
            continue
        if row_template:
            acc.row_count_per_template[row_template] += 1
        for entry in entries:
            if isinstance(entry, dict):
                _ingest_entry(entry, row_template, acc)
    return acc


def build_template_record(tpl: str, acc: _Accumulator) -> dict[str, Any]:
    slot_keys = sorted(k for k in acc.n if k[0] == tpl)
    slots = []
    for key in slot_keys:
        cand_mode = (
            acc.candidates[key].most_common(1)[0][0] if acc.candidates[key] else None
        )
        slots.append(
            {
                "slot_index": key[1],
                "slot_classes": sorted(acc.classes[key]),
                "n_candidates_typical": cand_mode,
                "n_observed": acc.n[key],
                "motifs_observed": dict(acc.motifs[key].most_common()),
                "motif_classes_observed": dict(acc.motif_classes[key].most_common()),
                "is_required": True,
            }
        )
    return {
        "template_name": tpl,
        "default_weight": DEFAULT_TEMPLATE_WEIGHTS.get(tpl),
        "in_static_registry": tpl in TEMPLATES,
        "n_observed_rows": acc.row_count_per_template.get(tpl, 0),
        "is_slot_opaque": not slots,
        "slots": slots,
    }


def build_inventory(acc: _Accumulator) -> dict[str, dict[str, Any]]:
    names = sorted(set(TEMPLATES.keys()) | acc.templates_observed)
    return {tpl: build_template_record(tpl, acc) for tpl in names}


def write_summary(out: dict[str, dict[str, Any]]) -> None:
    n_static = sum(1 for v in out.values() if v["in_static_registry"])
    n_opaque = sum(
        1 for v in out.values() if v["is_slot_opaque"] and v["in_static_registry"]
    )
    n_extra = sum(1 for v in out.values() if not v["in_static_registry"])
    print(f"Wrote {OUT}", file=sys.stderr)
    print(f"  Templates total: {len(out)}", file=sys.stderr)
    print(
        f"  In static TEMPLATES registry: {n_static} (opaque: {n_opaque})",
        file=sys.stderr,
    )
    print(f"  Observed in DB but NOT in registry: {n_extra}", file=sys.stderr)
    if n_extra:
        extras = sorted([n for n, v in out.items() if not v["in_static_registry"]])
        print(f"    sample: {extras[:10]}", file=sys.stderr)


def main() -> None:
    rows = load_slot_rows()
    print(f"Loaded {len(rows)} graph rows with non-empty slot_usage", file=sys.stderr)
    acc = aggregate(rows)
    out = build_inventory(acc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    write_summary(out)


if __name__ == "__main__":
    main()
