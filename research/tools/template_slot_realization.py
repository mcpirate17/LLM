"""Phase 1.2 — Slot realization stats.

Per (template_name, slot_index, motif): n, pass_rate, fail_rate, mean_sa,
mean_order_acc, Wilson 95% CI on pass_rate.

Pass cohort: controlled_lang_s05_sa_score >= 0.95
             AND COALESCE(failure_op,'') != 'nano_bind'
Fail cohort: controlled_lang_s05_sa_score <  0.30
             OR  failure_op = 'nano_bind'

Outputs:
  research/reports/slot_realization.csv      — typed-slot templates
  research/reports/template_overall.csv      — every template (incl. opaque)
  research/reports/slot_opaque.txt           — list of opaque template names
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
DB = f"file:{REPO / 'research/lab_notebook.db'}?mode=ro&immutable=0"
REPORTS = REPO / "research/reports"
INVENTORY = REPORTS / "slot_inventory.json"

PASS_SA = 0.95
FAIL_SA = 0.30
NANO_BIND = "nano_bind"
MIN_N_PUBLISH = 20
WILSON_Z = 1.96  # 95% CI


def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    # Mirrors GrammarAnalytics._wilson_interval (analytics_grammar.py:276); kept
    # inline here to avoid pulling the analytics package into a one-shot tool.
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def is_pass(sa: float | None, failure_op: str | None) -> bool:
    if sa is None:
        return False
    return sa >= PASS_SA and (failure_op or "") != NANO_BIND


def is_fail(sa: float | None, failure_op: str | None) -> bool:
    if (failure_op or "") == NANO_BIND:
        return True
    return sa is not None and sa < FAIL_SA


def fetch_cohort() -> list[dict]:
    """Cohort: rows with sa_score, non-reference. Includes screened_out."""
    conn = sqlite3.connect(DB, uri=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pr.result_id,
               pr.controlled_lang_s05_sa_score AS sa,
               pr.controlled_lang_s05_nb_order_acc AS order_acc,
               pr.failure_op AS failure_op,
               pgf.template_name AS row_template,
               pgf.slot_usage_json AS slot_usage_json
        FROM program_results pr
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        LEFT JOIN program_graph_features pgf ON pgf.result_id = pr.result_id
        WHERE pr.controlled_lang_s05_sa_score IS NOT NULL
          AND COALESCE(l.is_reference, 0) = 0
        """
    )
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


class SlotAcc:
    __slots__ = (
        "n",
        "n_pass",
        "n_fail",
        "sa_sum",
        "order_sum",
        "order_n",
        "motif_class",
    )

    def __init__(self) -> None:
        self.n = 0
        self.n_pass = 0
        self.n_fail = 0
        self.sa_sum = 0.0
        self.order_sum = 0.0
        self.order_n = 0
        self.motif_class: str | None = None

    def add(
        self,
        sa: float,
        order_acc: float | None,
        passed: bool,
        failed: bool,
        motif_class: str | None,
    ) -> None:
        self.n += 1
        if passed:
            self.n_pass += 1
        if failed:
            self.n_fail += 1
        self.sa_sum += sa
        if order_acc is not None:
            self.order_sum += order_acc
            self.order_n += 1
        if motif_class and self.motif_class is None:
            self.motif_class = motif_class

    def to_record(self) -> dict[str, float | int | str | None]:
        ci_lo, ci_hi = wilson(self.n_pass, self.n)
        return {
            "n": self.n,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "pass_rate": self.n_pass / self.n if self.n else 0.0,
            "fail_rate": self.n_fail / self.n if self.n else 0.0,
            "mean_sa": self.sa_sum / self.n if self.n else 0.0,
            "mean_order_acc": (self.order_sum / self.order_n) if self.order_n else None,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        }


SlotKey = tuple[str, int, str]  # (template, slot_index, motif)


def _ingest_slot_entries(
    row: dict, slot_acc: dict[SlotKey, SlotAcc], passed: bool, failed: bool
) -> None:
    raw = row["slot_usage_json"]
    if not raw or raw in ("", "[]", "null", "{}"):
        return
    try:
        entries = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return
    if not isinstance(entries, list):
        return
    sa = float(row["sa"])
    order = row["order_acc"]
    order_acc = float(order) if order is not None else None
    seen: set[SlotKey] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tpl = entry.get("template_name") or row["row_template"]
        slot_idx = entry.get("slot_index")
        motif = entry.get("selected_motif")
        motif_class = entry.get("selected_motif_class")
        if not tpl or slot_idx is None or not motif:
            continue
        key: SlotKey = (str(tpl), int(slot_idx), str(motif))
        if key in seen:
            continue
        seen.add(key)
        slot_acc.setdefault(key, SlotAcc()).add(
            sa, order_acc, passed, failed, motif_class
        )


def aggregate(
    rows: Iterable[dict],
) -> tuple[dict[SlotKey, SlotAcc], dict[str, SlotAcc]]:
    slot_acc: dict[SlotKey, SlotAcc] = {}
    template_acc: dict[str, SlotAcc] = {}
    for row in rows:
        sa = row["sa"]
        if sa is None:
            continue
        passed = is_pass(sa, row["failure_op"])
        failed = is_fail(sa, row["failure_op"])
        order_acc = float(row["order_acc"]) if row["order_acc"] is not None else None
        tpl = row["row_template"]
        if tpl:
            template_acc.setdefault(str(tpl), SlotAcc()).add(
                float(sa), order_acc, passed, failed, None
            )
        _ingest_slot_entries(row, slot_acc, passed, failed)
    return slot_acc, template_acc


def write_slot_csv(slot_acc: dict[SlotKey, SlotAcc], path: Path) -> int:
    fields = [
        "template_name",
        "slot_index",
        "motif",
        "motif_class",
        "n",
        "n_pass",
        "n_fail",
        "pass_rate",
        "fail_rate",
        "mean_sa",
        "mean_order_acc",
        "ci_low",
        "ci_high",
    ]
    n_published = 0
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (tpl, slot_idx, motif), acc in sorted(slot_acc.items()):
            if acc.n < MIN_N_PUBLISH:
                continue
            rec = acc.to_record()
            rec["template_name"] = tpl
            rec["slot_index"] = slot_idx
            rec["motif"] = motif
            rec["motif_class"] = acc.motif_class
            w.writerow(rec)
            n_published += 1
    return n_published


def write_template_csv(template_acc: dict[str, SlotAcc], path: Path) -> int:
    fields = [
        "template_name",
        "n",
        "n_pass",
        "n_fail",
        "pass_rate",
        "fail_rate",
        "mean_sa",
        "mean_order_acc",
        "ci_low",
        "ci_high",
    ]
    n = 0
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tpl, acc in sorted(template_acc.items()):
            rec = acc.to_record()
            rec["template_name"] = tpl
            w.writerow(rec)
            n += 1
    return n


def write_opaque_list(path: Path) -> int:
    inv = json.loads(INVENTORY.read_text())
    opaque = sorted(
        n
        for n, v in inv.items()
        if v.get("is_slot_opaque") and v.get("in_static_registry")
    )
    path.write_text("\n".join(opaque) + "\n")
    return len(opaque)


def main() -> None:
    rows = fetch_cohort()
    print(f"Cohort rows: {len(rows)}", file=sys.stderr)
    slot_acc, template_acc = aggregate(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    n_slot = write_slot_csv(slot_acc, REPORTS / "slot_realization.csv")
    n_tpl = write_template_csv(template_acc, REPORTS / "template_overall.csv")
    n_opaque = write_opaque_list(REPORTS / "slot_opaque.txt")
    print(f"slot_realization.csv: {n_slot} rows (n>={MIN_N_PUBLISH})", file=sys.stderr)
    print(f"  raw (template, slot, motif) keys: {len(slot_acc)}", file=sys.stderr)
    print(f"template_overall.csv: {n_tpl} templates", file=sys.stderr)
    print(f"slot_opaque.txt: {n_opaque} templates", file=sys.stderr)


if __name__ == "__main__":
    main()
