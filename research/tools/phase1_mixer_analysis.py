"""Phase 1.3 + 1.4 — Marginal mixer credit + op mixer certification.

One DB pass. Walks every cohort row's graph_json to extract the op set, then
emits two reports:

  research/reports/slot_mixer_credit.csv
    Per (template, slot_index, motif_class):
      n_a, pass_a — slot_motif_class is mixer AND graph has 1 mixer total
      n_b, pass_b — slot_motif_class non-mixer AND graph has >=1 mixer
      n_c, pass_c — slot_motif_class non-mixer AND graph has 0 mixers
      dominant_mixing  — TRUE iff pass_a >= 0.60 AND n_a >= 30

  research/reports/op_mixer_certification.csv
    Per op:
      n_total, n_no_other_mixer, pass_rate (no_other_mixer condition)
      class_certification: 'mixer' | 'non_mixer' | 'exotic_functional' | 'insufficient'

Pass: sa>=0.95 AND failure_op != 'nano_bind'

MIXER_SET (op-level, from the deep-dive doc plus aliases observed in DB):
  attention: softmax_attention, linear_attention, diff_attention, graph_attention,
             tropical_attention, clifford_attention, multiquery_attention,
             grouped_query_attention
  conv:      conv1d_seq, dilated_conv, separable_conv
  ssm:       selective_scan, mamba_block, gla, hyena_op
  recurrent: rwkv_time_mixing, rwkv_channel, retention, associative_memory
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "research/runs.db"
DB = f"file:{DB_PATH}?mode=ro&immutable=0"
REPORTS = REPO / "research/reports"

PASS_SA = 0.95
NANO_BIND = "nano_bind"
DOMINANT_MIN_N = 30
DOMINANT_MIN_PASS = 0.60
CERT_MIN_N = 30
CERT_PASS_THRESHOLD = 0.60
WILSON_Z = 1.96

MIXER_SET = frozenset(
    {
        "softmax_attention",
        "linear_attention",
        "diff_attention",
        "graph_attention",
        "tropical_attention",
        "clifford_attention",
        "multiquery_attention",
        "grouped_query_attention",
        "conv1d_seq",
        "dilated_conv",
        "separable_conv",
        "selective_scan",
        "mamba_block",
        "gla",
        "hyena_op",
        "rwkv_time_mixing",
        "rwkv_channel",
        "retention",
        "associative_memory",
    }
)

MIXER_CLASS_SET = frozenset(
    {
        "attention_core",
        "ssm_core",
        "conv_core",
        "channel_core",
    }
)


def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    # Same formula as analytics_grammar._wilson_interval (kept inline; Phase 1
    # tools do not depend on the analytics package).
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def is_pass(sa: float | None, failure_op: str | None) -> bool:
    return sa is not None and sa >= PASS_SA and (failure_op or "") != NANO_BIND


def fetch_cohort_rows() -> list[dict]:
    conn = sqlite3.connect(DB, uri=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pr.result_id,
               pr.language_control_s05_sentence_assoc_score AS sa,
               pr.failure_op AS failure_op,
               pr.graph_json AS graph_json,
               pgf.slot_usage_json AS slot_usage_json,
               pgf.template_name AS row_template
        FROM program_results pr
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        LEFT JOIN program_graph_features pgf ON pgf.result_id = pr.result_id
        WHERE pr.language_control_s05_sentence_assoc_score IS NOT NULL
          AND COALESCE(l.is_reference, 0) = 0
          AND pr.graph_json IS NOT NULL
        """
    )
    cols = [c[0] for c in cur.description]
    rows = []
    for raw in cur.fetchall():
        row = dict(zip(cols, raw))
        row["graph_json"] = resolve_graph_json_value(conn, DB_PATH, row["graph_json"])
        rows.append(row)
    conn.close()
    return rows


def extract_ops(graph_json_str: str) -> list[str]:
    try:
        g = json.loads(graph_json_str)
    except (json.JSONDecodeError, TypeError):
        return []
    nodes = g.get("nodes")
    if not nodes:
        return []
    iter_nodes = nodes.values() if isinstance(nodes, dict) else nodes
    ops: list[str] = []
    for node in iter_nodes:
        if isinstance(node, dict):
            op = node.get("op_name") or node.get("op") or node.get("type")
            if op and op != "input":
                ops.append(str(op))
    return ops


# --- Phase 1.3: marginal mixer credit per slot ---


class SlotCreditAcc:
    """Accumulator keyed by (template, slot_index, motif_class)."""

    __slots__ = ("n_a", "p_a", "n_b", "p_b", "n_c", "p_c")

    def __init__(self) -> None:
        self.n_a = 0
        self.p_a = 0
        self.n_b = 0
        self.p_b = 0
        self.n_c = 0
        self.p_c = 0


def _ingest_slot_credit(
    row: dict,
    ops: list[str],
    passed: bool,
    slot_acc: dict[tuple[str, int, str], SlotCreditAcc],
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
    mixer_count = sum(1 for op in ops if op in MIXER_SET)
    seen: set[tuple[str, int, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tpl = entry.get("template_name") or row["row_template"]
        slot_idx = entry.get("slot_index")
        motif_class = entry.get("selected_motif_class")
        if not tpl or slot_idx is None or not motif_class:
            continue
        key = (str(tpl), int(slot_idx), str(motif_class))
        if key in seen:
            continue
        seen.add(key)
        slot_is_mixer = motif_class in MIXER_CLASS_SET
        acc = slot_acc.setdefault(key, SlotCreditAcc())
        if slot_is_mixer and mixer_count == 1:
            acc.n_a += 1
            acc.p_a += int(passed)
        elif not slot_is_mixer and mixer_count >= 1:
            acc.n_b += 1
            acc.p_b += int(passed)
        elif not slot_is_mixer and mixer_count == 0:
            acc.n_c += 1
            acc.p_c += int(passed)


def write_slot_mixer_credit(slot_acc: dict, path: Path) -> tuple[int, int]:
    fields = [
        "template_name",
        "slot_index",
        "motif_class",
        "n_a",
        "pass_rate_a",
        "ci_low_a",
        "ci_high_a",
        "n_b",
        "pass_rate_b",
        "n_c",
        "pass_rate_c",
        "dominant_mixing",
    ]
    n_rows = 0
    n_dom = 0
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (tpl, slot_idx, mc), acc in sorted(slot_acc.items()):
            if acc.n_a + acc.n_b + acc.n_c == 0:
                continue
            pa = acc.p_a / acc.n_a if acc.n_a else 0.0
            pb = acc.p_b / acc.n_b if acc.n_b else 0.0
            pc = acc.p_c / acc.n_c if acc.n_c else 0.0
            ci_lo_a, ci_hi_a = wilson(acc.p_a, acc.n_a)
            dominant = acc.n_a >= DOMINANT_MIN_N and pa >= DOMINANT_MIN_PASS
            if dominant:
                n_dom += 1
            w.writerow(
                {
                    "template_name": tpl,
                    "slot_index": slot_idx,
                    "motif_class": mc,
                    "n_a": acc.n_a,
                    "pass_rate_a": pa,
                    "ci_low_a": ci_lo_a,
                    "ci_high_a": ci_hi_a,
                    "n_b": acc.n_b,
                    "pass_rate_b": pb,
                    "n_c": acc.n_c,
                    "pass_rate_c": pc,
                    "dominant_mixing": dominant,
                }
            )
            n_rows += 1
    return n_rows, n_dom


# --- Phase 1.4: op certification ---


class OpAcc:
    __slots__ = ("n_total", "n_no_other", "p_no_other", "n_with_other", "p_with_other")

    def __init__(self) -> None:
        self.n_total = 0
        self.n_no_other = 0
        self.p_no_other = 0
        self.n_with_other = 0
        self.p_with_other = 0


def _ingest_op_acc(ops: list[str], passed: bool, op_acc: dict[str, OpAcc]) -> None:
    op_set = set(ops)
    mixer_present = any(o in MIXER_SET for o in op_set)
    for op in op_set:
        acc = op_acc.setdefault(op, OpAcc())
        acc.n_total += 1
        # no_other_mixer: graph has no mixer aside from what THIS op contributes
        # if op is a mixer: graph has no OTHER mixers (i.e. mixer_count==1)
        # if op is non-mixer: graph has no mixer at all
        if op in MIXER_SET:
            mixer_count = sum(1 for o in ops if o in MIXER_SET)
            if mixer_count == 1:
                acc.n_no_other += 1
                acc.p_no_other += int(passed)
            else:
                acc.n_with_other += 1
                acc.p_with_other += int(passed)
        else:
            if not mixer_present:
                acc.n_no_other += 1
                acc.p_no_other += int(passed)
            else:
                acc.n_with_other += 1
                acc.p_with_other += int(passed)


def _certify(op: str, acc: OpAcc) -> str:
    if acc.n_no_other < CERT_MIN_N:
        return "insufficient"
    p = acc.p_no_other / acc.n_no_other
    if op in MIXER_SET:
        return "mixer" if p >= CERT_PASS_THRESHOLD else "mixer_underperforming"
    if p >= CERT_PASS_THRESHOLD:
        return "exotic_functional"
    return "non_mixer"


def write_op_certification(op_acc: dict[str, OpAcc], path: Path) -> int:
    fields = [
        "op",
        "in_mixer_set",
        "n_total",
        "n_no_other_mixer",
        "pass_rate_no_other",
        "ci_low_no_other",
        "ci_high_no_other",
        "n_with_other_mixer",
        "pass_rate_with_other",
        "certification",
    ]
    n_rows = 0
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for op, acc in sorted(op_acc.items(), key=lambda kv: -kv[1].n_total):
            if acc.n_total == 0:
                continue
            ci_lo, ci_hi = wilson(acc.p_no_other, acc.n_no_other)
            w.writerow(
                {
                    "op": op,
                    "in_mixer_set": op in MIXER_SET,
                    "n_total": acc.n_total,
                    "n_no_other_mixer": acc.n_no_other,
                    "pass_rate_no_other": (
                        acc.p_no_other / acc.n_no_other if acc.n_no_other else None
                    ),
                    "ci_low_no_other": ci_lo,
                    "ci_high_no_other": ci_hi,
                    "n_with_other_mixer": acc.n_with_other,
                    "pass_rate_with_other": (
                        acc.p_with_other / acc.n_with_other
                        if acc.n_with_other
                        else None
                    ),
                    "certification": _certify(op, acc),
                }
            )
            n_rows += 1
    return n_rows


def main() -> None:
    rows = fetch_cohort_rows()
    print(f"Cohort rows: {len(rows)}", file=sys.stderr)
    slot_acc: dict[tuple[str, int, str], SlotCreditAcc] = defaultdict(SlotCreditAcc)
    op_acc: dict[str, OpAcc] = {}
    skipped = 0
    for row in rows:
        sa = row["sa"]
        if sa is None:
            skipped += 1
            continue
        passed = is_pass(sa, row["failure_op"])
        ops = extract_ops(row["graph_json"])
        if not ops:
            continue
        _ingest_op_acc(ops, passed, op_acc)
        _ingest_slot_credit(row, ops, passed, slot_acc)
    REPORTS.mkdir(parents=True, exist_ok=True)
    n_slot, n_dom = write_slot_mixer_credit(slot_acc, REPORTS / "slot_mixer_credit.csv")
    n_op = write_op_certification(op_acc, REPORTS / "op_mixer_certification.csv")
    print(
        f"slot_mixer_credit.csv: {n_slot} rows; dominant-mixing slots: {n_dom}",
        file=sys.stderr,
    )
    print(f"op_mixer_certification.csv: {n_op} ops", file=sys.stderr)
    if n_dom < 5:
        print(
            "WARN: <5 dominant-mixing slots — STOP-AND-REPORT triggered "
            "(see deep-dive prompt §Constraints).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
