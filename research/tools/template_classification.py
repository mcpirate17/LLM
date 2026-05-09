"""Phase 2 — Template classification (Buckets A/B/C/D/E).

Combines Phase 1 outputs into a single per-template decision:

  research/reports/template_overall.csv          (n, mean_sa, ci_low/high)
  research/reports/slot_realization.csv          (per slot×motif pass rates)
  research/reports/slot_mixer_credit_v2.csv      (dominant-mixing slots)
  research/reports/op_mixer_certification_v2.csv (mixer/non-mixer/exotic op cert)
  research/reports/high_capability_slot_fills.csv(induction_intermediate max_auc per slot)
  research/reports/slot_opaque.txt               (typed-slot-absent templates)
  + per-template op presence derived from runs.db

Bucket criteria (deep-dive doc §Phase 2):
  A KEEP    : mean_sa>=0.80 AND n>=20 AND template has dominant-mixing slot
              AND template uses a 'mixer'-cert op (op-level guarantee)
  B CULL    : (mean_sa<0.40 AND n>=30 AND no dominant slot) OR ci_high<0.20
  C RESCUE  : has both pass-cohort (n_pass>=30) and fail-cohort (n_fail>=30)
              members, AND a slot exists where pass-fill ops differ from
              fail-fill ops by certification class
  D MINE    : failing parent + has at least one (exotic_functional, mixer)
              op pair somewhere in its passing-cohort graphs
  E INSUFFICIENT: n<20

Output: research/reports/template_classification.csv
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "research/runs.db"
REPORTS = REPO / "research/reports"

PASS_SA = 0.95
FAIL_SA = 0.30
NANO_BIND = "nano_bind"

KEEP_MIN_N = 20
KEEP_MIN_MEAN_SA = 0.80
CULL_MIN_N = 30
CULL_MAX_MEAN_SA = 0.40
CULL_MAX_CI_HIGH = 0.20
RESCUE_MIN_PASS_N = 30
RESCUE_MIN_FAIL_N = 30
INSUFFICIENT_MAX_N = 20


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def load_opaque(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


# --- per-template op presence + pass/fail counts derived from lab DB ---


def load_per_template_signals() -> dict[str, dict[str, Any]]:
    """template -> {pass_n, fail_n, ops_in_pass, ops_in_fail, ops_total}."""
    conn = sqlite3.connect(f"file:{LAB}?mode=ro&immutable=0", uri=True)
    cur = conn.execute(
        """
        SELECT pgf.template_name, pr.language_control_s05_sentence_assoc_score, pr.failure_op,
               pr.graph_json
        FROM program_results pr
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        LEFT JOIN program_graph_features pgf ON pgf.result_id = pr.result_id
        WHERE pr.language_control_s05_sentence_assoc_score IS NOT NULL
          AND COALESCE(l.is_reference, 0) = 0
          AND pr.graph_json IS NOT NULL
          AND pgf.template_name IS NOT NULL
        """
    )
    out: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "pass_n": 0,
            "fail_n": 0,
            "ops_in_pass": defaultdict(int),
            "ops_in_fail": defaultdict(int),
            "ops_total": defaultdict(int),
        }
    )
    for tpl, sa, failure_op, graph_json in cur.fetchall():
        if not tpl:
            continue
        passed = sa is not None and sa >= PASS_SA and (failure_op or "") != NANO_BIND
        failed = (failure_op or "") == NANO_BIND or (sa is not None and sa < FAIL_SA)
        try:
            g = json.loads(graph_json)
        except (json.JSONDecodeError, TypeError):
            continue
        nodes = g.get("nodes")
        if not nodes:
            continue
        iter_n = nodes.values() if isinstance(nodes, dict) else nodes
        ops: set[str] = set()
        for n in iter_n:
            if isinstance(n, dict):
                op = n.get("op_name") or n.get("op") or n.get("type")
                if op and op != "input":
                    ops.add(str(op))
        rec = out[tpl]
        if passed:
            rec["pass_n"] += 1
        if failed:
            rec["fail_n"] += 1
        for op in ops:
            rec["ops_total"][op] += 1
            if passed:
                rec["ops_in_pass"][op] += 1
            elif failed:
                rec["ops_in_fail"][op] += 1
    conn.close()
    return out


# --- Phase 1 input indexes ---


class Phase1Index:
    """Lookup tables built once from the Phase 1 CSVs."""

    def __init__(self) -> None:
        # Per-template overall stats
        self.overall: dict[str, dict[str, str]] = {
            r["template_name"]: r for r in load_csv(REPORTS / "template_overall.csv")
        }
        # Templates that have at least one dominant-mixing slot
        dom = load_csv(REPORTS / "slot_mixer_credit_v2.csv")
        self.dominant_templates: set[str] = {
            r["template_name"] for r in dom if r["dominant_mixing"] == "True"
        }
        # Op certification index
        self.op_cert: dict[str, str] = {
            r["op"]: r["certification"]
            for r in load_csv(REPORTS / "op_mixer_certification_v2.csv")
        }
        # Slot-realization rows grouped by template (for Bucket C analysis)
        self.slot_real_by_tpl: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in load_csv(REPORTS / "slot_realization.csv"):
            self.slot_real_by_tpl[r["template_name"]].append(r)
        # High-capability slots (template -> max_auc)
        self.high_cap_max_auc: dict[str, float] = {}
        for r in load_csv(REPORTS / "high_capability_slot_fills.csv"):
            tpl = r["template_name"]
            try:
                v = float(r["max_induction_intermediate_auc"])
            except (TypeError, ValueError):
                continue
            self.high_cap_max_auc[tpl] = max(self.high_cap_max_auc.get(tpl, 0.0), v)
        # Slot-opaque list
        self.opaque: set[str] = load_opaque(REPORTS / "slot_opaque.txt")


# --- bucket decision logic ---


def _has_mixer_op(
    rec: dict[str, Any], op_cert: dict[str, str]
) -> tuple[bool, str | None]:
    for op in rec["ops_total"]:
        if op_cert.get(op) == "mixer":
            return True, op
    return False, None


def _exotic_mixer_pair(rec: dict[str, Any], op_cert: dict[str, str]) -> str | None:
    """Bucket D pair: at least one exotic_functional op + at least one mixer op
    co-occurring in passing-cohort graphs of this template.
    """
    pass_ops = rec["ops_in_pass"]
    if rec["pass_n"] < 5:
        return None
    exotic = [
        op
        for op in pass_ops
        if op_cert.get(op) == "exotic_functional" and pass_ops[op] >= 3
    ]
    mixers = [op for op in pass_ops if op_cert.get(op) == "mixer" and pass_ops[op] >= 3]
    if exotic and mixers:
        return f"{exotic[0]}+{mixers[0]}"
    return None


def _rescue_slot_signal(tpl: str, idx: Phase1Index) -> tuple[bool, str | None]:
    """Bucket C: a slot has both passing fills and failing fills where the
    fills differ in op-cert or motif identity, with sufficient n on both sides.
    """
    rows = idx.slot_real_by_tpl.get(tpl, [])
    by_slot: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_slot[r["slot_index"]].append(r)
    for slot_idx, slot_rows in by_slot.items():
        passers = [
            r for r in slot_rows if float(r["pass_rate"]) >= 0.60 and int(r["n"]) >= 5
        ]
        failers = [
            r for r in slot_rows if float(r["pass_rate"]) <= 0.10 and int(r["n"]) >= 5
        ]
        if passers and failers:
            p_motifs = ",".join(r["motif"] for r in passers[:2])
            f_motifs = ",".join(r["motif"] for r in failers[:2])
            return True, f"slot{slot_idx}: pass={p_motifs} fail={f_motifs}"
    return False, None


class _Ctx:
    """Snapshot of all signals used by classification rules."""

    __slots__ = (
        "tpl",
        "n",
        "mean_sa",
        "ci_high",
        "pass_rate",
        "has_mixer",
        "mixer_op",
        "has_dom_slot",
        "is_opaque",
        "high_cap",
        "exotic_pair",
        "has_rescue",
        "rescue_signal",
        "pass_n",
        "fail_n",
    )

    def __init__(
        self,
        tpl: str,
        rec: dict[str, Any],
        idx: Phase1Index,
        overall: dict[str, str] | None,
    ) -> None:
        self.tpl = tpl
        self.n = int(overall["n"]) if overall else 0
        self.mean_sa = float(overall["mean_sa"]) if overall else 0.0
        self.ci_high = float(overall["ci_high"]) if overall else 0.0
        self.pass_rate = float(overall["pass_rate"]) if overall else 0.0
        self.has_mixer, self.mixer_op = _has_mixer_op(rec, idx.op_cert)
        self.has_dom_slot = tpl in idx.dominant_templates
        self.is_opaque = tpl in idx.opaque
        self.high_cap = idx.high_cap_max_auc.get(tpl, 0.0)
        self.exotic_pair = _exotic_mixer_pair(rec, idx.op_cert)
        self.has_rescue, self.rescue_signal = _rescue_slot_signal(tpl, idx)
        self.pass_n = int(rec["pass_n"])
        self.fail_n = int(rec["fail_n"])


def _try_e_insufficient(c: _Ctx) -> dict[str, Any] | None:
    if c.n < INSUFFICIENT_MAX_N:
        return {
            "bucket": "E",
            "primary_reason": f"n={c.n} < {INSUFFICIENT_MAX_N}",
            "action_summary": "leave; flag for backfill",
        }
    return None


def _try_b_cull(c: _Ctx) -> dict[str, Any] | None:
    bucket_b_a = (
        c.mean_sa < CULL_MAX_MEAN_SA and c.n >= CULL_MIN_N and not c.has_dom_slot
    )
    bucket_b_b = c.ci_high < CULL_MAX_CI_HIGH
    if not (bucket_b_a or bucket_b_b):
        return None
    reasons = []
    if bucket_b_a:
        reasons.append(f"mean_sa={c.mean_sa:.2f}<{CULL_MAX_MEAN_SA} no_dom_slot")
    if bucket_b_b:
        reasons.append(f"ci_high={c.ci_high:.2f}<{CULL_MAX_CI_HIGH}")
    return {
        "bucket": "B",
        "primary_reason": "; ".join(reasons),
        "action_summary": f"cull -> weight floor 0.5 (n={c.n})",
    }


def _try_a_plus(c: _Ctx) -> dict[str, Any] | None:
    if c.high_cap >= 0.95 and c.mean_sa >= 0.50 and c.n >= KEEP_MIN_N:
        return {
            "bucket": "A+",
            "primary_reason": f"v2_max_auc={c.high_cap:.2f} mean_sa={c.mean_sa:.2f}",
            "action_summary": f"weight boost +50% (mixer={c.mixer_op or 'none'})",
        }
    return None


def _try_a_keep(c: _Ctx) -> dict[str, Any] | None:
    """Bucket A — KEEP. Strict path requires dom_slot; soft path requires
    mixer_op + n >= 30 (more samples to compensate for missing dom_slot).
    """
    if c.mean_sa < KEEP_MIN_MEAN_SA or c.n < KEEP_MIN_N:
        return None
    if c.has_dom_slot and c.has_mixer:
        return {
            "bucket": "A",
            "primary_reason": (
                f"mean_sa={c.mean_sa:.2f}>={KEEP_MIN_MEAN_SA} dom_slot "
                f"mixer_op={c.mixer_op}"
            ),
            "action_summary": "keep; raise weight up to +50%",
        }
    # Soft A: mixer_op present, no dom_slot, but compensate by requiring
    # n >= 30 AND combined pass_rate >= 0.50 (so a template with high fluency
    # but failing nano_bind — e.g. token_merge_block — does not qualify).
    if c.has_mixer and c.n >= CULL_MIN_N and c.pass_rate >= 0.50:
        return {
            "bucket": "A",
            "primary_reason": (
                f"mean_sa={c.mean_sa:.2f}>={KEEP_MIN_MEAN_SA} pass_rate={c.pass_rate:.2f} "
                f"mixer_op={c.mixer_op} (soft-A: no dom_slot)"
            ),
            "action_summary": "keep; raise weight up to +50% (soft-A)",
        }
    return None


def _try_d_mine(c: _Ctx) -> dict[str, Any] | None:
    if c.exotic_pair and c.pass_rate < 0.60 and c.n >= CULL_MIN_N:
        return {
            "bucket": "D",
            "primary_reason": f"exotic_mixer_pair={c.exotic_pair} pass_rate={c.pass_rate:.2f}",
            "action_summary": f"mine sub-pattern -> new template ({c.exotic_pair})",
        }
    return None


def _try_c_rescue(c: _Ctx) -> dict[str, Any] | None:
    if c.has_rescue and c.pass_n >= RESCUE_MIN_PASS_N and c.fail_n >= RESCUE_MIN_FAIL_N:
        return {
            "bucket": "C",
            "primary_reason": c.rescue_signal,
            "action_summary": (
                f"tighten slot constraint -> mixer-only fills "
                f"(n_pass={c.pass_n}, n_fail={c.fail_n})"
            ),
        }
    return None


def _try_e_opaque(c: _Ctx) -> dict[str, Any] | None:
    if c.is_opaque:
        return {
            "bucket": "E",
            "primary_reason": "slot_opaque (no typed slots to decompose)",
            "action_summary": "leave; rely on op-level signals",
        }
    return None


_RULE_CHAIN = (
    _try_e_insufficient,
    _try_a_plus,
    _try_a_keep,
    _try_c_rescue,
    _try_d_mine,
    _try_b_cull,
    _try_e_opaque,
)


def _classify(
    tpl: str, rec: dict[str, Any], idx: Phase1Index, overall: dict[str, str] | None
) -> dict[str, Any]:
    c = _Ctx(tpl, rec, idx, overall)
    for rule in _RULE_CHAIN:
        decision = rule(c)
        if decision is not None:
            return decision
    return {
        "bucket": "HOLD",
        "primary_reason": (
            f"mean_sa={c.mean_sa:.2f} n={c.n} dom_slot={c.has_dom_slot} "
            f"mixer_op={c.has_mixer} exotic_pair={bool(c.exotic_pair)}"
        ),
        "action_summary": "no change pending review",
    }


def write_classification(
    per_tpl: dict[str, dict], idx: Phase1Index, path: Path
) -> dict[str, int]:
    fields = [
        "template_name",
        "bucket",
        "n",
        "mean_sa",
        "pass_rate",
        "fail_rate",
        "ci_low",
        "ci_high",
        "has_dom_slot",
        "mixer_op_present",
        "exotic_mixer_pair",
        "v2_max_auc",
        "is_opaque",
        "primary_reason",
        "action_summary",
    ]
    counts: dict[str, int] = defaultdict(int)
    all_templates = sorted(set(idx.overall.keys()) | set(per_tpl.keys()))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tpl in all_templates:
            rec = per_tpl.get(tpl) or {
                "pass_n": 0,
                "fail_n": 0,
                "ops_in_pass": {},
                "ops_in_fail": {},
                "ops_total": {},
            }
            overall = idx.overall.get(tpl)
            decision = _classify(tpl, rec, idx, overall)
            counts[decision["bucket"]] += 1
            _, mixer_op = _has_mixer_op(rec, idx.op_cert)
            exotic = _exotic_mixer_pair(rec, idx.op_cert)
            w.writerow(
                {
                    "template_name": tpl,
                    "bucket": decision["bucket"],
                    "n": int(overall["n"]) if overall else 0,
                    "mean_sa": float(overall["mean_sa"]) if overall else 0.0,
                    "pass_rate": float(overall["pass_rate"]) if overall else 0.0,
                    "fail_rate": float(overall["fail_rate"]) if overall else 0.0,
                    "ci_low": float(overall["ci_low"]) if overall else 0.0,
                    "ci_high": float(overall["ci_high"]) if overall else 0.0,
                    "has_dom_slot": tpl in idx.dominant_templates,
                    "mixer_op_present": mixer_op or "",
                    "exotic_mixer_pair": exotic or "",
                    "v2_max_auc": idx.high_cap_max_auc.get(tpl, 0.0),
                    "is_opaque": tpl in idx.opaque,
                    "primary_reason": decision["primary_reason"],
                    "action_summary": decision["action_summary"],
                }
            )
    return dict(counts)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    print("Loading per-template op presence + pass/fail counts...", file=sys.stderr)
    per_tpl = load_per_template_signals()
    print(f"  templates with cohort data: {len(per_tpl)}", file=sys.stderr)
    idx = Phase1Index()
    print(f"  overall stats: {len(idx.overall)} templates", file=sys.stderr)
    print(
        f"  dominant-mixing templates: {len(idx.dominant_templates)}", file=sys.stderr
    )
    print(f"  op cert mappings: {len(idx.op_cert)}", file=sys.stderr)
    print(f"  high-cap templates: {len(idx.high_cap_max_auc)}", file=sys.stderr)
    counts = write_classification(per_tpl, idx, REPORTS / "template_classification.csv")
    print("\nBucket counts:", file=sys.stderr)
    for b in ("A+", "A", "B", "C", "D", "HOLD", "E"):
        print(f"  {b:5s}  {counts.get(b, 0)}", file=sys.stderr)


if __name__ == "__main__":
    main()
