"""Phase 1.3 v2 + 1.4 v2 + high-capability slots — meta-DB rebuild.

Replaces:
  - phase1_mixer_analysis.py's hardcoded MIXER_SET (heuristic op list)
    with op_property_catalog.op_binding_range_class (hand-curated).
  - heuristic motif_class membership for "is mixer slot"
    with slot_property_catalog.slot_accepts_{attention,ssm,compression}.

Adds high-capability slot report keyed off induction_intermediate_auc
(coverage ~6% but per-position n up to 65 on the heaviest slots).

Inputs:
  research/meta_analysis.db (slot_observations, op_observations,
                             slot_property_catalog, op_property_catalog)
  research/runs.db (program_results — sa_score + failure_op + is_ref)

Outputs:
  research/reports/slot_mixer_credit_v2.csv
  research/reports/op_mixer_certification_v2.csv
  research/reports/high_capability_slot_fills.csv
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from research.stats import wilson_score_interval

REPO = Path(__file__).resolve().parents[2]
META = REPO / "research/meta_analysis.db"
LAB = REPO / "research/runs.db"
REPORTS = REPO / "research/reports"

PASS_SA = 0.95
NANO_BIND = "nano_bind"
DOMINANT_MIN_N = 30
DOMINANT_MIN_PASS = 0.60
CERT_MIN_N = 30
CERT_PASS_THRESHOLD = 0.60
CAPABILITY_AUC_MIN = 0.55

# op-binding-range-class values that count as "the graph has a mixer".
# 'full' = full-context mixer (attention, SSM, etc.); 'local' = conv1d_seq;
# 'medium' = conv variants. 'none' = non-mixer.
MIXER_BINDING_CLASSES = frozenset({"full", "local", "medium"})


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{META}?mode=ro&immutable=0", uri=True)
    conn.execute(f"ATTACH 'file:{LAB}?mode=ro&immutable=0' AS lab")
    return conn


def load_op_catalog(conn: sqlite3.Connection) -> dict[str, dict]:
    """op_name -> {category, binding_range, algebraic_space}."""
    cur = conn.execute(
        """SELECT op_name, op_category, op_binding_range_class, op_algebraic_space
           FROM op_property_catalog"""
    )
    return {
        row[0]: {
            "category": row[1],
            "binding_range": row[2],
            "algebraic_space": row[3],
        }
        for row in cur.fetchall()
    }


def load_slot_catalog(conn: sqlite3.Connection) -> dict[tuple[str, int], dict]:
    """(template_name, slot_index) -> {role_family, accepts_*}."""
    cur = conn.execute(
        """SELECT template_name, slot_index, slot_role_family,
                  slot_accepts_attention, slot_accepts_ssm, slot_accepts_routing,
                  slot_accepts_compression, slot_accepts_memory,
                  slot_is_wildcard, slot_allowed_class_count
           FROM slot_property_catalog"""
    )
    out: dict[tuple[str, int], dict] = {}
    for row in cur.fetchall():
        attn, ssm, routing, comp, mem = row[3], row[4], row[5], row[6], row[7]
        out[(row[0], int(row[1]))] = {
            "role_family": row[2],
            "is_mixer_slot": bool(attn or ssm or comp),
            "is_norm_slot": (attn, ssm, routing, comp, mem) == (0, 0, 0, 0, 0),
            "is_wildcard": bool(row[8]),
            "allowed_class_count": row[9],
            "accepts": {
                "attention": bool(attn),
                "ssm": bool(ssm),
                "routing": bool(routing),
                "compression": bool(comp),
                "memory": bool(mem),
            },
        }
    return out


# --- Per-graph op set + mixer-presence (from program_results.graph_json) ---


def load_per_graph_ops(
    conn: sqlite3.Connection, op_catalog: dict[str, dict]
) -> dict[str, dict]:
    """result_id -> {ops: set, mixer_count, sa, failure_op, passed}."""
    import json

    cur = conn.execute(
        """
        SELECT pr.result_id, pr.language_control_s05_sentence_assoc_score, pr.failure_op,
               pr.graph_json
        FROM lab.program_results pr
        LEFT JOIN lab.leaderboard l ON l.result_id = pr.result_id
        WHERE pr.language_control_s05_sentence_assoc_score IS NOT NULL
          AND COALESCE(l.is_reference, 0) = 0
          AND pr.graph_json IS NOT NULL
        """
    )
    out: dict[str, dict] = {}
    for result_id, sa, failure_op, graph_json in cur.fetchall():
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
        mixer_count = sum(
            1
            for o in ops
            if op_catalog.get(o, {}).get("binding_range") in MIXER_BINDING_CLASSES
        )
        passed = sa is not None and sa >= PASS_SA and (failure_op or "") != NANO_BIND
        out[result_id] = {
            "ops": ops,
            "mixer_count": mixer_count,
            "sa": sa,
            "failure_op": failure_op,
            "passed": passed,
        }
    return out


# --- Phase 1.3 v2: slot mixer credit using catalog bitfields ---


class SlotV2Acc:
    __slots__ = ("n_a", "p_a", "n_b", "p_b", "n_c", "p_c", "n_total", "p_total")

    def __init__(self) -> None:
        self.n_a = 0
        self.p_a = 0
        self.n_b = 0
        self.p_b = 0
        self.n_c = 0
        self.p_c = 0
        self.n_total = 0
        self.p_total = 0


def aggregate_slot_credit(
    conn: sqlite3.Connection,
    slot_catalog: dict[tuple[str, int], dict],
    per_graph: dict[str, dict],
) -> dict[tuple, SlotV2Acc]:
    cur = conn.execute(
        """SELECT result_id, template_name, slot_index, selected_motif_class
           FROM slot_observations
           WHERE selected_motif_class IS NOT NULL"""
    )
    acc: dict[tuple, SlotV2Acc] = defaultdict(SlotV2Acc)
    for result_id, tpl, slot_idx, motif_class in cur.fetchall():
        graph = per_graph.get(result_id)
        if graph is None:
            continue
        slot_props = slot_catalog.get((tpl, int(slot_idx)))
        if slot_props is None:
            continue
        if not slot_props["is_mixer_slot"]:
            continue
        passed = graph["passed"]
        mixer_count = graph["mixer_count"]
        key = (tpl, int(slot_idx), str(motif_class))
        a = acc[key]
        a.n_total += 1
        if passed:
            a.p_total += 1
        if mixer_count >= 1:
            a.n_a += 1
            if passed:
                a.p_a += 1
        if mixer_count >= 2:
            a.n_b += 1
            if passed:
                a.p_b += 1
        if mixer_count == 0:
            a.n_c += 1
            if passed:
                a.p_c += 1
    return acc


def write_slot_credit_v2(
    acc: dict[tuple, SlotV2Acc], slot_catalog: dict[tuple[str, int], dict], path: Path
) -> tuple[int, int]:
    fields = [
        "template_name",
        "slot_index",
        "motif_class",
        "slot_role_family",
        "n_total",
        "pass_rate_total",
        "n_a_has_mixer",
        "pass_rate_a",
        "ci_low_a",
        "ci_high_a",
        "n_b_has_extra_mixer",
        "pass_rate_b",
        "n_c_no_mixer",
        "pass_rate_c",
        "dominant_mixing",
    ]
    n_rows = 0
    n_dom = 0
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (tpl, slot_idx, mc), a in sorted(acc.items()):
            if a.n_total == 0:
                continue
            pa = a.p_a / a.n_a if a.n_a else 0.0
            pb = a.p_b / a.n_b if a.n_b else 0.0
            pc = a.p_c / a.n_c if a.n_c else 0.0
            pt = a.p_total / a.n_total
            ci_lo, ci_hi = wilson_score_interval(a.p_a, a.n_a)
            dominant = a.n_a >= DOMINANT_MIN_N and pa >= DOMINANT_MIN_PASS
            if dominant:
                n_dom += 1
            slot_props = slot_catalog.get((tpl, slot_idx), {})
            w.writerow(
                {
                    "template_name": tpl,
                    "slot_index": slot_idx,
                    "motif_class": mc,
                    "slot_role_family": slot_props.get("role_family", ""),
                    "n_total": a.n_total,
                    "pass_rate_total": pt,
                    "n_a_has_mixer": a.n_a,
                    "pass_rate_a": pa,
                    "ci_low_a": ci_lo,
                    "ci_high_a": ci_hi,
                    "n_b_has_extra_mixer": a.n_b,
                    "pass_rate_b": pb,
                    "n_c_no_mixer": a.n_c,
                    "pass_rate_c": pc,
                    "dominant_mixing": dominant,
                }
            )
            n_rows += 1
    return n_rows, n_dom


# --- Phase 1.4 v2: op cert using op_property_catalog ---


class OpV2Acc:
    __slots__ = ("n_total", "n_solo", "p_solo", "n_with_other", "p_with_other")

    def __init__(self) -> None:
        self.n_total = 0
        self.n_solo = 0
        self.p_solo = 0
        self.n_with_other = 0
        self.p_with_other = 0


def aggregate_op_cert(
    per_graph: dict[str, dict], op_catalog: dict[str, dict]
) -> dict[str, OpV2Acc]:
    op_acc: dict[str, OpV2Acc] = defaultdict(OpV2Acc)
    for graph in per_graph.values():
        ops = graph["ops"]
        passed = graph["passed"]
        mixer_count = graph["mixer_count"]
        for op in ops:
            a = op_acc[op]
            a.n_total += 1
            is_mixer = (
                op_catalog.get(op, {}).get("binding_range") in MIXER_BINDING_CLASSES
            )
            # solo = no other mixer in graph
            if is_mixer:
                solo = mixer_count == 1
            else:
                solo = mixer_count == 0
            if solo:
                a.n_solo += 1
                if passed:
                    a.p_solo += 1
            else:
                a.n_with_other += 1
                if passed:
                    a.p_with_other += 1
    return op_acc


def _certify_op(a: OpV2Acc, props: dict) -> str:
    if a.n_solo < CERT_MIN_N:
        return "insufficient"
    p = a.p_solo / a.n_solo
    is_mixer = props.get("binding_range") in MIXER_BINDING_CLASSES
    if is_mixer:
        return "mixer" if p >= CERT_PASS_THRESHOLD else "mixer_underperforming"
    if p >= CERT_PASS_THRESHOLD:
        return "exotic_functional"
    return "non_mixer"


def write_op_cert_v2(
    op_acc: dict[str, OpV2Acc], op_catalog: dict[str, dict], path: Path
) -> int:
    fields = [
        "op",
        "op_category",
        "op_binding_range_class",
        "op_algebraic_space",
        "n_total",
        "n_solo",
        "pass_rate_solo",
        "ci_low",
        "ci_high",
        "n_with_other",
        "pass_rate_with_other",
        "certification",
    ]
    n_rows = 0
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for op, a in sorted(op_acc.items(), key=lambda kv: -kv[1].n_total):
            props = op_catalog.get(op, {})
            ci_lo, ci_hi = wilson_score_interval(a.p_solo, a.n_solo)
            w.writerow(
                {
                    "op": op,
                    "op_category": props.get("category"),
                    "op_binding_range_class": props.get("binding_range"),
                    "op_algebraic_space": props.get("algebraic_space"),
                    "n_total": a.n_total,
                    "n_solo": a.n_solo,
                    "pass_rate_solo": (a.p_solo / a.n_solo if a.n_solo else None),
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "n_with_other": a.n_with_other,
                    "pass_rate_with_other": (
                        a.p_with_other / a.n_with_other if a.n_with_other else None
                    ),
                    "certification": _certify_op(a, props),
                }
            )
            n_rows += 1
    return n_rows


# --- High-capability slot fills (induction_intermediate_auc) ---


def write_high_capability(conn: sqlite3.Connection, path: Path) -> tuple[int, int]:
    fields = [
        "template_name",
        "slot_index",
        "selected_motif",
        "motif_class",
        "n_v2",
        "n_v2_pass",
        "mean_induction_intermediate_auc",
        "max_induction_intermediate_auc",
        "mean_binding_intermediate_auc",
    ]
    cur = conn.execute(
        """
        SELECT template_name, slot_index, selected_motif, selected_motif_class,
               COUNT(*) AS n_v2,
               SUM(CASE WHEN induction_intermediate_auc >= 0.55 THEN 1 ELSE 0 END) AS n_pass,
               AVG(induction_intermediate_auc) AS mean_auc,
               MAX(induction_intermediate_auc) AS max_auc,
               AVG(binding_intermediate_auc) AS mean_bind_auc
        FROM slot_observations
        WHERE induction_intermediate_auc IS NOT NULL
          AND selected_motif IS NOT NULL
        GROUP BY template_name, slot_index, selected_motif, selected_motif_class
        ORDER BY mean_auc DESC, n_v2 DESC
        """
    )
    rows = cur.fetchall()
    n_published = 0
    n_high = 0
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (
            tpl,
            slot_idx,
            motif,
            mc,
            n_v2,
            n_pass,
            mean_auc,
            max_auc,
            mean_bind,
        ) in rows:
            if n_v2 < 5:
                continue
            w.writerow(
                {
                    "template_name": tpl,
                    "slot_index": slot_idx,
                    "selected_motif": motif,
                    "motif_class": mc,
                    "n_v2": n_v2,
                    "n_v2_pass": n_pass,
                    "mean_induction_intermediate_auc": mean_auc,
                    "max_induction_intermediate_auc": max_auc,
                    "mean_binding_intermediate_auc": mean_bind,
                }
            )
            n_published += 1
            if (mean_auc or 0.0) >= CAPABILITY_AUC_MIN and n_v2 >= 5:
                n_high += 1
    return n_published, n_high


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    op_catalog = load_op_catalog(conn)
    slot_catalog = load_slot_catalog(conn)
    print(f"op_property_catalog: {len(op_catalog)} ops", file=sys.stderr)
    print(f"slot_property_catalog: {len(slot_catalog)} slots", file=sys.stderr)
    n_mixer_slots = sum(1 for s in slot_catalog.values() if s["is_mixer_slot"])
    print(f"  mixer-capable slots: {n_mixer_slots}", file=sys.stderr)

    per_graph = load_per_graph_ops(conn, op_catalog)
    print(f"cohort graphs: {len(per_graph)}", file=sys.stderr)

    slot_acc = aggregate_slot_credit(conn, slot_catalog, per_graph)
    n_slot, n_dom = write_slot_credit_v2(
        slot_acc, slot_catalog, REPORTS / "slot_mixer_credit_v2.csv"
    )
    print(
        f"slot_mixer_credit_v2.csv: {n_slot} rows; dominant-mixing: {n_dom}",
        file=sys.stderr,
    )

    op_acc = aggregate_op_cert(per_graph, op_catalog)
    n_op = write_op_cert_v2(
        op_acc, op_catalog, REPORTS / "op_mixer_certification_v2.csv"
    )
    print(f"op_mixer_certification_v2.csv: {n_op} ops", file=sys.stderr)

    n_pub, n_high = write_high_capability(
        conn, REPORTS / "high_capability_slot_fills.csv"
    )
    print(
        f"high_capability_slot_fills.csv: {n_pub} rows; AUC>=0.55: {n_high}",
        file=sys.stderr,
    )
    conn.close()


if __name__ == "__main__":
    main()
