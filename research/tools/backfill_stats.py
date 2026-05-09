"""Backfill template/op/motif/slot analytics from live notebook results.

Usage:
    python -m research.tools.backfill_stats [--db research/runs.db]

Reads canonicalized graph-analysis rows from the notebook so the steering
tables stay aligned with current continuous runs rather than waiting for an
offline corpus rebuild. The resulting analytics tables remain structural-unique
at the graph level while also carrying induction/binding/math-space signals.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from research.scientist.intelligence.ml_corpus import load_deduped_graph_analysis_rows
from research.tools._db_maintenance import connect_writer
from research.tools._script_audit import (
    complete_script_experiment,
    fail_script_experiment,
    start_script_experiment,
)

_RECENCY_HALF_LIFE_SECONDS = 14.0 * 24.0 * 3600.0
_RECENCY_WEIGHT_FLOOR = 0.25


def _unique_strings(values: object) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _extract_graph_info(
    graph_json: str,
) -> Tuple[List[str], List[str], List[str], List[dict]]:
    """Extract template names, motif names, op names, and slot usage from graph JSON."""
    try:
        g = json.loads(graph_json)
    except (json.JSONDecodeError, TypeError):
        return [], [], [], []

    metadata = g.get("metadata", {})
    templates = _unique_strings(metadata.get("templates_used", []))
    motifs = _unique_strings(metadata.get("motifs_used", []))
    slot_usage = metadata.get("template_slot_usage", [])

    ops = []
    nodes = g.get("nodes", {})
    node_iter = nodes.values() if isinstance(nodes, dict) else nodes
    for node in node_iter:
        if isinstance(node, dict):
            op = node.get("op_name")
            if op and op != "input":
                ops.append(op)

    return (
        templates,
        motifs,
        ops,
        slot_usage if isinstance(slot_usage, list) else [],
    )


def _sample_value(value) -> float:
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def _safe_std(values: List[float]) -> float:
    """Standard deviation, or 0.0 if fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    raw_values = [_sample_value(v) for v in values]
    mean = sum(raw_values) / len(raw_values)
    variance = sum((v - mean) ** 2 for v in raw_values) / (len(raw_values) - 1)
    return math.sqrt(variance)


def _mean_or_none(values: List[float]):
    if not values:
        return None
    first = values[0]
    if isinstance(first, tuple):
        weight_sum = sum(max(float(weight), 0.0) for _, weight in values)
        if weight_sum <= 0.0:
            return None
        return (
            sum(float(value) * max(float(weight), 0.0) for value, weight in values)
            / weight_sum
        )
    return sum(values) / len(values)


def _min_or_none(values: List[float]):
    return min((_sample_value(v) for v in values), default=None)


def _recency_weight(timestamp, now: float) -> float:
    ts = _normalize_metric(timestamp)
    if ts is None or ts <= 0.0:
        return 1.0
    age = max(0.0, now - ts)
    weight = 0.5 ** (age / _RECENCY_HALF_LIFE_SECONDS)
    return max(_RECENCY_WEIGHT_FLOOR, min(1.0, weight))


def _append_weighted(values: list, value, weight: float) -> None:
    metric = _normalize_metric(value)
    if metric is not None:
        values.append((metric, weight))


def _normalize_metric(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _ensure_tables(conn) -> None:
    """Ensure analytics tables exist via LabNotebook schema."""
    from research.scientist.notebook._shared import NOTEBOOK_SCHEMA

    conn.executescript(NOTEBOOK_SCHEMA)


def _ensure_generation_stats_columns(conn) -> None:
    expected = {
        "template_stats": (
            "avg_induction_screening_auc",
            "avg_binding_screening_auc",
            "avg_binding_screening_composite",
            "avg_ar_legacy_auc",
            "avg_hellaswag_acc",
            "avg_blimp_overall_accuracy",
            "avg_induction_intermediate_auc",
            "avg_binding_intermediate_auc",
            "math_space_rate",
        ),
        "op_stats": (
            "avg_induction_screening_auc",
            "avg_binding_screening_auc",
            "avg_binding_screening_composite",
            "avg_ar_legacy_auc",
            "avg_hellaswag_acc",
            "avg_blimp_overall_accuracy",
            "avg_induction_intermediate_auc",
            "avg_binding_intermediate_auc",
            "math_space_rate",
        ),
        "motif_stats": (
            "avg_induction_screening_auc",
            "avg_binding_screening_auc",
            "avg_binding_screening_composite",
            "avg_ar_legacy_auc",
            "avg_hellaswag_acc",
            "avg_blimp_overall_accuracy",
            "avg_induction_intermediate_auc",
            "avg_binding_intermediate_auc",
            "math_space_rate",
        ),
        "slot_stats": (
            "avg_induction_screening_auc",
            "avg_binding_screening_auc",
            "avg_binding_screening_composite",
            "avg_ar_legacy_auc",
            "avg_hellaswag_acc",
            "avg_blimp_overall_accuracy",
            "avg_induction_intermediate_auc",
            "avg_binding_intermediate_auc",
            "math_space_rate",
        ),
    }
    for table_name, columns in expected.items():
        existing = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for column in columns:
            if column in existing:
                continue
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} REAL")


def _load_stats_source_rows(db_path: str) -> List[Dict]:
    """Load canonical graph rows directly from notebook analysis data."""
    return load_deduped_graph_analysis_rows(db_path)


def _new_slot_outcome_bucket() -> dict:
    return {
        "n": 0,
        "s1": 0,
        "losses": [],
        "induction_screening_aucs": [],
        "binding_screening_aucs": [],
        "binding_screening_composites": [],
        "ar_legacy_aucs": [],
        "hellaswag_accs": [],
        "blimp_accuracies": [],
        "induction_intermediate_aucs": [],
        "binding_intermediate_aucs": [],
        "math_hits": [],
    }


def backfill(
    db_path: str = "research/runs.db",
    *,
    conn=None,
) -> Dict[str, int]:
    """Backfill analytics tables. Returns row counts inserted."""
    owns_connection = conn is None
    if conn is None:
        conn = connect_writer(Path(db_path))
    conn.execute("PRAGMA busy_timeout=15000")
    _ensure_tables(conn)
    _ensure_generation_stats_columns(conn)

    now = time.time()

    # Metric lists store (value, recency_weight) samples. Counts remain raw so
    # support thresholds still reflect actual observations.
    # [eval, s0, s1, losses, novelties, induction_screening_aucs, binding_screening_aucs,
    #  binding_screening_composites, ar_legacy_aucs, hellaswag_accs, blimp_accuracies,
    #  induction_intermediate_aucs, binding_intermediate_aucs, math_space_samples]
    tpl_data: Dict[str, list] = {}
    # same + co_occurrence counter at the end
    op_data: Dict[str, list] = {}
    # same + best template / best loss at the end
    motif_data: Dict[str, list] = {}
    slot_data: Dict[str, dict] = {}

    rows = _load_stats_source_rows(db_path)

    for row in rows:
        graph_json = str(row.get("graph_json") or "")
        if not graph_json:
            continue
        templates, motifs, ops, slot_usage = _extract_graph_info(graph_json)
        s0_pass = 1 if row.get("stage0_any_passed") else 0
        s1_pass = 1 if row.get("stage1_any_passed") else 0
        loss_ratio = _normalize_metric(row.get("loss_ratio"))
        novelty = _normalize_metric(row.get("novelty_score"))
        induction_screening_auc = _normalize_metric(row.get("induction_screening_auc"))
        binding_screening_auc = _normalize_metric(row.get("binding_screening_auc"))
        binding_screening_composite = _normalize_metric(
            row.get("binding_screening_composite")
        )
        ar_legacy_auc = _normalize_metric(row.get("ar_legacy_auc"))
        hellaswag_acc = _normalize_metric(row.get("hellaswag_acc"))
        blimp_accuracy = _normalize_metric(row.get("blimp_overall_accuracy"))
        induction_intermediate_auc = _normalize_metric(
            row.get("induction_intermediate_auc")
        )
        binding_intermediate_auc = _normalize_metric(
            row.get("binding_intermediate_auc")
        )
        math_space = 1 if row.get("graph_uses_math_spaces") else 0
        recency_weight = _recency_weight(
            row.get("latest_timestamp") or row.get("timestamp"),
            now,
        )

        for tpl in templates:
            if tpl not in tpl_data:
                tpl_data[tpl] = [0, 0, 0, [], [], [], [], [], [], [], [], [], [], []]
            d = tpl_data[tpl]
            d[0] += 1
            d[1] += s0_pass
            d[2] += s1_pass
            _append_weighted(d[3], loss_ratio, recency_weight)
            _append_weighted(d[4], novelty, recency_weight)
            _append_weighted(d[5], induction_screening_auc, recency_weight)
            _append_weighted(d[6], binding_screening_auc, recency_weight)
            _append_weighted(d[7], binding_screening_composite, recency_weight)
            _append_weighted(d[8], ar_legacy_auc, recency_weight)
            _append_weighted(d[9], hellaswag_acc, recency_weight)
            _append_weighted(d[10], blimp_accuracy, recency_weight)
            _append_weighted(d[11], induction_intermediate_auc, recency_weight)
            _append_weighted(d[12], binding_intermediate_auc, recency_weight)
            _append_weighted(d[13], math_space, recency_weight)

        op_set = set(ops)
        for op in op_set:
            if op not in op_data:
                op_data[op] = [
                    0,
                    0,
                    0,
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    Counter(),
                ]
            d = op_data[op]
            d[0] += 1
            d[1] += s0_pass
            d[2] += s1_pass
            _append_weighted(d[3], loss_ratio, recency_weight)
            _append_weighted(d[4], novelty, recency_weight)
            _append_weighted(d[5], induction_screening_auc, recency_weight)
            _append_weighted(d[6], binding_screening_auc, recency_weight)
            _append_weighted(d[7], binding_screening_composite, recency_weight)
            _append_weighted(d[8], ar_legacy_auc, recency_weight)
            _append_weighted(d[9], hellaswag_acc, recency_weight)
            _append_weighted(d[10], blimp_accuracy, recency_weight)
            _append_weighted(d[11], induction_intermediate_auc, recency_weight)
            _append_weighted(d[12], binding_intermediate_auc, recency_weight)
            _append_weighted(d[13], math_space, recency_weight)

        for a, b in itertools.combinations(op_set, 2):
            op_data[a][14][b] += 1
            op_data[b][14][a] += 1

        for motif in motifs:
            if motif not in motif_data:
                motif_data[motif] = [
                    0,
                    0,
                    0,
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    None,
                    float("inf"),
                ]
            d = motif_data[motif]
            d[0] += 1
            d[1] += s0_pass
            d[2] += s1_pass
            if loss_ratio is not None:
                _append_weighted(d[3], loss_ratio, recency_weight)
                if loss_ratio < d[15]:
                    d[15] = loss_ratio
                    d[14] = templates[0] if templates else None
            _append_weighted(d[4], novelty, recency_weight)
            _append_weighted(d[5], induction_screening_auc, recency_weight)
            _append_weighted(d[6], binding_screening_auc, recency_weight)
            _append_weighted(d[7], binding_screening_composite, recency_weight)
            _append_weighted(d[8], ar_legacy_auc, recency_weight)
            _append_weighted(d[9], hellaswag_acc, recency_weight)
            _append_weighted(d[10], blimp_accuracy, recency_weight)
            _append_weighted(d[11], induction_intermediate_auc, recency_weight)
            _append_weighted(d[12], binding_intermediate_auc, recency_weight)
            _append_weighted(d[13], math_space, recency_weight)

        for slot in slot_usage:
            if not isinstance(slot, dict):
                continue
            tpl_name = slot.get("template_name", "unknown")
            slot_idx = slot.get("slot_index", 0)
            sk = f"{tpl_name}.slot{slot_idx}"
            motif_cls = slot.get("selected_motif_class")
            is_wc = bool(slot.get("wildcard"))

            if sk not in slot_data:
                slot_data[sk] = {
                    "eval": 0,
                    "s1": 0,
                    "losses": [],
                    "induction_screening_aucs": [],
                    "binding_screening_aucs": [],
                    "binding_screening_composites": [],
                    "ar_legacy_aucs": [],
                    "hellaswag_accs": [],
                    "blimp_accuracies": [],
                    "induction_intermediate_aucs": [],
                    "binding_intermediate_aucs": [],
                    "math_hits": [],
                    "class_outcomes": {},
                    "wc_count": 0,
                    "wc_s1": 0,
                    "wc_class_outcomes": {},
                    "template_name": tpl_name,
                    "slot_index": slot_idx,
                    "slot_classes": slot.get("slot_classes", []),
                }
            sd = slot_data[sk]
            sd["eval"] += 1
            sd["s1"] += s1_pass
            _append_weighted(sd["losses"], loss_ratio, recency_weight)
            _append_weighted(
                sd["induction_screening_aucs"], induction_screening_auc, recency_weight
            )
            _append_weighted(
                sd["binding_screening_aucs"], binding_screening_auc, recency_weight
            )
            _append_weighted(
                sd["binding_screening_composites"],
                binding_screening_composite,
                recency_weight,
            )
            _append_weighted(sd["ar_legacy_aucs"], ar_legacy_auc, recency_weight)
            _append_weighted(sd["hellaswag_accs"], hellaswag_acc, recency_weight)
            _append_weighted(sd["blimp_accuracies"], blimp_accuracy, recency_weight)
            _append_weighted(
                sd["induction_intermediate_aucs"],
                induction_intermediate_auc,
                recency_weight,
            )
            _append_weighted(
                sd["binding_intermediate_aucs"],
                binding_intermediate_auc,
                recency_weight,
            )
            _append_weighted(sd["math_hits"], math_space, recency_weight)

            if motif_cls:
                co = sd["wc_class_outcomes"] if is_wc else sd["class_outcomes"]
                bucket = co.setdefault(motif_cls, _new_slot_outcome_bucket())
                bucket["n"] += 1
                bucket["s1"] += s1_pass
                _append_weighted(bucket["losses"], loss_ratio, recency_weight)
                _append_weighted(
                    bucket["induction_screening_aucs"],
                    induction_screening_auc,
                    recency_weight,
                )
                _append_weighted(
                    bucket["binding_screening_aucs"],
                    binding_screening_auc,
                    recency_weight,
                )
                _append_weighted(
                    bucket["binding_screening_composites"],
                    binding_screening_composite,
                    recency_weight,
                )
                _append_weighted(
                    bucket["ar_legacy_aucs"], ar_legacy_auc, recency_weight
                )
                _append_weighted(
                    bucket["hellaswag_accs"],
                    hellaswag_acc,
                    recency_weight,
                )
                _append_weighted(
                    bucket["blimp_accuracies"],
                    blimp_accuracy,
                    recency_weight,
                )
                _append_weighted(
                    bucket["induction_intermediate_aucs"],
                    induction_intermediate_auc,
                    recency_weight,
                )
                _append_weighted(
                    bucket["binding_intermediate_aucs"],
                    binding_intermediate_auc,
                    recency_weight,
                )
                _append_weighted(bucket["math_hits"], math_space, recency_weight)

            if is_wc:
                sd["wc_count"] += 1
                sd["wc_s1"] += s1_pass

    conn.execute("DELETE FROM template_stats")
    for tpl, (
        ev,
        s0,
        s1,
        losses,
        novs,
        inds,
        binds,
        bind_comp,
        ars,
        hellaswags,
        blimps,
        inds_v2,
        binds_v2,
        math_hits,
    ) in tpl_data.items():
        conn.execute(
            """INSERT INTO template_stats
               (template_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty,
                avg_induction_screening_auc, avg_binding_screening_auc, avg_binding_screening_composite,
                avg_ar_legacy_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                avg_induction_intermediate_auc, avg_binding_intermediate_auc,
                math_space_rate, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tpl,
                ev,
                s0,
                s1,
                _mean_or_none(losses),
                _min_or_none(losses),
                _safe_std(losses) if losses else None,
                _mean_or_none(novs),
                _mean_or_none(inds),
                _mean_or_none(binds),
                _mean_or_none(bind_comp),
                _mean_or_none(ars),
                _mean_or_none(hellaswags),
                _mean_or_none(blimps),
                _mean_or_none(inds_v2),
                _mean_or_none(binds_v2),
                _mean_or_none(math_hits),
                now,
            ),
        )

    conn.execute("DELETE FROM op_stats")
    for op, (
        ev,
        s0,
        s1,
        losses,
        novs,
        inds,
        binds,
        bind_comp,
        ars,
        hellaswags,
        blimps,
        inds_v2,
        binds_v2,
        math_hits,
        co_counter,
    ) in op_data.items():
        top20 = dict(co_counter.most_common(20))
        conn.execute(
            """INSERT INTO op_stats
               (op_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty,
                avg_induction_screening_auc, avg_binding_screening_auc, avg_binding_screening_composite,
                avg_ar_legacy_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                avg_induction_intermediate_auc, avg_binding_intermediate_auc,
                math_space_rate, co_occurrence_json, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                op,
                ev,
                s0,
                s1,
                _mean_or_none(losses),
                _min_or_none(losses),
                _safe_std(losses) if losses else None,
                _mean_or_none(novs),
                _mean_or_none(inds),
                _mean_or_none(binds),
                _mean_or_none(bind_comp),
                _mean_or_none(ars),
                _mean_or_none(hellaswags),
                _mean_or_none(blimps),
                _mean_or_none(inds_v2),
                _mean_or_none(binds_v2),
                _mean_or_none(math_hits),
                json.dumps(top20) if top20 else None,
                now,
            ),
        )

    conn.execute("DELETE FROM motif_stats")
    for motif, (
        ev,
        s0,
        s1,
        losses,
        novs,
        inds,
        binds,
        bind_comp,
        ars,
        hellaswags,
        blimps,
        inds_v2,
        binds_v2,
        math_hits,
        best_tpl,
        _,
    ) in motif_data.items():
        conn.execute(
            """INSERT INTO motif_stats
               (motif_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty,
                avg_induction_screening_auc, avg_binding_screening_auc, avg_binding_screening_composite,
                avg_ar_legacy_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                avg_induction_intermediate_auc, avg_binding_intermediate_auc,
                math_space_rate, best_template, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                motif,
                ev,
                s0,
                s1,
                _mean_or_none(losses),
                _min_or_none(losses),
                _safe_std(losses) if losses else None,
                _mean_or_none(novs),
                _mean_or_none(inds),
                _mean_or_none(binds),
                _mean_or_none(bind_comp),
                _mean_or_none(ars),
                _mean_or_none(hellaswags),
                _mean_or_none(blimps),
                _mean_or_none(inds_v2),
                _mean_or_none(binds_v2),
                _mean_or_none(math_hits),
                best_tpl,
                now,
            ),
        )

    conn.execute("DELETE FROM slot_stats")
    for sk, sd in slot_data.items():

        def _summarize_outcomes(outcomes: dict) -> dict:
            out = {}
            for cls, vals in outcomes.items():
                out[cls] = {
                    "n": vals["n"],
                    "s1": vals["s1"],
                    "mean_loss": _mean_or_none(vals["losses"]),
                    "mean_induction_screening_auc": _mean_or_none(
                        vals["induction_screening_aucs"]
                    ),
                    "mean_binding_screening_auc": _mean_or_none(
                        vals["binding_screening_aucs"]
                    ),
                    "mean_binding_screening_composite": _mean_or_none(
                        vals["binding_screening_composites"]
                    ),
                    "mean_ar_legacy_auc": _mean_or_none(vals["ar_legacy_aucs"]),
                    "mean_hellaswag_acc": _mean_or_none(vals["hellaswag_accs"]),
                    "mean_blimp_overall_accuracy": _mean_or_none(
                        vals["blimp_accuracies"]
                    ),
                    "mean_induction_intermediate_auc": _mean_or_none(
                        vals["induction_intermediate_aucs"]
                    ),
                    "mean_binding_intermediate_auc": _mean_or_none(
                        vals["binding_intermediate_aucs"]
                    ),
                    "math_space_rate": _mean_or_none(vals["math_hits"]),
                }
            return out

        losses = sd["losses"]
        conn.execute(
            """INSERT INTO slot_stats
               (slot_key, template_name, slot_index, slot_classes,
                eval_count, s1_pass_count, mean_loss, min_loss,
                avg_induction_screening_auc, avg_binding_screening_auc, avg_binding_screening_composite,
                avg_ar_legacy_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                avg_induction_intermediate_auc, avg_binding_intermediate_auc,
                math_space_rate, class_outcomes, wildcard_count,
                wildcard_s1_count, wildcard_class_outcomes, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sk,
                sd["template_name"],
                sd["slot_index"],
                json.dumps(sd["slot_classes"]),
                sd["eval"],
                sd["s1"],
                _mean_or_none(losses),
                _min_or_none(losses),
                _mean_or_none(sd["induction_screening_aucs"]),
                _mean_or_none(sd["binding_screening_aucs"]),
                _mean_or_none(sd["binding_screening_composites"]),
                _mean_or_none(sd["ar_legacy_aucs"]),
                _mean_or_none(sd["hellaswag_accs"]),
                _mean_or_none(sd["blimp_accuracies"]),
                _mean_or_none(sd["induction_intermediate_aucs"]),
                _mean_or_none(sd["binding_intermediate_aucs"]),
                _mean_or_none(sd["math_hits"]),
                json.dumps(_summarize_outcomes(sd["class_outcomes"])),
                sd["wc_count"],
                sd["wc_s1"],
                json.dumps(_summarize_outcomes(sd["wc_class_outcomes"])),
                now,
            ),
        )

    conn.commit()
    if owns_connection:
        conn.close()

    counts = {
        "template_stats": len(tpl_data),
        "op_stats": len(op_data),
        "motif_stats": len(motif_data),
        "slot_stats": len(slot_data),
    }
    print(f"Backfilled: {counts}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill analytics stats tables")
    parser.add_argument("--db", default="research/runs.db")
    parser.add_argument(
        "--refresh-models",
        action="store_true",
        help="Also retrain ML predictors (Bayesian tracker, graph predictor) after stats backfill",
    )
    args = parser.parse_args()
    nb, exp_id = start_script_experiment(
        db_path=args.db,
        experiment_type="analytics_backfill",
        config={"db": args.db, "refresh_models": bool(args.refresh_models)},
        source_script="backfill_stats",
        hypothesis="Backfill analytics stats tables",
    )
    try:
        counts = backfill(args.db)

        if args.refresh_models:
            print("\nRefreshing ML models...")
            try:
                from research.tools.train_predictors import (
                    train_bayesian,
                    train_graph_predictor,
                )

                train_bayesian(save=True)
                print("  Bayesian tracker refreshed")
                train_graph_predictor()
                print("  Graph predictor refreshed")
            except Exception as e:
                print(f"  ML model refresh failed: {e}")
                counts["model_refresh_error"] = str(e)

        complete_script_experiment(
            nb,
            exp_id,
            results={**counts, "refresh_models": bool(args.refresh_models)},
            summary=(
                f"Analytics backfill complete: templates={counts['template_stats']} "
                f"ops={counts['op_stats']}"
            ),
        )
    except KeyboardInterrupt:
        fail_script_experiment(nb, exp_id, error="KeyboardInterrupt")
        nb.close()
        raise
    except Exception as exc:
        fail_script_experiment(nb, exp_id, error=str(exc))
        nb.close()
        raise
    nb.close()


if __name__ == "__main__":
    main()
