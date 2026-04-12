"""Backfill template_stats, op_stats, motif_stats from the deduped training corpus.

Usage:
    python -m research.tools.backfill_stats [--db research/lab_notebook.db]

Reads the shared deduped ML corpus, supplements it with any canonical graphs
present in the notebook but missing from the corpus snapshot, then extracts
templates_used/motifs_used/op names and populates the analytics tables with
structural-unique statistics. Idempotent and safe to re-run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sqlite3
import time
from collections import Counter
from typing import Dict, List, Tuple

from research.scientist.intelligence.ml_corpus import (
    _fallback_graph_analysis_rows,
    load_deduped_graph_training_rows,
)
from research.tools._script_audit import (
    complete_script_experiment,
    fail_script_experiment,
    start_script_experiment,
)


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


def _safe_std(values: List[float]) -> float:
    """Standard deviation, or 0.0 if fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _mean_or_none(values: List[float]):
    return sum(values) / len(values) if values else None


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Ensure analytics tables exist via LabNotebook schema."""
    from research.scientist.notebook._shared import NOTEBOOK_SCHEMA

    conn.executescript(NOTEBOOK_SCHEMA)


def _load_stats_source_rows(db_path: str) -> List[Dict]:
    """Combine corpus-backed rows with notebook-backed rows for missing canonicals.

    The deduped ML corpus is the preferred source because it applies training
    eligibility filters, but it can lag behind active notebook families. For
    analytics observability we union in canonical graphs present in
    ``program_results`` but absent from the corpus snapshot.
    """
    corpus_rows = load_deduped_graph_training_rows(db_path)
    rows_by_canonical: Dict[str, Dict] = {}
    for row in corpus_rows:
        canonical = str(row.get("canonical_fingerprint") or "")
        if canonical:
            rows_by_canonical[canonical] = dict(row)

    for row in _fallback_graph_analysis_rows(db_path):
        canonical = str(row.get("canonical_fingerprint") or "")
        if not canonical or canonical in rows_by_canonical:
            continue
        rows_by_canonical[canonical] = {
            "canonical_fingerprint": canonical,
            "graph_json": row.get("graph_json"),
            "stage0_any_passed": row.get("stage0_any_passed"),
            "stage1_any_passed": row.get("stage1_any_passed"),
            "loss_ratio_best": row.get("loss_ratio"),
            "n_rows": row.get("n_rows", 1),
        }

    return list(rows_by_canonical.values())


def backfill(db_path: str = "research/lab_notebook.db") -> Dict[str, int]:
    """Backfill analytics tables. Returns row counts inserted."""
    conn = sqlite3.connect(db_path, timeout=15.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    _ensure_tables(conn)

    now = time.time()

    # Accumulators — using lists for losses/novelties, counters for co-occurrence
    tpl_data: Dict[str, list] = {}  # [eval, s0, s1, [losses], [novelties]]
    op_data: Dict[str, list] = {}  # [eval, s0, s1, [losses], [novelties], Counter]
    motif_data: Dict[
        str, list
    ] = {}  # [eval, s0, s1, [losses], [novelties], best_tpl, best_loss]
    # Slot stats: slot_key → {eval, s1, [losses], class_outcomes, wc_count, wc_s1, wc_class_outcomes, template_name, slot_index, slot_classes}
    slot_data: Dict[str, dict] = {}

    rows = _load_stats_source_rows(db_path)

    for row in rows:
        graph_json = str(row.get("graph_json") or "")
        if not graph_json:
            continue
        templates, motifs, ops, slot_usage = _extract_graph_info(graph_json)
        s0_pass = 1 if row.get("stage0_any_passed") else 0
        s1_pass = 1 if row.get("stage1_any_passed") else 0
        loss_ratio = row.get("loss_ratio_best")
        novelty = None
        valid_loss = loss_ratio is not None and math.isfinite(float(loss_ratio))
        valid_nov = novelty is not None and math.isfinite(float(novelty))

        for tpl in templates:
            if tpl not in tpl_data:
                tpl_data[tpl] = [0, 0, 0, [], []]
            d = tpl_data[tpl]
            d[0] += 1
            d[1] += s0_pass
            d[2] += s1_pass
            if valid_loss:
                d[3].append(loss_ratio)
            if valid_nov:
                d[4].append(novelty)

        op_set = set(ops)
        for op in op_set:
            if op not in op_data:
                op_data[op] = [0, 0, 0, [], [], Counter()]
            d = op_data[op]
            d[0] += 1
            d[1] += s0_pass
            d[2] += s1_pass
            if valid_loss:
                d[3].append(loss_ratio)
            if valid_nov:
                d[4].append(novelty)

        # Co-occurrence: iterate pairs once via combinations (not O(n²) nested loop)
        for a, b in itertools.combinations(op_set, 2):
            op_data[a][5][b] += 1
            op_data[b][5][a] += 1

        for motif in motifs:
            if motif not in motif_data:
                motif_data[motif] = [0, 0, 0, [], [], None, float("inf")]
            d = motif_data[motif]
            d[0] += 1
            d[1] += s0_pass
            d[2] += s1_pass
            if valid_loss:
                d[3].append(loss_ratio)
                if loss_ratio < d[6]:
                    d[6] = loss_ratio
                    d[5] = templates[0] if templates else None
            if valid_nov:
                d[4].append(novelty)

        # Accumulate slot-level stats
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
            if valid_loss:
                sd["losses"].append(loss_ratio)

            if motif_cls:
                # Track per-class outcomes
                co = sd["wc_class_outcomes"] if is_wc else sd["class_outcomes"]
                if motif_cls not in co:
                    co[motif_cls] = {"n": 0, "s1": 0, "losses": []}
                co[motif_cls]["n"] += 1
                co[motif_cls]["s1"] += s1_pass
                if valid_loss:
                    co[motif_cls]["losses"].append(loss_ratio)

            if is_wc:
                sd["wc_count"] += 1
                sd["wc_s1"] += s1_pass

    # Write template_stats
    conn.execute("DELETE FROM template_stats")
    for tpl, (ev, s0, s1, losses, novs) in tpl_data.items():
        conn.execute(
            """INSERT INTO template_stats
               (template_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tpl,
                ev,
                s0,
                s1,
                _mean_or_none(losses),
                min(losses) if losses else None,
                _safe_std(losses) if losses else None,
                _mean_or_none(novs),
                now,
            ),
        )

    # Write op_stats
    conn.execute("DELETE FROM op_stats")
    for op, (ev, s0, s1, losses, novs, co_counter) in op_data.items():
        top20 = dict(co_counter.most_common(20))
        conn.execute(
            """INSERT INTO op_stats
               (op_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty,
                co_occurrence_json, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                op,
                ev,
                s0,
                s1,
                _mean_or_none(losses),
                min(losses) if losses else None,
                _safe_std(losses) if losses else None,
                _mean_or_none(novs),
                json.dumps(top20) if top20 else None,
                now,
            ),
        )

    # Write motif_stats
    conn.execute("DELETE FROM motif_stats")
    for motif, (ev, s0, s1, losses, novs, best_tpl, _) in motif_data.items():
        conn.execute(
            """INSERT INTO motif_stats
               (motif_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty,
                best_template, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                motif,
                ev,
                s0,
                s1,
                _mean_or_none(losses),
                min(losses) if losses else None,
                _safe_std(losses) if losses else None,
                _mean_or_none(novs),
                best_tpl,
                now,
            ),
        )

    # Write slot_stats
    conn.execute("DELETE FROM slot_stats")
    for sk, sd in slot_data.items():
        # Summarize class_outcomes: replace loss lists with mean_loss
        def _summarize_outcomes(outcomes: dict) -> dict:
            out = {}
            for cls, vals in outcomes.items():
                out[cls] = {
                    "n": vals["n"],
                    "s1": vals["s1"],
                    "mean_loss": _mean_or_none(vals["losses"]),
                }
            return out

        losses = sd["losses"]
        conn.execute(
            """INSERT INTO slot_stats
               (slot_key, template_name, slot_index, slot_classes,
                eval_count, s1_pass_count, mean_loss, min_loss,
                class_outcomes, wildcard_count, wildcard_s1_count,
                wildcard_class_outcomes, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sk,
                sd["template_name"],
                sd["slot_index"],
                json.dumps(sd["slot_classes"]),
                sd["eval"],
                sd["s1"],
                _mean_or_none(losses),
                min(losses) if losses else None,
                json.dumps(_summarize_outcomes(sd["class_outcomes"])),
                sd["wc_count"],
                sd["wc_s1"],
                json.dumps(_summarize_outcomes(sd["wc_class_outcomes"])),
                now,
            ),
        )

    conn.commit()
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
    parser.add_argument("--db", default="research/lab_notebook.db")
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
