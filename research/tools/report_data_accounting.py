#!/usr/bin/env python3
"""Report notebook accounting with explicit row/run/graph separation."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from research.scientist.intelligence.ml_corpus import (
    _graph_fingerprint,
    load_deduped_graph_training_rows,
    load_deduped_predictor_training_rows,
    load_deduped_screening_predictor_rows,
)
from research.scientist.notebook import LabNotebook


def _fingerprint_mismatch_summary(db_path: str) -> dict:
    conn = sqlite3.connect(db_path, timeout=10.0)
    rows = conn.execute(
        """
        SELECT result_id, graph_fingerprint, graph_json
        FROM program_results
        WHERE TRIM(COALESCE(graph_json, '')) <> ''
          AND graph_json <> '{}'
        """
    ).fetchall()
    conn.close()

    checked = 0
    mismatch = 0
    examples: list[dict[str, str]] = []
    for result_id, stored_fp, graph_json in rows:
        checked += 1
        canonical = _graph_fingerprint(str(graph_json))
        if str(stored_fp or "") != canonical:
            mismatch += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "result_id": str(result_id or ""),
                        "stored_graph_fingerprint": str(stored_fp or ""),
                        "recomputed_canonical_fingerprint": canonical,
                    }
                )
    return {
        "checked_rows": checked,
        "mismatch_rows": mismatch,
        "mismatch_rate": (mismatch / checked) if checked else 0.0,
        "examples": examples,
    }


def build_report(db_path: str) -> dict:
    nb = LabNotebook(db_path)
    try:
        accounting = nb.get_data_accounting_summary()
    finally:
        nb.close()

    return {
        "db_path": db_path,
        "data_accounting": accounting,
        "predictor_corpora": {
            "graph_training_graphs": len(
                load_deduped_graph_training_rows(db_path, validate=True)
            ),
            "screening_predictor_graphs": len(
                load_deduped_screening_predictor_rows(db_path, validate=True)
            ),
            "predictor_training_graphs": len(
                load_deduped_predictor_training_rows(db_path, validate=True)
            ),
        },
        "graph_identity_audit": _fingerprint_mismatch_summary(db_path),
        "notes": [
            "training_curves is a per-step log table keyed by result_id, not a graph table",
            "program_results is a run-level table keyed by result_id",
            "canonical predictor corpora dedupe by recomputed canonical_fingerprint from graph_json",
        ],
    }


def _markdown_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Metric | Value |", "| --- | ---: |"]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    accounting = report["data_accounting"]
    row_volume = accounting["row_volume"]
    run_volume = accounting["run_volume"]
    graph_volume = accounting["graph_volume"]
    filtering = accounting["filtering"]
    curves = accounting["training_curve_density"]
    corpora = report["predictor_corpora"]
    identity = report["graph_identity_audit"]

    lines = [
        "# Data Accounting Audit",
        "",
        "This report separates raw log rows from run records and graph entities.",
        "",
        "## Core Counts",
        "",
        _markdown_table(
            [
                ("program_results rows", str(row_volume["program_result_rows"])),
                ("training_curves rows", str(row_volume["training_curve_rows"])),
                ("leaderboard rows", str(row_volume["leaderboard_rows"])),
                ("unique runs", str(run_volume["unique_runs"])),
                ("unique graphs", str(graph_volume["unique_graphs"])),
                (
                    "unique graph x protocol",
                    str(graph_volume["unique_graph_protocols"]),
                ),
                (
                    "unique graph x protocol x budget",
                    str(graph_volume["unique_graph_protocol_budgets"]),
                ),
            ]
        ),
        "",
        "## Filtering And Coverage",
        "",
        _markdown_table(
            [
                ("runs filtered pre-S0", str(filtering["runs_filtered_pre_s0"])),
                ("runs filtered pre-S0.5", str(filtering["runs_filtered_pre_s05"])),
                ("runs filtered pre-S1", str(filtering["runs_filtered_pre_s1"])),
                ("runs reaching S1 pass", str(filtering["runs_reaching_s1_pass"])),
                (
                    "graphs all filtered pre-S0",
                    str(filtering["graphs_all_filtered_pre_s0"]),
                ),
                (
                    "graphs all filtered pre-S0.5",
                    str(filtering["graphs_all_filtered_pre_s05"]),
                ),
                (
                    "graphs all filtered pre-S1",
                    str(filtering["graphs_all_filtered_pre_s1"]),
                ),
                ("graphs with any S1 pass", str(filtering["graphs_any_s1_pass"])),
                (
                    "trusted comparable graphs",
                    str(graph_volume["trusted_comparable_graphs"]),
                ),
                ("promotable graphs", str(graph_volume["promotable_graphs"])),
                (
                    "screening-model eligible graphs",
                    str(graph_volume["screening_model_eligible_graphs"]),
                ),
                (
                    "graphs with any downstream eval",
                    str(graph_volume["downstream_eval_graphs"]),
                ),
                (
                    "graphs with full downstream bundle",
                    str(graph_volume["downstream_full_bundle_graphs"]),
                ),
            ]
        ),
        "",
        "## Training Curve Density",
        "",
        _markdown_table(
            [
                ("runs with training_curves", str(curves["runs_with_training_curves"])),
                (
                    "runs without training_curves",
                    str(curves["runs_without_training_curves"]),
                ),
                (
                    "avg rows per run with curve",
                    f"{curves['avg_rows_per_run_with_curve']:.2f}",
                ),
                (
                    "median rows per run with curve",
                    f"{curves['median_rows_per_run_with_curve']:.2f}",
                ),
                (
                    "max rows per run with curve",
                    str(curves["max_rows_per_run_with_curve"]),
                ),
            ]
        ),
        "",
        "## Predictor Corpora",
        "",
        _markdown_table(
            [
                ("graph training corpus graphs", str(corpora["graph_training_graphs"])),
                (
                    "screening predictor graphs",
                    str(corpora["screening_predictor_graphs"]),
                ),
                (
                    "predictor training graphs",
                    str(corpora["predictor_training_graphs"]),
                ),
            ]
        ),
        "",
        "## Graph Identity Audit",
        "",
        _markdown_table(
            [
                ("rows checked", str(identity["checked_rows"])),
                ("stored fingerprint mismatches", str(identity["mismatch_rows"])),
                ("mismatch rate", f"{identity['mismatch_rate']:.4%}"),
            ]
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="research/lab_notebook.db")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = build_report(args.db)
    output = (
        json.dumps(report, indent=2, sort_keys=True)
        if args.json
        else render_markdown(report)
    )
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
