#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
import random
from typing import Any

from research.scientist.notebook.notebook_misc import _MiscMixin
from research.synthesis.graph import ComputationGraph
from research.synthesis.templates import TEMPLATES
from research.synthesis.validator import validate_graph
from research.tools.backfill_templates import _NON_ROUTING_TEMPLATES

DEFAULT_NOTEBOOK_DB = Path("research/lab_notebook.db")
DEFAULT_PROFILING_DB = Path("research/profiling/component_profiles.db")
DEFAULT_OUT = Path(
    "research/reports/template_component_priors/template_component_priors.csv"
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _extract_graph_paths(
    graph: ComputationGraph,
) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str, str]]]:
    op_names: list[str] = []
    pairs: set[tuple[str, str]] = set()
    triplets: set[tuple[str, str, str]] = set()

    topo = graph.topological_order()
    for node_id in topo:
        node = graph.nodes[node_id]
        if node.is_input:
            continue
        op_names.append(node.op_name)
        for parent_id in node.input_ids:
            parent = graph.nodes[parent_id]
            if parent.is_input:
                continue
            pairs.add((parent.op_name, node.op_name))
            for grandparent_id in parent.input_ids:
                grandparent = graph.nodes[grandparent_id]
                if grandparent.is_input:
                    continue
                triplets.add((grandparent.op_name, parent.op_name, node.op_name))
    return op_names, sorted(pairs), sorted(triplets)


def _load_op_profiles(profiling_db: Path) -> dict[str, dict[str, float]]:
    if not profiling_db.exists():
        return {}
    conn = sqlite3.connect(str(profiling_db), timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT op_name, forward_time_us, grad_exploding, grad_vanishing,
                   output_has_nan, grad_has_nan, jacobian_spectral_norm
            FROM op_profiles
            WHERE error IS NULL
            """
        ).fetchall()
    finally:
        conn.close()
    return {
        str(row["op_name"]): {
            "forward_time_us": float(row["forward_time_us"] or 0.0),
            "grad_exploding": float(row["grad_exploding"] or 0.0),
            "grad_vanishing": float(row["grad_vanishing"] or 0.0),
            "output_has_nan": float(row["output_has_nan"] or 0.0),
            "grad_has_nan": float(row["grad_has_nan"] or 0.0),
            "jacobian_spectral_norm": float(row["jacobian_spectral_norm"] or 0.0),
        }
        for row in rows
    }


def _load_pair_profiles(profiling_db: Path) -> dict[tuple[str, str], dict[str, float]]:
    if not profiling_db.exists():
        return {}
    conn = sqlite3.connect(str(profiling_db), timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT op_a, op_b,
                   (output_has_nan = 0 AND grad_has_nan = 0 AND grad_vanishing = 0) AS stable,
                   grad_exploding, jacobian_spectral_norm
            FROM pair_profiles
            WHERE error IS NULL AND composition = 'sequential'
            """
        ).fetchall()
    finally:
        conn.close()
    return {
        (str(row["op_a"]), str(row["op_b"])): {
            "stable": float(row["stable"] or 0.0),
            "grad_exploding": float(row["grad_exploding"] or 0.0),
            "jacobian_spectral_norm": float(row["jacobian_spectral_norm"] or 0.0),
        }
        for row in rows
    }


def _load_triplet_profiles(
    profiling_db: Path,
) -> dict[tuple[str, str, str], dict[str, float]]:
    if not profiling_db.exists():
        return {}
    conn = sqlite3.connect(str(profiling_db), timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT op_a, op_b, op_c, triplet_stable, grad_exploding, lipschitz_estimate
            FROM triplet_profiles
            WHERE error IS NULL
            """
        ).fetchall()
    finally:
        conn.close()
    return {
        (str(row["op_a"]), str(row["op_b"]), str(row["op_c"])): {
            "stable": float(row["triplet_stable"] or 0.0),
            "grad_exploding": float(row["grad_exploding"] or 0.0),
            "lipschitz_estimate": float(row["lipschitz_estimate"] or 0.0),
        }
        for row in rows
    }


def _persisted_template_counts(notebook_db: Path) -> dict[str, int]:
    if not notebook_db.exists():
        return {}
    conn = sqlite3.connect(str(notebook_db), timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT template_name, COUNT(*) AS n
            FROM program_graph_features
            WHERE template_name IS NOT NULL AND TRIM(template_name) <> ''
            GROUP BY template_name
            """
        ).fetchall()
    finally:
        conn.close()
    return {str(row["template_name"]): int(row["n"] or 0) for row in rows}


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _build_template_row(
    name: str,
    op_profiles: dict[str, dict[str, float]],
    pair_profiles: dict[tuple[str, str], dict[str, float]],
    triplet_profiles: dict[tuple[str, str, str], dict[str, float]],
    persisted_counts: dict[str, int],
) -> dict[str, Any]:
    graph = ComputationGraph(model_dim=256)
    inp = graph.add_input()
    out = TEMPLATES[name](graph, inp, random.Random(42), None)
    graph.set_output(out)
    validation = validate_graph(graph, max_ops=24, max_depth=18)
    ops, pairs, triplets = _extract_graph_paths(graph)
    slot_usage = graph.metadata.get("template_slot_usage") or []

    covered_ops = [op for op in ops if op in op_profiles]
    covered_pairs = [pair for pair in pairs if pair in pair_profiles]
    covered_triplets = [triplet for triplet in triplets if triplet in triplet_profiles]

    op_forward = [op_profiles[op]["forward_time_us"] for op in covered_ops]
    op_exploding = [op_profiles[op]["grad_exploding"] for op in covered_ops]
    pair_stability = [pair_profiles[pair]["stable"] for pair in covered_pairs]
    triplet_stability = [
        triplet_profiles[triplet]["stable"] for triplet in covered_triplets
    ]

    unstable_ops = sorted(
        {op for op in covered_ops if op_profiles[op]["grad_exploding"] > 0}
    )
    unstable_pairs = sorted(
        f"{a}->{b}" for (a, b) in covered_pairs if pair_profiles[(a, b)]["stable"] < 1.0
    )

    flags: list[str] = []
    if not validation.valid:
        flags.append("validator_invalid")
    if len(slot_usage) == 0:
        flags.append("no_slot_observability")
    if len(covered_pairs) < max(1, len(pairs) // 3):
        flags.append("weak_pair_profile_coverage")
    if len(covered_triplets) == 0 and len(triplets) > 0:
        flags.append("no_triplet_profile_coverage")
    if unstable_ops:
        flags.append("contains_grad_exploding_ops")
    if unstable_pairs:
        flags.append("contains_unstable_pairs")
    if persisted_counts.get(name, 0) == 0:
        flags.append("no_persisted_runs")

    return {
        "template_name": name,
        "registered": True,
        "non_routing_backfill": name in _NON_ROUTING_TEMPLATES,
        "persisted_graph_feature_rows": int(persisted_counts.get(name, 0)),
        "slot_count_declared": int(
            _MiscMixin._infer_template_slot_counts().get(name, 0) or 0
        ),
        "slot_count_emitted": len(slot_usage),
        "valid_screening_graph": bool(validation.valid),
        "validator_errors": "|".join(validation.errors or []),
        "n_ops": graph.n_ops(),
        "depth": graph.depth(),
        "n_unique_ops": len(set(ops)),
        "n_pairs": len(pairs),
        "n_triplets": len(triplets),
        "op_profile_coverage": round(len(covered_ops) / max(1, len(ops)), 3),
        "pair_profile_coverage": round(len(covered_pairs) / max(1, len(pairs)), 3),
        "triplet_profile_coverage": round(
            len(covered_triplets) / max(1, len(triplets)), 3
        ),
        "mean_op_forward_time_us": _mean(op_forward),
        "mean_op_grad_exploding": _mean(op_exploding),
        "mean_pair_stability": _mean(pair_stability),
        "min_pair_stability": min(pair_stability) if pair_stability else None,
        "mean_triplet_stability": _mean(triplet_stability),
        "unstable_ops": "|".join(unstable_ops),
        "unstable_pairs": "|".join(unstable_pairs),
        "ops": "|".join(ops),
        "flags": "|".join(flags),
    }


def build_template_component_prior_rows(
    template_names: list[str],
    *,
    notebook_db: Path = DEFAULT_NOTEBOOK_DB,
    profiling_db: Path = DEFAULT_PROFILING_DB,
) -> list[dict[str, Any]]:
    op_profiles = _load_op_profiles(profiling_db)
    pair_profiles = _load_pair_profiles(profiling_db)
    triplet_profiles = _load_triplet_profiles(profiling_db)
    persisted_counts = _persisted_template_counts(notebook_db)
    return [
        _build_template_row(
            name, op_profiles, pair_profiles, triplet_profiles, persisted_counts
        )
        for name in template_names
        if name in TEMPLATES
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report template scaffold priors from component observability data"
    )
    parser.add_argument(
        "--templates", nargs="*", help="Specific template names to score"
    )
    parser.add_argument("--notebook-db", default=str(DEFAULT_NOTEBOOK_DB))
    parser.add_argument("--profiling-db", default=str(DEFAULT_PROFILING_DB))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    template_names = args.templates or sorted(TEMPLATES.keys())
    rows = build_template_component_prior_rows(
        template_names,
        notebook_db=Path(args.notebook_db),
        profiling_db=Path(args.profiling_db),
    )
    out_path = Path(args.out)
    _write_csv(out_path, rows)
    print(out_path)
    print(json.dumps(rows[: min(10, len(rows))], indent=2))


if __name__ == "__main__":
    main()
