"""Miner V2 — novelty filter + Python code skeleton generation.

Companion to `mine_template_subpatterns.py` (V1). V1 emits ranked chains
that pass cohort baseline; V2 cross-references each chain against the
chains that *registered TEMPLATES already produce in their passing graphs*,
filters to genuinely novel patterns, and emits Python skeletons for the
top-K novel candidates so a human can adapt them into actual templates.

Pipeline (single read pass):
  1. Group passing-cohort graphs by `program_graph_features.template_name`.
  2. For each template, sample up to MAX_GRAPHS_PER_TEMPLATE graphs and
     extract their length-3/4 op chains.
  3. Build covered_chains: {template_name: set[chain_tuple]}.
  4. Read V1 output (mined_chain_proposals.csv); annotate each chain with
     covered_by = [templates that already produce it].
  5. Filter to novel chains (covered_by == [] or restricted to a single
     low-weight template).
  6. Emit research/reports/mined_novel_chain_proposals.json with skeletons.

Skeletons are NOT auto-applied. They live in the JSON for human review.

Outputs:
  research/reports/mined_novel_chain_proposals.json   (top-K novel + skeletons)
  research/reports/mined_chain_coverage.csv           (full annotation table)
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.tools.mine_template_subpatterns import (  # noqa: E402
    DEFAULT_CHAIN_LENGTHS,
    NANO_BIND,
    PASS_SA,
    STRUCTURAL_OPS,
    extract_chains,
    parse_graph,
)

LAB = REPO / "research/lab_notebook.db"
REPORTS = REPO / "research/reports"

V1_OUTPUT = REPORTS / "mined_chain_proposals.csv"

MAX_GRAPHS_PER_TEMPLATE = 100


def fetch_template_grouped() -> dict[str, list[str]]:
    """Return {template_name: [graph_json, ...]} for passing-cohort graphs."""
    conn = sqlite3.connect(f"file:{LAB}?mode=ro&immutable=0", uri=True)
    cur = conn.execute(
        f"""
        SELECT pgf.template_name, pr.graph_json
        FROM program_graph_features pgf
        JOIN program_results pr ON pr.result_id = pgf.result_id
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE pr.controlled_lang_s05_sa_score >= {PASS_SA}
          AND COALESCE(pr.failure_op, '') != '{NANO_BIND}'
          AND COALESCE(l.is_reference, 0) = 0
          AND pr.graph_json IS NOT NULL
          AND pgf.template_name IS NOT NULL
        """
    )
    by_template: dict[str, list[str]] = defaultdict(list)
    for tpl, graph_json in cur.fetchall():
        if not tpl:
            continue
        if len(by_template[tpl]) >= MAX_GRAPHS_PER_TEMPLATE:
            continue
        by_template[tpl].append(graph_json)
    conn.close()
    return by_template


def build_coverage_index(
    by_template: dict[str, list[str]],
    chain_lengths: tuple[int, ...],
) -> dict[str, set[tuple[str, ...]]]:
    """For each template, the set of chains its passing graphs produce."""
    coverage: dict[str, set[tuple[str, ...]]] = {}
    for tpl, graphs in by_template.items():
        chains: set[tuple[str, ...]] = set()
        for raw in graphs:
            ops_by_id, inputs_by_id = parse_graph(raw)
            if not ops_by_id:
                continue
            for L in chain_lengths:
                chains.update(extract_chains(ops_by_id, inputs_by_id, L))
        coverage[tpl] = chains
    return coverage


def load_v1_records() -> list[dict]:
    if not V1_OUTPUT.exists():
        print(
            f"ERROR: {V1_OUTPUT} not found. Run mine_template_subpatterns.py first.",
            file=sys.stderr,
        )
        sys.exit(2)
    return list(csv.DictReader(V1_OUTPUT.open()))


def annotate_with_coverage(
    records: list[dict],
    coverage: dict[str, set[tuple[str, ...]]],
) -> list[dict]:
    out: list[dict] = []
    for r in records:
        chain = tuple(r["chain"].split("|"))
        covered_by = sorted(tpl for tpl, chains in coverage.items() if chain in chains)
        n_covered = len(covered_by)
        novelty_label = (
            "novel" if n_covered == 0 else ("rare" if n_covered <= 2 else "common")
        )
        rec = dict(r)
        rec["covered_by_count"] = n_covered
        rec["covered_by_templates"] = ",".join(covered_by[:5])
        rec["covered_by_more"] = max(0, n_covered - 5)
        rec["novelty_label"] = novelty_label
        out.append(rec)
    return out


def write_coverage_csv(records: list[dict], path: Path) -> int:
    fields = list(records[0].keys()) if records else []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(r)
    return len(records)


def _emit_skeleton(chain: tuple[str, ...], anchor: str, name: str) -> str:
    """Generate a Python tpl_<name> skeleton from an op-chain.

    The skeleton chains ops linearly with input_id → op0 → op1 → ... → out,
    wrapping with a residual add. Configs and binary-op handling are left
    as TODOs because the chain alone doesn't carry that info.
    """
    body_lines = [
        f"def tpl_{name}(",
        "    graph: 'ComputationGraph',",
        "    input_id: int,",
        "    rng: random.Random,",
        "    weights: MotifWeights = None,",
        ") -> int:",
        f'    """Auto-mined skeleton (anchor={anchor}, chain={chain}).',
        "",
        "    NOT auto-registered. Adapt before adding to TEMPLATES:",
        "    - Inspect each add_op for binary inputs / config requirements.",
        "    - Add norm wrapping where the constituent ops require it",
        "      (see _motif_rules.py for must_precede constraints).",
        '    """',
        "    current = input_id",
    ]
    for i, op in enumerate(chain):
        if op in STRUCTURAL_OPS:
            body_lines.append(
                f'    current = graph.add_op("{op}", [current])  # step {i}'
            )
        else:
            body_lines.append(
                f'    current = graph.add_op("{op}", [current])  # step {i} (anchor)'
            )
    body_lines.extend(
        [
            "    current = _fix_dim(graph, current)",
            "    return _residual(graph, input_id, current, context=__name__)",
        ]
    )
    return "\n".join(body_lines)


def write_novel_proposals(
    records: list[dict],
    path: Path,
    top_k: int,
) -> tuple[int, int]:
    novel = [r for r in records if r["novelty_label"] == "novel"]
    rare = [r for r in records if r["novelty_label"] == "rare"]
    novel.sort(key=lambda r: -float(r["novelty_score"]))
    rare.sort(key=lambda r: -float(r["novelty_score"]))

    out = {
        "version": "v2",
        "miner": "directed_op_chains_with_novelty_filter",
        "novelty_classes": {
            "novel": "no registered template's passing graphs contain this chain",
            "rare": "1-2 templates produce it (worth a dedicated template)",
            "common": "3+ templates already produce it (skip)",
        },
        "novel_candidates": [],
        "rare_candidates": [],
    }
    for r in novel[:top_k]:
        chain = tuple(r["chain"].split("|"))
        anchor = r["anchor_op"]
        name = f"mined_{anchor}_block"
        out["novel_candidates"].append(
            {
                "chain": chain,
                "length": int(r["length"]),
                "n_total": int(r["n_total"]),
                "n_pass": int(r["n_pass"]),
                "pass_rate": round(float(r["pass_rate"]), 3),
                "lift_vs_cohort": round(float(r["lift_vs_cohort"]), 2),
                "anchor_op": anchor,
                "covered_by_templates": [],
                "proposed_template_name": name,
                "code_skeleton": _emit_skeleton(chain, anchor, name),
            }
        )
    for r in rare[:top_k]:
        chain = tuple(r["chain"].split("|"))
        anchor = r["anchor_op"]
        name = f"mined_{anchor}_variant_block"
        out["rare_candidates"].append(
            {
                "chain": chain,
                "length": int(r["length"]),
                "n_total": int(r["n_total"]),
                "n_pass": int(r["n_pass"]),
                "pass_rate": round(float(r["pass_rate"]), 3),
                "lift_vs_cohort": round(float(r["lift_vs_cohort"]), 2),
                "anchor_op": anchor,
                "covered_by_templates": (
                    r["covered_by_templates"].split(",")
                    if r["covered_by_templates"]
                    else []
                ),
                "proposed_template_name": name,
                "code_skeleton": _emit_skeleton(chain, anchor, name),
            }
        )
    path.write_text(json.dumps(out, indent=2))
    return len(out["novel_candidates"]), len(out["rare_candidates"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHAIN_LENGTHS),
    )
    args = parser.parse_args()

    print("Loading V1 mined chain proposals...", file=sys.stderr)
    records = load_v1_records()
    print(f"  V1 candidates: {len(records)}", file=sys.stderr)

    print("Building per-template coverage index...", file=sys.stderr)
    by_template = fetch_template_grouped()
    n_graphs = sum(len(v) for v in by_template.values())
    print(
        f"  templates: {len(by_template)} | sampled graphs: {n_graphs} "
        f"(cap {MAX_GRAPHS_PER_TEMPLATE}/template)",
        file=sys.stderr,
    )
    coverage = build_coverage_index(by_template, tuple(args.lengths))
    print(
        f"  coverage chains: {sum(len(c) for c in coverage.values())} "
        f"(distinct over {len(coverage)} templates)",
        file=sys.stderr,
    )

    annotated = annotate_with_coverage(records, coverage)
    REPORTS.mkdir(parents=True, exist_ok=True)
    n_csv = write_coverage_csv(annotated, REPORTS / "mined_chain_coverage.csv")
    n_novel, n_rare = write_novel_proposals(
        annotated, REPORTS / "mined_novel_chain_proposals.json", args.top_k
    )
    by_label: dict[str, int] = defaultdict(int)
    for r in annotated:
        by_label[r["novelty_label"]] += 1
    print(f"\nmined_chain_coverage.csv: {n_csv} rows", file=sys.stderr)
    for label, n in sorted(by_label.items()):
        print(f"  {label}: {n}", file=sys.stderr)
    print(
        f"\nmined_novel_chain_proposals.json: novel={n_novel}, rare={n_rare}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
