"""Phase 5 — Auto-mine sub-patterns from passing-cohort graphs.

V1 miner: walks each cohort graph's directed dataflow, extracts all directed
op-chains of length 3 and 4, canonicalizes each as a tuple of op_names, and
ranks chains by overrepresentation in the pass cohort vs the global cohort.

Outputs:
  research/reports/mined_chain_proposals.csv  — full ranked table
  research/reports/mined_chain_proposals.json — top-K with skeleton sketches

Cohort filter (matches template_pass classification):
  Pass: language_control_s05_sentence_assoc_score >= 0.95 AND failure_op != 'nano_bind'
  Fail: language_control_s05_sentence_assoc_score < 0.30 OR failure_op = 'nano_bind'

Significance criteria (default — overridable via CLI):
  - n >= 30  (chain occurred in at least 30 cohort graphs)
  - pass_rate (over cohort graphs containing the chain) >= 0.60
  - lift = pass_rate / cohort_pass_rate >= 1.5

Skips chains whose op-set is dominated by structural ops (linear_proj,
add, rmsnorm) — those are universal scaffolding, not novel patterns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "research/runs.db"
REPORTS = REPO / "research/reports"

PASS_SA = 0.95
FAIL_SA = 0.30
NANO_BIND = "nano_bind"

# Ops we don't want to surface as pattern anchors — pure structural scaffolding.
STRUCTURAL_OPS = frozenset(
    {
        "linear_proj",
        "linear_proj_down",
        "linear_proj_up",
        "add",
        "rmsnorm",
        "layernorm",
        "norm",
        "_fix_dim",
        "fix_dim",
        "concat",
        "split2",
        "split3",
        "input",
    }
)

# Default significance gates
DEFAULT_MIN_N = 30
DEFAULT_MIN_PASS_RATE = 0.60
DEFAULT_MIN_LIFT = 1.5
DEFAULT_CHAIN_LENGTHS = (3, 4)


def fetch_cohort() -> list[dict]:
    conn = sqlite3.connect(f"file:{LAB}?mode=ro&immutable=0", uri=True)
    cur = conn.execute(
        """
        SELECT pr.result_id,
               pr.language_control_s05_sentence_assoc_score AS sa,
               pr.failure_op,
               pr.graph_json
        FROM program_results_compat pr
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE pr.language_control_s05_sentence_assoc_score IS NOT NULL
          AND COALESCE(l.is_reference, 0) = 0
          AND pr.graph_json IS NOT NULL
        """
    )
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for row in rows:
        row["graph_json"] = resolve_graph_json_value(
            conn,
            LAB,
            row.get("graph_json"),
        )
    conn.close()
    return rows


def is_pass(sa: float | None, failure_op: str | None) -> bool:
    return sa is not None and sa >= PASS_SA and (failure_op or "") != NANO_BIND


def is_fail(sa: float | None, failure_op: str | None) -> bool:
    return (failure_op or "") == NANO_BIND or (sa is not None and sa < FAIL_SA)


def parse_graph(graph_json_str: str) -> tuple[dict[int, str], dict[int, list[int]]]:
    """Return (node_id -> op_name) and (node_id -> list of upstream input_ids)."""
    try:
        g = json.loads(graph_json_str)
    except (json.JSONDecodeError, TypeError):
        return {}, {}
    nodes_raw = g.get("nodes")
    if not nodes_raw:
        return {}, {}
    iter_n = nodes_raw.values() if isinstance(nodes_raw, dict) else nodes_raw
    op_by_id: dict[int, str] = {}
    inputs_by_id: dict[int, list[int]] = {}
    for n in iter_n:
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        if nid is None:
            continue
        op = n.get("op_name") or n.get("op") or n.get("type") or ""
        if not op:
            continue
        op_by_id[int(nid)] = str(op)
        ins = n.get("input_ids") or []
        if isinstance(ins, list):
            inputs_by_id[int(nid)] = [int(i) for i in ins if isinstance(i, int)]
    return op_by_id, inputs_by_id


def extract_chains(
    op_by_id: dict[int, str], inputs_by_id: dict[int, list[int]], length: int
) -> set[tuple[str, ...]]:
    """Return the set of directed op-chains of given length (no node repeats).

    Chains follow dataflow upstream: a chain (op0, op1, op2) means
    op2 has op1 as an input (directly or by chaining), op1 has op0 as an input.
    """
    chains: set[tuple[str, ...]] = set()
    for tail_id in op_by_id:
        # DFS walking input edges, recording the op sequence (downstream to upstream)
        stack = [(tail_id, [tail_id])]
        while stack:
            node, path = stack.pop()
            if len(path) == length:
                ops = tuple(op_by_id[nid] for nid in reversed(path))
                chains.add(ops)
                continue
            for upstream in inputs_by_id.get(node, ()):
                if upstream in path:
                    continue
                if upstream not in op_by_id:
                    continue
                stack.append((upstream, path + [upstream]))
    return chains


def chain_is_interesting(chain: tuple[str, ...]) -> bool:
    """Drop chains that are entirely structural scaffolding."""
    return any(op not in STRUCTURAL_OPS for op in chain)


def aggregate(
    rows: list[dict], chain_lengths: tuple[int, ...]
) -> tuple[Counter[tuple[str, ...]], Counter[tuple[str, ...]], int, int, int]:
    """Count chain occurrences per cohort row (not per occurrence in graph).

    Returns:
      n_total[chain], n_pass[chain], total_rows, total_pass_rows, total_fail_rows
    """
    n_total: Counter[tuple[str, ...]] = Counter()
    n_pass: Counter[tuple[str, ...]] = Counter()
    rows_seen = 0
    pass_seen = 0
    fail_seen = 0
    for row in rows:
        op_by_id, inputs_by_id = parse_graph(row["graph_json"])
        if not op_by_id:
            continue
        chains: set[tuple[str, ...]] = set()
        for L in chain_lengths:
            chains.update(extract_chains(op_by_id, inputs_by_id, L))
        chains = {c for c in chains if chain_is_interesting(c)}
        passed = is_pass(row["sa"], row["failure_op"])
        failed = is_fail(row["sa"], row["failure_op"])
        rows_seen += 1
        if passed:
            pass_seen += 1
        elif failed:
            fail_seen += 1
        for c in chains:
            n_total[c] += 1
            if passed:
                n_pass[c] += 1
    return n_total, n_pass, rows_seen, pass_seen, fail_seen


def write_csv(records: list[dict], path: Path) -> int:
    fields = [
        "chain",
        "length",
        "n_total",
        "n_pass",
        "pass_rate",
        "lift_vs_cohort",
        "novelty_score",
        "anchor_op",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(r)
    return len(records)


def write_top_json(records: list[dict], path: Path, top_k: int = 50) -> int:
    out = {
        "version": "v1",
        "miner": "directed_op_chains",
        "naming_convention": "<anchor_op>_<chain_summary>_block",
        "candidates": [],
    }
    for r in records[:top_k]:
        chain = r["chain"]
        # propose a name from the anchor op (most distinctive non-structural)
        anchor = r["anchor_op"]
        name = f"mined_{anchor}_block"
        out["candidates"].append(
            {
                "chain": chain,
                "length": r["length"],
                "n_total": r["n_total"],
                "n_pass": r["n_pass"],
                "pass_rate": round(r["pass_rate"], 3),
                "lift_vs_cohort": round(r["lift_vs_cohort"], 2),
                "novelty_score": round(r["novelty_score"], 3),
                "anchor_op": anchor,
                "proposed_template_name": name,
                "proposed_skeleton": [f"chain: {' -> '.join(chain.split('|'))}"],
            }
        )
    path.write_text(json.dumps(out, indent=2))
    return len(out["candidates"])


def _record_for(
    chain: tuple[str, ...], n_total: int, n_pass: int, cohort_pass_rate: float
) -> dict:
    pass_rate = n_pass / n_total if n_total else 0.0
    lift = pass_rate / cohort_pass_rate if cohort_pass_rate > 0 else 0.0
    # Novelty: which op in the chain is most distinctive (least structural)?
    # Pick the first non-structural op as the anchor.
    anchor = next((op for op in chain if op not in STRUCTURAL_OPS), chain[0])
    # Score = lift * sqrt(n_total) — higher when both effect size + sample
    novelty = lift * math.sqrt(n_total) / 10.0
    return {
        "chain": "|".join(chain),
        "length": len(chain),
        "n_total": n_total,
        "n_pass": n_pass,
        "pass_rate": pass_rate,
        "lift_vs_cohort": lift,
        "novelty_score": novelty,
        "anchor_op": anchor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    parser.add_argument("--min-pass-rate", type=float, default=DEFAULT_MIN_PASS_RATE)
    parser.add_argument("--min-lift", type=float, default=DEFAULT_MIN_LIFT)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHAIN_LENGTHS),
    )
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    rows = fetch_cohort()
    print(f"Cohort rows: {len(rows)}", file=sys.stderr)
    n_total, n_pass, rows_seen, pass_seen, fail_seen = aggregate(
        rows, tuple(args.lengths)
    )
    cohort_pass_rate = pass_seen / rows_seen if rows_seen else 0.0
    print(
        f"Cohort: {rows_seen} rows, {pass_seen} pass, {fail_seen} fail, "
        f"pass_rate={cohort_pass_rate:.3f}",
        file=sys.stderr,
    )
    print(f"Distinct chains observed: {len(n_total)}", file=sys.stderr)

    records: list[dict] = []
    for chain, count in n_total.items():
        if count < args.min_n:
            continue
        rec = _record_for(chain, count, n_pass[chain], cohort_pass_rate)
        if rec["pass_rate"] < args.min_pass_rate:
            continue
        if rec["lift_vs_cohort"] < args.min_lift:
            continue
        records.append(rec)
    records.sort(key=lambda r: -r["novelty_score"])

    REPORTS.mkdir(parents=True, exist_ok=True)
    n_csv = write_csv(records, REPORTS / "mined_chain_proposals.csv")
    n_json = write_top_json(records, REPORTS / "mined_chain_proposals.json", args.top_k)
    print(
        f"mined_chain_proposals.csv: {n_csv} chains "
        f"(min_n={args.min_n}, min_pass_rate={args.min_pass_rate}, "
        f"min_lift={args.min_lift})",
        file=sys.stderr,
    )
    print(f"mined_chain_proposals.json: top {n_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
