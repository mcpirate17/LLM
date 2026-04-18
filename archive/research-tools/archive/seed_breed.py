"""Seed mutation children from known-good donor graphs.

Two target donors from the candidate-comparable frontier analysis
(2026-04-16):

* ``bb120386-3bc`` — ppl=11.83, hs=0.30, ind=0.044. Retrieval-capable
  substring (``token_type_classifier → matmul → entropy_score → mul``)
  embedded in a linear trunk. Mutate under ``capability_first`` grammar
  to preserve the retrieval ops and shuffle the surrounding trunk.

* ``903157e5-219`` — ppl=32.78 (bad), but clears GPT-2 on binding /
  hellaswag / induction. Treat as a retrieval-module donor: keep the
  content-addressed ops, swap the trunk for a ppl-winner family
  (``conv1d_seq``, ``selective_scan``, ``swiglu_mlp``).

The tool:

1. Loads each donor's graph_json from the LabNotebook.
2. Generates ``--n-mutants`` children per donor under the
   ``capability_first`` grammar (which boosts retrieval ops and enforces
   the new ``gate8_retrieval_dead`` screener).
3. Writes the mutants to a JSON file ready for manual inspection or
   ingestion by the regular experiment runner.

Usage::

    python -m research.tools.seed_breed \\
        --donors bb120386-3bc 903157e5-219 \\
        --n-mutants 32 \\
        --out research/reports/seed_breed_mutants.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("seed_breed")


# Reference donors from the 2026-04-16 capability-first analysis.
DEFAULT_DONORS: tuple[str, ...] = (
    "bb120386-3bc",
    "903157e5-219",
)


def _load_donor_graph(nb, result_id: str):
    from ..synthesis.serializer import graph_from_json

    # Tolerate both full result_ids and short prefixes — the analysis
    # memos use truncated IDs.
    row = nb.get_program_detail(result_id)
    if row is None:
        cursor = nb.conn.execute(
            "SELECT result_id, graph_json FROM program_results "
            "WHERE result_id LIKE ? LIMIT 1",
            (f"{result_id}%",),
        ).fetchone()
        if cursor is None:
            return None, None
        row = dict(cursor)
    graph_json = row.get("graph_json")
    if not graph_json:
        log.warning("donor %s has no graph_json", result_id)
        return row.get("result_id"), None
    return row.get("result_id"), graph_from_json(graph_json)


def _summarize(graph) -> Dict[str, object]:
    op_counts: Dict[str, int] = {}
    for _, node in graph.nodes.items():
        if getattr(node, "is_input", False):
            continue
        name = node.op_name
        op_counts[name] = op_counts.get(name, 0) + 1
    return {
        "n_ops": graph.n_ops(),
        "depth": graph.depth(),
        "fingerprint": graph.fingerprint(),
        "op_counts": dict(sorted(op_counts.items())),
    }


def _breed_from_donor(
    donor_id: str,
    parent_graph,
    n_mutants: int,
    seed: int,
) -> List[Dict]:
    """Generate N mutation children from a donor under capability_first."""

    from ..search.evolution import _mutate_graph, _local_mutate_graph
    from ..synthesis.grammar import GrammarConfig
    from ..synthesis.serializer import graph_to_json

    rng = random.Random(seed)
    grammar = GrammarConfig.capability_first(model_dim=parent_graph.model_dim)

    mutants: List[Dict] = []
    # Mix of global grammar mutations and local (topology-preserving) swaps.
    # Global mutations sample the grammar — they will lean on the
    # capability_first op priors to inject retrieval structure. Local
    # mutations keep the donor's topology and swap individual ops.
    for i in range(n_mutants):
        try:
            if i % 2 == 0:
                child = _mutate_graph(parent_graph, grammar, rng)
                kind = "grammar"
            else:
                child = _local_mutate_graph(parent_graph, rng)
                kind = "local"
        except Exception as exc:  # noqa: BLE001
            # Fail loudly but keep breeding — a single bad mutation shouldn't
            # kill the batch.
            log.warning("donor %s mutation %d failed (%s): %s", donor_id, i, kind, exc)
            continue

        mutants.append(
            {
                "mutant_index": i,
                "mutation_kind": kind,
                "donor_id": donor_id,
                "summary": _summarize(child),
                "graph_json": graph_to_json(child),
            }
        )
    return mutants


def _ensure_out_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--donors",
        nargs="+",
        default=list(DEFAULT_DONORS),
        help="Donor result_ids (full or short prefix).",
    )
    ap.add_argument(
        "--n-mutants",
        type=int,
        default=32,
        help="Number of mutation children per donor.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("research/reports/seed_breed_mutants.json"),
        help="Output JSON path (default: research/reports/...).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=2026_04_16,
        help="RNG seed for mutation sampling.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("ARIA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from ..scientist.notebook import LabNotebook

    nb = LabNotebook()
    nb.flush_writes()

    results: Dict[str, object] = {
        "donors": {},
        "mutants": [],
        "grammar_preset": "capability_first",
        "n_mutants_per_donor": args.n_mutants,
    }

    for donor_short in args.donors:
        resolved_id, parent = _load_donor_graph(nb, donor_short)
        if parent is None:
            log.error(
                "donor %s not found in notebook (resolved=%r) — skipping",
                donor_short,
                resolved_id,
            )
            results["donors"][donor_short] = {"status": "missing"}
            continue
        summary = _summarize(parent)
        log.info(
            "donor %s resolved=%s ops=%d depth=%d fp=%s",
            donor_short,
            resolved_id,
            summary["n_ops"],
            summary["depth"],
            summary["fingerprint"][:12],
        )
        results["donors"][donor_short] = {
            "status": "loaded",
            "resolved_id": resolved_id,
            "summary": summary,
        }
        mutants = _breed_from_donor(
            donor_id=resolved_id or donor_short,
            parent_graph=parent,
            n_mutants=args.n_mutants,
            seed=args.seed + hash(donor_short) & 0xFFFF,
        )
        log.info("donor %s produced %d mutants", donor_short, len(mutants))
        results["mutants"].extend(mutants)

    _ensure_out_dir(args.out)
    args.out.write_text(json.dumps(results, indent=2))
    log.info(
        "wrote %d mutants from %d donor(s) to %s",
        len(results["mutants"]),
        len(args.donors),
        args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
