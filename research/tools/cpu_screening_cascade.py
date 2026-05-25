#!/usr/bin/env python
"""GPU-FREE virtual screening cascade: millions of graphs in → high-quality shortlist out.

Runs entirely on CPU. Generates candidate graphs from the factory, scores each by MEASURED-mechanism
structure (no GPU, no training, no capability labels), and emits a small shortlist of FULL graphs
worth the expensive real probe. Built on the validated signals from the closed-book arc:

  - gate:    n_mixers_on_path >= 1 (a cross-position skill needs a sequence-mixer on an input→output
             path; keeps 95.7% of induction-capable, prunes the structurally-dead). `static_capability_gate`.
  - exploit: mechanism_score = n_mix + 1.5*sum_memory + n_global  (validated label-free composite, ROC
             0.907 vs induction). Deliberately NOT depth/n_ops — those score higher (0.93) but are the
             SIZE confound (the zero-cost-NAS "#params baseline" trap) that won't generalize to small
             novel-good designs.
  - explore: novelty = n_novel_mixers_on_path + algebra_diversity  (label-free). Reserves shortlist
             slots for the unknown-good so the cascade doesn't collapse onto the familiar — the trap
             every label-trained predictor falls into here (screener anti-correlated on novel winners).

Output is an explore∪exploit shortlist with full graph dicts, ready to hand to the real probe.

Usage::
    python -m research.tools.cpu_screening_cascade generate --pool 200000 --exploit 200 --explore 100
    python -m research.tools.cpu_screening_cascade validate     # recall@topK on labeled corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from research.synthesis.grammar import GrammarConfig, generate_layer_graph
from research.synthesis.op_roles import OpRole, get_role
from research.tools.generate_novel_screened import (
    _graph_features,
    _historical_fingerprints,
    _novel_mixers,
)
from research.tools.graph_semantic_features import (
    _MEMORY_ORDINAL,
    GraphSemanticExtractor,
)
from research.tools.static_capability_gate import on_path_op_names

logger = logging.getLogger(__name__)
_RUNS_DB = "research/runs.db"
_META_DB = "research/meta_analysis.db"


@dataclass
class MechProfile:
    n_mix: int
    sum_mem: float
    n_global: int
    alg_div: int
    n_novel_mix: int
    mech_score: float
    novelty: float


class CpuMechanismScorer:
    """Label-free, GPU-free mechanism scorer over input→output-path ops (catalog loaded once)."""

    def __init__(self, runs_db: str = _RUNS_DB, meta_db: str = _META_DB) -> None:
        self.ext = GraphSemanticExtractor(runs_db, meta_db)
        self.novel: Set[str] = _novel_mixers(runs_db)

    def profile(self, nodes: Dict[str, Any] | List[Any]) -> MechProfile:
        ops = on_path_op_names(nodes)
        mixers = [op for op in ops if get_role(op) is OpRole.MIX]
        mem = [_MEMORY_ORDINAL.get(self.ext.op_memory.get(m, ""), 0.0) for m in mixers]
        n_global = sum(1 for m in mixers if self.ext.op_receptive.get(m) == "global")
        alg_div = len({self.ext.op_algebra.get(m, "") for m in mixers if m})
        n_novel = sum(1 for m in mixers if m in self.novel)
        n_mix = len(mixers)
        return MechProfile(
            n_mix=n_mix,
            sum_mem=float(sum(mem)),
            n_global=n_global,
            alg_div=alg_div,
            n_novel_mix=n_novel,
            mech_score=n_mix + 1.5 * float(sum(mem)) + n_global,
            novelty=float(n_novel + alg_div),
        )


# --------------------------------------------------------------------------- #
# generate mode — the cascade
# --------------------------------------------------------------------------- #
@dataclass
class Scored:
    fingerprint: str
    ops: List[str]
    profile: MechProfile
    graph_dict: Dict[str, Any]


def _generate_pool(
    scorer: CpuMechanismScorer,
    hist: Set[str],
    pool: int,
    max_attempts: int,
    seed0: int,
    progress_every: int,
) -> Tuple[List[Scored], Counter]:
    """Generate → validity/novel-fp/mixer gate → mechanism-score. CPU only."""
    cfg = GrammarConfig()
    stats: Counter = Counter()
    seen: Set[str] = set()
    kept: List[Scored] = []
    t0 = time.time()
    for i in range(max_attempts):
        try:
            g = generate_layer_graph(cfg, seed=seed0 + i)
        except Exception:
            stats["invalid"] += 1
            continue
        op_set, _, _, fp = _graph_features(g)
        if fp in hist or fp in seen:
            stats["already_seen"] += 1
            continue
        seen.add(fp)
        gd = g.to_dict()
        prof = scorer.profile(gd["nodes"])
        if prof.n_mix < 1:  # structural gate
            stats["no_mixer_on_path"] += 1
            continue
        kept.append(Scored(fp, sorted(op_set), prof, gd))
        stats["kept"] += 1
        if progress_every and (i + 1) % progress_every == 0:
            logger.info(
                "  attempts=%d kept=%d (%.0f/s)",
                i + 1,
                len(kept),
                (i + 1) / max(time.time() - t0, 1e-9),
            )
        if len(kept) >= pool:
            break
    return kept, stats


def _select(kept: List[Scored], n_exploit: int, n_explore: int) -> List[Scored]:
    """Explore∪exploit shortlist: top mechanism_score ∪ top novelty (dedup, exploit wins ties)."""
    by_mech = sorted(kept, key=lambda s: -s.profile.mech_score)[:n_exploit]
    by_nov = sorted(kept, key=lambda s: -s.profile.novelty)[:n_explore]
    out: Dict[str, Scored] = {s.fingerprint: s for s in by_mech}
    for s in by_nov:
        out.setdefault(s.fingerprint, s)
    return list(out.values())


def run_generate(args: argparse.Namespace) -> Dict[str, Any]:
    scorer = CpuMechanismScorer(args.db, args.meta)
    hist = _historical_fingerprints(args.db)
    t0 = time.time()
    kept, stats = _generate_pool(
        scorer, hist, args.pool, args.max_attempts, args.seed0, args.progress_every
    )
    shortlist = _select(kept, args.exploit, args.explore)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for s in shortlist:
            f.write(
                json.dumps(
                    {
                        "fingerprint": s.fingerprint,
                        "ops": s.ops,
                        "mech_score": round(s.profile.mech_score, 3),
                        "novelty": s.profile.novelty,
                        "n_mixers_on_path": s.profile.n_mix,
                        "n_novel_mixers": s.profile.n_novel_mix,
                        "graph": s.graph_dict,
                    }
                )
                + "\n"
            )
    return {
        "elapsed_s": round(time.time() - t0, 1),
        "graphs_per_s": round(args.max_attempts / max(time.time() - t0, 1e-9)),
        "stats": dict(stats),
        "pool_kept": len(kept),
        "shortlist": len(shortlist),
        "out": out.as_posix(),
        "top_by_mech": [
            {"fp": s.fingerprint, "mech": round(s.profile.mech_score, 2), "ops": s.ops}
            for s in sorted(kept, key=lambda s: -s.profile.mech_score)[:5]
        ],
    }


# --------------------------------------------------------------------------- #
# validate mode — recall of capable in the shortlist, on the labeled corpus
# --------------------------------------------------------------------------- #
def run_validate(args: argparse.Namespace) -> Dict[str, Any]:
    import sqlite3

    scorer = CpuMechanismScorer(args.db, args.meta)
    con = sqlite3.connect(args.db)
    rows = con.execute(
        """SELECT g.graph_json, AVG(r.induction_screening_auc)
           FROM graphs g JOIN graph_runs r ON g.graph_fingerprint=r.graph_fingerprint
           WHERE g.graph_json_is_placeholder=0 AND r.induction_screening_auc IS NOT NULL
           GROUP BY g.graph_fingerprint"""
    ).fetchall()
    con.close()
    mech: List[float] = []
    y: List[float] = []
    t0 = time.time()
    for gj, auc in rows:
        try:
            nodes = json.loads(gj)["nodes"]
        except Exception:
            continue
        mech.append(scorer.profile(nodes).mech_score)
        y.append(float(auc))
    mech_a = np.array(mech)
    y_a = np.array(y)
    pos = y_a > 0.35
    order = np.argsort(-mech_a)
    n = len(y_a)
    out: Dict[str, Any] = {
        "n": n,
        "n_capable": int(pos.sum()),
        "graphs_per_s": round(n / max(time.time() - t0, 1e-9)),
        "recall_at_topk": {},
    }
    base_rate = pos.mean()
    for frac in (0.05, 0.10, 0.20, 0.30):
        k = int(n * frac)
        top = order[:k]
        recall = pos[top].sum() / max(pos.sum(), 1)
        precision = pos[top].mean()
        out["recall_at_topk"][f"top_{int(frac * 100)}pct"] = {
            "recall_of_capable": round(float(recall), 3),
            "precision": round(float(precision), 3),
            "enrichment_vs_base": round(float(precision / max(base_rate, 1e-9)), 2),
        }
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["generate", "validate"])
    p.add_argument("--db", default=_RUNS_DB)
    p.add_argument("--meta", default=_META_DB)
    p.add_argument("--pool", type=int, default=50000)
    p.add_argument("--max-attempts", type=int, default=200000)
    p.add_argument("--seed0", type=int, default=11_000_000)
    p.add_argument("--exploit", type=int, default=200)
    p.add_argument("--explore", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=20000)
    p.add_argument("--out", default="research/reports/cpu_cascade_shortlist.jsonl")
    args = p.parse_args()
    report = run_generate(args) if args.mode == "generate" else run_validate(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
