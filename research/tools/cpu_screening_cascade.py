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
from research.tools.annotate_literature_attribution import (
    DEFAULT_MAPPING,
    classify_graph_family,
)
from research.tools.generate_novel_screened import (
    _graph_features,
    _historical_fingerprints,
    _novel_mixers,
)
from research.tools.graph_semantic_features import (
    _MEMORY_ORDINAL,
    GraphSemanticExtractor,
)
from research.tools.learned_rules import score_template_quality
from research.tools.static_capability_gate import (
    mixer_chain_depth,
    on_path_op_names,
)

logger = logging.getLogger(__name__)
_RUNS_DB = "research/runs.db"
_META_DB = "research/meta_analysis.db"


@dataclass
class MechProfile:
    n_mix: int
    mixer_depth: int
    sum_mem: float
    n_global: int
    alg_div: int
    n_novel_mix: int
    mech_score: float
    novelty: float
    lit_family: str  # closest published-architecture family (dominant mixer)
    lit_model: str  # the published model name it resembles
    lit_match_type: str  # exact | family | partial | novel


class CpuMechanismScorer:
    """Label-free, GPU-free mechanism scorer over input→output-path ops (catalog loaded once)."""

    def __init__(self, runs_db: str = _RUNS_DB, meta_db: str = _META_DB) -> None:
        self.ext = GraphSemanticExtractor(runs_db, meta_db)
        self.novel: Set[str] = _novel_mixers(runs_db)
        self.lit_families: Dict[str, Any] = json.loads(DEFAULT_MAPPING.read_text())[
            "graph_families"
        ]

    def profile(self, nodes: Dict[str, Any] | List[Any]) -> MechProfile:
        ops = on_path_op_names(nodes)
        all_ops = {
            str(n["op_name"])
            for n in (nodes.values() if isinstance(nodes, dict) else nodes)
            if not n.get("is_input")
        }
        fam = classify_graph_family(
            all_ops
        )  # same logic as literature_attribution pass
        lit = self.lit_families.get(fam, {})
        mixers = [op for op in ops if get_role(op) is OpRole.MIX]
        mem = [_MEMORY_ORDINAL.get(self.ext.op_memory.get(m, ""), 0.0) for m in mixers]
        n_global = sum(1 for m in mixers if self.ext.op_receptive.get(m) == "global")
        alg_div = len({self.ext.op_algebra.get(m, "") for m in mixers if m})
        n_novel = sum(1 for m in mixers if m in self.novel)
        n_mix = len(mixers)
        depth = mixer_chain_depth(
            nodes
        )  # ROUTING depth (chained mixing stages), not param count
        return MechProfile(
            n_mix=n_mix,
            mixer_depth=depth,
            sum_mem=float(sum(mem)),
            n_global=n_global,
            alg_div=alg_div,
            n_novel_mix=n_novel,
            # routing-composition led (induction circuit is depth>=2) + per-stage quality
            mech_score=2.0 * depth + float(sum(mem)) + n_global + 0.5 * n_mix,
            novelty=float(n_novel + alg_div),
            lit_family=fam,
            lit_model=str(lit.get("external_model_name", "?")),
            lit_match_type=str(lit.get("match_type", "?")),
        )


# --------------------------------------------------------------------------- #
# generate mode — the cascade
# --------------------------------------------------------------------------- #
@dataclass
class Scored:
    fingerprint: str
    ops: List[str]
    profile: MechProfile
    quality: Dict[str, Any]  # learned_rules.score_template_quality output
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
        q = score_template_quality(
            gd["nodes"]
        )  # good-template + data-mined failure rules
        if not q["passes_must"]:  # mixer-on-path + norm + residual + no-double-gating
            stats["bad_template"] += 1
            continue
        if q["failure_risk"]["compile"] >= 0.4 or q["failure_risk"]["lookahead"] >= 0.4:
            stats["high_failure_risk"] += 1
            continue
        kept.append(Scored(fp, sorted(op_set), prof, q, gd))
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


def _context_rule_clean(s: "Scored") -> bool:
    """Hard backstop: drop any shortlisted graph that violates the grammar's context/adjacency
    rules (forbidden prev/next-op pairs, local_window_attn successor reqs, etc.). The motif
    grammar already enforces these upstream (empirically 0/651), but this guarantees the OUTPUT."""
    from research.synthesis._context_validation import find_graph_context_violations
    from research.synthesis.serializer import graph_from_json

    try:
        return not find_graph_context_violations(
            graph_from_json(json.dumps(s.graph_dict))
        )
    except Exception:
        return False  # un-checkable ⇒ exclude (don't ship a graph we can't validate)


def run_generate(args: argparse.Namespace) -> Dict[str, Any]:
    scorer = CpuMechanismScorer(args.db, args.meta)
    hist = _historical_fingerprints(args.db)
    t0 = time.time()
    kept, stats = _generate_pool(
        scorer, hist, args.pool, args.max_attempts, args.seed0, args.progress_every
    )
    selected = _select(kept, args.exploit, args.explore)
    shortlist = [s for s in selected if _context_rule_clean(s)]
    stats["context_rule_dropped"] = len(selected) - len(shortlist)
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
                        "mixer_depth": s.profile.mixer_depth,
                        "n_mixers_on_path": s.profile.n_mix,
                        "n_novel_mixers": s.profile.n_novel_mix,
                        "lit_family": s.profile.lit_family,
                        "lit_model": s.profile.lit_model,
                        "lit_match_type": s.profile.lit_match_type,
                        "template_quality": s.quality["score"],
                        "failure_risk": s.quality["failure_risk"],
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
        "shortlist_vs_published": dict(
            Counter(s.profile.lit_match_type for s in shortlist)
        ),
        "shortlist_contains_novel_mixer": sum(
            1 for s in shortlist if s.profile.n_novel_mix > 0
        ),
        "shortlist_mean_template_quality": round(
            float(np.mean([s.quality["score"] for s in shortlist]))
            if shortlist
            else 0.0,
            3,
        ),
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


_MUST_CHECKS = (
    "has_mixer_on_path",
    "has_normalization",
    "has_residual",
    "no_double_gating",
)


def run_rescreen(args: argparse.Namespace) -> Dict[str, Any]:
    """Re-check an already-emitted shortlist jsonl against the FULL rules (encoded context rules +
    good-template must-checks + data-mined failure-risk) — how many now fail, by layer."""
    from research.synthesis._context_validation import find_graph_context_violations
    from research.synthesis.serializer import graph_from_json

    rows = [
        json.loads(line) for line in Path(args.in_path).read_text().splitlines() if line
    ]
    ctx_kinds: Counter = Counter()
    check_fail: Counter = Counter()
    n_ctx = n_must = n_risk = n_any = 0
    failing: List[Dict[str, Any]] = []
    clean: List[Dict[str, Any]] = []
    for r in rows:
        nodes = r["graph"]["nodes"]
        q = score_template_quality(nodes)
        try:
            viol = find_graph_context_violations(
                graph_from_json(json.dumps(r["graph"]))
            )
        except Exception:
            viol = ["UNCHECKABLE"]
        risk = q["failure_risk"]
        hi_risk = risk["compile"] >= 0.4 or risk["lookahead"] >= 0.4
        reasons: List[str] = []
        if viol:
            n_ctx += 1
            reasons.append("context_violation")
            for v in viol:
                ctx_kinds[v.split(":")[0][:50]] += 1
        if not q["passes_must"]:
            n_must += 1
        for chk, ok in q["checks"].items():
            if not ok:
                check_fail[chk] += 1
                if chk in _MUST_CHECKS:
                    reasons.append(chk)
        if hi_risk:
            n_risk += 1
            reasons.append("high_failure_risk")
        if reasons:
            n_any += 1
            if len(failing) < 40:
                failing.append(
                    {
                        "fp": r.get("fingerprint"),
                        "reasons": reasons,
                        "failure_risk": risk,
                    }
                )
        else:
            clean.append(r)
    clean_path = Path(args.in_path).with_name(Path(args.in_path).stem + "_clean.jsonl")
    clean_path.write_text("".join(json.dumps(r) + "\n" for r in clean))
    return {
        "in": args.in_path,
        "clean_out": clean_path.as_posix(),
        "n_total": len(rows),
        "n_context_violation": n_ctx,
        "context_violation_kinds": dict(ctx_kinds.most_common(10)),
        "n_fail_must": n_must,
        "must_check_failures": {
            k: check_fail[k] for k in _MUST_CHECKS if check_fail[k]
        },
        "all_check_failures": dict(check_fail.most_common()),
        "n_high_failure_risk": n_risk,
        "n_fail_any": n_any,
        "n_clean": len(rows) - n_any,
        "failing_fingerprints": failing,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["generate", "validate", "rescreen"])
    p.add_argument("--db", default=_RUNS_DB)
    p.add_argument("--meta", default=_META_DB)
    p.add_argument("--pool", type=int, default=50000)
    p.add_argument("--max-attempts", type=int, default=200000)
    p.add_argument("--seed0", type=int, default=11_000_000)
    p.add_argument("--exploit", type=int, default=200)
    p.add_argument("--explore", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=20000)
    p.add_argument("--out", default="research/reports/cpu_cascade_shortlist.jsonl")
    p.add_argument(
        "--in",
        dest="in_path",
        default="research/reports/cpu_cascade_large_shortlist.jsonl",
        help="shortlist jsonl to rescreen (rescreen mode)",
    )
    args = p.parse_args()
    if args.mode == "generate":
        report = run_generate(args)
    elif args.mode == "validate":
        report = run_validate(args)
    else:
        report = run_rescreen(args)
        Path("research/reports/cascade_shortlist_rescreen.json").write_text(
            json.dumps(report, indent=2, sort_keys=True)
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
