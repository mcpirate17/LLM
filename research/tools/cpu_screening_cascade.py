#!/usr/bin/env python
"""GPU-FREE virtual screening cascade: millions of graphs in → high-quality shortlist out.

Runs entirely on CPU. Generates candidate graphs from the factory, scores each by MEASURED-mechanism
structure (no GPU, no training, no capability labels), and emits a small shortlist of FULL graphs
worth the expensive real probe. Built on the validated signals from the closed-book arc:

  - gate:    n_mixers_on_path >= 1 (a cross-position skill needs a sequence-mixer on an input→output
             path; keeps 95.7% of induction-capable, prunes the structurally-dead). `static_capability_gate`.
  - ML gate: predicted ar_gate must clear its trained threshold; AR is a no-go gate,
             not an exploit score.
  - ML downstream gate: at least one non-AR probe axis must clear its trained threshold.
  - exploit: AR+downstream-gated survivors rank by non-AR probe-axis predictions plus mechanism score.
             Deliberately NOT depth/n_ops — those score higher but are the SIZE confound
             (the zero-cost-NAS "#params baseline" trap) that won't generalize to small
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
from scipy.stats import spearmanr

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
from research.tools.label_free_probe_oracle import (
    LabelFreeProbeOracleScorer,
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

    def __init__(
        self,
        runs_db: str = _RUNS_DB,
        meta_db: str = _META_DB,
        *,
        use_probe_oracle: bool = True,
    ) -> None:
        self.ext = GraphSemanticExtractor(runs_db, meta_db)
        self.novel: Set[str] = _novel_mixers(runs_db)
        self.lit_families: Dict[str, Any] = json.loads(DEFAULT_MAPPING.read_text())[
            "graph_families"
        ]
        self.probe_oracle = (
            LabelFreeProbeOracleScorer.try_load(runs_db=runs_db, meta_db=meta_db)
            if use_probe_oracle
            else None
        )

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

    def probe_oracle_score(self, graph_dict: Dict[str, Any]) -> Dict[str, Any] | None:
        if self.probe_oracle is None:
            return None
        try:
            return self.probe_oracle.score_graph_dict(graph_dict)
        except Exception as exc:  # noqa: BLE001
            logger.debug("label-free probe oracle scoring failed: %s", exc)
            return None


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
    probe_oracle: Dict[str, Any] | None = None
    measured_score: float | None = (
        None  # OOD-robust measured capability rank (top-K only)
    )


# Recall-99 PREDICTION cuts from `research/tools/nas_gate_calibration --target-recall 0.99`
# (leave-family-out OOD, 2026-06-03). This is now the funnel's COARSE high-recall stage: the
# declared oracle is OOD-anti-predictive on novel archs, so it only prunes obvious junk while
# retaining ~99% of in-corpus winners; the MEASURED precision ranker (`_measured_rank_topk`) does
# the real selection downstream. Validated against the novel-winner holdout: the recall-95 cuts
# (induction 0.0214, nano 0.108) REJECTED 2/3 known novel winners (7fd270, 818545); these recall-99
# cuts retain 3/3. The LABEL thresholds (induction 0.35, nano 0.5) reject ~99% of winners as
# prediction cuts (GBM regresses to base rate) — never use them as gates. ar_gate/ar_curriculum did
# NOT earn a gate (axis-mismatched). Regenerate after any oracle retrain.
_CAPABILITY_GATE_CUTS: Dict[str, float] = {
    "induction": 0.004931,
    "nano_induction_nearest": 0.072749,
}


def _capability_gate_on_predictions(preds: Dict[str, Any]) -> bool:
    """Core coarse gate: keep iff induction OR nano clears its recall-99 cut.

    OR-of-capable, so a winner flagged by either retrieval axis survives; reject only when BOTH
    axes are predictable AND both fall below their cut. Fall open when neither is predictable —
    never silently drop an unmeasured candidate. Operates on a raw ``{axis: prediction}`` dict so
    it serves both the single-graph and the batched scoring paths.
    """
    checked = False
    for axis, thr in _CAPABILITY_GATE_CUTS.items():
        p = preds.get(axis)
        if p is None:
            continue
        checked = True
        try:
            if float(p) >= thr:
                return True
        except (TypeError, ValueError):
            continue
    return not checked  # neither axis predictable ⇒ fall open


def _capability_gate_passes(probe: Dict[str, Any] | None) -> bool:
    """Axis-matched hard gate for the induction-family target (probe-dict wrapper).

    ar_gate is deliberately NOT a gate here (axis-mismatched OOD); it remains in the output for
    ranking/inspection only. See `_capability_gate_on_predictions` for the operating-point logic.
    """
    if not probe:
        return True
    preds = probe.get("label_free_probe_predictions")
    if not isinstance(preds, dict):
        return True
    return _capability_gate_on_predictions(preds)


def _batch_oracle_predict(
    probe_oracle: Any, feats: List[Dict[str, float]]
) -> List[Dict[str, float]]:
    """Batched per-axis predictions for many feature dicts — amortizes LightGBM predict ~120x.

    Uses the loaded oracle's public models/feature_names. Deliberately does NOT compute the
    per-row novelty percentile (non-batchable, ~0.7ms/graph): that is deferred to the few gate
    survivors via `score_graph_dict`. Returns one ``{axis: predicted}`` dict per input row.
    """
    oracle = probe_oracle.oracle
    names = oracle.feature_names
    X = np.array([[f.get(n, 0.0) for n in names] for f in feats], dtype=np.float64)
    cols = {ax: np.asarray(m.predict(X)) for ax, m in oracle.models.items()}
    return [
        {ax: round(float(cols[ax][i]), 4) for ax in cols} for i in range(len(feats))
    ]


def _flush_oracle_batch(
    scorer: CpuMechanismScorer,
    pend: List[tuple],
    kept: List[Scored],
    stats: Counter,
    pool: int,
) -> bool:
    """Batch-predict pending candidates, apply the capability gate, full-score survivors.

    Returns True once ``kept`` reaches ``pool``. The full probe dict (with novelty/recommendation
    for the output) is computed via ``score_graph_dict`` only for graphs that pass the gate — not
    for every generated graph, which is the whole point of the batching.
    """
    if not pend:
        return False
    preds = _batch_oracle_predict(scorer.probe_oracle, [p[5] for p in pend])
    for (fp, ops, prof, q, gd, _), pred in zip(pend, preds):
        if not _capability_gate_on_predictions(pred):
            stats["capability_no_go"] += 1
            continue
        kept.append(Scored(fp, ops, prof, q, gd, scorer.probe_oracle_score(gd)))
        stats["kept"] += 1
        if len(kept) >= pool:
            pend.clear()
            return True
    pend.clear()
    return False


def _generate_pool(
    scorer: CpuMechanismScorer,
    hist: Set[str],
    pool: int,
    max_attempts: int,
    seed0: int,
    progress_every: int,
    oracle_batch: int = 512,
) -> Tuple[List[Scored], Counter]:
    """Generate → cheap CPU gates → BATCHED oracle gate → mechanism-score. CPU only.

    The cheap gates (validity, novel-fp, template quality, failure risk) run per graph; graphs
    that survive them accumulate into a batch whose oracle predictions are computed in one
    vectorized call (LightGBM predict ~120x faster batched than per-row). The capability gate is
    applied on the batched predictions; only survivors pay the per-row novelty/full-probe cost.
    """
    cfg = GrammarConfig()
    stats: Counter = Counter()
    seen: Set[str] = set()
    kept: List[Scored] = []
    pend: List[tuple] = []  # cheap-gate survivors awaiting the batched oracle gate
    po = scorer.probe_oracle
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
        fr = q["failure_risk"]
        if fr["compile"] >= 0.4 or fr["lookahead"] >= 0.4 or fr["resource"] >= 0.4:
            stats["high_failure_risk"] += 1
            continue
        feats = None
        if po is not None:
            try:
                feats = po.extractor.features(gd["nodes"])
            except Exception:  # noqa: BLE001
                feats = None  # un-measurable ⇒ fall open below
        if feats is not None:
            pend.append((fp, sorted(op_set), prof, q, gd, feats))
            if len(pend) >= oracle_batch and _flush_oracle_batch(
                scorer, pend, kept, stats, pool
            ):
                break
        else:  # oracle disabled, or features unmeasurable: keep (fall open)
            kept.append(Scored(fp, sorted(op_set), prof, q, gd, None))
            stats["kept"] += 1
        if progress_every and (i + 1) % progress_every == 0:
            logger.info(
                "  attempts=%d kept=%d pend=%d (%.0f/s)",
                i + 1,
                len(kept),
                len(pend),
                (i + 1) / max(time.time() - t0, 1e-9),
            )
        if len(kept) >= pool:
            break
    if len(kept) < pool:  # final partial batch
        _flush_oracle_batch(scorer, pend, kept, stats, pool)
    return kept, stats


# --------------------------------------------------------------------------- #
# parallel generation — generation (~1ms/graph) is the throughput floor; the
# cheap CPU stages (generate → profile → template-quality → features) are
# embarrassingly parallel and torch-free, so fan them across processes. The
# oracle predict + capability gate + dedup stay in the (single) main process.
# --------------------------------------------------------------------------- #
_WORKER_SCORER: "CpuMechanismScorer | None" = None


def _worker_init(db: str, meta: str) -> None:
    """Per-worker scorer (no oracle — oracle scoring is centralized in the main process)."""
    global _WORKER_SCORER
    _WORKER_SCORER = CpuMechanismScorer(db, meta, use_probe_oracle=False)


def _cheap_screen_chunk(task: tuple[int, int]) -> Tuple[List[tuple], Counter]:
    """Worker: generate a seed range, apply the cheap CPU gates, return survivor tuples + stats.

    Survivors carry (fp, ops, profile, quality, graph_dict, features) — everything the main
    process needs for the batched oracle gate. No torch, no oracle: fork-safe and pickle-light.
    """
    seed0, n = task
    scorer = _WORKER_SCORER
    assert scorer is not None, "worker not initialized"
    cfg = GrammarConfig()
    stats: Counter = Counter()
    out: List[tuple] = []
    for i in range(seed0, seed0 + n):
        try:
            g = generate_layer_graph(cfg, seed=i)
        except Exception:  # noqa: BLE001
            stats["invalid"] += 1
            continue
        op_set, _, _, fp = _graph_features(g)
        gd = g.to_dict()
        q = score_template_quality(gd["nodes"])
        if not q["passes_must"]:
            stats["bad_template"] += 1
            continue
        fr = q["failure_risk"]
        if fr["compile"] >= 0.4 or fr["lookahead"] >= 0.4 or fr["resource"] >= 0.4:
            stats["high_failure_risk"] += 1
            continue
        prof = scorer.profile(gd["nodes"])
        out.append((fp, sorted(op_set), prof, q, gd, scorer.ext.features(gd["nodes"])))
    return out, stats


def _generate_pool_parallel(
    scorer: CpuMechanismScorer,
    hist: Set[str],
    pool: int,
    max_attempts: int,
    seed0: int,
    oracle_batch: int,
    workers: int,
    chunk: int,
    db: str,
    meta: str,
    progress_every: int,
) -> Tuple[List[Scored], Counter]:
    """Same funnel as ``_generate_pool`` but cheap stages fan out across ``workers`` processes.

    Chunks are consumed in seed order (ordered ``imap``) so dedup + pool-fill are deterministic and
    the result matches the serial path graph-for-graph (under a fixed hash seed). Oracle predict +
    capability gate run only in this process.
    """
    import multiprocessing as mp

    seen: Set[str] = set()
    kept: List[Scored] = []
    pend: List[tuple] = []
    stats: Counter = Counter()
    oracle_on = scorer.probe_oracle is not None
    tasks = [
        (seed0 + c, min(chunk, max_attempts - c)) for c in range(0, max_attempts, chunk)
    ]
    t0 = time.time()
    done = 0
    with mp.Pool(workers, initializer=_worker_init, initargs=(db, meta)) as p:
        for survivors, cstats in p.imap(_cheap_screen_chunk, tasks):
            stats.update(cstats)
            done += 1
            for fp, ops, prof, q, gd, feats in survivors:
                if fp in hist or fp in seen:
                    stats["already_seen"] += 1
                    continue
                seen.add(fp)
                if not oracle_on:  # no oracle gate → keep cheap survivors directly
                    kept.append(Scored(fp, ops, prof, q, gd, None))
                    stats["kept"] += 1
                    if len(kept) >= pool:
                        p.terminate()
                        return kept, stats
                    continue
                pend.append((fp, ops, prof, q, gd, feats))
                if len(pend) >= oracle_batch and _flush_oracle_batch(
                    scorer, pend, kept, stats, pool
                ):
                    p.terminate()
                    return kept, stats
            if progress_every and done % max(progress_every // chunk, 1) == 0:
                logger.info(
                    "  chunks=%d/%d kept=%d (%.0f graphs/s)",
                    done,
                    len(tasks),
                    len(kept),
                    (done * chunk) / max(time.time() - t0, 1e-9),
                )
            if len(kept) >= pool:
                p.terminate()
                break
    if len(kept) < pool:
        _flush_oracle_batch(scorer, pend, kept, stats, pool)
    return kept, stats


def _declared_key(s: Scored) -> tuple[float, float]:
    """Cheap coarse rank (declared-oracle rank_score, then mech) — selects the top-K to measure."""
    rank = (
        float(s.probe_oracle.get("label_free_probe_rank_score", 0.0))
        if s.probe_oracle
        else 0.0
    )
    return (rank, s.profile.mech_score)


def _exploit_key(s: Scored) -> tuple[float, float, float]:
    """Measured capability first (OOD-robust precision), then declared rank, then mech.

    Survivors scored by the measured ranker sort above un-scored ones (measured=-inf), so the
    exploit shortlist is chosen by the measured signal; declared rank / mech break ties and order
    the un-scored tail (e.g. when the measured ranker is disabled).
    """
    measured = s.measured_score if s.measured_score is not None else float("-inf")
    rank, mech = _declared_key(s)
    return (measured, rank, mech)


def _select(kept: List[Scored], n_exploit: int, n_explore: int) -> List[Scored]:
    """Explore∪exploit shortlist: exploit by measured-first capability, explore by novelty."""
    by_cap = sorted(kept, key=_exploit_key, reverse=True)[:n_exploit]
    by_nov = sorted(kept, key=lambda s: -s.profile.novelty)[:n_explore]
    out: Dict[str, Scored] = {s.fingerprint: s for s in by_cap}
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


def _measured_rank_topk(
    kept: List[Scored], k: int, enabled: bool, tau: float, device: str | None
) -> Tuple[List[Scored], int]:
    """Measured PRECISION ranker — the funnel's OOD-robust second stage.

    The declared oracle is anti-predictive on novel archs (it ranks known novel winners LOW, e.g.
    rank_score ~0.18 vs ~0.6 for declared-favored graphs), so probing only the top-K-by-declared
    would never measure them. We therefore measure the UNION of the top-K-by-declared survivors AND
    the top-K-by-novelty survivors — the latter is exactly where novel winners sit — set each one's
    measured ``capability_score`` (→ the primary exploit key, so a high-measured novel arch can win
    an exploit slot, not just the explore reserve), and drop the structurally dead
    (``long_range_reach < tau``). Bounded to ≤2K probes at ~0.4s each; fail-open on probe failure.
    """
    if not enabled or not kept or k <= 0:
        return kept, 0
    try:
        from research.tools.measured_descriptors import (
            MeasuredDescriptorExtractor,
            capability_score_from_descriptors,
        )

        mdx = MeasuredDescriptorExtractor(device=device, n_seeds=1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("measured ranker unavailable (%s) — declared ranking only", exc)
        return kept, 0
    measure: Dict[int, Scored] = {
        id(s): s for s in sorted(kept, key=_declared_key, reverse=True)[:k]
    }
    for s in sorted(kept, key=lambda x: -x.profile.novelty)[:k]:
        measure.setdefault(id(s), s)  # novel winners live here, not in declared-top
    dead: Set[int] = set()
    for s in measure.values():
        try:
            d = mdx.descriptors(json.dumps(s.graph_dict, separators=(",", ":")))
        except Exception:  # noqa: BLE001
            d = None
        if d is None:
            continue  # fail-open: leave un-scored (ranks below scored survivors)
        if float(d.get("long_range_reach", 0.0)) < tau:
            dead.add(id(s))  # structurally can't route ⇒ drop
            continue
        s.measured_score = capability_score_from_descriptors(d)
    if dead:
        kept = [s for s in kept if id(s) not in dead]
    return kept, len(dead)


def _measured_confirm_shortlist(
    shortlist: List[Scored], enabled: bool, tau: float, device: str | None
) -> Tuple[List[Scored], int]:
    """Structural confirmation of the shortlist's UN-scored (explore) picks only.

    Exploit picks already carry a ``measured_score`` from ``_measured_rank_topk`` (they survived the
    structural floor there). This drops only the explore/novelty picks the ranker never probed whose
    MEASURED computation can't route information backward (``long_range_reach < tau``). Fail-open on
    probe failure; bounded to the shortlist.
    """
    if not enabled or not shortlist:
        return shortlist, 0
    try:
        from research.tools.measured_descriptors import MeasuredDescriptorExtractor

        mdx = MeasuredDescriptorExtractor(device=device, n_seeds=1)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "measured confirmation unavailable (%s) — shortlist unfiltered", exc
        )
        return shortlist, 0
    kept: List[Scored] = []
    for s in shortlist:
        if s.measured_score is not None:
            kept.append(s)  # already structurally confirmed in the rank stage
            continue
        try:
            ok = mdx.induction_capable(
                json.dumps(s.graph_dict, separators=(",", ":")), threshold=tau
            )
        except Exception:  # noqa: BLE001
            ok = True  # fail-open: never drop a graph we could not measure
        if ok:
            kept.append(s)
    return kept, len(shortlist) - len(kept)


def _write_shortlist(out: Path, shortlist: List[Scored]) -> None:
    """Write the shortlist jsonl: structural profile + measured score + full probe dict + graph."""
    with out.open("w") as f:
        for s in shortlist:
            f.write(
                json.dumps(
                    {
                        "fingerprint": s.fingerprint,
                        "ops": s.ops,
                        "mech_score": round(s.profile.mech_score, 3),
                        "measured_score": (
                            round(s.measured_score, 6)
                            if s.measured_score is not None
                            else None
                        ),
                        "novelty": s.profile.novelty,
                        "mixer_depth": s.profile.mixer_depth,
                        "n_mixers_on_path": s.profile.n_mix,
                        "n_novel_mixers": s.profile.n_novel_mix,
                        "lit_family": s.profile.lit_family,
                        "lit_model": s.profile.lit_model,
                        "lit_match_type": s.profile.lit_match_type,
                        "template_quality": s.quality["score"],
                        "failure_risk": s.quality["failure_risk"],
                        **(s.probe_oracle or {}),
                        "graph": s.graph_dict,
                    }
                )
                + "\n"
            )


def run_generate(args: argparse.Namespace) -> Dict[str, Any]:
    scorer = CpuMechanismScorer(
        args.db, args.meta, use_probe_oracle=bool(args.probe_oracle)
    )
    hist = _historical_fingerprints(args.db)
    t0 = time.time()
    if args.workers and args.workers > 1:
        kept, stats = _generate_pool_parallel(
            scorer,
            hist,
            args.pool,
            args.max_attempts,
            args.seed0,
            args.oracle_batch,
            args.workers,
            args.chunk,
            args.db,
            args.meta,
            args.progress_every,
        )
    else:
        kept, stats = _generate_pool(
            scorer,
            hist,
            args.pool,
            args.max_attempts,
            args.seed0,
            args.progress_every,
            oracle_batch=args.oracle_batch,
        )
    # Funnel stage 2: measured PRECISION rank of the top-K declared survivors (OOD-robust).
    kept, n_structural_dead = _measured_rank_topk(
        kept,
        args.measured_rank_k,
        bool(args.measured_confirm),
        args.measured_tau,
        args.device,
    )
    selected = _select(kept, args.exploit, args.explore)
    shortlist = [s for s in selected if _context_rule_clean(s)]
    stats["context_rule_dropped"] = len(selected) - len(shortlist)
    shortlist, n_explore_dead = _measured_confirm_shortlist(
        shortlist, bool(args.measured_confirm), args.measured_tau, args.device
    )
    stats["measured_structural_dropped"] = n_structural_dead + n_explore_dead
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_shortlist(out, shortlist)
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
        "label_free_probe_oracle": {
            "enabled": bool(args.probe_oracle),
            "loaded": scorer.probe_oracle is not None,
            "scored_kept": sum(1 for s in kept if s.probe_oracle),
            "scored_shortlist": sum(1 for s in shortlist if s.probe_oracle),
        },
        "top_by_mech": [
            {
                "fp": s.fingerprint,
                "mech": round(s.profile.mech_score, 2),
                "label_free_probe_rank_score": (s.probe_oracle or {}).get(
                    "label_free_probe_rank_score"
                ),
                "label_free_probe_score": (s.probe_oracle or {}).get(
                    "label_free_probe_score"
                ),
                "label_free_probe_gate": (s.probe_oracle or {}).get(
                    "label_free_probe_gate"
                ),
                "ops": s.ops,
            }
            for s in (
                sorted(kept, key=_exploit_key, reverse=True)
                if any(s.probe_oracle for s in kept)
                else sorted(kept, key=lambda s: -s.profile.mech_score)
            )[:5]
        ],
    }


# --------------------------------------------------------------------------- #
# validate mode — recall of capable in the shortlist, on the labeled corpus
# --------------------------------------------------------------------------- #
# Allowlisted capability axes + their positive-class threshold (interpolation-safe SQL).
# mech_score is a structural induction-FAMILY circuit detector: it concentrates the capable
# 8-13x on induction / nano_induction_nearest / binding_curriculum (all "retrieve a token
# seen earlier"), but is BLIND to AR (ar_gate/ar_curriculum, a different mechanism — enrich
# <1x). AR Gate is handled as a no-go filter before non-AR exploit ranking.
_VALIDATE_LABELS: Dict[str, float] = {
    "induction_screening_auc": 0.35,
    "nano_induction_nearest_max_accuracy": 0.5,
    "binding_curriculum_auc": 0.5,
    "ar_gate_score": 0.8,
    "ar_curriculum_auc_pair_final": 0.5,
}


def _validate_axis(
    scorer: CpuMechanismScorer, con: Any, label_col: str, thr: float
) -> Dict[str, Any]:
    """Mech-score recall@topK + enrichment + global spearman against one capability axis."""
    if (
        label_col not in _VALIDATE_LABELS
    ):  # allowlist ⇒ f-string interpolation is injection-safe
        raise ValueError(f"label_col must be one of {tuple(_VALIDATE_LABELS)}")
    query = (
        f"SELECT g.graph_json, AVG(r.{label_col}) "
        "FROM graphs g JOIN graph_runs r ON g.graph_fingerprint=r.graph_fingerprint "
        f"WHERE g.graph_json_is_placeholder=0 AND r.{label_col} IS NOT NULL "
        "GROUP BY g.graph_fingerprint"
    )
    rows = con.execute(query).fetchall()  # nosec B608  # nosemgrep: python-sql-string-formatting
    mech: List[float] = []
    y: List[float] = []
    t0 = time.time()
    for gj, val in rows:
        try:
            nodes = json.loads(gj)["nodes"]
        except Exception:
            continue
        mech.append(scorer.profile(nodes).mech_score)
        y.append(float(val))
    mech_a = np.array(mech)
    y_a = np.array(y)
    pos = y_a > thr
    order = np.argsort(-mech_a)
    n = len(y_a)
    base_rate = float(pos.mean()) if n else 0.0
    rk: Dict[str, Any] = {}
    for frac in (0.05, 0.10, 0.20, 0.30):
        k = max(int(n * frac), 1)
        top = order[:k]
        rk[f"top_{int(frac * 100)}pct"] = {
            "recall_of_capable": round(float(pos[top].sum() / max(pos.sum(), 1)), 3),
            "precision": round(float(pos[top].mean()), 3),
            "enrichment_vs_base": round(
                float(pos[top].mean() / max(base_rate, 1e-9)), 2
            ),
        }
    sp = (
        float(spearmanr(mech_a, y_a)[0])  # type: ignore[arg-type]  # scipy stub returns opaque tuple
        if n > 2 and pos.any()
        else float("nan")
    )
    return {
        "label_col": label_col,
        "threshold": thr,
        "n": n,
        "n_capable": int(pos.sum()),
        "base_rate": round(base_rate, 4),
        "spearman_mech_label": round(sp, 3),
        "graphs_per_s": round(n / max(time.time() - t0, 1e-9)),
        "recall_at_topk": rk,
    }


def run_validate(args: argparse.Namespace) -> Dict[str, Any]:
    import sqlite3

    scorer = CpuMechanismScorer(args.db, args.meta, use_probe_oracle=False)
    cols = list(_VALIDATE_LABELS) if args.label_col == "all" else [args.label_col]
    con = sqlite3.connect(args.db)
    try:
        axes = {
            col: _validate_axis(
                scorer,
                con,
                col,
                args.thr
                if (args.thr is not None and args.label_col != "all")
                else _VALIDATE_LABELS[col],
            )
            for col in cols
        }
    finally:
        con.close()
    if len(axes) == 1:
        return next(iter(axes.values()))
    return {
        "axes": axes,
        "note": (
            "mech_score is a structural induction-family detector (strong on induction / "
            "nano_induction_nearest / binding_curriculum); AR axes are served by the trained "
            "probe-oracle max-axis layer, not the structural score."
        ),
    }


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
    p.add_argument(
        "--label-col",
        default="induction_screening_auc",
        choices=(*_VALIDATE_LABELS, "all"),
        help="validate mode: capability axis to score mech_score against ('all' = every axis).",
    )
    p.add_argument(
        "--thr",
        type=float,
        default=None,
        help="validate mode: positive-class threshold override (single-axis only).",
    )
    p.add_argument("--pool", type=int, default=50000)
    p.add_argument("--max-attempts", type=int, default=200000)
    p.add_argument("--seed0", type=int, default=11_000_000)
    p.add_argument("--exploit", type=int, default=200)
    p.add_argument("--explore", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=20000)
    p.add_argument(
        "--oracle-batch",
        type=int,
        default=512,
        help="cheap-gate survivors per batched oracle-predict call (amortizes LightGBM ~120x).",
    )
    p.add_argument("--out", default="research/reports/cpu_cascade_shortlist.jsonl")
    p.add_argument(
        "--no-probe-oracle",
        dest="probe_oracle",
        action="store_false",
        help="Disable persisted label-free AR/nano probe-oracle scoring.",
    )
    p.set_defaults(probe_oracle=True)
    p.add_argument(
        "--no-measured-confirm",
        dest="measured_confirm",
        action="store_false",
        help="Disable the measured long_range_reach structural confirmation on the shortlist.",
    )
    p.set_defaults(measured_confirm=True)
    p.add_argument(
        "--measured-tau",
        type=float,
        default=0.01,
        help="long_range_reach floor for the measured shortlist confirmation (validated n=1102).",
    )
    p.add_argument(
        "--measured-rank-k",
        type=int,
        default=1500,
        help="top-K declared survivors to measured-rank (OOD-robust precision; ~0.4s/graph).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes for the cheap generation stage (1 = serial; >1 fans out, near-linear).",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=2000,
        help="seeds per worker task (parallel generation).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="device for the measured confirmation probe (default: auto cuda/cpu).",
    )
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
