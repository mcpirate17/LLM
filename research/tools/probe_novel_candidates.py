#!/usr/bin/env python
"""Ground-truth a stratified sample of novel-mixer graphs with the REAL induction probe.

Closes the discovery loop: the screener only PREDICTS induction; this runs the canonical
500-step screening probe (induction_score_gold — the exact metric the screener was trained
to predict) on actual compiled models, so we learn whether the screener's bets pay off and
whether any genuinely-novel mixer actually learns.

Two strata are probed:
  - top_predicted : highest screener score (usually novel-mixer + softmax/rope hybrids).
  - exotic_mixer  : graphs whose novel mixer is genuinely exotic (tropical/clifford/
                    ultrametric/stdp) — where the screener is skeptical/uninformative.

Standalone: results go to research/reports/, NOTHING is written to the notebook/runs.db
(so the partial-data S1 guardrails are not involved).

Usage::

    python -m research.tools.probe_novel_candidates --n-top 6 --n-exotic 6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from research.defaults import RUNS_DB
from research.eval.native_induction import induction_score_gold
from research.synthesis.compiler import compile_model
from research.synthesis.grammar import GrammarConfig, generate_layer_graph
from research.tools.capability_screener import featurize_op_sets, load_screener
from research.tools.generate_novel_screened import (
    _build_op_weights,
    _graph_features,
    _historical_fingerprints,
    _novel_mixers,
    _scaffold_boost,
)

logger = logging.getLogger(__name__)

# Genuinely exotic mixers (not attention-flavoured) — where novelty really lives.
_EXOTIC = {
    "tropical_attention",
    "tropical_softmax",
    "ultrametric_attention",
    "stdp_attention",
    "clifford_attention",
}


def _collect_pool(
    db_path: str, pool: int, max_attempts: int, seed0: int
) -> List[Dict[str, Any]]:
    """Generate novel + novel-mixer graphs (keeping the graph objects), screen them."""
    hist = _historical_fingerprints(db_path)
    novel_mixers = _novel_mixers(db_path)
    scaffold = _scaffold_boost(db_path, 0.35)
    op_weights = _build_op_weights(novel_mixers, scaffold, 6.0, 2.5)
    cfg = GrammarConfig(op_weights=op_weights)
    booster, meta = load_screener()
    vocab = list(meta["op_vocab"])

    cand: List[Dict[str, Any]] = []
    seen_new: set = set()
    for i in range(max_attempts):
        try:
            g = generate_layer_graph(cfg, seed=seed0 + i)
        except Exception:
            continue
        op_set, op_count, pair_count, fp = _graph_features(g)
        if fp in hist or fp in seen_new:
            continue
        seen_new.add(fp)
        hits = sorted(op_set & novel_mixers)
        if not hits:
            continue
        cand.append(
            {
                "graph": g,
                "fingerprint": fp,
                "ops": sorted(op_set),
                "op_count": op_count,
                "pair_count": pair_count,
                "novel_mixers": hits,
            }
        )
        if len(cand) >= pool:
            break
    X = featurize_op_sets(
        [set(c["ops"]) for c in cand],
        [c["op_count"] for c in cand],
        [c["pair_count"] for c in cand],
        vocab,
    )
    for c, s in zip(cand, np.asarray(booster.predict(X), dtype=np.float64)):
        c["predicted"] = round(float(s), 4)
    return cand


def _stratified(
    cand: List[Dict[str, Any]], n_top: int, n_exotic: int
) -> List[Dict[str, Any]]:
    by_score = sorted(cand, key=lambda c: -c["predicted"])
    picked: Dict[str, Dict[str, Any]] = {}
    for c in by_score[:n_top]:
        c["stratum"] = "top_predicted"
        picked[c["fingerprint"]] = c
    exotic = [c for c in by_score if set(c["novel_mixers"]) & _EXOTIC]
    for c in exotic[:n_exotic]:
        picked.setdefault(c["fingerprint"], {**c, "stratum": "exotic_mixer"})
    return list(picked.values())


def _probe_one(c: Dict[str, Any], n_layers: int, device: str) -> Dict[str, Any]:
    rec = {
        k: c[k] for k in ("fingerprint", "ops", "novel_mixers", "predicted", "stratum")
    }
    t0 = time.time()
    try:
        model = compile_model([c["graph"]] * n_layers)
        res = induction_score_gold(model, device=device, seed=0)
        rec["actual_induction_auc"] = round(float(res.auc), 4)
        rec["status"] = res.status
        rec["gap_accuracies"] = {
            str(k): round(float(v), 3) for k, v in (res.gap_accuracies or {}).items()
        }
    except Exception as exc:
        rec["actual_induction_auc"] = None
        rec["status"] = f"probe_failed: {type(exc).__name__}: {exc}"
    rec["probe_s"] = round(time.time() - t0, 1)
    return rec


def _maybe_measured_prefilter(
    cand: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """OPT-IN closed-book pre-probe filter (ARIA_MEASURED_PREFILTER=1; default off).

    Drops structurally-induction-incapable candidates (cheap 1-layer long_range_reach < tau)
    before the expensive gold probe. Validated safe (prospective: 0 capable skipped, ROC 0.91 >
    screener 0.77; keeps the high-reach novel-winner region). tau via ARIA_MEASURED_PREFILTER_TAU.
    """
    if os.environ.get("ARIA_MEASURED_PREFILTER") != "1":
        return cand, {"enabled": False}
    from research.tools.measured_descriptors import MeasuredDescriptorExtractor

    tau = float(os.environ.get("ARIA_MEASURED_PREFILTER_TAU", "0.01"))
    mdx = MeasuredDescriptorExtractor(n_seeds=1)
    n0 = len(cand)
    kept = [
        c for c in cand if mdx.induction_capable(json.dumps(c["graph"].to_dict()), tau)
    ]
    logger.info(
        "measured prefilter ON (tau=%.3g): kept %d/%d, skipped %d",
        tau,
        len(kept),
        n0,
        n0 - len(kept),
    )
    return kept, {
        "enabled": True,
        "tau": tau,
        "pool_pre": n0,
        "kept": len(kept),
        "skipped": n0 - len(kept),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    logger.info("collecting candidate pool ...")
    cand = _collect_pool(args.db, args.pool, args.max_attempts, args.seed0)
    cand, prefilter_stats = _maybe_measured_prefilter(cand)
    sample = _stratified(cand, args.n_top, args.n_exotic)
    logger.info(
        "probing %d graphs (real induction, %d layers, %s) ...",
        len(sample),
        args.n_layers,
        args.device,
    )
    results = []
    for i, c in enumerate(sample):
        r = _probe_one(c, args.n_layers, args.device)
        logger.info(
            "  [%d/%d] pred=%.3f actual=%s %s mixers=%s",
            i + 1,
            len(sample),
            r["predicted"],
            r["actual_induction_auc"],
            r["status"],
            r["novel_mixers"],
        )
        results.append(r)

    ok = [r for r in results if r["actual_induction_auc"] is not None]
    pred = np.array([r["predicted"] for r in ok]) if ok else np.array([])
    act = np.array([r["actual_induction_auc"] for r in ok]) if ok else np.array([])
    summary: Dict[str, Any] = {
        "pool_size": len(cand),
        "measured_prefilter": prefilter_stats,
        "probed": len(results),
        "probe_ok": len(ok),
        "actual_capable_count": int((act > 0.35).sum()) if ok else 0,
        "actual_max": round(float(act.max()), 4) if ok else None,
        "pred_vs_actual_corr": round(float(np.corrcoef(pred, act)[0, 1]), 3)
        if len(ok) > 2 and act.std() > 0
        else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    return {
        "summary": summary,
        "out": out.as_posix(),
        "results": sorted(results, key=lambda r: -(r["actual_induction_auc"] or -1)),
    }


def _multiseed_probe(
    c: Dict[str, Any], n_layers: int, device: str, seeds: List[int]
) -> Dict[str, Any]:
    aucs: List[Any] = []
    for s in seeds:
        try:
            model = compile_model([c["graph"]] * n_layers)
            res = induction_score_gold(model, device=device, seed=s)
            aucs.append(round(float(res.auc), 4))
        except Exception as exc:
            logger.info("    seed %d failed: %s", s, exc)
            aucs.append(None)
    valid = [a for a in aucs if a is not None]
    arr = np.array(valid) if valid else np.array([])
    return {
        "fingerprint": c["fingerprint"],
        "novel_mixers": c["novel_mixers"],
        "ops": c["ops"],
        "per_seed_auc": aucs,
        "median": round(float(np.median(arr)), 4) if valid else None,
        "min": round(float(arr.min()), 4) if valid else None,
        "max": round(float(arr.max()), 4) if valid else None,
        "std": round(float(arr.std()), 4) if valid else None,
        "capable_seeds": int((arr > 0.35).sum()) if valid else 0,
        "n_seeds": len(valid),
    }


def confirm(args: argparse.Namespace, fps: List[str]) -> Dict[str, Any]:
    """Recover specific graphs by fingerprint and re-probe each across N seeds."""
    cand = _collect_pool(args.db, args.pool, args.max_attempts, args.seed0)
    by = {c["fingerprint"]: c for c in cand}
    seeds = list(range(args.n_seeds))
    logger.info("confirming %d fingerprints across seeds %s ...", len(fps), seeds)
    results = []
    for fp in fps:
        c = by.get(fp)
        if c is None:
            results.append({"fingerprint": fp, "error": "not_recovered_in_pool"})
            continue
        r = _multiseed_probe(c, args.n_layers, args.device, seeds)
        logger.info(
            "  %s  median=%s  range=[%s,%s]  capable_seeds=%d/%d  per_seed=%s",
            fp[:12],
            r["median"],
            r["min"],
            r["max"],
            r["capable_seeds"],
            r["n_seeds"],
            r["per_seed_auc"],
        )
        results.append(r)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"confirmation": results, "seeds": seeds}, indent=2))
    return {"confirmation": results, "out": out.as_posix()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--pool", type=int, default=400)
    p.add_argument("--max-attempts", type=int, default=8000)
    p.add_argument("--seed0", type=int, default=3_000_000)
    p.add_argument("--n-top", type=int, default=6)
    p.add_argument("--n-exotic", type=int, default=6)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="research/reports/novel_candidate_probes.json")
    p.add_argument(
        "--confirm-fps",
        default="",
        help="comma-separated fingerprints to multi-seed confirm",
    )
    p.add_argument("--n-seeds", type=int, default=5)
    args = p.parse_args()
    if args.confirm_fps:
        report = confirm(
            args, [f.strip() for f in args.confirm_fps.split(",") if f.strip()]
        )
    else:
        report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
