#!/usr/bin/env python
"""Capability oracle — the operational synthesis: predict + confidence + route the unknown.

Ties the framework together into one decision per architecture:
  1. capability prediction from a SELECTED top-k math+structure feature set (feature_count_sweep
     showed ~10-20 selected features generalize best — accretion hurts).
  2. novelty / OOD score in property space (novelty_scorer).
  3. a recommendation:
       - high novelty (>= novelty_pctile_thr)  -> EXPLORE_PROBE  (model has no basis; mint a label)
       - else predicted >= op_thr               -> PREDICT_GOOD
       - else                                    -> PREDICT_BAD

This is how the framework handles invented/never-seen math: confidently judge what it knows, and
route the genuinely-novel to the probe rather than mis-predicting it. Works on any graph (live or
historical graph_json) via the semantic extractor.

Usage::  python -m research.tools.capability_oracle               # train + demo on probed winners
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from research.defaults import RUNS_DB
from research.tools.backfill_graph_semantics import load_semantic_features
from research.tools.capability_shrinkage_denoise import (
    _NONE_CLUSTER,
    _shrink,
    _template_map,
)
from research.tools.graph_semantic_features import GraphSemanticExtractor
from research.tools.induction_predictor_foundation import _build_corpus, _fit_gbm
from research.tools.novelty_scorer import NoveltyScorer
from research.tools.probe_novel_candidates import _collect_pool


class CapabilityOracle:
    def __init__(
        self,
        model,
        sel_names: List[str],
        scorer: NoveltyScorer,
        op_thr: float,
        novelty_pctile_thr: float,
    ) -> None:
        self.model = model
        self.sel_names = sel_names
        self.scorer = scorer
        self.op_thr = op_thr
        self.novelty_pctile_thr = novelty_pctile_thr

    @classmethod
    def train(
        cls,
        db_path: str,
        top_k: int,
        shrink_f: float,
        op_thr: float,
        novelty_pctile_thr: float,
    ) -> "CapabilityOracle":
        _, _, ind_all, _, fps_all = _build_corpus(db_path)
        X, names, present = load_semantic_features(fps_all)
        pos = {fp: i for i, fp in enumerate(fps_all)}
        ind = ind_all[np.array([pos[fp] for fp in present])]
        tmpl = _template_map(db_path)
        clusters = [tmpl.get(fp, _NONE_CLUSTER) for fp in present]
        y = _shrink(ind, clusters, np.ones(len(present), dtype=bool), shrink_f)
        # select top-k features by importance (full-fit), then refit on the subset
        order = np.argsort(
            -np.asarray(_fit_gbm(X, y).feature_importances_, dtype=np.float64)
        )
        sel = order[:top_k]
        sel_names = [names[i] for i in sel]
        model = _fit_gbm(X[:, sel], y)
        scorer = NoveltyScorer(X[:, sel], sel_names, k=10)
        return cls(model, sel_names, scorer, op_thr, novelty_pctile_thr)

    def evaluate_features(self, feats: Dict[str, float]) -> Dict[str, Any]:
        x = np.array([[feats.get(n, 0.0) for n in self.sel_names]], dtype=np.float64)
        pred = float(self.model.predict(x)[0])
        nov = float(self.scorer.novelty(x)[0])
        pctile = self.scorer.percentile(nov)
        if pctile >= self.novelty_pctile_thr:
            rec = "EXPLORE_PROBE"
        elif pred >= self.op_thr:
            rec = "PREDICT_GOOD"
        else:
            rec = "PREDICT_BAD"
        return {
            "predicted_capability": round(pred, 4),
            "novelty_pctile": round(pctile, 3),
            "recommendation": rec,
        }


def run(
    db_path: str,
    top_k: int,
    shrink_f: float,
    op_thr: float,
    novelty_pctile_thr: float,
    probes_file: str,
) -> Dict[str, Any]:
    oracle = CapabilityOracle.train(
        db_path, top_k, shrink_f, op_thr, novelty_pctile_thr
    )
    out: Dict[str, Any] = {
        "selected_features": oracle.sel_names,
        "op_threshold": op_thr,
        "novelty_pctile_threshold": novelty_pctile_thr,
    }
    pf = Path(probes_file)
    if not pf.exists():
        out["note"] = f"{probes_file} missing; trained only"
        return out
    probes = [
        r
        for r in json.loads(pf.read_text())["results"]
        if r.get("actual_induction_auc") is not None
    ]
    cand = {c["fingerprint"]: c for c in _collect_pool(db_path, 600, 12000, 3_000_000)}
    ext = GraphSemanticExtractor(db_path)
    rows = []
    for r in probes:
        c = cand.get(r["fingerprint"])
        if c is None:
            continue
        decision = oracle.evaluate_features(ext.features(c["graph"].to_dict()["nodes"]))
        rows.append(
            {
                "fingerprint": r["fingerprint"],
                "actual_induction": round(float(r["actual_induction_auc"]), 4),
                **decision,
                "novel_mixers": r.get("novel_mixers", []),
            }
        )
    rows.sort(key=lambda x: -x["actual_induction"])
    out["probed_decisions"] = rows
    routed = sum(1 for r in rows if r["recommendation"] == "EXPLORE_PROBE")
    out["summary"] = {
        "n": len(rows),
        "routed_to_probe": routed,
        "capable_correctly_routed_or_predicted_good": sum(
            1
            for r in rows
            if r["actual_induction"] > 0.35
            and r["recommendation"] in ("EXPLORE_PROBE", "PREDICT_GOOD")
        ),
        "n_capable": sum(1 for r in rows if r["actual_induction"] > 0.35),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--shrink", type=float, default=0.75)
    p.add_argument("--op-thr", type=float, default=0.15)
    p.add_argument("--novelty-pctile", type=float, default=0.9)
    p.add_argument("--probes", default="research/reports/novel_candidate_probes.json")
    args = p.parse_args()
    print(
        json.dumps(
            run(
                args.db,
                args.top_k,
                args.shrink,
                args.op_thr,
                args.novelty_pctile,
                args.probes,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
