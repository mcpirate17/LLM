#!/usr/bin/env python
"""Novelty / out-of-distribution scorer in math-property space — for the truly unknown.

A label-trained model cannot predict the capability of a genuinely-novel design (proven:
score_probed_with_reps gave the 0.94 stdp winner ~0.04). The right response is NOT a
confident-wrong prediction — it is to flag the design as OUT-OF-DISTRIBUTION and route it to
the probe (active exploration). This scorer measures how far a graph sits from the explored
region in standardized math+structure property space (mean distance to k nearest training
graphs). High novelty + non-trivial capability prior = best exploration candidate.

Exploration policy (per candidate):
  - low novelty,  pred high -> trust: confident-good.
  - low novelty,  pred low  -> skip: confident-bad.
  - high novelty            -> PROBE: the model has no basis; mint a label here.

Usage::  python -m research.tools.novelty_scorer          # demo: novelty of probed winners
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from research.defaults import RUNS_DB
from research.tools.backfill_graph_semantics import load_semantic_features
from research.tools.graph_semantic_features import GraphSemanticExtractor
from research.tools.induction_predictor_foundation import _build_corpus
from research.tools.probe_novel_candidates import _collect_pool


class NoveltyScorer:
    """kNN distance in standardized property space = distance from the explored region."""

    def __init__(
        self, X_train: np.ndarray, feature_names: List[str], k: int = 10
    ) -> None:
        self.names = feature_names
        self.k = k
        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        self.Z = (X_train - self.mean) / self.std
        self._train_self = self._knn_mean(self.Z, exclude_self=True)

    def _knn_mean(self, Z: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        out = np.empty(len(Z), dtype=np.float64)
        for i in range(len(Z)):
            d = np.sqrt(((self.Z - Z[i]) ** 2).sum(axis=1))
            if exclude_self:
                d = np.sort(d)[1 : self.k + 1]
            else:
                d = np.sort(d)[: self.k]
            out[i] = float(d.mean())
        return out

    def novelty(self, X: np.ndarray) -> np.ndarray:
        return self._knn_mean((X - self.mean) / self.std)

    def percentile(self, value: float) -> float:
        return float((self._train_self < value).mean())


def _features_dict_to_matrix(
    dicts: List[Dict[str, float]], names: List[str]
) -> np.ndarray:
    return np.array([[d.get(n, 0.0) for n in names] for d in dicts], dtype=np.float64)


def run(db_path: str, probes_file: str, k: int) -> Dict[str, Any]:
    _, _, _, _, fps_all = _build_corpus(db_path)
    Xtrain, names, _present = load_semantic_features(fps_all)
    scorer = NoveltyScorer(Xtrain, names, k=k)
    train_nov = scorer._train_self
    out: Dict[str, Any] = {
        "n_train": len(Xtrain),
        "train_novelty_p50": round(float(np.percentile(train_nov, 50)), 3),
        "train_novelty_p95": round(float(np.percentile(train_nov, 95)), 3),
    }
    pf = Path(probes_file)
    if not pf.exists():
        out["note"] = f"{probes_file} not found; ran train-only novelty stats"
        return out
    probes = [
        r
        for r in json.loads(pf.read_text())["results"]
        if r.get("actual_induction_auc") is not None
    ]
    cand = _collect_pool(db_path, pool=600, max_attempts=12000, seed0=3_000_000)
    by = {c["fingerprint"]: c for c in cand}
    ext = GraphSemanticExtractor(db_path)
    rows: List[Dict[str, Any]] = []
    for r in probes:
        c = by.get(r["fingerprint"])
        if c is None:
            continue
        feats = ext.features(c["graph"].to_dict()["nodes"])
        nov = float(scorer.novelty(_features_dict_to_matrix([feats], names))[0])
        rows.append(
            {
                "fingerprint": r["fingerprint"],
                "actual_induction": round(float(r["actual_induction_auc"]), 4),
                "novelty": round(nov, 3),
                "novelty_pctile_vs_train": round(scorer.percentile(nov), 3),
                "novel_mixers": r.get("novel_mixers", []),
            }
        )
    rows.sort(key=lambda x: -x["novelty"])
    out["probed_graphs"] = rows
    capable = [x for x in rows if x["actual_induction"] > 0.35]
    out["capable_mean_novelty_pctile"] = (
        round(float(np.mean([x["novelty_pctile_vs_train"] for x in capable])), 3)
        if capable
        else None
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--probes", default="research/reports/novel_candidate_probes.json")
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()
    print(json.dumps(run(args.db, args.probes, args.k), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
