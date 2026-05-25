#!/usr/bin/env python
"""Score the ground-truth-probed novel graphs with name vs math+structure models.

The leave-op-out test showed the historical labels contain almost no capable examples of
the novel mechanisms (stdp/clifford/ultrametric max induction ~0.02) — yet we probed a
generated stdp graph at 0.94. So this asks the transfer question directly: when a model is
trained on the historical labels and shown the ACTUAL probed winners, does the math+structure
representation predict their measured capability better than op-names? Names see "stdp =
historically bad"; math+structure can transfer the induction recipe via the winner's scaffold
(attention + rope + norm + right receptive field) that it learned from softmax graphs.

Regenerates the probed pool (deterministic), matches the probed fingerprints, computes both
representations live, scores with both trained models, compares to measured induction.

Usage::  python -m research.tools.score_probed_with_reps
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from research.defaults import RUNS_DB
from research.tools.backfill_graph_semantics import load_semantic_features
from research.tools.capability_screener import _static_matrix, featurize_op_sets
from research.tools.capability_shrinkage_denoise import (
    _NONE_CLUSTER,
    _shrink,
    _template_map,
)
from research.tools.graph_semantic_features import GraphSemanticExtractor
from research.tools.induction_predictor_foundation import _build_corpus, _fit_gbm
from research.tools.probe_novel_candidates import _collect_pool

logger = logging.getLogger(__name__)


def _train_models(db_path: str, shrink_f: float):
    _, _, ind_all, _, fps_all = _build_corpus(db_path)
    Xsem, sem_names, present = load_semantic_features(fps_all)
    pos = {fp: i for i, fp in enumerate(fps_all)}
    idx = np.array([pos[fp] for fp in present])
    ind = ind_all[idx]
    Xname, _name_names, name_vocab = _static_matrix(db_path, present)
    tmpl = _template_map(db_path)
    clusters = [tmpl.get(fp, _NONE_CLUSTER) for fp in present]
    y = _shrink(ind, clusters, np.ones(len(present), dtype=bool), shrink_f)
    name_model = _fit_gbm(Xname, y)
    sem_model = _fit_gbm(Xsem, y)
    return name_model, name_vocab, sem_model, sem_names


def run(db_path: str, shrink_f: float, probes_file: str) -> Dict[str, Any]:
    actuals = {
        r["fingerprint"]: r["actual_induction_auc"]
        for r in json.loads(Path(probes_file).read_text())["results"]
        if r.get("actual_induction_auc") is not None
    }
    name_model, name_vocab, sem_model, sem_names = _train_models(db_path, shrink_f)
    ext = GraphSemanticExtractor(db_path)

    # Regenerate the probed pool (same seed0/pool as the probe run) to recover graphs.
    cand = _collect_pool(db_path, pool=200, max_attempts=4000, seed0=3_000_000)
    by = {c["fingerprint"]: c for c in cand}

    rows: List[Dict[str, Any]] = []
    for fp, actual in actuals.items():
        c = by.get(fp)
        if c is None:
            continue
        nodes = c["graph"].to_dict()["nodes"]
        Xn = featurize_op_sets(
            [set(c["ops"])], [c["op_count"]], [c["pair_count"]], name_vocab
        )
        feats = ext.features(nodes)
        Xs = np.array([[feats.get(n, 0.0) for n in sem_names]], dtype=np.float64)
        rows.append(
            {
                "fingerprint": fp,
                "actual_induction": round(float(actual), 4),
                "pred_name": round(float(name_model.predict(Xn)[0]), 4),
                "pred_math_structure": round(float(sem_model.predict(Xs)[0]), 4),
                "novel_mixers": c["novel_mixers"],
            }
        )
    rows.sort(key=lambda r: -r["actual_induction"])
    act = np.array([r["actual_induction"] for r in rows])
    pn = np.array([r["pred_name"] for r in rows])
    ps = np.array([r["pred_math_structure"] for r in rows])

    def _corr(a, b):
        return (
            round(float(np.corrcoef(a, b)[0, 1]), 3)
            if len(a) > 2 and a.std() > 0 and b.std() > 0
            else None
        )

    return {
        "n_probed_scored": len(rows),
        "corr_name_vs_actual": _corr(pn, act),
        "corr_math_structure_vs_actual": _corr(ps, act),
        "rows": rows,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--shrink", type=float, default=0.75)
    p.add_argument("--probes", default="research/reports/novel_candidate_probes.json")
    args = p.parse_args()
    print(json.dumps(run(args.db, args.shrink, args.probes), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
