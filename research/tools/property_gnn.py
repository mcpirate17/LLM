#!/usr/bin/env python
"""Property-conditioned Graph Neural Network with uncertainty — the RIGHT model.

Replaces the flat GBM-on-aggregated-features (which overfits as properties grow — a verdict on the
model, not on features) with a learner that is compositional, scales with properties, and knows what
it does not know:

  - ML type:  representation-learning GNN (not fixed-capacity trees).
  - cascade:  K message-passing layers over the op-DAG — prediction is compositional like the graph.
  - gating:   learned attention readout over nodes (contextual, not global feature selection).
  - features: each op is a NODE carrying its full math-property vector — a never-seen op is just a
              node with a vector, so generalization to novel ops is structural, no one-hot needed.
  - unknown:  a deep ensemble gives (mean, std). OOD designs -> high std -> "don't trust, PROBE".
              This unifies capability prediction and novelty in one principled model.

The point: this model should IMPROVE (or not degrade) as properties are added — the thing that makes
infinite property extensibility real. Trains on the labeled induction corpus; evaluates temporal ROC
and ensemble uncertainty on the probed novel winners.

Usage::  python -m research.tools.property_gnn --epochs 60 --ensemble 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from research.defaults import RUNS_DB
from research.synthesis.op_roles import OpRole, get_role
from research.tools.capability_shrinkage_denoise import (
    _NONE_CLUSTER,
    _shrink,
    _template_map,
)
from research.tools.graph_semantic_features import (
    _ALGEBRA_SPACES,
    _MEMORY_ORDINAL,
    _NUMERIC_PROPS,
    GraphSemanticExtractor,
)
from research.tools.induction_predictor_foundation import _build_corpus

logger = logging.getLogger(__name__)
_ROLES = [r.value for r in OpRole]
_CFG_KEYS = {
    "cfg_topk": ("top_k", "k", "num_experts", "n_experts"),
    "cfg_depth": ("max_depth", "max_iterations", "recursion_depth"),
    "cfg_mlp_ratio": ("mlp_ratio", "compression_ratio"),
    "cfg_temp": ("route_temperature", "temperature", "tau"),
    "cfg_span": ("span_width", "window", "kernel_size"),
    "cfg_lanes": ("lane_count", "n_lanes"),
}
_MAX_NODES = 32


class PerOpFeaturizer:
    """Per-op node feature vector: math props + algebra/role/receptive/memory + config scalars."""

    def __init__(self, ext: GraphSemanticExtractor) -> None:
        self.ext = ext
        self.names = (
            [f"num_{p}" for p in _NUMERIC_PROPS]
            + [f"alg_{a}" for a in _ALGEBRA_SPACES]
            + [f"role_{r}" for r in _ROLES]
            + ["receptive_global", "memory_ord"]
            + list(_CFG_KEYS.keys())
        )
        self.dim = len(self.names)

    def vector(self, op: str, config: Dict[str, Any]) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        num = self.ext._op_numeric(op)
        for i, p in enumerate(_NUMERIC_PROPS):
            x = num.get(p)
            if x is not None and np.isfinite(x):
                v[i] = x
        off = len(_NUMERIC_PROPS)
        alg = self.ext.op_algebra.get(op, "")
        if alg in _ALGEBRA_SPACES:
            v[off + _ALGEBRA_SPACES.index(alg)] = 1.0
        off += len(_ALGEBRA_SPACES)
        role = get_role(op).value
        if role in _ROLES:
            v[off + _ROLES.index(role)] = 1.0
        off += len(_ROLES)
        v[off] = 1.0 if self.ext.op_receptive.get(op) == "global" else 0.0
        v[off + 1] = _MEMORY_ORDINAL.get(self.ext.op_memory.get(op, ""), 0.0)
        off += 2
        for j, keys in enumerate(_CFG_KEYS.values()):
            for kk in keys:
                if kk in (config or {}) and config[kk] is not None:
                    try:
                        v[off + j] = float(config[kk])
                        break
                    except (TypeError, ValueError):
                        pass
        return v


def _graph_tensors(nodes, feat: PerOpFeaturizer):
    node_list = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
    node_list = node_list[:_MAX_NODES]
    ids = {n["id"]: i for i, n in enumerate(node_list)}
    len(node_list)
    X = np.zeros((_MAX_NODES, feat.dim), dtype=np.float32)
    A = np.zeros((_MAX_NODES, _MAX_NODES), dtype=np.float32)
    mask = np.zeros(_MAX_NODES, dtype=np.float32)
    for i, n in enumerate(node_list):
        X[i] = feat.vector(str(n["op_name"]), n.get("config") or {})
        mask[i] = 1.0
        A[i, i] = 1.0
        for src in n.get("input_ids", []) or []:
            if src in ids:
                j = ids[src]
                A[i, j] = 1.0
                A[j, i] = 1.0
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    A = A / deg  # row-normalized mean aggregation
    return X, A, mask


def _load_graph_json(db_path: str, fps: List[str]) -> Dict[str, Any]:
    con = sqlite3.connect(db_path)
    out = {}
    qmarks = ",".join("?" * len(fps))
    for fp, gj in con.execute(
        f"SELECT graph_fingerprint, graph_json FROM graphs WHERE graph_fingerprint IN ({qmarks})",  # nosec B608  # nosemgrep: python-sql-string-formatting
        fps,
    ):
        try:
            out[str(fp)] = json.loads(gj)["nodes"]
        except Exception:
            pass
    con.close()
    return out


class PropertyGNN(nn.Module):
    def __init__(
        self, in_dim: int, hidden: int = 96, layers: int = 3, dropout: float = 0.1
    ):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.msg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout)
                )
                for _ in range(layers)
            ]
        )
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, X, A, mask):
        h = torch.relu(self.embed(X))
        m = mask.unsqueeze(-1)
        for layer in self.msg:
            agg = torch.bmm(A, h)
            h = h + layer(torch.cat([h, agg], dim=-1))
            h = h * m
        scores = self.attn(h).masked_fill(m == 0, -1e9)
        w = torch.softmax(scores, dim=1)
        pooled = torch.cat(
            [(w * h).sum(1), (h * m).sum(1) / m.sum(1).clamp(min=1)], dim=-1
        )
        return self.head(pooled).squeeze(-1)


def _train_one(Xtr, Atr, mtr, ytr, in_dim, epochs, seed, device):
    torch.manual_seed(seed)
    model = PropertyGNN(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    n = len(ytr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, 128):
            idx = perm[s : s + 128]
            opt.zero_grad()
            pred = model(Xtr[idx], Atr[idx], mtr[idx])
            loss = lossf(pred, ytr[idx])
            loss.backward()
            opt.step()
    return model


def _predict(models, X, A, m):
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds.append(model(X, A, m).cpu().numpy())
    P = np.stack(preds)
    return P.mean(0), P.std(0)


def _build(db_path, fps, feat, gj_map):
    Xs, As, Ms, keep = [], [], [], []
    for i, fp in enumerate(fps):
        nodes = gj_map.get(fp)
        if nodes is None:
            continue
        X, A, mk = _graph_tensors(nodes, feat)
        Xs.append(X)
        As.append(A)
        Ms.append(mk)
        keep.append(i)
    return np.stack(Xs), np.stack(As), np.stack(Ms), np.array(keep)


def run(
    db_path: str, epochs: int, ensemble: int, shrink_f: float, thr: float
) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = GraphSemanticExtractor(db_path)
    feat = PerOpFeaturizer(ext)
    _, _, ind_all, _, fps_all = _build_corpus(db_path)
    gj = _load_graph_json(db_path, fps_all)
    X, A, M, keep = _build(db_path, fps_all, feat, gj)
    fps = [fps_all[i] for i in keep]
    ind = ind_all[keep]
    tmpl = _template_map(db_path)
    clusters = [tmpl.get(fp, _NONE_CLUSTER) for fp in fps]

    n = len(fps)
    cut = int(n * 0.8)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:cut] = True
    y = _shrink(ind, clusters, train_mask, shrink_f).astype(np.float32)

    # normalize node features on train nodes only
    flatX = X[:cut][M[:cut].astype(bool)]
    mu, sd = flatX.mean(0), flatX.std(0)
    sd[sd < 1e-6] = 1.0
    Xn = ((X - mu) / sd).astype(np.float32)

    t = lambda a: torch.tensor(a, device=device)  # noqa: E731
    Xtr, Atr, Mtr, ytr = t(Xn[:cut]), t(A[:cut]), t(M[:cut]), t(y[:cut])
    Xte, Ate, Mte = t(Xn[cut:]), t(A[cut:]), t(M[cut:])
    models = [
        _train_one(Xtr, Atr, Mtr, ytr, feat.dim, epochs, s, device)
        for s in range(ensemble)
    ]
    pred_te, _ = _predict(models, Xte, Ate, Mte)
    cap = ind[cut:] > thr
    roc = float(roc_auc_score(cap, pred_te)) if 0 < cap.sum() < len(cap) else None

    out: Dict[str, Any] = {
        "device": device,
        "n_graphs": n,
        "node_feature_dim": feat.dim,
        "ensemble": ensemble,
        "temporal_roc": round(roc, 4) if roc else None,
        "gbm_temporal_roc_for_reference": 0.89,
    }
    # uncertainty on probed novel winners vs in-distribution
    from research.tools.probe_novel_candidates import _collect_pool

    probes = json.loads(open("research/reports/novel_candidate_probes.json").read())[
        "results"
    ]
    cand = {c["fingerprint"]: c for c in _collect_pool(db_path, 200, 4000, 3_000_000)}
    rows = []
    for r in probes:
        c = cand.get(r["fingerprint"])
        if c is None or r.get("actual_induction_auc") is None:
            continue
        Xg, Ag, mg = _graph_tensors(c["graph"].to_dict()["nodes"], feat)
        Xg = ((Xg - mu) / sd).astype(np.float32)
        pm, ps = _predict(models, t(Xg[None]), t(Ag[None]), t(mg[None]))
        rows.append(
            {
                "fingerprint": r["fingerprint"],
                "actual_induction": round(float(r["actual_induction_auc"]), 4),
                "pred": round(float(pm[0]), 4),
                "uncertainty_std": round(float(ps[0]), 4),
            }
        )
    _, std_te = _predict(models, Xte, Ate, Mte)
    rows.sort(key=lambda x: -x["actual_induction"])
    out["in_dist_mean_uncertainty"] = round(float(std_te.mean()), 4)
    out["probed_novel_decisions"] = rows
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--ensemble", type=int, default=4)
    p.add_argument("--shrink", type=float, default=0.75)
    p.add_argument("--thr", type=float, default=0.35)
    args = p.parse_args()
    print(
        json.dumps(
            run(args.db, args.epochs, args.ensemble, args.shrink, args.thr),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
