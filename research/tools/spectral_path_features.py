#!/usr/bin/env python
"""Spectral + path abstraction of a computation DAG — the higher-order rep v2 lacks.

`graph_semantic_features.py` (v2) is per-NODE aggregates + 2-node role bigrams. This module
adds the *relational* abstraction its own PCA/PLS analysis named as the remaining headroom:
graph-spectral descriptors of topology and path/motif descriptors ("what computation a PATH
performs"). It composes `GraphSemanticExtractor` for catalog access (no duplicated loading) and
emits a disjoint ``spath_*`` namespace, so it is purely additive to the existing feature blob.

Four blocks:
  A. spectral topology     — normalized-Laplacian spectrum of the op-DAG (gap/λmax/entropy/
                             near-zero/adjacency-radius): an isomorphism-robust topology print.
  B. computational spectrum — the SAME spectral summaries on an adjacency RE-WEIGHTED by op
                             math-priors (Lipschitz gain, low-pass smoothing). The eigenstructure
                             summarizes the whole path-ensemble's gain/smoothing at once — the
                             literal "spectral path" object: paths abstracted spectrally.
  C. path functionals      — DAG dynamic-programming (O(V+E), no path enumeration) over all
                             input→output paths: log-gain max/min/spread, low-pass cascade depth,
                             algebra-switch count, memory accumulation, a PATHWISE induction recipe
                             (one path through retrieval∧position∧norm∧residual — precise vs v2's
                             global AND), log path-count, mean/longest path length.
  D. motifs / trigrams     — 3-node topology motifs (diamond/chain3/fan-in) + role trigrams.

Same node-dict input as the v2 extractor (ComputationGraph.to_dict() / graphs.graph_json).
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from research.synthesis.op_roles import OpRole, get_role
from research.tools.graph_semantic_features import (
    _MEMORY_ORDINAL,
    GraphSemanticExtractor,
)

FEATURE_VERSION = "spectral_path_v1"

# recipe bits accumulated along a single path (pathwise induction recipe)
_BIT_RETRIEVAL = 1
_BIT_POSITION = 2
_BIT_NORM = 4
_BIT_RESIDUAL = 8
_BIT_ALL = _BIT_RETRIEVAL | _BIT_POSITION | _BIT_NORM | _BIT_RESIDUAL
_LIP_DEFAULT = 1.0  # missing Lipschitz prior → norm-preserving
_EPS = 1e-9


class SpectralPathExtractor:
    """Spectral + path abstraction of a graph. Reuses the catalog via GraphSemanticExtractor."""

    def __init__(
        self, runs_db: str, meta_db: str = "research/meta_analysis.db"
    ) -> None:
        self.base = GraphSemanticExtractor(runs_db, meta_db)

    # ── public ──────────────────────────────────────────────────────
    def features(self, nodes: Dict[str, Any] | List[Any]) -> Dict[str, float]:
        node_list = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
        order, preds, succs = _topo(node_list)
        idx = {n["id"]: i for i, n in enumerate(node_list)}
        f: Dict[str, float] = {}
        self._spectral(node_list, idx, f)
        self._paths(node_list, order, preds, succs, f)
        self._motifs(node_list, preds, succs, f)
        return f

    # ── A+B. spectra (topology + property-weighted) ─────────────────
    def _spectral(
        self, node_list: List[Any], idx: Dict[Any, int], f: Dict[str, float]
    ) -> None:
        n = len(node_list)
        adj = np.zeros((n, n), dtype=np.float64)  # undirected structure
        gain = np.zeros((n, n), dtype=np.float64)  # Lipschitz-weighted
        lowp = np.zeros((n, n), dtype=np.float64)  # low-pass-weighted
        lip = np.array([self._lipschitz(node_list[i]) for i in range(n)])
        lp = np.array([self._lowpass(node_list[i]) for i in range(n)])
        for n_ in node_list:
            j = idx[n_["id"]]
            for src in n_.get("input_ids", []) or []:
                if src not in idx:
                    continue
                i = idx[src]
                adj[i, j] = adj[j, i] = 1.0
                gw = math.sqrt(max(lip[i] * lip[j], _EPS))  # path-gain weight
                gain[i, j] = gain[j, i] = gw
                lw = 1.0 + 0.5 * (lp[i] + lp[j])  # smoothing weight
                lowp[i, j] = lowp[j, i] = lw
        _spectrum_block(adj, "topo", f)
        _spectrum_block(gain, "gain", f)
        _spectrum_block(lowp, "lowpass", f)

    # ── C. path functionals (DAG dynamic programming) ───────────────
    def _paths(
        self,
        node_list: List[Any],
        order: List[int],
        preds: Dict[int, List[int]],
        succs: Dict[int, List[int]],
        f: Dict[str, float],
    ) -> None:
        by_id = {n["id"]: n for n in node_list}
        attrs = {nid: self._node_attrs(by_id[nid]) for nid in by_id}
        st = _PathState()
        for nid in order:
            _dp_node(st, nid, preds[nid], attrs)
        sinks = [
            nid for nid in by_id if not succs[nid] and not by_id[nid].get("is_input")
        ] or list(by_id)
        _write_path_features(st, sinks, f)

    def _node_attrs(self, node: Any) -> "_NodeAttrs":
        name = str(node["op_name"])
        return _NodeAttrs(
            alg=self.base.op_algebra.get(name, ""),
            loglip=math.log(max(self._lipschitz(node), _EPS)),
            lp=self._lowpass(node),
            mem=_MEMORY_ORDINAL.get(self.base.op_memory.get(name, ""), 0.0),
            bit=self._recipe_bit(node),
        )

    # ── D. motifs + role trigrams ───────────────────────────────────
    def _motifs(
        self,
        node_list: List[Any],
        preds: Dict[int, List[int]],
        succs: Dict[int, List[int]],
        f: Dict[str, float],
    ) -> None:
        by_id = {n["id"]: n for n in node_list}
        role = {nid: get_role(str(by_id[nid]["op_name"])).value for nid in by_id}
        n_edges = sum(len(preds[nid]) for nid in by_id) or 1
        chain3 = diamond = fanin3 = 0
        trigrams: Counter = Counter()
        for b in by_id:
            for a in preds[b]:
                for c in succs[b]:  # a→b→c chains
                    chain3 += 1
                    trigrams[(role[a], role[b], role[c])] += 1
            if len(preds[b]) >= 3:
                fanin3 += 1
            # diamond: two preds of b share a common ancestor (branch→merge)
            forebears = [set(preds[p]) for p in preds[b]]
            for i in range(len(forebears)):
                for j in range(i + 1, len(forebears)):
                    if forebears[i] & forebears[j]:
                        diamond += 1
        f["spath_motif_chain3"] = chain3 / n_edges
        f["spath_motif_diamond"] = float(diamond)
        f["spath_motif_fanin3"] = float(fanin3)
        for a, b, c in (
            ("normalize", "mix", "gate"),
            ("project", "mix", "project"),
            ("normalize", "mix", "normalize"),
            ("position", "mix", "normalize"),
            ("mix", "gate", "project"),
        ):
            f[f"spath_trigram_{a}_{b}_{c}"] = trigrams.get((a, b, c), 0) / n_edges

    # ── per-op priors (catalog via composed base) ───────────────────
    def _lipschitz(self, node: Any) -> float:
        v = self.base._op_numeric(str(node["op_name"])).get(
            "op_geometric_lipschitz_prior"
        )
        return _LIP_DEFAULT if v is None or math.isnan(v) else float(v)

    def _lowpass(self, node: Any) -> float:
        v = self.base._op_numeric(str(node["op_name"])).get(
            "op_spectral_low_pass_strength"
        )
        return 0.0 if v is None or math.isnan(v) else float(v)

    def _recipe_bit(self, node: Any) -> int:
        name = str(node["op_name"])
        r = get_role(name)
        b = 0
        if "attention" in name or (
            self.base.op_receptive.get(name) == "global" and r is OpRole.MIX
        ):
            b |= _BIT_RETRIEVAL
        if r is OpRole.POSITION:
            b |= _BIT_POSITION
        if r is OpRole.NORMALIZE:
            b |= _BIT_NORM
        if len(node.get("input_ids", []) or []) > 1:  # merge ≈ residual junction
            b |= _BIT_RESIDUAL
        return b


# --------------------------------------------------------------------------- #
# path dynamic-programming helpers (module-level, pure)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _NodeAttrs:
    """Static per-op quantities consumed by the path DP."""

    alg: str
    loglip: float
    lp: float
    mem: float
    bit: int


@dataclass
class _PathState:
    """Per-node accumulators over all source→node paths (filled in topo order)."""

    gmax: Dict[int, float] = field(default_factory=dict)  # max log-gain
    gmin: Dict[int, float] = field(default_factory=dict)  # min log-gain
    lpc: Dict[int, float] = field(default_factory=dict)  # max low-pass cascade
    sw: Dict[int, float] = field(default_factory=dict)  # max algebra switches
    mem: Dict[int, float] = field(default_factory=dict)  # max memory accumulation
    masks: Dict[int, int] = field(default_factory=dict)  # union of recipe bitmasks
    npaths: Dict[int, float] = field(default_factory=dict)  # path count
    sumlen: Dict[int, float] = field(default_factory=dict)  # Σ path lengths
    maxlen: Dict[int, int] = field(default_factory=dict)  # longest path (edges)


def _dp_node(
    st: _PathState, nid: int, ps: List[int], attrs: Dict[int, _NodeAttrs]
) -> None:
    """One topo-order step: fold predecessor accumulators into node ``nid``."""
    a = attrs[nid]
    if not ps:  # source (input)
        st.gmax[nid] = st.gmin[nid] = a.loglip
        st.lpc[nid], st.sw[nid], st.mem[nid] = a.lp, 0.0, a.mem
        st.masks[nid], st.npaths[nid] = a.bit, 1.0
        st.sumlen[nid], st.maxlen[nid] = 0.0, 0
        return
    st.gmax[nid] = max(st.gmax[p] for p in ps) + a.loglip
    st.gmin[nid] = min(st.gmin[p] for p in ps) + a.loglip
    st.lpc[nid] = max(st.lpc[p] for p in ps) + a.lp
    st.sw[nid] = max(
        st.sw[p] + (attrs[p].alg != a.alg if attrs[p].alg and a.alg else 0) for p in ps
    )
    st.mem[nid] = max(max(st.mem[p] for p in ps), a.mem)
    union = 0
    for p in ps:
        union |= st.masks[p]
    st.masks[nid] = union | a.bit
    st.npaths[nid] = sum(st.npaths[p] for p in ps)
    st.sumlen[nid] = sum(st.sumlen[p] + st.npaths[p] for p in ps)
    st.maxlen[nid] = max(st.maxlen[p] for p in ps) + 1


def _write_path_features(st: _PathState, sinks: List[int], f: Dict[str, float]) -> None:
    """Aggregate the DP accumulators over sink (output) nodes into spath_* features."""
    f["spath_path_loggain_max"] = max(st.gmax[s] for s in sinks)
    f["spath_path_loggain_min"] = min(st.gmin[s] for s in sinks)
    f["spath_path_loggain_spread"] = (
        f["spath_path_loggain_max"] - f["spath_path_loggain_min"]
    )
    f["spath_path_lowpass_cascade_max"] = max(st.lpc[s] for s in sinks)
    f["spath_path_alg_switches_max"] = max(st.sw[s] for s in sinks)
    f["spath_path_memory_max"] = max(st.mem[s] for s in sinks)
    f["spath_recipe_induction_pathwise"] = (
        1.0 if any((st.masks[s] & _BIT_ALL) == _BIT_ALL for s in sinks) else 0.0
    )
    tot = sum(st.npaths[s] for s in sinks)
    f["spath_n_io_paths_log"] = math.log1p(tot)
    f["spath_longest_path"] = float(max(st.maxlen[s] for s in sinks))
    f["spath_mean_path_len"] = (
        sum(st.sumlen[s] for s in sinks) / tot if tot > 0 else 0.0
    )


# --------------------------------------------------------------------------- #
# graph helpers (module-level, pure)
# --------------------------------------------------------------------------- #
def _topo(
    node_list: List[Any],
) -> Tuple[List[int], Dict[int, List[int]], Dict[int, List[int]]]:
    """Kahn topological order + pred/succ adjacency keyed by node id (edges to known ids)."""
    ids = {n["id"] for n in node_list}
    preds: Dict[int, List[int]] = {n["id"]: [] for n in node_list}
    succs: Dict[int, List[int]] = {n["id"]: [] for n in node_list}
    for n in node_list:
        for src in n.get("input_ids", []) or []:
            if src in ids:
                preds[n["id"]].append(src)
                succs[src].append(n["id"])
    indeg = {nid: len(preds[nid]) for nid in preds}
    q = deque(nid for nid, d in indeg.items() if d == 0)
    order: List[int] = []
    while q:
        nid = q.popleft()
        order.append(nid)
        for s in succs[nid]:
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    if len(order) != len(node_list):  # cycle guard: append remainder by id
        order += [nid for nid in preds if nid not in set(order)]
    return order, preds, succs


def _spectrum_block(adj: np.ndarray, tag: str, f: Dict[str, float]) -> None:
    """Normalized-Laplacian spectral summaries of a symmetric (weighted) adjacency."""
    n = adj.shape[0]
    if n < 2 or adj.sum() == 0:
        for k in ("gap", "lambda_max", "entropy", "near_zero_frac", "adj_radius"):
            f[f"spath_{tag}_{k}"] = 0.0
        return
    deg = adj.sum(axis=1)
    dinv = np.where(deg > _EPS, 1.0 / np.sqrt(deg), 0.0)
    lap = np.eye(n) - (adj * dinv[:, None]) * dinv[None, :]
    lap = 0.5 * (lap + lap.T)  # symmetrize against float drift
    eig = np.clip(np.linalg.eigvalsh(lap), 0.0, None)
    eig_sorted = np.sort(eig)
    f[f"spath_{tag}_gap"] = float(eig_sorted[1])  # algebraic connectivity
    f[f"spath_{tag}_lambda_max"] = float(eig_sorted[-1])
    f[f"spath_{tag}_near_zero_frac"] = float((eig < 1e-6).mean())
    s = eig.sum()
    if s > _EPS:
        p = eig[eig > _EPS] / s
        f[f"spath_{tag}_entropy"] = float(-(p * np.log(p)).sum() / math.log(n))
    else:
        f[f"spath_{tag}_entropy"] = 0.0
    f[f"spath_{tag}_adj_radius"] = float(np.abs(np.linalg.eigvalsh(adj)).max())
