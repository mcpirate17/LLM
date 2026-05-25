#!/usr/bin/env python
"""Comprehensive graph semantic feature extractor — math + structure + control-flow.

Represents an architecture by WHAT IT COMPUTES, not the op NAMES. One feature vector
combining five sources, designed to generalize to never-seen ops:

  1. op-math aggregate  — op_property_catalog (algebra/spectral/geometric/dynamical/
                          numerical/activation/expressivity/composition); category-mean
                          fallback for ops absent from the catalog (novelty-robust).
  2. structural topology — depth/width/branching/joining/skipping/dispersion/degree +
                          role & category histograms (computed from the graph; future-proof).
  3. control-flow        — routing (top_k/lanes/temperature), recursion (depth), compression
                          (mlp_ratio/out_dim), span, stochastic temperature (from node configs).
  4. information-flow     — sequence- vs channel-mixers, has_{attention,ssm,spiking,tropical},
                          mixer-algebra diversity, norm/gate/position counts.
  5. novelty             — novel-op fraction + distinct math families (literature_attribution).

Operates on the node dict from ComputationGraph.to_dict() OR graphs.graph_json — the same
format — so the SAME representation is used for live generation and historical backfill.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from typing import Any, Dict, List

from research.synthesis.op_roles import OpRole, get_role

FEATURE_VERSION = "graph_semantic_v2"

# Catalog columns coerced to float and averaged over a graph's ops.
_NUMERIC_PROPS: List[str] = [
    "op_n_inputs",
    "op_has_params",
    "op_is_parameterized",
    "op_is_stateless",
    "op_byte_safe",
    "op_standalone",
    "op_min_layer_depth",
    "op_preserves_gradient_declared",
    "op_numerically_risky_declared",
    "op_empirical_probe_needed",
    "op_algebraic_idempotent",
    "op_algebraic_involutive",
    "op_algebraic_commutes_with_norm",
    "op_spectral_low_pass_strength",
    "op_spectral_diagonalizable_prior",
    "op_spectral_radius_init_prior",
    "op_geometric_lipschitz_prior",
    "op_geometric_jacobian_rank_prior",
    "op_geometric_jacobian_cond_prior",
    "op_geometric_curvature_prior",
    "op_dynamical_causal",
    "op_dynamical_contraction_factor_prior",
    "op_dynamical_exponential_decay_rate_prior",
    "op_dynamical_has_state",
    "op_numerical_hessian_conditioning_prior",
    "op_numerical_grad_vanish_propensity",
    "op_numerical_grad_explode_propensity",
    "op_numerical_fp16_stable_prior",
    "op_numerical_init_sensitivity",
    "op_activation_effective_rank_prior",
    "op_expressivity_depth_required_for_xor_prior",
    "op_expressivity_params_for_identity_prior",
    "op_composition_parallel_safe",
    "op_composition_residual_safe",
    "op_composition_norm_required",
    "op_composition_max_stack_depth_before_collapse",
    "op_differentiability_needs_surrogate",
]
_ALGEBRA_SPACES = ["euclidean", "clifford", "poincare", "spiking", "padic", "tropical"]
_MEMORY_ORDINAL = {
    "O(1)": 0.0,
    "O(log L)": 0.5,
    "O(L)": 1.0,
    "O(L log L)": 1.5,
    "O(L^2)": 2.0,
}
_ALL_ROLES = [r.value for r in OpRole]

# config keys that encode control-flow semantics (future-proof: generic key patterns).
_ROUTING_K = ("top_k", "k", "num_experts", "n_experts")
_LANES = ("lane_count", "n_lanes", "n_experts", "num_experts")
_RECURSION = ("max_depth", "max_iterations", "n_iterations", "recursion_depth")
_COMPRESS = ("mlp_ratio", "compression_ratio", "rank", "latent_dim", "bottleneck_dim")
_TEMP = ("route_temperature", "temperature", "tau")
_SPAN = ("span_width", "window", "window_size", "kernel_size")


class GraphSemanticExtractor:
    """Loads reference tables once; produces a semantic feature dict per graph."""

    def __init__(
        self, runs_db: str, meta_db: str = "research/meta_analysis.db"
    ) -> None:
        self.op_props: Dict[str, Dict[str, Any]] = {}
        self.cat_mean: Dict[str, Dict[str, float]] = {}
        self.op_algebra: Dict[str, str] = {}
        self.op_memory: Dict[str, str] = {}
        self.op_receptive: Dict[str, str] = {}
        self.op_category: Dict[str, str] = {}
        self.novel_ops: set = set()
        self._load_catalog(meta_db)
        self._load_novelty(runs_db)

    def _load_catalog(self, meta_db: str) -> None:
        con = sqlite3.connect(meta_db)
        cols = [r[1] for r in con.execute("PRAGMA table_info(op_property_catalog)")]
        rows = con.execute("SELECT * FROM op_property_catalog").fetchall()
        con.close()
        cat_acc: Dict[str, List[Dict[str, float]]] = {}
        for r in rows:
            d = dict(zip(cols, r))
            op = str(d["op_name"])
            self.op_algebra[op] = str(d.get("op_algebraic_space") or "unknown")
            self.op_memory[op] = str(d.get("op_dynamical_memory_length_class") or "")
            self.op_receptive[op] = str(
                d.get("op_geometric_receptive_field") or "unspecified"
            )
            cat = str(d.get("op_category") or "unknown")
            self.op_category[op] = cat
            numeric = {k: _coerce(d.get(k)) for k in _NUMERIC_PROPS}
            self.op_props[op] = numeric
            cat_acc.setdefault(cat, []).append(numeric)
        for cat, vecs in cat_acc.items():
            self.cat_mean[cat] = {
                k: float(_nanmean([v[k] for v in vecs])) for k in _NUMERIC_PROPS
            }

    def _load_novelty(self, runs_db: str) -> None:
        con = sqlite3.connect(runs_db)
        try:
            for k, m in con.execute(
                "SELECT entity_key, match_type FROM literature_attribution WHERE entity_type='op'"
            ):
                if str(m) in ("novel", "partial"):
                    self.novel_ops.add(str(k))
        except sqlite3.OperationalError:
            pass
        con.close()

    def _op_numeric(self, op: str) -> Dict[str, float]:
        if op in self.op_props:
            return self.op_props[op]
        return self.cat_mean.get(self.op_category.get(op, ""), {})  # novelty fallback

    def features(
        self, nodes: Dict[str, Any] | List[Any], model_dim: int = 256
    ) -> Dict[str, float]:
        node_list = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
        ops = [n for n in node_list if not n.get("is_input")]
        f: Dict[str, float] = {}
        self._math_aggregate(ops, f)
        self._topology(node_list, ops, f)
        self._control_flow(ops, f, model_dim)
        self._info_flow(ops, f)
        self._novelty(ops, f)
        self._abstract(node_list, ops, f)
        return f

    # ── 6. abstract / derived (v2): recipe conjunctions, spreads, transitions ──
    def _abstract(
        self, node_list: List[Any], ops: List[Any], f: Dict[str, float]
    ) -> None:
        names = [str(n["op_name"]) for n in ops]
        # Recipe-level functional requirements (transfer across mechanisms).
        has_retrieval = (
            1.0
            if (
                f.get("has_attention", 0.0) > 0
                or any(
                    self.op_receptive.get(n) == "global" and get_role(n) is OpRole.MIX
                    for n in names
                )
            )
            else 0.0
        )
        has_pos = 1.0 if f.get("role_frac_position", 0.0) > 0 else 0.0
        has_norm = 1.0 if f.get("role_frac_normalize", 0.0) > 0 else 0.0
        has_resid = (
            1.0
            if (f.get("n_skips", 0.0) > 0 or f.get("role_frac_residual", 0.0) > 0)
            else 0.0
        )
        f["recipe_has_content_retrieval"] = has_retrieval
        f["recipe_retrieval_x_position"] = has_retrieval * has_pos
        f["recipe_induction_score"] = has_retrieval * has_pos * has_norm * has_resid
        # Property spreads = "derivatives" of math priors across the graph's ops.
        for prop in (
            "op_geometric_lipschitz_prior",
            "op_spectral_low_pass_strength",
            "op_geometric_jacobian_rank_prior",
            "op_numerical_grad_vanish_propensity",
        ):
            vals = [self._op_numeric(n).get(prop) for n in names]
            f[f"spread_{prop}"] = _nanstd([v for v in vals if v is not None])
        # Algebra transitions + role bigrams along edges.
        ids = {n["id"]: str(n["op_name"]) for n in node_list}
        n_edges = alg_trans = 0
        bigrams: Counter = Counter()
        for n in node_list:
            if n.get("is_input"):
                continue
            dst = str(n["op_name"])
            for src_id in n.get("input_ids", []) or []:
                src = ids.get(src_id)
                if src is None:
                    continue
                n_edges += 1
                sa, da = self.op_algebra.get(src, ""), self.op_algebra.get(dst, "")
                if sa and da and "unknown" not in (sa, da) and sa != da:
                    alg_trans += 1
                bigrams[(get_role(src).value, get_role(dst).value)] += 1
        f["algebra_transition_frac"] = alg_trans / max(n_edges, 1)
        for a, b in (
            ("normalize", "mix"),
            ("mix", "normalize"),
            ("mix", "gate"),
            ("project", "mix"),
            ("mix", "project"),
        ):
            f[f"bigram_{a}_to_{b}"] = bigrams.get((a, b), 0) / max(n_edges, 1)
        # Structural abstractions.
        f["parallelism_ratio"] = f.get("n_ops", 0.0) / max(f.get("depth", 1.0), 1.0)
        f["residual_density"] = f.get("n_skips", 0.0) / max(f.get("n_ops", 1.0), 1.0)
        f["branch_merge_balance"] = f.get("n_branches", 0.0) - f.get("n_merges", 0.0)

    # ── 1. op-math aggregate (catalog) ─────────────────────────────
    def _math_aggregate(self, ops: List[Any], f: Dict[str, float]) -> None:
        n = max(len(ops), 1)
        acc: Dict[str, List[float]] = {k: [] for k in _NUMERIC_PROPS}
        algebra = {a: 0 for a in _ALGEBRA_SPACES}
        recept_global = 0
        mem_vals: List[float] = []
        for node in ops:
            op = str(node["op_name"])
            num = self._op_numeric(op)
            for k in _NUMERIC_PROPS:
                v = num.get(k)
                if v is not None and not math.isnan(v):
                    acc[k].append(v)
            algebra[self.op_algebra.get(op, "")] = (
                algebra.get(self.op_algebra.get(op, ""), 0) + 1
            )
            if self.op_receptive.get(op) == "global":
                recept_global += 1
            mem_vals.append(_MEMORY_ORDINAL.get(self.op_memory.get(op, ""), 0.0))
        for k in _NUMERIC_PROPS:
            f[f"math_mean_{k}"] = float(_nanmean(acc[k]))
        for a in _ALGEBRA_SPACES:
            f[f"algebra_frac_{a}"] = algebra.get(a, 0) / n
        f["receptive_global_frac"] = recept_global / n
        f["memory_len_mean"] = sum(mem_vals) / n
        f["memory_len_max"] = max(mem_vals) if mem_vals else 0.0

    # ── 2. structural topology ─────────────────────────────────────
    def _topology(
        self, node_list: List[Any], ops: List[Any], f: Dict[str, float]
    ) -> None:
        ids = {n["id"]: n for n in node_list}
        out_deg: Dict[int, int] = {i: 0 for i in ids}
        n_merge = 0
        n_skip = 0
        for n in node_list:
            ins = n.get("input_ids", []) or []
            if len(ins) > 1:
                n_merge += 1
            for src in ins:
                out_deg[src] = out_deg.get(src, 0) + 1
                if n.get("depth", 0) - ids.get(src, {}).get("depth", 0) > 1:
                    n_skip += 1
        depths = [n.get("depth", 0) for n in node_list]
        f["n_ops"] = float(len(ops))
        f["n_edges"] = float(sum(len(n.get("input_ids", []) or []) for n in node_list))
        f["depth"] = float(max(depths) if depths else 0)
        f["max_fanout"] = float(max(out_deg.values()) if out_deg else 0)
        f["n_branches"] = float(sum(1 for d in out_deg.values() if d > 1))
        f["n_merges"] = float(n_merge)
        f["n_skips"] = float(n_skip)
        f["width_est"] = f["n_ops"] / max(f["depth"], 1.0)
        roles = [get_role(str(n["op_name"])).value for n in ops]
        for r in _ALL_ROLES:
            f[f"role_frac_{r}"] = roles.count(r) / max(len(ops), 1)
        cats = [self.op_category.get(str(n["op_name"]), "unknown") for n in ops]
        for cat in (
            "mixing",
            "math_space",
            "parameterized",
            "functional",
            "structural",
            "reduction",
            "sequence",
            "frequency",
            "linear_algebra",
        ):
            f[f"cat_frac_{cat}"] = cats.count(cat) / max(len(ops), 1)

    # ── 3. control-flow (node configs) ─────────────────────────────
    def _control_flow(
        self, ops: List[Any], f: Dict[str, float], model_dim: int
    ) -> None:
        topk, lanes, recur, comp_ratio, temps, spans = [], [], [], [], [], []
        out_dims = []
        n_router = n_recursive = n_stochastic = 0
        for node in ops:
            conf = node.get("config") or {}
            kk = _first(conf, _ROUTING_K)
            if kk is not None:
                topk.append(float(kk))
                n_router += 1
            ll = _first(conf, _LANES)
            if ll is not None:
                lanes.append(float(ll))
            rr = _first(conf, _RECURSION)
            if rr is not None:
                recur.append(float(rr))
                n_recursive += 1
            cr = _first(conf, _COMPRESS)
            if cr is not None:
                comp_ratio.append(float(cr))
            tt = _first(conf, _TEMP)
            if tt is not None:
                temps.append(float(tt))
                n_stochastic += 1
            sp = _first(conf, _SPAN)
            if sp is not None:
                spans.append(float(sp))
            if "out_dim" in conf:
                out_dims.append(float(conf["out_dim"]))
        f["route_max_topk"] = max(topk) if topk else 0.0
        f["route_n_routers"] = float(n_router)
        f["route_max_lanes"] = max(lanes) if lanes else 0.0
        f["route_mean_temperature"] = (sum(temps) / len(temps)) if temps else 0.0
        f["recursion_max_depth"] = max(recur) if recur else 0.0
        f["recursion_n_ops"] = float(n_recursive)
        f["compress_mean_ratio"] = (
            (sum(comp_ratio) / len(comp_ratio)) if comp_ratio else 0.0
        )
        f["compress_min_outdim_frac"] = (min(out_dims) / model_dim) if out_dims else 1.0
        f["span_max"] = max(spans) if spans else 0.0
        f["stochastic_n_ops"] = float(n_stochastic)

    # ── 4. information-flow / derived ──────────────────────────────
    def _info_flow(self, ops: List[Any], f: Dict[str, float]) -> None:
        names = [str(n["op_name"]) for n in ops]
        algebras = {self.op_algebra.get(n, "") for n in names}
        f["mixer_algebra_diversity"] = float(
            len([a for a in algebras if a and a != "unknown"])
        )
        f["n_seq_mixers"] = float(sum(1 for n in names if get_role(n) is OpRole.MIX))
        f["n_gates"] = float(sum(1 for n in names if get_role(n) is OpRole.GATE))
        f["n_norms"] = float(sum(1 for n in names if get_role(n) is OpRole.NORMALIZE))
        f["n_positions"] = float(
            sum(1 for n in names if get_role(n) is OpRole.POSITION)
        )
        f["has_attention"] = float(
            any("attention" in n or n == "softmax_attention" for n in names)
        )
        f["has_ssm"] = float(
            any(n in ("state_space", "selective_scan", "mlstm_cell") for n in names)
        )
        f["has_spiking"] = float(
            any(self.op_algebra.get(n) == "spiking" for n in names)
        )
        f["has_tropical"] = float(
            any(self.op_algebra.get(n) == "tropical" for n in names)
        )
        f["has_recursion"] = 1.0 if f.get("recursion_n_ops", 0) > 0 else 0.0
        f["has_routing"] = 1.0 if f.get("route_n_routers", 0) > 0 else 0.0

    # ── 5. novelty ─────────────────────────────────────────────────
    def _novelty(self, ops: List[Any], f: Dict[str, float]) -> None:
        names = [str(n["op_name"]) for n in ops]
        n = max(len(names), 1)
        f["novel_op_frac"] = sum(1 for x in names if x in self.novel_ops) / n
        f["novel_op_count"] = float(sum(1 for x in names if x in self.novel_ops))
        f["uncataloged_op_frac"] = sum(1 for x in names if x not in self.op_props) / n


def _coerce(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _nanmean(vals: List[float]) -> float:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(clean) / len(clean) if clean else 0.0


def _nanstd(vals: List[float]) -> float:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    if len(clean) < 2:
        return 0.0
    m = sum(clean) / len(clean)
    return math.sqrt(sum((v - m) ** 2 for v in clean) / len(clean))


def _first(conf: Dict[str, Any], keys: tuple) -> Any:
    for k in keys:
        if k in conf and conf[k] is not None:
            try:
                return float(conf[k])
            except (TypeError, ValueError):
                continue
    return None
