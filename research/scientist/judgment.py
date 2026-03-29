"""
Shared Judgment Engine

Unified scoring module for evaluating architecture candidates.
Used by both the Designer (suggestions, mutations) and the Research
runner (screening, promotion). Consumes Phase 1 research aggregates
and produces structured decisions with evidence.

Two entry points calling the same pipeline:
  - score_candidate(graph, ctx, signals) → JudgmentResult
  - recommend_components(ctx, signals, candidates) → ranked list
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple


@dataclass(slots=True)
class JudgmentContext:
    """Input context for judgment scoring."""

    fingerprint_bucket: str = ""
    active_op_pairs: Tuple[str, ...] = ()
    parent_fingerprint: Optional[str] = None
    parent_scores: Optional[Dict[str, Any]] = None
    intent: Optional[str] = None
    matched_insights: Tuple[Dict[str, Any], ...] = ()
    novelty_context: Dict[str, Any] = field(default_factory=dict)
    performance_context: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JudgmentResult:
    """Output from judgment scoring."""

    total_score: float = 0.0
    signal_breakdown: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    risk_flags: List[str] = field(default_factory=list)
    recommended_action: str = "hold"
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    support_counts: Dict[str, int] = field(default_factory=dict)


# ── Signal scorer type ────────────────────────────────────────────────
# Each returns (score_delta, confidence, evidence_items)

_ScorerReturn = Tuple[float, float, List[Dict[str, Any]]]
_ScorerFn = Callable[
    [Dict[str, Any], JudgmentContext, Dict[str, Any]],
    _ScorerReturn,
]

# Maximum influence any single signal can have (±30% of total)
_SIGNAL_CAP = 0.30

# Low-support threshold — below this, weight signal toward neutral
_MIN_SUPPORT = 5

# Exploration budget — reserve this fraction for under-sampled candidates
_EXPLORATION_BUDGET = 0.15


# ── Individual signal scorers ─────────────────────────────────────────


def _score_op_priors(
    candidate: Dict[str, Any],
    ctx: JudgmentContext,
    signals: Dict[str, Any],
) -> _ScorerReturn:
    """Score based on per-op success rates from op_success_rates table."""
    op_priors = signals.get("op_priors", [])
    if not op_priors:
        return 0.0, 0.0, []

    prior_map = {p["op_name"]: p for p in op_priors if "op_name" in p}
    ops = candidate.get("ops", [])
    if not ops:
        return 0.0, 0.0, []

    score_sum = 0.0
    n_matched = 0
    evidence = []
    for op_name in ops:
        prior = prior_map.get(op_name)
        if not prior:
            continue
        s1_rate = float(prior.get("s1_rate", 0.0))
        n_used = int(prior.get("n_used", 0))
        # Positive signal: high s1 rate
        delta = (s1_rate - 0.5) * 2.0  # normalize to [-1, 1]
        if n_used < _MIN_SUPPORT:
            delta *= n_used / _MIN_SUPPORT  # dampen low-support
        score_sum += delta
        n_matched += 1
        if abs(delta) > 0.2:
            evidence.append(
                {
                    "signal": "op_priors",
                    "op": op_name,
                    "s1_rate": s1_rate,
                    "support": n_used,
                    "delta": round(delta, 4),
                }
            )

    if n_matched == 0:
        return 0.0, 0.0, []

    avg_score = score_sum / n_matched
    confidence = min(1.0, n_matched / 5.0)
    return avg_score, confidence, evidence


def _score_op_pairs(
    candidate: Dict[str, Any],
    ctx: JudgmentContext,
    signals: Dict[str, Any],
) -> _ScorerReturn:
    """Score based on op-pair success rates from Phase 1 aggregates."""
    pair_priors = signals.get("op_pair_priors", [])
    if not pair_priors:
        return 0.0, 0.0, []

    pair_map = {p["signature"]: p for p in pair_priors if "signature" in p}
    active_pairs = ctx.active_op_pairs or candidate.get("op_pairs", ())
    if not active_pairs:
        return 0.0, 0.0, []

    score_sum = 0.0
    n_matched = 0
    evidence = []
    for pair_sig in active_pairs:
        prior = pair_map.get(pair_sig)
        if not prior:
            continue
        success_rate = float(prior.get("success_rate", 0.0))
        support = int(prior.get("support", 0))
        delta = (success_rate - 0.3) * 2.0  # baseline ~30%
        if support < _MIN_SUPPORT:
            delta *= support / _MIN_SUPPORT
        score_sum += delta
        n_matched += 1
        if abs(delta) > 0.15:
            evidence.append(
                {
                    "signal": "op_pairs",
                    "pair": pair_sig,
                    "success_rate": success_rate,
                    "support": support,
                    "delta": round(delta, 4),
                }
            )

    if n_matched == 0:
        return 0.0, 0.0, []

    avg_score = score_sum / n_matched
    confidence = min(1.0, n_matched / 3.0)
    return avg_score, confidence, evidence


def _score_fingerprint_bucket(
    candidate: Dict[str, Any],
    ctx: JudgmentContext,
    signals: Dict[str, Any],
) -> _ScorerReturn:
    """Bonus/penalty based on fingerprint bucket performance and category distribution."""
    buckets = signals.get("fingerprint_buckets", [])
    if not buckets or not ctx.fingerprint_bucket:
        return 0.0, 0.0, []

    bucket_map = {b["bucket"]: b for b in buckets if "bucket" in b}
    bucket = bucket_map.get(ctx.fingerprint_bucket)
    if not bucket:
        return 0.0, 0.0, []

    s1_rate = float(bucket.get("s1_rate", 0.0))
    n_graphs = int(bucket.get("n_graphs", 0))
    delta = (s1_rate - 0.3) * 1.5
    confidence = min(1.0, n_graphs / 20.0) if n_graphs >= _MIN_SUPPORT else 0.3

    # Bonus for buckets with routing diversity (higher-resolution feature)
    top_routing = bucket.get("top_routing_ops", [])
    if top_routing:
        routing_bonus = min(0.05, 0.02 * len(top_routing))
        delta += routing_bonus

    # Category distribution: penalize single-category dominance (op-soup),
    # reward spread across multiple categories (structured architectures)
    cat_dist = bucket.get("op_category_distribution", {})
    if cat_dist:
        max_cat_share = max(cat_dist.values()) if cat_dist.values() else 0.0
        n_active_cats = sum(1 for v in cat_dist.values() if v >= 0.05)
        # Penalize: >60% in one category = likely op-soup
        if max_cat_share > 0.6:
            delta -= 0.03 * (max_cat_share - 0.6)
        # Reward: 4+ active categories = structurally diverse
        if n_active_cats >= 4:
            delta += 0.02 * min(1.0, (n_active_cats - 3) / 4.0)

    evidence = (
        [
            {
                "signal": "fingerprint_bucket",
                "bucket": ctx.fingerprint_bucket,
                "s1_rate": s1_rate,
                "n_graphs": n_graphs,
                "top_routing_ops": top_routing,
                "template_signature": bucket.get("template_signature", ""),
                "category_diversity": n_active_cats if cat_dist else 0,
                "delta": round(delta, 4),
            }
        ]
        if abs(delta) > 0.1
        else []
    )

    return delta, confidence, evidence


def _score_lineage(
    candidate: Dict[str, Any],
    ctx: JudgmentContext,
    signals: Dict[str, Any],
) -> _ScorerReturn:
    """Boost patterns that historically improved parent scores."""
    successors = signals.get("lineage_successors", [])
    if not successors or not ctx.parent_fingerprint:
        return 0.0, 0.0, []

    # Find transitions from parent fingerprint
    relevant = [
        s for s in successors if s.get("parent_fingerprint") == ctx.parent_fingerprint
    ]
    if not relevant:
        return 0.0, 0.0, []

    best = max(relevant, key=lambda s: float(s.get("improved_rate", 0.0)))
    improved_rate = float(best.get("improved_rate", 0.0))
    support = int(best.get("support", 0))
    delta = (improved_rate - 0.5) * 1.5
    if support < _MIN_SUPPORT:
        delta *= support / _MIN_SUPPORT

    confidence = min(1.0, support / 5.0)
    evidence = (
        [
            {
                "signal": "lineage",
                "parent_fp": ctx.parent_fingerprint,
                "improved_rate": improved_rate,
                "support": support,
                "delta": round(delta, 4),
            }
        ]
        if abs(delta) > 0.1
        else []
    )

    return delta, confidence, evidence


def _score_failure_risk(
    candidate: Dict[str, Any],
    ctx: JudgmentContext,
    signals: Dict[str, Any],
) -> _ScorerReturn:
    """Graduated penalty from failure-risk signatures."""
    risk_sigs = signals.get("failure_risk_signatures", [])
    critical = signals.get("critical_failures", [])
    if not risk_sigs and not critical:
        return 0.0, 0.0, []

    # Build lookup sets
    critical_set = frozenset(c["signature"] for c in critical if "signature" in c)
    risk_map = {r["signature"]: r for r in risk_sigs if "signature" in r}

    ops = candidate.get("ops", [])
    pairs = ctx.active_op_pairs or candidate.get("op_pairs", ())

    # Check all signatures (individual ops and pairs)
    all_sigs = list(ops) + list(pairs)

    penalty = 0.0
    evidence = []
    risk_flags = []

    for sig in all_sigs:
        if sig in critical_set:
            penalty -= 0.8
            risk_flags.append(f"critical_failure:{sig}")
            evidence.append(
                {
                    "signal": "failure_risk",
                    "signature": sig,
                    "severity": "critical",
                    "delta": -0.8,
                }
            )
        elif sig in risk_map:
            weight = float(risk_map[sig].get("weight", 1.0))
            # weight is 0.25-0.65 scale; convert to soft penalty.
            # Multiplier kept low — failure correlations are often
            # template-level, not proof the pair itself is broken.
            sig_penalty = -(1.0 - weight) * 0.15
            penalty += sig_penalty
            if abs(sig_penalty) > 0.05:
                evidence.append(
                    {
                        "signal": "failure_risk",
                        "signature": sig,
                        "weight": weight,
                        "delta": round(sig_penalty, 4),
                    }
                )

    confidence = 0.8 if evidence else 0.0
    return max(-1.0, penalty), confidence, evidence


def _score_insight_interactions(
    candidate: Dict[str, Any],
    ctx: JudgmentContext,
    signals: Dict[str, Any],
) -> _ScorerReturn:
    """Boost synergistic pairs from insights interaction table."""
    interactions = signals.get("insight_interactions", [])
    if not interactions or not ctx.matched_insights:
        return 0.0, 0.0, []

    insight_ids = frozenset(i.get("insight_id", "") for i in ctx.matched_insights)
    if not insight_ids:
        return 0.0, 0.0, []

    score_sum = 0.0
    n_matched = 0
    evidence = []
    for interaction in interactions:
        a = interaction.get("insight_a", "")
        b = interaction.get("insight_b", "")
        if a in insight_ids and b in insight_ids:
            reward = float(interaction.get("mean_reward", 0.5))
            delta = (reward - 0.5) * 2.0
            n_trials = int(interaction.get("n_trials", 0))
            if n_trials < 3:
                delta *= n_trials / 3.0
            score_sum += delta
            n_matched += 1
            if abs(delta) > 0.1:
                evidence.append(
                    {
                        "signal": "insight_interactions",
                        "pair": f"{a}+{b}",
                        "mean_reward": reward,
                        "delta": round(delta, 4),
                    }
                )

    if n_matched == 0:
        return 0.0, 0.0, []
    return score_sum / n_matched, min(1.0, n_matched / 2.0), evidence


def _score_novelty(
    candidate: Dict[str, Any],
    ctx: JudgmentContext,
    signals: Dict[str, Any],
) -> _ScorerReturn:
    """Reward under-explored fingerprint regions."""
    novelty = ctx.novelty_context
    if not novelty:
        return 0.0, 0.0, []

    novelty_score = float(novelty.get("novelty_score", 0.5))
    confidence_val = float(novelty.get("confidence", 0.5))

    # Higher novelty = exploration bonus
    delta = (novelty_score - 0.5) * 0.8  # moderate influence
    evidence = (
        [
            {
                "signal": "novelty",
                "novelty_score": novelty_score,
                "delta": round(delta, 4),
            }
        ]
        if abs(delta) > 0.05
        else []
    )

    return delta, confidence_val, evidence


def _score_intent_alignment(
    candidate: Dict[str, Any],
    ctx: JudgmentContext,
    signals: Dict[str, Any],
) -> _ScorerReturn:
    """Boost candidates that match the stated intent."""
    if not ctx.intent:
        return 0.0, 0.0, []

    _INTENT_OP_AFFINITIES: Dict[str, FrozenSet[str]] = {
        "refine_compression": frozenset(
            {
                "ternary_projection",
                "nm_sparse_linear",
                "block_sparse_linear",
                "semi_structured_2_4_linear",
                "low_rank_proj",
                "bottleneck_proj",
                "token_merging",
            }
        ),
        "improve_stability": frozenset(
            {
                "rmsnorm",
                "layernorm",
                "dynamic_norm",
                "learnable_scale",
            }
        ),
        "expand_capacity": frozenset(
            {
                "moe_topk",
                "moe_2expert",
                "linear_proj_up",
                "softmax_attention",
                "graph_attention",
                "linear_attention",
            }
        ),
        "beat_benchmarks": frozenset(
            {
                "softmax_attention",
                "selective_scan",
                "swiglu_mlp",
                "linear_attention",
                "conv1d_seq",
            }
        ),
    }

    affinity_ops = _INTENT_OP_AFFINITIES.get(ctx.intent, frozenset())
    if not affinity_ops:
        return 0.0, 0.0, []

    ops = set(candidate.get("ops", []))
    if not ops:
        return 0.0, 0.0, []

    matches = ops & affinity_ops
    if not matches:
        return (
            -0.1,
            0.5,
            [
                {
                    "signal": "intent_alignment",
                    "intent": ctx.intent,
                    "matched": 0,
                    "delta": -0.1,
                }
            ],
        )

    alignment = len(matches) / max(len(ops), 1)
    delta = alignment * 0.6
    evidence = [
        {
            "signal": "intent_alignment",
            "intent": ctx.intent,
            "matched_ops": sorted(matches),
            "alignment": round(alignment, 4),
            "delta": round(delta, 4),
        }
    ]
    return delta, 0.7, evidence


def _score_peer_comparison(
    candidate: Dict[str, Any],
    ctx: JudgmentContext,
    signals: Dict[str, Any],
) -> _ScorerReturn:
    """Score based on how nearest historical peers performed.

    Penalizes candidates whose op-set is similar to historically-failed peers.
    Rewards similarity to successful peers.
    """
    peers = signals.get("nearest_peers", [])
    if not peers:
        return 0.0, 0.0, []

    score_sum = 0.0
    weight_sum = 0.0
    evidence = []

    for peer in peers:
        sim = float(peer.get("jaccard_similarity", 0.0))
        if sim < 0.1:
            continue
        s1 = bool(peer.get("stage1_passed", False))
        loss = peer.get("loss_ratio")
        # Positive signal from successful peers, negative from failed
        if s1 and loss is not None:
            peer_quality = max(0.0, 1.0 - min(float(loss), 1.5))
            delta = sim * (peer_quality - 0.3)
        elif not s1:
            delta = -sim * 0.4
        else:
            continue
        score_sum += delta
        weight_sum += sim

    if weight_sum < 0.1:
        return 0.0, 0.0, []

    avg_delta = score_sum / weight_sum
    confidence = min(1.0, len(peers) / 3.0)

    if abs(avg_delta) > 0.05:
        evidence.append(
            {
                "signal": "peer_comparison",
                "n_peers": len(peers),
                "avg_similarity": round(weight_sum / max(len(peers), 1), 4),
                "delta": round(avg_delta, 4),
            }
        )

    return avg_delta, confidence, evidence


# ── Scorer registry (dict dispatch) ──────────────────────────────────

_SIGNAL_SCORERS: Dict[str, _ScorerFn] = {
    "op_priors": _score_op_priors,
    "op_pairs": _score_op_pairs,
    "fingerprint_bucket": _score_fingerprint_bucket,
    "lineage": _score_lineage,
    "failure_risk": _score_failure_risk,
    "insight_interactions": _score_insight_interactions,
    "novelty": _score_novelty,
    "intent_alignment": _score_intent_alignment,
    "peer_comparison": _score_peer_comparison,
}


# ── Scoring pipeline ─────────────────────────────────────────────────


def _run_scoring_pipeline(
    candidate: Dict[str, Any],
    ctx: JudgmentContext,
    signals: Dict[str, Any],
) -> JudgmentResult:
    """Run all signal scorers and compose into a JudgmentResult."""
    total = 0.0
    total_weight = 0.0
    breakdown: Dict[str, float] = {}
    all_evidence: List[Dict[str, Any]] = []
    support_counts: Dict[str, int] = {}
    risk_flags: List[str] = []

    for name, scorer in _SIGNAL_SCORERS.items():
        delta, confidence, evidence = scorer(candidate, ctx, signals)

        # Cap any single signal's influence at ±_SIGNAL_CAP
        capped = max(-_SIGNAL_CAP, min(_SIGNAL_CAP, delta))
        weighted = capped * confidence
        total += weighted
        total_weight += confidence

        breakdown[name] = round(weighted, 6)
        all_evidence.extend(evidence)
        support_counts[name] = len(evidence)

        # Collect risk flags from failure_risk scorer
        if name == "failure_risk":
            for ev in evidence:
                if ev.get("severity") == "critical":
                    risk_flags.append(ev.get("signature", "unknown"))

    # Normalize to [0, 1] range
    if total_weight > 0:
        normalized = 0.5 + (total / (total_weight * 2.0))
    else:
        normalized = 0.5
    normalized = max(0.0, min(1.0, normalized))

    # Determine recommended action
    action = _recommend_action(normalized, risk_flags, ctx)

    return JudgmentResult(
        total_score=round(normalized, 6),
        signal_breakdown=breakdown,
        confidence=round(min(1.0, total_weight / len(_SIGNAL_SCORERS)), 4),
        risk_flags=risk_flags,
        recommended_action=action,
        evidence=all_evidence,
        support_counts=support_counts,
    )


def _recommend_action(score: float, risk_flags: List[str], ctx: JudgmentContext) -> str:
    """Determine recommended action from score and flags."""
    if risk_flags:
        return "discard"
    if score >= 0.7:
        return "promote"
    if score >= 0.55:
        return "mutate"
    if score >= 0.4:
        return "hold"
    return "discard"


# ── Public entry points ───────────────────────────────────────────────


def _extract_candidate_dict(graph: Any) -> Dict[str, Any]:
    """Extract a candidate dict from a ComputationGraph or dict."""
    if isinstance(graph, dict):
        return graph

    ops: List[str] = []
    pairs: List[str] = []
    nodes = getattr(graph, "nodes", {})

    # Extract ops from topological order
    topo = getattr(graph, "topological_order", None)
    if callable(topo):
        try:
            order = topo()
        except Exception:
            order = list(nodes.keys())
    else:
        order = list(nodes.keys())

    prev_op = None
    for nid in order:
        node = nodes.get(nid)
        if node is None:
            continue
        op_name = getattr(node, "op_name", "")
        if op_name and op_name not in ("input", "output"):
            ops.append(op_name)
            if prev_op:
                pairs.append(f"{prev_op}->{op_name}")
            prev_op = op_name

    return {"ops": ops, "op_pairs": pairs}


def score_candidate(
    candidate_graph: Any,
    ctx: JudgmentContext,
    signals: Dict[str, Any],
) -> JudgmentResult:
    """Score a single candidate graph.

    Args:
        candidate_graph: ComputationGraph or dict with 'ops' and 'op_pairs' keys.
        ctx: Judgment context with fingerprint, parent, intent info.
        signals: Research signals dict (from recommendation-signals endpoint).

    Returns:
        JudgmentResult with score, breakdown, evidence, and recommendation.
    """
    candidate = _extract_candidate_dict(candidate_graph)
    return _run_scoring_pipeline(candidate, ctx, signals)


def recommend_components(
    ctx: JudgmentContext,
    signals: Dict[str, Any],
    candidates: Sequence[Any],
) -> List[Tuple[Any, JudgmentResult]]:
    """Rank multiple candidates by judgment score.

    Returns list of (candidate, JudgmentResult) sorted by total_score descending,
    with exploration budget applied.

    Args:
        ctx: Shared judgment context.
        signals: Research signals dict.
        candidates: Sequence of ComputationGraph or candidate dicts.

    Returns:
        Ranked list of (candidate, result) tuples.
    """
    scored: List[Tuple[Any, JudgmentResult]] = []
    for c in candidates:
        result = score_candidate(c, ctx, signals)
        scored.append((c, result))

    # Sort by total_score descending
    scored.sort(key=lambda pair: pair[1].total_score, reverse=True)

    if not scored:
        return scored

    # Apply exploration budget: reserve bottom slots for under-sampled candidates
    n = len(scored)
    n_explore = max(1, int(n * _EXPLORATION_BUDGET))
    n_exploit = n - n_explore

    # Partition: top exploit slots stay ranked, explore slots are drawn from
    # candidates with lowest support (most under-explored)
    if n > n_explore + 1:
        exploit = scored[:n_exploit]
        explore_pool = scored[n_exploit:]
        # Sort explore pool by min support count (most novel first)
        explore_pool.sort(
            key=lambda pair: sum(pair[1].support_counts.values()),
        )
        return exploit + explore_pool

    return scored
