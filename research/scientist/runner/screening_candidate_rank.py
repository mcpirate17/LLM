from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

from ..notebook import LabNotebook

logger = logging.getLogger(__name__)

try:
    from ..judgment import JudgmentContext, score_candidate

    _HAS_JUDGMENT = True
except ImportError:
    _HAS_JUDGMENT = False

_EXPLORATION_BUDGET = 0.15


def _infer_fingerprint_bucket(graph: Any, bucket_names: set[str]) -> str:
    if not bucket_names:
        return ""
    ops = {
        getattr(node, "op_name", "")
        for node in getattr(graph, "nodes", {}).values()
        if not getattr(node, "is_input", False) and getattr(node, "op_name", "")
    }
    has_attention = any("attention" in op for op in ops)
    has_mixing = any(
        token in op for op in ops for token in ("state_space", "scan", "conv", "mix")
    )
    if has_attention and has_mixing and "hybrid" in bucket_names:
        return "hybrid"
    if has_attention and "attention-heavy" in bucket_names:
        return "attention-heavy"
    if has_mixing and "mixing-heavy" in bucket_names:
        return "mixing-heavy"
    if (
        any(token in op for op in ops for token in ("sparse", "gate", "topk", "moe"))
        and "sparse" in bucket_names
    ):
        return "sparse"
    if "exotic" in bucket_names:
        return "exotic"
    return ""


def judgment_rerank(
    graphs: List[Any],
    nb: LabNotebook,
    log: logging.Logger,
    log_event: Callable[..., None] | None = None,
) -> List[Tuple[Any, float]]:
    if not _HAS_JUDGMENT or not graphs:
        return [(graph, 0.5) for graph in graphs]

    try:
        signals: Dict[str, Any] = {
            "op_pair_priors": nb.get_op_pair_priors(min_support=5, limit=50),
            "fingerprint_buckets": nb.get_fingerprint_buckets(limit=5),
            "lineage_successors": nb.get_lineage_successor_stats(limit=50),
        }
        risk = nb.get_failure_risk_signatures(limit=50)
        signals["failure_risk_signatures"] = risk.get("failure_risk_signatures", [])
        signals["critical_failures"] = risk.get("critical_failures", [])
    except Exception as exc:
        log.debug("judgment_rerank: signals fetch failed (%s), using original order", exc)
        return [(graph, 0.5) for graph in graphs]

    bucket_names = {
        bucket["bucket"]
        for bucket in signals.get("fingerprint_buckets", [])
        if bucket.get("bucket")
    }

    scored: List[tuple[Any, float, int]] = []
    skipped = 0
    for graph in graphs:
        try:
            candidate_signals = signals
            fp = graph.fingerprint() if hasattr(graph, "fingerprint") else None
            if fp and hasattr(nb, "get_nearest_peers"):
                try:
                    peers = nb.get_nearest_peers(fp, n=5)
                    if peers:
                        candidate_signals = {**signals, "nearest_peers": peers}
                except (AttributeError, TypeError, KeyError) as exc:
                    logger.debug("nearest_peers lookup failed: %s", exc)
            ctx = JudgmentContext(
                fingerprint_bucket=_infer_fingerprint_bucket(graph, bucket_names)
            )
            result = score_candidate(graph, ctx, candidate_signals)
        except Exception as exc:
            logger.debug("score_candidate failed, using neutral score: %s", exc)
            scored.append((graph, 0.5, 0))
            continue

        if result.risk_flags:
            log.info(
                "judgment_rerank: discarding candidate with risk flags: %s",
                result.risk_flags,
            )
            skipped += 1
            continue
        scored.append((graph, result.total_score, sum(result.support_counts.values())))

    if skipped:
        log.info(
            "judgment_rerank: filtered %d candidates with critical failures", skipped
        )
    if not scored:
        return [(graph, 0.5) for graph in graphs]

    scored.sort(key=lambda row: row[1], reverse=True)
    n = len(scored)
    n_explore = max(1, int(n * _EXPLORATION_BUDGET))
    n_exploit = n - n_explore
    if n > n_explore + 1:
        exploit = scored[:n_exploit]
        explore_pool = scored[n_exploit:]
        explore_pool.sort(key=lambda row: row[2])
        scored = exploit + explore_pool

    if log_event is not None:
        try:
            scores = [row[1] for row in scored]
            log_event(
                nb,
                "judgment_rerank",
                f"Reranked {n} candidates ({skipped} filtered)",
                n_candidates=n,
                n_filtered=skipped,
                score_min=round(min(scores), 3),
                score_max=round(max(scores), 3),
                score_mean=round(sum(scores) / len(scores), 3),
                n_explore=n_explore,
            )
        except (TypeError, ValueError, AttributeError) as exc:
            logger.debug("Failed logging judgment_rerank event: %s", exc)

    return [(row[0], row[1]) for row in scored]
