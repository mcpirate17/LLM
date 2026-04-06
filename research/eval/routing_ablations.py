from __future__ import annotations

from typing import Any, Dict, Iterable, List


_ROUTING_VARIANT_ORDER = (
    "no_routing",
    "single_token",
    "dense_pair",
    "dense_triplet",
    "sparse_pair",
    "sparse_triplet",
    "token_byte",
)


def summarize_routing_variant(metrics: Dict[str, Any]) -> Dict[str, Any]:
    quality = float(metrics.get("quality", 0.0) or 0.0)
    compute = float(metrics.get("compute", 0.0) or 0.0)
    quality_per_compute = quality / compute if compute > 0 else 0.0
    out = dict(metrics)
    out["quality_per_compute"] = round(quality_per_compute, 6)
    return out


def rank_matched_budget_variants(
    rows: Iterable[Dict[str, Any]], budget_tolerance: float = 0.1
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    normalized = [summarize_routing_variant(dict(row)) for row in rows]
    for row in normalized:
        compute = float(row.get("compute", 0.0) or 0.0)
        peers = [
            peer
            for peer in normalized
            if peer is not row
            and compute > 0
            and abs(float(peer.get("compute", 0.0) or 0.0) - compute)
            <= compute * budget_tolerance
        ]
        row["matched_budget_peer_count"] = len(peers)
        row["matched_budget"] = len(peers) > 0
        ranked.append(row)
    ranked.sort(
        key=lambda row: (
            not bool(row.get("matched_budget")),
            -float(row.get("quality_per_compute", 0.0) or 0.0),
            _ROUTING_VARIANT_ORDER.index(row.get("variant"))
            if row.get("variant") in _ROUTING_VARIANT_ORDER
            else len(_ROUTING_VARIANT_ORDER),
        )
    )
    return ranked
