"""
Normalized Compression Distance (NCD) — Information-Theoretic Reward Signal

Replaces quality_per_byte proxy with NCD, which measures how much a graph
description "explains" its loss curve. Low NCD = the graph structure and
training behavior are informationally redundant (good — compact description
captures the behavior). High NCD = they are unrelated (bad — graph structure
doesn't predict training dynamics).

NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

Where C = zlib.compress(level=9).
"""

from __future__ import annotations

import json
import zlib
from typing import Any, Dict, List, Optional, Union


def _compress_len(data: bytes) -> int:
    """Return the compressed length of data using zlib level 9."""
    return len(zlib.compress(data, level=9))


def compute_ncd(x: bytes, y: bytes) -> float:
    """Compute Normalized Compression Distance between two byte strings.

    Returns a value in [0, 1] where 0 = maximally similar and 1 = maximally different.
    """
    cx = _compress_len(x)
    cy = _compress_len(y)
    cxy = _compress_len(x + y)
    denominator = max(cx, cy)
    if denominator == 0:
        return 0.0
    ncd = (cxy - min(cx, cy)) / denominator
    # Clamp to [0, 1] — NCD can slightly exceed 1 due to compression artifacts
    return max(0.0, min(1.0, ncd))


def compute_graph_ncd(
    graph_json: str,
    loss_curve: Union[List[float], List[Dict[str, Any]]],
    n_params: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute NCD between a graph description and its training loss curve.

    Args:
        graph_json: JSON string of the computation graph.
        loss_curve: Either a list of loss floats, or a list of dicts with "loss" key.
        n_params: Optional parameter count for per-param description length.

    Returns:
        Dict with keys:
            ncd_score: float in [0, 1]
            description_length: int (compressed graph bytes)
            description_length_per_param: float or None
    """
    # Serialize graph
    graph_bytes = graph_json.encode("utf-8")

    # Serialize loss curve
    if loss_curve and isinstance(loss_curve[0], dict):
        losses = [float(d.get("loss", 0)) for d in loss_curve if d.get("loss") is not None]
    else:
        losses = [float(v) for v in loss_curve]

    curve_str = ",".join(f"{v:.6f}" for v in losses)
    curve_bytes = curve_str.encode("utf-8")

    ncd_score = compute_ncd(graph_bytes, curve_bytes)
    description_length = _compress_len(graph_bytes)

    dl_per_param = None
    if n_params and n_params > 0:
        dl_per_param = description_length / n_params

    return {
        "ncd_score": ncd_score,
        "description_length": description_length,
        "description_length_per_param": dl_per_param,
    }
