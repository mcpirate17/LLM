"""
Architecture Naming — Human-readable names for discovered architectures.

Generates display names from architecture family + distinguishing ops + short
hash suffix, e.g. "Conv-Mixer #d043" instead of "d043261be7cd9f5c".
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set


def architecture_display_name(
    family: Optional[str],
    graph_fingerprint: Optional[str],
    graph_json: Optional[str] = None,
    architecture_desc: Optional[str] = None,
) -> str:
    """Generate a human-readable architecture name.

    Priority:
    1. Family + distinguishing suffix from graph ops + short hash
    2. Architecture desc (if provided and non-empty) + short hash
    3. Short hash alone
    """
    short_hash = _short_hash(graph_fingerprint)

    if family and family != "Unknown":
        suffix = _distinguishing_suffix(graph_json, family)
        base = f"{family}{suffix}"
        if short_hash:
            return f"{base} #{short_hash}"
        return base

    if architecture_desc and architecture_desc.strip():
        desc = architecture_desc.strip()
        if len(desc) > 30:
            desc = desc[:27] + "..."
        if short_hash:
            return f"{desc} #{short_hash}"
        return desc

    if short_hash:
        return f"Architecture #{short_hash}"
    return "Unknown Architecture"


def _short_hash(fingerprint: Optional[str]) -> str:
    """Extract a 4-character hash suffix from a fingerprint."""
    if not fingerprint or not isinstance(fingerprint, str):
        return ""
    fp = fingerprint.strip()
    if len(fp) >= 4:
        return fp[:4]
    return fp


def _distinguishing_suffix(graph_json: Optional[str], family: str) -> str:
    """Extract distinguishing ops that aren't implied by the family name."""
    if not graph_json:
        return ""

    try:
        graph = json.loads(graph_json)
        nodes = graph.get("nodes")
        if isinstance(nodes, dict):
            node_iter = [n for n in nodes.values() if isinstance(n, dict)]
        elif isinstance(nodes, list):
            node_iter = [n for n in nodes if isinstance(n, dict)]
        else:
            return ""
        ops = {str(n.get("op_name", "")).strip() for n in node_iter}
        ops.discard("")
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""

    if not ops:
        return ""

    # Ops that are interesting differentiators but not implied by the family
    notable_ops: Dict[str, str] = {
        "layer_norm": "LN",
        "rmsnorm": "RMS",
        "batch_norm": "BN",
        "dropout": "Drop",
        "residual": "Res",
        "skip_connection": "Skip",
        "concat": "Cat",
        "split2": "Split",
        "swiglu": "SwiGLU",
        "gelu": "GELU",
        "silu": "SiLU",
        "fourier_mix": "FFT",
        "depthwise_conv1d": "DWConv",
        "softmax_attention": "SoftAttn",
        "linear_attention": "LinAttn",
        "shared_qk_attention": "SharedQK",
    }

    # Family-implied ops to skip
    family_lower = family.lower()
    skip_labels: Set[str] = set()
    if "conv" in family_lower:
        skip_labels.update({"DWConv"})
    if "attention" in family_lower:
        skip_labels.update({"SoftAttn", "LinAttn", "SharedQK"})
    if "spectral" in family_lower:
        skip_labels.update({"FFT"})
    if "gated" in family_lower or "nonlinear" in family_lower:
        skip_labels.update({"SwiGLU", "GELU", "SiLU"})

    suffixes = []
    for op, label in notable_ops.items():
        if op in ops and label not in skip_labels:
            suffixes.append(label)
        if len(suffixes) >= 2:
            break

    if suffixes:
        return " (" + "+".join(suffixes) + ")"
    return ""


def annotate_display_names(entries: List[Dict[str, Any]]) -> None:
    """Add 'display_name' field to a list of candidate/leaderboard entries in-place."""
    for entry in entries:
        entry["display_name"] = architecture_display_name(
            family=entry.get("architecture_family"),
            graph_fingerprint=entry.get("graph_fingerprint"),
            graph_json=entry.get("_graph_json"),
            architecture_desc=entry.get("architecture_desc"),
        )
