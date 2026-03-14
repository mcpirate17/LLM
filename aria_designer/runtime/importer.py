"""
Survivor Importer: research/ lab notebook → aria_designer workflows.

Reads top-performing architectures (stage1 survivors) from the research
LabNotebook and converts them into editable aria_designer workflow JSON.

This is the reverse of bridge.py (which goes workflow → ComputationGraph).

Usage:
    from runtime.importer import import_survivors, graph_to_workflow

    workflows = import_survivors(n=10, sort_by="validation_loss_ratio")
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

_RESEARCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "research"))
if _RESEARCH_ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(_RESEARCH_ROOT))

from research.synthesis.graph import ComputationGraph
from research.synthesis.primitives import PRIMITIVE_REGISTRY
from research.synthesis.workflow_converter import graph_to_workflow as _g2w
try:
    from aria_designer.api.app.component_identity import canonicalize_workflow
except ImportError:
    # Fallback for when running from within aria_designer/ as cwd
    import importlib
    _ci = importlib.import_module("api.app.component_identity")
    canonicalize_workflow = _ci.canonicalize_workflow

def graph_to_workflow(
    graph: ComputationGraph,
    workflow_id: Optional[str] = None,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    wf = _g2w(graph, workflow_id, name, metadata)
    return canonicalize_workflow(wf)

# ── Notebook access ──────────────────────────────────────────────────

def _get_notebook():
    """Get a LabNotebook connection to the research database."""
    from research.scientist.notebook import LabNotebook
    db_path = os.path.join(_RESEARCH_ROOT, "lab_notebook.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Lab notebook not found at {db_path}")
    return LabNotebook(db_path)


def import_survivors(
    n: int = 10,
    sort_by: str = "validation_loss_ratio",
    min_novelty: float = 0.0,
) -> List[Dict[str, Any]]:
    """Import top survivors from the research lab notebook as aria_designer workflows.

    Args:
        n: number of survivors to import
        sort_by: ranking metric ("validation_loss_ratio", "discovery_loss_ratio",
                                 "loss_ratio", "novelty_score",
                                 "structural_novelty", "behavioral_novelty")
        min_novelty: minimum novelty_score threshold

    Returns:
        List of workflow dicts in workflow_graph.v1 format, enriched with
        research metadata (scores, fingerprint, etc.)
    """
    nb = _get_notebook()

    survivors = nb.get_top_programs(n=n * 2, sort_by=sort_by)  # fetch extra for filtering

    workflows = []
    seen_fingerprints = set()

    for prog in survivors:
        if len(workflows) >= n:
            break

        # Filter by novelty
        nov = prog.get("novelty_score") or 0
        if nov < min_novelty:
            continue

        # Deduplicate by graph fingerprint
        fp = prog.get("graph_fingerprint", "")
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)

        # Parse graph
        graph_json_str = prog.get("graph_json")
        if not graph_json_str:
            continue

        try:
            from research.synthesis.serializer import graph_from_json
            graph = graph_from_json(graph_json_str)
        except Exception:
            continue

        # Check if all ops are in PRIMITIVE_REGISTRY
        unknown_ops = []
        for node in graph.nodes.values():
            if not node.is_input and node.op_name not in PRIMITIVE_REGISTRY:
                unknown_ops.append(node.op_name)

        # Convert to workflow
        result_id = prog.get("result_id", uuid4().hex[:8])
        wf = graph_to_workflow(
            graph,
            workflow_id=f"survivor_{result_id}",
            name=f"Survivor {result_id} (novelty={nov:.2f})",
            metadata={
                "result_id": result_id,
                "experiment_id": prog.get("experiment_id", ""),
                "novelty_score": nov,
                "structural_novelty": prog.get("structural_novelty", 0),
                "behavioral_novelty": prog.get("behavioral_novelty", 0),
                "loss_ratio": prog.get("loss_ratio", 0),
                "validation_loss_ratio": prog.get("validation_loss_ratio"),
                "discovery_loss_ratio": prog.get("discovery_loss_ratio"),
                "generalization_gap": prog.get("generalization_gap"),
                "param_count": prog.get("param_count", 0),
                "final_loss": prog.get("final_loss", 0),
                "compatible": len(unknown_ops) == 0,
                "unknown_ops": unknown_ops,
            },
        )
        wf["result_id"] = result_id
        workflows.append(wf)

    return workflows


def import_single(result_id: str) -> Dict[str, Any]:
    """Import a single program by its result_id.

    Returns:
        Workflow dict in workflow_graph.v1 format.

    Raises:
        ValueError: if the result_id is not found or graph is invalid.
    """
    nb = _get_notebook()
    detail = nb.get_program_detail(result_id)
    if detail is None:
        raise ValueError(f"Program result '{result_id}' not found")

    graph_json_str = detail.get("graph_json")
    if not graph_json_str:
        raise ValueError(f"Program '{result_id}' has no graph_json")

    from research.synthesis.serializer import graph_from_json
    graph = graph_from_json(graph_json_str)

    return graph_to_workflow(
        graph,
        workflow_id=f"imported_{result_id}",
        name=f"Imported {result_id}",
        metadata={
            "result_id": result_id,
            "experiment_id": detail.get("experiment_id", ""),
            "novelty_score": detail.get("novelty_score", 0),
            "loss_ratio": detail.get("loss_ratio", 0),
            "validation_loss_ratio": detail.get("validation_loss_ratio"),
            "discovery_loss_ratio": detail.get("discovery_loss_ratio"),
        },
    )
