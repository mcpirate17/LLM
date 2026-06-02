"""Auto-register validated mined chains as live templates.

The chain-mining → promotion → validation pipeline emits
``research/data/synthesis_candidates/validated_template_candidates.json``. When the
``ARIA_ENABLE_MINED_TEMPLATES`` environment flag (or its sibling caller
hook) is set, this module materializes each ``ready_for_registration``
candidate into a real template callable and registers it into the
TEMPLATES dict at import time.

Each mined template is a flat sequence of single-input ops with a
trailing dimension fix. Mined templates record slot bindings under
``selected_motif_class="mined_op"`` so analytics can split mined-vs-static
template performance.

Gate by feature flag, never replace existing templates, and skip silently
if the validated-candidates artifact is missing.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from typing import TYPE_CHECKING

from ._template_helpers import (
    record_template_slot_binding,
    template_add_op as _add,
)

if TYPE_CHECKING:
    from .graph import ComputationGraph
else:
    ComputationGraph = Any

logger = logging.getLogger(__name__)

_ENV_FLAG = "ARIA_ENABLE_MINED_TEMPLATES"
_ENV_PATH = "ARIA_MINED_TEMPLATES_PATH"  # optional override (tests + manual)
_DEFAULT_VALIDATED_PATH = Path(
    "research/data/synthesis_candidates/validated_template_candidates.json"
)
_DEFAULT_MINED_WEIGHT = 0.5  # below default to prevent crowding out human templates
# Hard cap on number of mined templates registered per process. Prevents a
# runaway candidate JSON from doubling TEMPLATES size.
_MAX_MINED_TEMPLATES = 32


TemplateFn = Callable[["ComputationGraph", int, random.Random, Any], int]


def _make_chain_template(name: str, chain: Tuple[str, ...]) -> TemplateFn:
    """Construct a TEMPLATES callable from a flat chain of arity-1 ops.

    Mirrors the wrapper validator's ``_build_chain_graph`` topology:
      input → rmsnorm → [chain ops] → fix_dim
    minus the input/output node creation (caller handles those).
    """

    def _template(
        graph: "ComputationGraph",
        input_id: int,
        rng: random.Random,
        weights: Any = None,
    ) -> int:
        del rng, weights  # mined templates have no slot lottery
        template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
        current = _add(graph, "rmsnorm", [input_id], context=f"{name}.input_norm")
        for i, op in enumerate(chain):
            current = _add(graph, op, [current], context=f"{name}.step{i}.{op}")
            record_template_slot_binding(
                graph,
                template_name=name,
                template_instance=template_instance,
                slot_index=i,
                slot_key=f"{name}[{template_instance}].step{i}",
                slot_classes=("mined_step",),
                selected_name=op,
                selected_class="mined_op",
                input_node_id=current,
            )
        cur_dim = graph.nodes[current].output_shape.dim
        if cur_dim != graph.model_dim:
            fix_op = (
                "linear_proj_down" if cur_dim > graph.model_dim else "linear_proj_up"
            )
            current = _add(
                graph,
                fix_op,
                [current],
                {"out_dim": graph.model_dim},
                context=f"{name}.fix_dim",
            )
        return current

    _template.__name__ = name
    _template.__qualname__ = name
    return _template


def _load_validated_candidates(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    ready = payload.get("ready_for_registration")
    if isinstance(ready, list):
        return [c for c in ready if isinstance(c, dict)]
    # Fall back to the full candidate list when the validator didn't emit a
    # ready bucket (e.g. early Phase 1 output).
    fallback = payload.get("candidates") or []
    return [c for c in fallback if isinstance(c, dict)]


def _candidate_is_eligible(candidate: Dict[str, Any]) -> bool:
    """Require the chain to have actually passed forward + backward smoke."""
    validation = candidate.get("validation") or {}
    if not validation.get("compile_passed"):
        return False
    # backward_passed implies forward_passed AND finite gradients.
    if not validation.get("backward_passed"):
        return False
    return True


def register_mined_templates(
    templates_dict: Dict[str, TemplateFn],
    weights_dict: Dict[str, float],
    *,
    json_path: str | Path | None = None,
    enable: bool | None = None,
    max_register: int = _MAX_MINED_TEMPLATES,
) -> List[str]:
    """Merge mined templates into the existing template registry.

    Args:
        templates_dict: target dict (typically ``synthesis.templates.TEMPLATES``).
        weights_dict: target weight dict (``DEFAULT_TEMPLATE_WEIGHTS``).
        json_path: validated candidates JSON; defaults to synthesis_candidates.
        enable: explicit override; when None, falls back to the env flag.
        max_register: hard cap on number of templates registered.

    Returns:
        List of template names actually registered (subset; may be empty).
    """
    if enable is None:
        enable = os.environ.get(_ENV_FLAG, "0") not in ("", "0", "false", "False")
    if not enable:
        return []

    if json_path is not None:
        path = Path(json_path)
    elif os.environ.get(_ENV_PATH):
        path = Path(os.environ[_ENV_PATH])
    else:
        path = _DEFAULT_VALIDATED_PATH
    candidates = _load_validated_candidates(path)
    if not candidates:
        return []

    registered: List[str] = []
    for candidate in candidates:
        if len(registered) >= max_register:
            break
        if not _candidate_is_eligible(candidate):
            continue
        name = str(candidate.get("proposed_template_name") or "").strip()
        chain = tuple(str(op) for op in candidate.get("chain") or ())
        if not name or not chain:
            continue
        if name in templates_dict:
            # Existing template wins; mined name collision is silent skip.
            continue
        templates_dict[name] = _make_chain_template(name, chain)
        weights_dict.setdefault(name, _DEFAULT_MINED_WEIGHT)
        registered.append(name)

    if registered:
        logger.info(
            "Registered %d mined templates from %s: %s",
            len(registered),
            path,
            ", ".join(registered[:5]) + ("..." if len(registered) > 5 else ""),
        )
    return registered
