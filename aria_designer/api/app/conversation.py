"""Conversational chat manager for Aria Designer.

Handles multi-turn conversations where users describe goals and Aria
proposes graph modifications iteratively. Reuses intent_parser for
classification — never duplicates that logic.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from . import database as db
from .component_identity import (
    discover_concepts,
)
from .intent_parser import (
    _COMPONENT_GROUPS,
    component_groups,
    parse_intent_constraints,
)
from .models import utc_now_iso

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages multi-turn Aria chat sessions.

    Tracks applied changes, rejected suggestions, and user goals to
    produce increasingly relevant suggestions over the conversation.
    """

    __slots__ = ()

    # ── Session lifecycle ─────────────────────────────────────────────

    @staticmethod
    def start_session(workflow_json: Optional[Dict[str, Any]] = None) -> str:
        """Create a new conversation session. Returns session_id."""
        session_id = f"chat_{uuid4().hex[:12]}"
        now = utc_now_iso()
        workflow_id = (workflow_json or {}).get("workflow_id")
        db.create_conversation(session_id, workflow_id, now)
        db.add_message(
            session_id,
            "system",
            "Conversation started. I can help you design, modify, and improve your architecture.",
            now,
        )
        return session_id

    @staticmethod
    def get_session(session_id: str) -> Optional[Dict[str, Any]]:
        """Get session info including message count."""
        conv = db.get_conversation(session_id)
        if conv is None:
            return None
        messages = db.get_messages(session_id)
        conv["message_count"] = len(messages)
        return conv

    @staticmethod
    def end_session(session_id: str) -> bool:
        """End a conversation session."""
        return db.end_conversation(session_id, utc_now_iso())

    @staticmethod
    def get_history(session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get conversation history."""
        rows = db.get_messages(session_id, limit=limit)
        result: list[Dict[str, Any]] = []
        for row in rows:
            entry: Dict[str, Any] = {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            meta_raw = row.get("metadata_json")
            if meta_raw:
                try:
                    entry["metadata"] = json.loads(meta_raw)
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(entry)
        return result

    # ── Message processing ────────────────────────────────────────────

    @staticmethod
    def process_message(
        session_id: str,
        message: str,
        workflow_json: Optional[Dict[str, Any]] = None,
        research_signals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a user message and generate a response.

        Returns a dict with: content, patch_proposal (optional),
        suggestions (list), needs_clarification (bool).
        """
        now = utc_now_iso()

        # Store user message
        db.add_message(session_id, "user", message, now)
        db.update_conversation_timestamp(session_id, now)

        # Load conversation context
        history = db.get_messages(session_id, limit=50)
        context = _build_context(history, workflow_json)

        # Classify intent
        intent_result = _classify_message(message, context)

        # Generate response based on intent
        response = _generate_response(
            intent_result,
            message,
            context,
            workflow_json,
            research_signals,
        )

        # Store Aria's response
        response_meta = {}
        if response.get("patch_proposal"):
            response_meta["patch_proposal"] = response["patch_proposal"]
        if response.get("suggestions"):
            response_meta["suggestions_count"] = len(response["suggestions"])

        db.add_message(
            session_id,
            "aria",
            response["content"],
            utc_now_iso(),
            metadata=response_meta if response_meta else None,
        )

        return response


# ── Internal helpers ──────────────────────────────────────────────────

_CLARIFICATION_TRIGGERS = frozenset(
    {
        "help",
        "what",
        "how",
        "why",
        "explain",
        "confused",
        "don't understand",
        "not sure",
        "which",
        "should i",
        "difference between",
    }
)

_ACTION_TRIGGERS = frozenset(
    {
        "add",
        "insert",
        "remove",
        "delete",
        "replace",
        "swap",
        "change",
        "use",
        "try",
        "put",
        "connect",
        "wire",
    }
)

_GOAL_TRIGGERS = frozenset(
    {
        "want",
        "need",
        "goal",
        "build",
        "create",
        "design",
        "make",
        "improve",
        "optimize",
        "faster",
        "better",
        "smaller",
        "stable",
    }
)

# ── Architectural pattern detectors ──────────────────────────────────
# Each pattern is (keywords, description, nodes) where
# nodes is a list of (role_label, component_type, params).

_ARCH_PATTERNS: List[Dict[str, Any]] = [
    {
        "keywords": {"split", "path", "easy", "hard", "difficulty", "scorer"},
        "min_match": 3,
        "name": "difficulty-gated split",
        "description": (
            "Token difficulty scorer → lane router → fast easy lane + "
            "existing hard lane → gather back to output"
        ),
        "nodes": [
            ("difficulty_scorer", "routing/difficulty_scorer", {}),
            ("lane_router", "routing/lane_router", {"num_lanes": 2}),
            (
                "easy_dispatch",
                "structural/conditional_dispatch",
                {"num_lanes": 2, "lane": 0},
            ),
            ("easy_path", "linear_algebra/linear_proj", {}),
            (
                "hard_dispatch",
                "structural/conditional_dispatch",
                {"num_lanes": 2, "lane": 1},
            ),
            ("gather", "structural/conditional_gather", {"num_lanes": 2}),
        ],
    },
    {
        "keywords": {"transformer", "attention", "residual"},
        "min_match": 1,
        "name": "transformer block",
        "description": (
            "Classic transformer: norm → attention → residual → norm → FFN → residual"
        ),
        "nodes": [
            ("norm1", "linear_algebra/rmsnorm", {}),
            ("attn", "mixing/softmax_attention", {"heads": 4}),
            ("residual1", "math/add", {}),
            ("norm2", "linear_algebra/rmsnorm", {}),
            ("ffn_up", "linear_algebra/linear_proj_up", {"out_dim": 1024}),
            ("act", "math/gelu", {}),
            ("ffn_down", "linear_algebra/linear_proj_down", {"out_dim": 256}),
            ("residual2", "math/add", {}),
        ],
    },
    {
        "keywords": {"ssm", "mamba", "state space", "scan"},
        "min_match": 1,
        "name": "SSM block",
        "description": (
            "Mamba-style SSM: norm → linear_proj → selective_scan → "
            "silu → linear_proj_down"
        ),
        "nodes": [
            ("norm", "linear_algebra/rmsnorm", {}),
            ("proj_in", "linear_algebra/linear_proj_up", {"out_dim": 512}),
            ("scan", "linear_algebra/selective_scan", {}),
            ("act", "math/silu", {}),
            ("proj_out", "linear_algebra/linear_proj_down", {"out_dim": 256}),
        ],
    },
    {
        "keywords": {"hybrid", "parallel", "attention", "ssm"},
        "min_match": 2,
        "name": "hybrid attention+SSM",
        "description": ("Parallel attention and SSM paths with learned merge"),
        "nodes": [
            ("norm", "linear_algebra/rmsnorm", {}),
            ("split", "structural/split2", {}),
            ("attn_path", "mixing/softmax_attention", {"heads": 4}),
            ("ssm_path", "linear_algebra/selective_scan", {}),
            ("merge", "structural/concat", {}),
            ("proj_out", "linear_algebra/linear_proj_down", {"out_dim": 256}),
        ],
    },
    {
        "keywords": {"moe", "mixture", "expert", "routing"},
        "min_match": 1,
        "name": "mixture-of-experts",
        "description": "MoE: gate → route to expert FFNs → merge",
        "nodes": [
            ("gate", "channel_mixing/moe_topk", {"top_k": 2, "num_experts": 4}),
            ("split", "structural/split2", {}),
            ("expert1", "linear_algebra/linear_proj", {"out_dim": 256}),
            ("expert2", "linear_algebra/linear_proj", {"out_dim": 256}),
            ("merge", "structural/concat", {}),
            ("proj_out", "linear_algebra/linear_proj_down", {"out_dim": 256}),
        ],
    },
    {
        "keywords": {"fast", "simple", "ffn", "feedforward"},
        "min_match": 1,
        "name": "simple FFN",
        "description": "Feed-forward: linear_up → activation → linear_down",
        "nodes": [
            ("ffn_up", "linear_algebra/linear_proj_up", {"out_dim": 1024}),
            ("act", "math/gelu", {}),
            ("ffn_down", "linear_algebra/linear_proj_down", {"out_dim": 256}),
        ],
    },
]


def _build_context(
    history: List[Dict[str, Any]],
    workflow_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build conversation context from history and workflow state."""
    ctx: Dict[str, Any] = {
        "turn_count": sum(1 for m in history if m.get("role") == "user"),
        "applied_changes": [],
        "rejected_suggestions": [],
        "user_goals": [],
    }

    for msg in history:
        meta_raw = msg.get("metadata_json")
        if not meta_raw:
            continue
        try:
            meta = json.loads(meta_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if meta.get("patch_applied"):
            ctx["applied_changes"].append(meta["patch_applied"])
        if meta.get("rejected"):
            ctx["rejected_suggestions"].append(meta["rejected"])

    if workflow_json:
        nodes = workflow_json.get("nodes", [])
        edges = workflow_json.get("edges", [])
        ctx["node_count"] = len(nodes)
        ctx["edge_count"] = len(edges)
        ctx["component_types"] = [str(n.get("component_type", "")) for n in nodes]
        ctx["nodes"] = nodes
        ctx["edges"] = edges
    return ctx


def _resolve_concepts(message: str) -> List[Dict[str, Any]]:
    """Extract component concepts from natural language."""
    return discover_concepts(message)


def _match_pattern(message: str) -> Optional[Dict[str, Any]]:
    """Match message against architectural patterns. Returns best match."""
    lower = message.lower()
    best: Optional[Dict[str, Any]] = None
    best_score = 0

    for pattern in _ARCH_PATTERNS:
        keywords = pattern["keywords"]
        matches = sum(1 for kw in keywords if kw in lower)
        if matches >= pattern["min_match"] and matches > best_score:
            best = pattern
            best_score = matches

    return best


def _classify_message(
    message: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify user message intent for response routing."""
    lower = message.lower()
    tokens = set(lower.split())

    result: Dict[str, Any] = {"type": "general", "components_mentioned": []}

    # Resolve natural language concepts to components
    resolved = _resolve_concepts(message)
    result["resolved_concepts"] = resolved
    result["components_mentioned"] = [r["component_type"] for r in resolved]

    # Check for architectural pattern match
    pattern = _match_pattern(message)
    if pattern:
        result["matched_pattern"] = pattern

    for cat in _COMPONENT_GROUPS:
        if cat in lower:
            result.setdefault("categories_mentioned", []).append(cat)

    # Classify intent type — pattern match overrides simple keyword checks
    if pattern and (tokens & (_GOAL_TRIGGERS | _ACTION_TRIGGERS)):
        result["type"] = "architecture"
    elif tokens & _ACTION_TRIGGERS and result["components_mentioned"]:
        result["type"] = "action"
    elif tokens & _GOAL_TRIGGERS:
        result["type"] = "goal"
    elif result["components_mentioned"] and len(result["components_mentioned"]) >= 2:
        # Multiple components mentioned implies architectural intent
        result["type"] = "architecture"
    elif tokens & _CLARIFICATION_TRIGGERS:
        result["type"] = "question"
    elif context.get("turn_count", 0) == 0:
        result["type"] = "greeting"

    # Use intent_parser for constraint extraction
    try:
        constraints = parse_intent_constraints(message)
        result["intent_key"] = constraints.intent_key
        result["preferred_groups"] = list(constraints.preferred_component_groups)
    except (KeyError, TypeError):
        result["intent_key"] = "balanced_refine"

    return result


def _find_endpoints(
    nodes: List[Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    input_node = next(
        (n for n in nodes if "input" in str(n.get("component_type", "")).lower()),
        None,
    )
    output_node = next(
        (n for n in nodes if "output" in str(n.get("component_type", "")).lower()),
        None,
    )
    return input_node, output_node


def _find_primary_path_edges(
    input_node: Optional[Dict[str, Any]],
    output_node: Optional[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if input_node is None or output_node is None:
        return None, None
    outgoing = [edge for edge in edges if str(edge.get("source")) == input_node["id"]]
    incoming = [edge for edge in edges if str(edge.get("target")) == output_node["id"]]
    input_edge = outgoing[0] if len(outgoing) == 1 else None
    output_edge = incoming[0] if len(incoming) == 1 else None
    return input_edge, output_edge


def _append_add_node(
    ops: List[Dict[str, Any]],
    node_id: str,
    component_type: str,
    params: Dict[str, Any],
    x: int,
    y: int,
) -> None:
    ops.append(
        {
            "op": "add_node",
            "node_id": node_id,
            "payload": {
                "id": node_id,
                "component_type": component_type,
                "params": params,
                "ui_meta": {"x": x, "y": y},
            },
        }
    )


def _append_rewire(
    ops: List[Dict[str, Any]],
    source: str,
    target: str,
    *,
    remove_edge_id: Optional[str] = None,
    source_port: Optional[str] = None,
    target_port: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {"source": source, "target": target}
    if remove_edge_id:
        payload["remove_edge_id"] = remove_edge_id
    if source_port:
        payload["source_port"] = source_port
    if target_port:
        payload["target_port"] = target_port
    ops.append({"op": "rewire", "node_id": target, "payload": payload})


def _build_difficulty_routed_patch(
    workflow_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    nodes = (workflow_json or {}).get("nodes", [])
    edges = (workflow_json or {}).get("edges", [])
    ops: list[Dict[str, Any]] = []

    input_node, output_node = _find_endpoints(nodes)
    input_edge, output_edge = _find_primary_path_edges(input_node, output_node, edges)

    base_x = 200
    base_y = 0
    if input_node:
        position = input_node.get("ui_meta", {}).get("position", {})
        base_x = position.get("x", 200) + 180
        base_y = position.get("y", 0) - 120

    difficulty_id = f"aria_difficulty_{uuid4().hex[:4]}"
    router_id = f"aria_router_{uuid4().hex[:4]}"
    easy_dispatch_id = f"aria_easy_dispatch_{uuid4().hex[:4]}"
    hard_dispatch_id = f"aria_hard_dispatch_{uuid4().hex[:4]}"
    easy_lane_id = f"aria_fast_lane_{uuid4().hex[:4]}"
    gather_id = f"aria_gather_{uuid4().hex[:4]}"

    _append_add_node(
        ops, difficulty_id, "routing/difficulty_scorer", {}, base_x, base_y
    )
    _append_add_node(
        ops, router_id, "routing/lane_router", {"num_lanes": 2}, base_x + 180, base_y
    )
    _append_add_node(
        ops,
        easy_dispatch_id,
        "structural/conditional_dispatch",
        {"num_lanes": 2, "lane": 0},
        base_x + 360,
        base_y - 90,
    )
    _append_add_node(
        ops, easy_lane_id, "linear_algebra/linear_proj", {}, base_x + 540, base_y - 90
    )
    _append_add_node(
        ops,
        hard_dispatch_id,
        "structural/conditional_dispatch",
        {"num_lanes": 2, "lane": 1},
        base_x + 360,
        base_y + 90,
    )
    _append_add_node(
        ops,
        gather_id,
        "structural/conditional_gather",
        {"num_lanes": 2},
        base_x + 720,
        base_y,
    )

    if input_node:
        _append_rewire(ops, input_node["id"], difficulty_id)
    _append_rewire(ops, difficulty_id, router_id)
    _append_rewire(ops, router_id, easy_dispatch_id)
    _append_rewire(ops, easy_dispatch_id, easy_lane_id)
    _append_rewire(ops, easy_lane_id, gather_id, target_port="a")
    _append_rewire(ops, router_id, hard_dispatch_id)

    if input_edge:
        _append_rewire(
            ops,
            hard_dispatch_id,
            str(input_edge.get("target")),
            remove_edge_id=str(input_edge.get("id") or ""),
            target_port=str(input_edge.get("target_port") or "x"),
        )
    else:
        hard_lane_id = f"aria_hard_lane_{uuid4().hex[:4]}"
        _append_add_node(
            ops, hard_lane_id, "mixing/softmax_attention", {}, base_x + 540, base_y + 90
        )
        _append_rewire(ops, hard_dispatch_id, hard_lane_id)
        _append_rewire(ops, hard_lane_id, gather_id, target_port="b")

    if output_edge:
        _append_rewire(
            ops,
            str(output_edge.get("source")),
            gather_id,
            remove_edge_id=str(output_edge.get("id") or ""),
            source_port=str(output_edge.get("source_port") or "y"),
            target_port="b",
        )

    if output_node:
        _append_rewire(ops, gather_id, output_node["id"])

    return {
        "rationale": (
            "Difficulty-routed refinement: score token difficulty, route easy tokens "
            "through a fast linear lane, keep the existing path as the hard lane, "
            "then gather before the output."
        ),
        "ops": ops,
    }


def _build_patch_from_pattern(
    pattern: Dict[str, Any],
    workflow_json: Optional[Dict[str, Any]],
    message: str,
) -> Dict[str, Any]:
    """Build a multi-node patch proposal from an architectural pattern."""
    nodes = (workflow_json or {}).get("nodes", [])
    edges = (workflow_json or {}).get("edges", [])
    ops: list[Dict[str, Any]] = []

    if pattern.get("name") == "difficulty-gated split":
        return _build_difficulty_routed_patch(workflow_json)

    # Find where to insert — after input or at end of existing chain
    input_node, output_node = _find_endpoints(nodes)

    # Base position for new nodes
    base_x = 200
    base_y = 0
    if input_node:
        position = input_node.get("ui_meta", {}).get("position", {})
        base_x = position.get("x", 200) + 200
        base_y = position.get("y", 0)

    prev_id = input_node["id"] if input_node else None
    created_ids: list[str] = []

    for i, (role, comp_type, params) in enumerate(pattern["nodes"]):
        node_id = f"aria_{role}_{uuid4().hex[:4]}"
        ops.append(
            {
                "op": "add_node",
                "node_id": node_id,
                "payload": {
                    "id": node_id,
                    "component_type": comp_type,
                    "params": params,
                    "ui_meta": {"x": base_x + (i * 180), "y": base_y},
                },
            }
        )
        # Wire sequentially (the UI handles split/merge topology on apply)
        if prev_id:
            ops.append(
                {
                    "op": "rewire",
                    "node_id": node_id,
                    "payload": {
                        "source": prev_id,
                        "target": node_id,
                    },
                }
            )
        prev_id = node_id
        created_ids.append(node_id)

    # Wire last created node to output if it exists
    if output_node and prev_id:
        # Remove existing edge to output first
        for edge in edges:
            if str(edge.get("target", "")) == output_node["id"]:
                ops.append(
                    {
                        "op": "rewire",
                        "node_id": output_node["id"],
                        "payload": {
                            "source": prev_id,
                            "target": output_node["id"],
                            "remove_edge_id": edge.get("id"),
                        },
                    }
                )
                break
        else:
            ops.append(
                {
                    "op": "rewire",
                    "node_id": output_node["id"],
                    "payload": {
                        "source": prev_id,
                        "target": output_node["id"],
                    },
                }
            )

    return {
        "rationale": f"{pattern['name']}: {pattern['description']}",
        "ops": ops,
    }


def _generate_response(
    intent: Dict[str, Any],
    message: str,
    context: Dict[str, Any],
    workflow_json: Optional[Dict[str, Any]],
    research_signals: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a response based on classified intent."""
    msg_type = intent.get("type", "general")
    components = intent.get("components_mentioned", [])

    if msg_type == "greeting" or context.get("turn_count", 0) == 0:
        return _respond_greeting(context)

    if msg_type == "question":
        return _respond_question(message, components, context)

    if msg_type == "architecture":
        return _respond_architecture(message, intent, context, workflow_json)

    if msg_type == "action" and components and workflow_json:
        return _respond_action(message, components, context, workflow_json)

    if msg_type == "goal":
        return _respond_goal(message, intent, context, workflow_json)

    # Fallback: try concept resolution before giving up
    resolved = intent.get("resolved_concepts", [])
    if resolved:
        return _respond_resolved_concepts(message, resolved, context, workflow_json)

    return _respond_general(message, context, workflow_json)


def _respond_greeting(context: Dict[str, Any]) -> Dict[str, Any]:
    """Respond to first message or greeting."""
    node_count = context.get("node_count", 0)
    if node_count > 0:
        comp_types = context.get("component_types", [])
        # Summarize what's on the canvas
        type_summary = ", ".join(
            t.split("/")[-1]
            for t in comp_types[:6]
            if "input" not in t and "output" not in t
        )
        content = f"I see your graph with {node_count} nodes"
        if type_summary:
            content += f" ({type_summary})"
        content += (
            ". I can help you:\n"
            "- Describe what you want in plain language — "
            'e.g. "add a difficulty gate that splits easy and hard tokens"\n'
            '- Modify the graph — "replace the relu with silu"\n'
            '- Build patterns — "add a transformer block after the input"'
        )
    else:
        content = (
            "Welcome! Describe what you want to build in plain language. "
            "For example:\n"
            '- "I want a transformer with multi-head attention"\n'
            '- "Build a model that splits tokens by difficulty — '
            'easy path gets a fast linear layer, hard path gets full attention"\n'
            '- "Create an SSM-based architecture with gating"\n\n'
            "I'll translate your description into a graph patch you can apply."
        )
    return {"content": content, "needs_clarification": False, "suggestions": []}


def _respond_architecture(
    message: str,
    intent: Dict[str, Any],
    context: Dict[str, Any],
    workflow_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Handle complex architectural descriptions with pattern matching."""
    pattern = intent.get("matched_pattern")

    if pattern:
        # Build the multi-node patch
        patch = _build_patch_from_pattern(pattern, workflow_json, message)
        node_list = pattern["nodes"]
        node_desc = " → ".join(f"**{comp}** ({role})" for role, comp, _ in node_list)

        content = (
            f"I'll build a **{pattern['name']}** pattern:\n\n"
            f"{node_desc}\n\n"
            f"{pattern['description']}\n\n"
            f"This adds {len(node_list)} nodes. Apply this patch?"
        )
        return {
            "content": content,
            "patch_proposal": patch,
            "needs_clarification": False,
            "suggestions": [],
        }

    # No exact pattern match — build from resolved concepts
    resolved = intent.get("resolved_concepts", [])
    if resolved:
        return _respond_resolved_concepts(message, resolved, context, workflow_json)

    return _respond_goal(message, intent, context, workflow_json)


def _respond_resolved_concepts(
    message: str,
    resolved: List[Dict[str, Any]],
    context: Dict[str, Any],
    workflow_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a patch from individually resolved concepts."""
    nodes = (workflow_json or {}).get("nodes", [])
    (workflow_json or {}).get("edges", [])
    ops: list[Dict[str, Any]] = []

    input_node = next(
        (n for n in nodes if "input" in str(n.get("component_type", "")).lower()),
        None,
    )
    output_node = next(
        (n for n in nodes if "output" in str(n.get("component_type", "")).lower()),
        None,
    )

    base_x = 200
    base_y = 0
    if input_node:
        ui = input_node.get("ui_meta", {})
        base_x = ui.get("x", 200) + 200
        base_y = ui.get("y", 0)

    prev_id = input_node["id"] if input_node else None
    descriptions: list[str] = []

    for i, r in enumerate(resolved):
        comp_type = r["component_type"]
        # Skip IO types that are already present
        if comp_type in ("graph_input", "graph_output"):
            continue
        node_id = f"aria_{uuid4().hex[:6]}"
        ops.append(
            {
                "op": "add_node",
                "node_id": node_id,
                "payload": {
                    "id": node_id,
                    "component_type": comp_type,
                    "params": {},
                    "ui_meta": {"x": base_x + (i * 180), "y": base_y},
                },
            }
        )
        if prev_id:
            ops.append(
                {
                    "op": "rewire",
                    "node_id": node_id,
                    "payload": {"source": prev_id, "target": node_id},
                }
            )
        prev_id = node_id
        descriptions.append(f'**{comp_type}** (from "{r["concept"]}")')

    if output_node and prev_id:
        ops.append(
            {
                "op": "rewire",
                "node_id": output_node["id"],
                "payload": {"source": prev_id, "target": output_node["id"]},
            }
        )

    if not ops:
        return _respond_general(message, context, workflow_json)

    content = (
        "Based on your description, I'll add:\n\n"
        + "\n".join(f"- {d}" for d in descriptions)
        + f"\n\n{len(descriptions)} node(s) total. Apply this patch?"
    )

    return {
        "content": content,
        "patch_proposal": {"rationale": message, "ops": ops},
        "needs_clarification": False,
        "suggestions": [],
    }


def _respond_question(
    message: str,
    components: List[str],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Answer questions about components or architecture."""
    if components:
        comp = components[0]
        cats = component_groups(comp)
        cat_str = ", ".join(cats) if cats else "general"
        content = f"**{comp}** belongs to the {cat_str} category. "
        # Add adjacency info
        from .help_content import get_component_tips

        tips = get_component_tips(comp)
        works_with = tips.get("works_well_with", [])[:5]
        if works_with:
            content += f"It works well with: {', '.join(works_with)}. "
        patterns = tips.get("patterns", [])
        if patterns:
            content += patterns[0]
    else:
        content = (
            "I can help with:\n"
            '- Component info — "what does linear_proj do?"\n'
            '- Compatibility — "what goes well with rmsnorm?"\n'
            '- Architecture patterns — "how do I build a transformer block?"\n'
            '- Building — "I want a model that splits easy and hard tokens"'
        )
        return {"content": content, "needs_clarification": True, "suggestions": []}

    return {"content": content, "needs_clarification": False, "suggestions": []}


def _respond_action(
    message: str,
    components: List[str],
    context: Dict[str, Any],
    workflow_json: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a patch proposal for an action request."""
    lower = message.lower()
    nodes = workflow_json.get("nodes", [])

    is_remove = any(w in lower for w in ("remove", "delete"))
    is_replace = any(w in lower for w in ("replace", "swap", "change"))

    ops: list[Dict[str, Any]] = []
    comp_id = components[0]

    if is_remove:
        target_node = next(
            (n for n in nodes if comp_id in str(n.get("component_type", "")).lower()),
            None,
        )
        if target_node:
            ops.append(
                {"op": "remove_node", "node_id": target_node["id"], "payload": {}}
            )
            content = f"I'll remove the **{comp_id}** node. Apply this change?"
        else:
            content = f"I couldn't find a **{comp_id}** node in your graph."
            return {"content": content, "needs_clarification": True, "suggestions": []}

    elif is_replace:
        target_node = next(
            (
                n
                for n in nodes
                if any(
                    c in str(n.get("component_type", "")).lower()
                    for c in components[:1]
                )
            ),
            None,
        )
        replacement = components[1] if len(components) > 1 else None
        if target_node and replacement:
            ops.append(
                {
                    "op": "replace_node",
                    "node_id": target_node["id"],
                    "payload": {"component_type": replacement},
                }
            )
            content = f"I'll replace **{target_node.get('component_type')}** with **{replacement}**. Apply?"
        else:
            content = "Which component should I replace it with?"
            return {"content": content, "needs_clarification": True, "suggestions": []}

    else:
        new_id = f"aria_{uuid4().hex[:6]}"
        ops.append(
            {
                "op": "add_node",
                "node_id": new_id,
                "payload": {"id": new_id, "component_type": comp_id, "params": {}},
            }
        )
        content = f"I'll add a **{comp_id}** node to your graph. Apply this change?"

    patch_proposal = {"rationale": message, "ops": ops} if ops else None

    return {
        "content": content,
        "patch_proposal": patch_proposal,
        "needs_clarification": False,
        "suggestions": [],
    }


def _respond_goal(
    message: str,
    intent: Dict[str, Any],
    context: Dict[str, Any],
    workflow_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Respond to goal-oriented requests — try pattern matching first."""
    # Try to match an architectural pattern from the goal description
    pattern = _match_pattern(message)
    if pattern and workflow_json:
        patch = _build_patch_from_pattern(pattern, workflow_json, message)
        node_list = pattern["nodes"]
        node_desc = " → ".join(f"**{comp}**" for _, comp, _ in node_list)

        content = (
            f"I'll build a **{pattern['name']}** for you:\n\n"
            f"{node_desc}\n\n"
            f"This adds {len(node_list)} nodes. Apply this patch?"
        )
        return {
            "content": content,
            "patch_proposal": patch,
            "needs_clarification": False,
            "suggestions": [],
        }

    # Try resolved concepts
    resolved = intent.get("resolved_concepts", [])
    if resolved and workflow_json:
        return _respond_resolved_concepts(message, resolved, context, workflow_json)

    intent_key = intent.get("intent_key", "balanced_refine")
    node_count = context.get("node_count", 0)

    if node_count == 0:
        content = (
            "Let's start building! Describe your architecture and I'll create a patch.\n\n"
            "Some ideas:\n"
            '- "Build a transformer with multi-head attention and residual connections"\n'
            '- "I want a model that routes easy tokens through a fast linear path '
            'and hard tokens through full attention"\n'
            '- "Create an SSM model with gating"\n'
            '- "Make a mixture-of-experts with 4 expert paths"'
        )
        return {"content": content, "needs_clarification": True, "suggestions": []}

    comp_types = context.get("component_types", [])
    content = f"Your graph has {node_count} nodes. "

    if intent_key == "improve_stability":
        has_norm = any("norm" in t.lower() for t in comp_types)
        if not has_norm:
            content += (
                "For stability, I recommend adding **rmsnorm** before mixing layers. "
                'Say "add normalization before the attention" and I\'ll create the patch.'
            )
        else:
            content += (
                "You already have normalization. Consider adding residual connections — "
                'say "add residual connections" and I\'ll wire them up.'
            )
    elif intent_key == "refine_compression":
        content += (
            "To compress, I can replace linear_proj with **bottleneck_proj** or "
            '**low_rank_proj**. Say "compress the projections" for a patch.'
        )
    elif intent_key == "expand_capacity":
        content += (
            "To expand capacity, describe what you want — e.g. "
            '"add another attention layer" or "split into parallel paths".'
        )
    else:
        content += (
            "Tell me specifically what you'd like — for example:\n"
            '- "Add a difficulty gate that splits into easy and hard paths"\n'
            '- "Put normalization before each mixing layer"\n'
            '- "Replace the linear with a bottleneck for compression"'
        )

    return {"content": content, "needs_clarification": True, "suggestions": []}


def _respond_general(
    message: str,
    context: Dict[str, Any],
    workflow_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fallback response — try harder to find something useful."""
    # Last resort: try concept resolution
    resolved = _resolve_concepts(message)
    if resolved and workflow_json:
        return _respond_resolved_concepts(message, resolved, context, workflow_json)

    node_count = context.get("node_count", 0)
    if node_count > 0:
        content = (
            "I can understand natural language descriptions of architectures. Try:\n"
            '- "I want a difficulty gate that splits tokens into easy and hard paths"\n'
            '- "Add attention after the normalization"\n'
            '- "Replace linear with a low-rank projection"\n'
            '- "Build a transformer block with residuals"\n'
            '- "What does softmax_attention do?"'
        )
    else:
        content = (
            "Describe the architecture you want to build. For example:\n"
            '- "Build a transformer with multi-head attention"\n'
            '- "I want a model with difficulty-based token routing"\n'
            '- "Create a hybrid SSM + attention model"\n\n'
            "I'll translate your description into graph operations."
        )
    return {"content": content, "needs_clarification": True, "suggestions": []}
