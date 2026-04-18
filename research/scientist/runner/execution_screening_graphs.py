"""Hot-loop graph screening helpers.

These helpers keep per-graph structural analysis in one focused place so the
screening loop can reuse a single pass over graph nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Tuple

from ...eval._eval_native import load_eval_native


@dataclass(frozen=True, slots=True)
class ScreeningGraphAnalysis:
    """Cached graph facts used repeatedly during screening."""

    op_names: FrozenSet[str]
    counted_ops: Tuple[str, ...]
    toxic_bigrams: Tuple[str, ...]
    has_parameterized_op: bool

    @property
    def has_content_addressed_op(self) -> bool:
        """Graph contains an attention-class op capable of content-based retrieval."""
        return bool(self.op_names & CONTENT_ADDRESSED_OPS)

    @property
    def has_sequence_mixing(self) -> bool:
        """Graph contains any op that mixes information across positions."""
        return bool(self.op_names & SEQUENCE_MIXING_OPS)


def analyze_graph_for_screening(
    graph: Any,
    get_primitive: Callable[[str], Any] | None,
) -> ScreeningGraphAnalysis:
    """Collect reusable graph facts in one pass.

    The screening loop checks parameterized ops, routing ops, toxic bigrams,
    and stage-0 op accounting for nearly every candidate. Doing those in a
    single scan avoids repeated dictionary iteration on large batches.
    """

    nodes = graph.nodes
    ordered_nodes = list(nodes.items())
    has_params_flags = []

    if get_primitive is not None:
        primitive_cache: Dict[str, bool] = {}
        for _node_id, node in ordered_nodes:
            op_name = node.op_name
            if not op_name or getattr(node, "is_input", False):
                has_params_flags.append(False)
                continue
            if op_name not in primitive_cache:
                try:
                    primitive = get_primitive(op_name)
                except (KeyError, ValueError):
                    primitive = None
                primitive_cache[op_name] = bool(
                    primitive is not None and getattr(primitive, "has_params", False)
                )
            has_params_flags.append(primitive_cache[op_name])
    else:
        has_params_flags = [False] * len(ordered_nodes)

    try:
        native = load_eval_native()
        analysis = native.screening_graph_analysis_native(
            [int(node_id) for node_id, _ in ordered_nodes],
            [str(node.op_name or "") for _, node in ordered_nodes],
            [
                [int(parent_id) for parent_id in node.input_ids]
                for _, node in ordered_nodes
            ],
            [bool(getattr(node, "is_input", False)) for _, node in ordered_nodes],
            [bool(getattr(node, "is_output", False)) for _, node in ordered_nodes],
            [bool(flag) for flag in has_params_flags],
        )
        return ScreeningGraphAnalysis(
            op_names=frozenset(analysis["op_names"]),
            counted_ops=tuple(analysis["counted_ops"]),
            toxic_bigrams=tuple(analysis["toxic_bigrams"]),
            has_parameterized_op=bool(analysis["has_parameterized_op"]),
        )
    except Exception:
        pass

    op_names = set()
    counted_ops = []
    toxic_bigrams = set()
    has_parameterized_op = False

    for idx, (_node_id, node) in enumerate(ordered_nodes):
        if node.is_input:
            continue

        op_name = node.op_name
        if op_name:
            counted_ops.append(op_name)

        if getattr(node, "is_output", False):
            continue

        op_names.add(op_name)
        if not has_parameterized_op and has_params_flags[idx]:
            has_parameterized_op = True

        for parent_id in node.input_ids:
            parent = nodes.get(parent_id)
            if (
                parent is not None
                and not parent.is_input
                and not getattr(parent, "is_output", False)
            ):
                toxic_bigrams.add(f"{parent.op_name}->{op_name}")

    return ScreeningGraphAnalysis(
        op_names=frozenset(op_names),
        counted_ops=tuple(counted_ops),
        toxic_bigrams=tuple(sorted(toxic_bigrams)),
        has_parameterized_op=has_parameterized_op,
    )


# ── Sequence mixing capability tiers ─────────────────────────────────
#
# Attention computes content-based all-to-all similarity (Q·K^T) and
# routes information accordingly (softmax · V). This is the ONLY
# mechanism that enables:
#   - Induction heads (copy patterns from earlier in context)
#   - Binding (associate and retrieve distant tokens by content)
#   - Long-range dependency (subject-verb agreement across 100+ tokens)
#
# SSM/recurrent ops provide long-range mixing through state accumulation
# but with exponential decay — they can approximate but not sharply
# retrieve. Conv provides only local mixing (window 3-5 tokens).
#
# We use two tiers:
#   CONTENT_ADDRESSED_OPS: Can learn content-based retrieval (attention
#       family). Required for investigation/validation promotion.
#   SEQUENCE_MIXING_OPS: Any cross-position information flow (attention +
#       SSM + conv + accumulation). Required to pass screening.

CONTENT_ADDRESSED_OPS: FrozenSet[str] = frozenset(
    {
        "softmax_attention",
        "latent_attention_compressor",
        "graph_attention",
        "local_window_attn",
        "linear_attention",
        "diff_attention",
        "tropical_attention",
        "ultrametric_attention",
        "stdp_attention",
        "clifford_attention",
        # Bilinear / retrieval-family ops that, when wired into a query-key-value
        # style path, enable exact content-addressed retrieval. Not every use of
        # these ops is retrieval-capable; gate8 treats their presence as a
        # necessary (not sufficient) condition and the deeper binding probe does
        # the final check.
        "matmul",
        "outer_product",
        "gather_topk",
        "cosine_similarity",
        # New attention-class ops (2026-04-15)
        "difficulty_routed_attention",
        "strided_attention",
        "gated_progressive_attention",
        "gated_linear_attention",
        "associative_memory",
    }
)

SEQUENCE_MIXING_OPS: FrozenSet[str] = CONTENT_ADDRESSED_OPS | frozenset(
    {
        # SSM / recurrent (long-range but lossy)
        "state_space",
        "selective_scan",
        "rwkv_channel",
        "rwkv_time_mixing",
        # Convolution (local mixing only)
        "conv1d_seq",
        # Accumulation (proto-attention)
        "cumsum",
        "cumprod_safe",
        # Token interaction (local)
        "token_merge",
        "adjacent_token_merge",
        "sliding_window_mask",
        "causal_mix",
        # New mixing ops (2026-04-15) — long_conv_hyena and mixture_of_recursions
        # are SSM-class (long-range mixing without content addressing)
        "long_conv_hyena",
        "mixture_of_recursions",
    }
)


def structural_gate_failure(
    graph: Any,
    *,
    routing_mandatory: bool,
    efficiency_ops: FrozenSet[str],
    analysis: ScreeningGraphAnalysis,
    binding_capable_required: bool = False,
) -> str | None:
    """Return the first failing structural gate code, or ``None``."""

    if graph.n_ops() <= 7:
        return "gate1_min_ops"
    if not graph.has_gradient_path():
        return "gate2_no_grad"
    if not graph.has_residual_path():
        return "gate3_no_residual"
    if not analysis.has_parameterized_op:
        return "gate4_no_params"
    if routing_mandatory and not (analysis.op_names & efficiency_ops):
        return "gate5_no_routing"
    # Gate 6: must contain at least one sequence-mixing op (attention, SSM,
    # conv, or equivalent). Graphs without mixing cannot learn token
    # relationships — they achieve low loss by memorization but score zero
    # on induction, binding, and associative recall probes.
    if not (analysis.op_names & SEQUENCE_MIXING_OPS):
        return "gate6_no_mixing"
    # Gate 7: need minimum op diversity — a graph of 8 linear_proj ops
    # has no functional variety. Require at least 4 distinct op types
    # (e.g., norm + attention + FFN + residual).
    if len(analysis.op_names) < 4:
        return "gate7_low_diversity"
    # Gate 8: when the grammar preset declares binding-capable required,
    # reject graphs that contain no content-addressed retrieval op. SSM and
    # conv can mix sequences but cannot bind — they pass gate 6 yet will
    # score zero on binding/induction probes and waste investigation compute.
    # Keeping this gate opt-in preserves backward compatibility for presets
    # that deliberately explore retrieval-free trunks (e.g. exploration).
    if binding_capable_required and not (analysis.op_names & CONTENT_ADDRESSED_OPS):
        return "gate8_retrieval_dead"
    return None


def toxic_failure_ratio(
    failure_blocklist: Dict[str, float],
    analysis: ScreeningGraphAnalysis,
) -> float:
    """Compute toxic bigram ratio from cached graph analysis."""

    if not failure_blocklist or not analysis.toxic_bigrams:
        return 0.0
    toxic_weight = sum(
        1.0 - failure_blocklist[bigram]
        for bigram in analysis.toxic_bigrams
        if bigram in failure_blocklist
    )
    return toxic_weight / len(analysis.toxic_bigrams)
