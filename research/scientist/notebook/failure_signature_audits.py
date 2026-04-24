from __future__ import annotations

"""Audited failure-signature overrides.

These signatures were manually reviewed against current context rules,
template wiring, slot observability, and failure provenance. Their historical
failure-only counts were caused by upstream generator/runtime issues rather than
the pair itself, so they must not be auto-deweighted as toxic op pairs.
"""

AUDITED_FALSE_FAILURE_SIGNATURES: dict[str, str] = {
    "adjacent_token_merge->add": (
        "Historical failures were dominated by routing_dead_path/causality "
        "violations and over-depth graphs; add was a residual sink, not the cause."
    ),
    "block_sparse_linear->calibrated_branch_merge": (
        "Failures were dominated by over-budget hybrid routing graphs, not this adjacency."
    ),
    "block_sparse_linear->linear_proj": (
        "Sparse projection followed by linear restoration is valid; historical failures "
        "came from oversized routing assemblies."
    ),
    "conv1d_seq->rmsnorm": (
        "The audited failures were downstream selective_scan/template regressions, "
        "not the conv-to-norm adjacency."
    ),
    "depth_weighted_proj->add": (
        "Residual add was a terminal sink; failures were dominated by no-learning "
        "patterns in downstream motifs."
    ),
    "feature_sparsity->swiglu_mlp": (
        "Current feature_sparse templates intentionally use this pair; historical "
        "failures came from unrelated shape/template issues."
    ),
    "fused_linear_gelu->add": (
        "Residual add was not the failing component; failures were dominated by "
        "downstream no-learning behavior."
    ),
    "grouped_linear->neg": (
        "Mixed historical failures were driven by unrelated validation/runtime issues, "
        "not a stable grouped_linear->neg contract break."
    ),
    "hetero_moe->linear_proj": (
        "Historical rows came from outdated motif/template assembly; current routing "
        "rules already prevent the bad direct adjacency."
    ),
    "hyp_distance->linear_proj": (
        "The current dedicated template restores with linear_proj_up; the old "
        "hyp_distance->linear_proj failures were stale template artifacts."
    ),
    "kronecker_linear->relu": (
        "Historical failures were diffuse no-learning/unknown failures, not a specific "
        "kronecker_linear->relu implementation break."
    ),
    "layernorm->rope_rotate": "Standard normalization-before-RoPE pattern.",
    "layernorm->signal_conditioned_compression": (
        "Standard normalization-before-compression pattern; failures were dominated by "
        "routing telemetry issues upstream."
    ),
    "layernorm->transpose_sd": (
        "Cross-dim templates intentionally use norm before transpose; failures were "
        "from unrelated downstream scan/no-learning issues."
    ),
    "lif_neuron->stdp_attention": (
        "The spiking stack is valid; direct failures were historical attribution noise, "
        "not a broken lif_neuron/stdp_attention contract."
    ),
    "linear_proj_down->mul": (
        "Historical failures were dominated by downstream runtime issues; the pair itself "
        "is not a proven toxic adjacency."
    ),
    "linear_proj_up->moe_topk": (
        "Current context rules already forbid the stale bad adjacency; penalizing it again "
        "as a toxic pair misattributes a generator-rule issue to the pair."
    ),
    "linear_proj_up->sigmoid": (
        "Historical failures were dominated by downstream runtime/unknown errors, not a "
        "stable linear_proj_up->sigmoid break."
    ),
    "moe_2expert->rmsnorm": (
        "Post-MoE normalization is a valid pattern; the failures were not rooted in this pair."
    ),
    "rmsnorm->calibrated_branch_merge": (
        "Failures were dominated by oversized routing assemblies; the norm-to-merge edge "
        "was not the root cause."
    ),
    "gated_lane_blend->linear_proj": (
        "Historical failures were contaminated by alias/canonicalization split with "
        "route_lanes plus name-sensitive wiring behavior; this is not stable pair evidence."
    ),
    "layernorm->gated_linear_attention": (
        "Failures are dominated by gated_linear_attention's near-global weakness; the pair "
        "does not isolate a norm-specific adjacency failure."
    ),
    "layernorm->mixture_of_recursions": (
        "Failures are receiver-dominated because mixture_of_recursions currently has near-zero "
        "global S1 success; this is not pair-specific evidence."
    ),
    "linear_proj->calibrated_branch_merge": (
        "Observed failures cluster in weak multiscale routing templates; the pair is being "
        "penalized for template-family behavior rather than adjacency alone."
    ),
    "linear_proj->gated_lane_blend": (
        "Historical failures were contaminated by alias/canonicalization split with "
        "route_lanes plus name-sensitive wiring behavior; this is not stable pair evidence."
    ),
    "mixture_of_recursions->linear_proj": (
        "Failures are source-dominated because mixture_of_recursions itself is globally failing; "
        "the pair does not add pair-specific explanatory power."
    ),
    "rmsnorm->gated_linear_attention": (
        "Failures are dominated by gated_linear_attention's near-global weakness; the pair "
        "does not isolate a norm-specific adjacency failure."
    ),
    "route_recursion->linear_proj": (
        "Historical failures were contaminated by deprecated alias storage for depth_gated_transform "
        "and name-sensitive wiring behavior, not a stable pair break."
    ),
    "rwkv_channel->rmsnorm": (
        "Current productive RWKV template families use this neighborhood successfully; the "
        "historical penalty reflects placement/template effects, not a toxic adjacency."
    ),
    "semi_structured_2_4_linear->calibrated_branch_merge": (
        "Observed failures cluster in weak multiscale routing templates; the pair is being "
        "penalized for template-family behavior rather than adjacency alone."
    ),
    "semi_structured_2_4_linear->linear_proj": (
        "Observed failures cluster in weak multiscale routing templates; this is not strong "
        "evidence that sparse linear restoration is intrinsically toxic."
    ),
    "signal_conditioned_compression->add": (
        "Historical failures were contaminated by routing_conditioned_compression alias usage "
        "and routing-template assembly effects; add was often just the residual sink."
    ),
    "split2->rmsnorm": (
        "Failures were dominated by split-slot placement misuse and downstream full-width contract "
        "breaks, not by the normalization edge itself."
    ),
}

AUDITED_FALSE_FAILURE_SIGNATURE_SET = frozenset(AUDITED_FALSE_FAILURE_SIGNATURES)
