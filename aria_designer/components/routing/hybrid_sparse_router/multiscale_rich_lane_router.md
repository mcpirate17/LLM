# Multiscale Rich Lane Router

## Summary

`multiscale_rich_lane_router` is a three-tier intelligent routing template built on top of the `hybrid_sparse_router` / Lane Router component. It is centered around routing tokens into:

- a cheap `default_path` for low-risk traffic,
- a richer medium path that evaluates pair, triplet, and quartet sparse groupings,
- a hard path that uses a difficulty signal plus compression or expert-style compute for the most difficult tokens.

At the manifest level, it is described as a "richer three-tier router with multi-span medium routing and broad medium/hard lane component menus."

## High-Level Flow

1. Optionally normalize the input.
2. Create a `default_path` fallback branch for cheap handling.
3. Apply `hybrid_token_gate` to separate routed traffic from fallback-oriented traffic.
4. Build three sparse span views with widths 2, 3, and 4.
5. Run three nested `hybrid_sparse_router` branches for pair, triplet, and quartet routing.
6. Merge those three routed branches into a medium lane.
7. Normalize the medium lane, choose one medium-lane component from the approved pool, then project back to model width.
8. Build a hard-token signal with `token_class_proj`.
9. Seed the hard lane with `signal_conditioned_compression`.
10. Choose one hard-lane component from the approved heavy-compute pool, then project back to model width.
11. Merge the default, medium, hard, gated skip, and original input through residual additions.

## Core Components

- `default_path`
  Cheap default or early-exit residual branch for tokens that do not justify richer routing.

- `hybrid_token_gate`
  Cheap token-level gate that separates default-path traffic from sparse-routed traffic.

- `sparse_span_builder`
  Builds sparse fused span features. This template instantiates it three times for widths 2, 3, and 4.

- `hybrid_sparse_router`
  Nested lane router used for pair, triplet, and quartet routing branches.

- `layernorm`
  Normalizes the merged medium branch before medium-lane compute.

- `linear_proj`
  Projects medium and hard outputs back to `model_dim` before merge.

- `token_class_proj`
  Learned classifier that produces routing or difficulty scores for hard-token separation.

- `signal_conditioned_compression`
  Compresses features conditioned on the routing signal before hard-lane compute.

## Template Slots

These are the tracked template-level slots recorded by synthesis for `multiscale_rich_lane_router`:

- `default_path`
  Bound to `default_path`. This is the cheap fallback branch.

- `pair_spans`
  Bound to `sparse_span_builder` with `span_width=2`. Builds pair-token sparse spans.

- `triplet_spans`
  Bound to `sparse_span_builder` with `span_width=3`. Builds triplet-token sparse spans.

- `quartet_spans`
  Bound to `sparse_span_builder` with `span_width=4`. Builds quartet-token sparse spans.

- `medium_router`
  Bound to one medium-lane component selected from the approved medium pool after the pair, triplet, and quartet router outputs are merged.

- `hard_router`
  Bound to one hard-lane component selected from the approved hard pool after difficulty classification and compression setup.

At the manifest level, the template advertises these slot bindings:

- `default_path`
- `medium_router`
- `sparse_spans`
- `difficulty_signal`
- `compression_router`
- `hard_router`

## Parent Lane Router Slots

The parent `hybrid_sparse_router` component exposes the following slots:

- `pre_router`
  Lightweight stem before routing to shape token features.

- `default_path`
  Cheap residual path merged when routed confidence is low.

- `easy_router`
  Cheap real compute lane for easy tokens before medium and hard routing take over.

- `medium_router`
  Sparse pair, triplet, or quartet router used for medium-difficulty token groups.

- `routed_lane`
  Lane-conditioned compute branch selected by sparse span routing.

- `sparse_spans`
  Packed sparse pair or triplet features produced by the span builder.

- `difficulty_signal`
  Difficulty features that separate medium and hard traffic.

- `compression_router`
  Difficulty-conditioned compression before the hard expert branch.

- `hard_router`
  Heavy expert path for the hardest tokens.

- `token_merge`
  Merge stage after easy, medium, and hard lanes recombine.

- `post_merge`
  Stabilizing post-merge compute before the output residual.

## Mandatory Slots And Their Meaning

For the underlying `hybrid_sparse_router` component, these slots are mandatory:

- `default_path`
  Required because the router always needs a cheap confidence fallback path.

- `routed_lane`
  Required because sparse routing must dispatch into a lane-conditioned compute branch.

- `sparse_spans`
  Required because the router depends on packed sparse span features for grouped routing.

For `multiscale_rich_lane_router` specifically:

- `default_path` is explicitly instantiated at the top level.
- `sparse_spans` is explicitly instantiated three times as `pair_spans`, `triplet_spans`, and `quartet_spans`.
- `routed_lane` is satisfied inside each nested `hybrid_sparse_router` branch rather than exposed as a top-level template slot.

## Medium Router Candidate Components

The medium branch can choose one of these approved components:

- `route_lanes`
  Multi-lane dispatch.

- `adaptive_lane_mixer`
  Learned difficulty-based lane mixing. In the designer manifests this corresponds to `difficulty_blend_3way`.

- `semi_structured_2_4_linear`
  Semi-structured sparse linear projection.

- `block_sparse_linear`
  Block-sparse linear projection.

- `rwkv_time_mixing`
  RWKV-style time mixing.

- `nm_sparse_linear`
  N:M structured sparse linear projection.

- `default_path`
  Cheap fallback-style processing.

- `cheap_verify_blend`
  Speculative cheap path plus verification path.

- `conv1d_seq`
  Depthwise sequence convolution.

- `conv_only`
  Pure convolutional token-mixing stack.

## Hard Router Candidate Components

The hard branch can choose one of these approved components:

- `compression_mixture_experts`
  Routing into compression-specific experts. In the designer manifests this corresponds to `dual_compression_blend`.

- `routing_conditioned_compression`
  Compression conditioned directly by routing signal. In the designer manifests this corresponds to `signal_conditioned_compression`.

- `dual_compression_blend`
  Method-specific compression experts.

- `route_recursion`
  Adaptive recursion depth control.

- `adaptive_recursion`
  Per-token adaptive recursion depth. In the designer manifests this corresponds to `depth_weighted_proj`.

- `mixed_recursion_gate`
  Recursion-step transforms chosen conditionally by difficulty. In the designer manifests this corresponds to `score_depth_blend`.

- `moe_topk`
  Top-k mixture-of-experts.

- `moe_2expert`
  Lightweight two-expert MoE.

- `n_way_sparse_router`
  N-way sparse bottleneck routing. In the designer manifests this corresponds to `sparse_bottleneck_moe`.

- `state_space`
  Mamba-style state-space sequence mixer.

## Template Description

`multiscale_rich_lane_router` is a hierarchical intelligent routing template that first gates tokens into fallback versus routed traffic, then evaluates medium-difficulty tokens across pair, triplet, and quartet sparse routing branches, and finally sends hard tokens through a difficulty-conditioned compression plus expert or recursion-style compute path. Its defining property is that both the medium and hard lanes are selected from broader approved component menus, making it more expressive than the simpler multiscale router variants.

## Source References

- Template assembly: `research/synthesis/_templates_routing.py`
- Template manifest bindings: `aria_designer/components/routing/hybrid_sparse_router/manifest.yaml`
- Alias mapping for renamed ops: `research/synthesis/primitives.py`
