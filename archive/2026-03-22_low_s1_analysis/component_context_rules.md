# Component Context Rules

## graph_attention
- Preferred predecessors: rmsnorm, layernorm, split2
- Forbidden predecessors: collapsed dims
- Preferred successors: linear_proj, add
- Forbidden successors: output_head directly
- Required graph motifs: residual_block, attention_variants
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe after compiler param cast fix
- Residual/branch requirements: use inside residual attention block
- Causal constraints: causal-only
- Search mode: general search

## local_window_attn
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: collapsed dims, unnormalized deep residuals
- Preferred successors: linear_proj, add
- Forbidden successors: output_head directly
- Required graph motifs: residual_block, attention_variants
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: run attention math in fp32 under bf16 autocast
- Residual/branch requirements: must sit inside a residual attention block
- Causal constraints: causal-only local window
- Search mode: restricted-use

## state_space
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## sliding_window_mask
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: non-standard rank, collapsed dims
- Preferred successors: linear_attention, linear_proj
- Forbidden successors: output_head directly
- Required graph motifs: attn_sliding_window
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe after compiler param cast fix
- Residual/branch requirements: use inside residual attention blocks
- Causal constraints: causal-only local window
- Search mode: restricted-use

## causal_mask
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: already non-causal or seq-reordered tensors
- Preferred successors: softmax_attention, linear_proj
- Forbidden successors: output_head directly
- Required graph motifs: attn_causal_mask
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe after compiler param cast fix
- Residual/branch requirements: use inside attention/residual stacks
- Causal constraints: causal-only
- Search mode: restricted-use

## tropical_center
- Preferred predecessors: tropical_attention, tropical_gate
- Forbidden predecessors: plain euclidean ops without tropical bridge
- Preferred successors: tropical_gate, add, linear_proj
- Forbidden successors: output_head directly
- Required graph motifs: tropical_core
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: fp32 preferred for stability probes
- Residual/branch requirements: keep inside tropical residual block
- Causal constraints: non-causal / structural-only in current formulation
- Search mode: niche mode

## early_exit
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: niche mode

## gated_delta
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## diff_attention
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## fused_linear_gelu
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## integral_kernel
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## split3
- Preferred predecessors: linear_proj to 3-divisible dim, rmsnorm
- Forbidden predecessors: non-3-divisible dims, collapsed_dim tensors
- Preferred successors: three_way_split branches, concat
- Forbidden successors: output_head directly
- Required graph motifs: three_way_split
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: none
- Residual/branch requirements: must rejoin via concat + projection or residual add
- Causal constraints: feature split only; do not interpret as seq reordering
- Search mode: restricted-use

## exp
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## cumprod_safe
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## lif_neuron
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: niche mode

## sparse_threshold
- Preferred predecessors: lif_neuron, spike_rate_code, rmsnorm
- Forbidden predecessors: plain dense euclidean activations
- Preferred successors: stdp_attention, linear_proj, add
- Forbidden successors: output_head directly
- Required graph motifs: spiking_neuromorphic
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: fp32 preferred around thresholding
- Residual/branch requirements: use inside spiking residual block
- Causal constraints: causal-only spike timing
- Search mode: niche mode

## stdp_attention
- Preferred predecessors: lif_neuron, spike_rate_code, sparse_threshold
- Forbidden predecessors: plain dense euclidean activations
- Preferred successors: linear_proj, add
- Forbidden successors: output_head directly
- Required graph motifs: spiking_neuromorphic
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: fp32 preferred around exponential decay
- Residual/branch requirements: use inside spiking residual block
- Causal constraints: causal-only spike timing
- Search mode: niche mode

## cumsum
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## sub
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## chebyshev_spectral_mix
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## minimum
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## embedding_lookup
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## geometric_product
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: niche mode

## n_way_sparse_router
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: niche mode

## sign_ste
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## tropical_matmul
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: niche mode

## norm_last
- Preferred predecessors: rmsnorm, linear_proj_down
- Forbidden predecessors: already-collapsed dims
- Preferred successors: linear_proj_up, mul
- Forbidden successors: output_head directly
- Required graph motifs: reduce_norm
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: none
- Residual/branch requirements: must rebroadcast or reproject before residual merge
- Causal constraints: feature reduction only
- Search mode: restricted-use

## sqrt
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## log
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## mod_topk
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## sum_last
- Preferred predecessors: rmsnorm, linear_proj_down
- Forbidden predecessors: already-collapsed dims
- Preferred successors: linear_proj_up, mul
- Forbidden successors: output_head directly
- Required graph motifs: reduce_sum
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: fp32 preferred only for extreme values
- Residual/branch requirements: must rebroadcast or reproject before residual merge
- Causal constraints: feature reduction only
- Search mode: restricted-use

## max_last
- Preferred predecessors: rmsnorm, linear_proj_down
- Forbidden predecessors: already-collapsed dims
- Preferred successors: linear_proj_up, mul
- Forbidden successors: output_head directly
- Required graph motifs: reduce_max
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: none
- Residual/branch requirements: must rebroadcast or reproject before residual merge
- Causal constraints: feature reduction only
- Search mode: restricted-use

## mean_last
- Preferred predecessors: rmsnorm, linear_proj_down
- Forbidden predecessors: already-collapsed dims
- Preferred successors: linear_proj_up, mul
- Forbidden successors: output_head directly
- Required graph motifs: reduce_mean
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: none
- Residual/branch requirements: must rebroadcast or reproject before residual merge
- Causal constraints: feature reduction only
- Search mode: restricted-use

## reciprocal
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## linear_attention
- Preferred predecessors: sliding_window_mask, rmsnorm, layernorm
- Forbidden predecessors: non-standard rank, collapsed dims
- Preferred successors: linear_proj, add
- Forbidden successors: output_head directly
- Required graph motifs: attn_sliding_window, residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe after compiler param cast fix
- Residual/branch requirements: use inside residual attention block
- Causal constraints: causal-only
- Search mode: general search

## identity
- Preferred predecessors: route scaffolding only
- Forbidden predecessors: general learned blocks
- Preferred successors: selective_scan, conv1d_seq, rwkv_channel
- Forbidden successors: standalone output_head
- Required graph motifs: routing scaffolds only
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: none
- Residual/branch requirements: structural only; do not judge by S1 alone
- Causal constraints: feature-preserving only
- Search mode: restricted-use

## progressive_compression_gate
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: niche mode

## softmax_attention
- Preferred predecessors: rope_rotate, causal_mask, rmsnorm, layernorm
- Forbidden predecessors: odd-dim rope misuse
- Preferred successors: linear_proj, add
- Forbidden successors: output_head directly
- Required graph motifs: attn_rope, attn_causal_mask, transformer_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe after compiler param cast fix
- Residual/branch requirements: use inside residual attention block
- Causal constraints: causal-only
- Search mode: general search

## linear_proj_down
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## linear_proj_up
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## concat
- Preferred predecessors: parallel branch outputs of matching seq len
- Forbidden predecessors: mismatched seq or rank
- Preferred successors: linear_proj, add
- Forbidden successors: output_head directly
- Required graph motifs: parallel_split, three_way_split
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: none
- Residual/branch requirements: should be followed by projection back to model dim
- Causal constraints: preserve per-token alignment
- Search mode: restricted-use

## hyp_linear
- Preferred predecessors: exp_map
- Forbidden predecessors: plain euclidean ops without exp_map
- Preferred successors: hyp_tangent_nonlinear
- Forbidden successors: output_head directly, plain euclidean mixers
- Required graph motifs: hyperbolic_poincare
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: fp32 preferred in hyperbolic region if norms spike
- Residual/branch requirements: return through log_map before residual merge
- Causal constraints: causal-neutral
- Search mode: niche mode

## hyp_tangent_nonlinear
- Preferred predecessors: hyp_linear
- Forbidden predecessors: plain euclidean ops
- Preferred successors: log_map
- Forbidden successors: output_head directly
- Required graph motifs: hyperbolic_poincare
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: fp32 preferred in hyperbolic region if norms spike
- Residual/branch requirements: return through log_map before residual merge
- Causal constraints: causal-neutral
- Search mode: niche mode

## rwkv_time_mixing
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## rope_rotate
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## split2
- Preferred predecessors: input, rmsnorm, rope_rotate
- Forbidden predecessors: collapsed_dim tensors, reduce_last outputs
- Preferred successors: parallel branches, concat
- Forbidden successors: output_head directly
- Required graph motifs: parallel_split
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: none
- Residual/branch requirements: must rejoin via concat + projection or residual add
- Causal constraints: feature split only; do not interpret as seq reordering
- Search mode: restricted-use

## fixed_point_iter
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## transpose_sd
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search

## grouped_linear
- Preferred predecessors: rmsnorm, layernorm
- Forbidden predecessors: sum_last, mean_last, max_last, norm_last
- Preferred successors: linear_proj, add
- Forbidden successors: split2, split3
- Required graph motifs: residual_block
- Forbidden graph motifs: standalone_single_op
- Dtype constraints: bf16-safe unless noted
- Residual/branch requirements: should sit inside a residual block
- Causal constraints: must preserve causal token ordering
- Search mode: general search
