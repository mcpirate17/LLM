# Before / After

## graph_attention
- Root cause: B. stale evidence from fixed bug, M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: yes
- New result: works_now
- Recommendation: rerun-needed

## local_window_attn
- Root cause: B. stale evidence from fixed bug
- Code fix applied: yes
- Rule fix applied: no
- Rerun performed: yes
- New result: fixed_now
- Recommendation: restricted-use

## state_space
- Root cause: C. vocab/gate mismatch artifact, H. numerical instability, M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: yes
- New result: works_now
- Recommendation: rerun-needed

## sliding_window_mask
- Root cause: B. stale evidence from fixed bug, K. structural/non-learnable op, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: yes
- New result: structural
- Recommendation: structural-only

## causal_mask
- Root cause: B. stale evidence from fixed bug, K. structural/non-learnable op, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: yes
- New result: structural
- Recommendation: structural-only

## tropical_center
- Root cause: C. vocab/gate mismatch artifact, D. graph-context misuse, H. numerical instability, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: yes
- New result: works_now
- Recommendation: restricted-use

## early_exit
- Root cause: D. graph-context misuse, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: no
- New result: works_now
- Recommendation: restricted-use

## gated_delta
- Root cause: M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: genuinely weak

## diff_attention
- Root cause: M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: genuinely weak

## fused_linear_gelu
- Root cause: M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: genuinely weak

## integral_kernel
- Root cause: M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: genuinely weak

## split3
- Root cause: K. structural/non-learnable op, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: no
- New result: structural
- Recommendation: structural-only

## exp
- Root cause: H. numerical instability, M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: genuinely weak

## cumprod_safe
- Root cause: unclassified
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: restricted-use

## lif_neuron
- Root cause: D. graph-context misuse, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: no
- New result: works_now
- Recommendation: restricted-use

## sparse_threshold
- Root cause: D. graph-context misuse, G. shape/contract mismatch, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: no
- New result: works_now
- Recommendation: restricted-use

## stdp_attention
- Root cause: D. graph-context misuse, G. shape/contract mismatch, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: no
- New result: works_now
- Recommendation: restricted-use

## cumsum
- Root cause: M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: genuinely weak

## sub
- Root cause: M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: genuinely weak

## chebyshev_spectral_mix
- Root cause: M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: genuinely weak

## minimum
- Root cause: M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: genuinely weak

## embedding_lookup
- Root cause: unclassified
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: restricted-use

## geometric_product
- Root cause: D. graph-context misuse, G. shape/contract mismatch, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: no
- New result: works_now
- Recommendation: restricted-use

## n_way_sparse_router
- Root cause: D. graph-context misuse, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: no
- New result: works_now
- Recommendation: restricted-use

## sign_ste
- Root cause: H. numerical instability, M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: genuinely weak

## tropical_matmul
- Root cause: D. graph-context misuse, G. shape/contract mismatch, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: no
- New result: works_now
- Recommendation: restricted-use

## norm_last
- Root cause: K. structural/non-learnable op, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: no
- New result: structural
- Recommendation: structural-only

## sqrt
- Root cause: H. numerical instability, M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: genuinely weak

## log
- Root cause: H. numerical instability, M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: genuinely weak

## mod_topk
- Root cause: unclassified
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: restricted-use

## sum_last
- Root cause: K. structural/non-learnable op, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: no
- New result: structural
- Recommendation: structural-only

## max_last
- Root cause: K. structural/non-learnable op, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: no
- New result: structural
- Recommendation: structural-only

## mean_last
- Root cause: K. structural/non-learnable op, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: no
- New result: structural
- Recommendation: structural-only

## reciprocal
- Root cause: unclassified
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: restricted-use

## linear_attention
- Root cause: B. stale evidence from fixed bug, M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: yes
- New result: works_now
- Recommendation: rerun-needed

## identity
- Root cause: C. vocab/gate mismatch artifact, K. structural/non-learnable op, L. niche op needing restricted placement, N. telemetry/reporting ambiguity
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: yes
- New result: structural
- Recommendation: structural-only

## progressive_compression_gate
- Root cause: B. stale evidence from fixed bug, D. graph-context misuse, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: yes
- New result: works_now
- Recommendation: restricted-use

## softmax_attention
- Root cause: B. stale evidence from fixed bug, M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: yes
- New result: works_now
- Recommendation: rerun-needed

## linear_proj_down
- Root cause: B. stale evidence from fixed bug, C. vocab/gate mismatch artifact
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: yes
- New result: works_now
- Recommendation: rerun-needed

## linear_proj_up
- Root cause: unclassified
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: restricted-use

## concat
- Root cause: B. stale evidence from fixed bug, C. vocab/gate mismatch artifact, K. structural/non-learnable op, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: yes
- New result: structural
- Recommendation: structural-only

## hyp_linear
- Root cause: C. vocab/gate mismatch artifact, D. graph-context misuse, G. shape/contract mismatch, I. gradient explosion, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: yes
- New result: works_now
- Recommendation: restricted-use

## hyp_tangent_nonlinear
- Root cause: C. vocab/gate mismatch artifact, D. graph-context misuse, G. shape/contract mismatch, I. gradient explosion, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: yes
- New result: works_now
- Recommendation: restricted-use

## rwkv_time_mixing
- Root cause: B. stale evidence from fixed bug, M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: yes
- New result: works_now
- Recommendation: rerun-needed

## rope_rotate
- Root cause: B. stale evidence from fixed bug
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: yes
- New result: works_now
- Recommendation: rerun-needed

## split2
- Root cause: B. stale evidence from fixed bug, C. vocab/gate mismatch artifact, K. structural/non-learnable op, L. niche op needing restricted placement
- Code fix applied: no
- Rule fix applied: yes
- Rerun performed: yes
- New result: structural
- Recommendation: structural-only

## fixed_point_iter
- Root cause: C. vocab/gate mismatch artifact, M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: yes
- New result: works_now
- Recommendation: rerun-needed

## transpose_sd
- Root cause: unclassified
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: no
- New result: works_now
- Recommendation: restricted-use

## grouped_linear
- Root cause: C. vocab/gate mismatch artifact, I. gradient explosion, M. valid but weak
- Code fix applied: no
- Rule fix applied: no
- Rerun performed: yes
- New result: works_now
- Recommendation: rerun-needed
