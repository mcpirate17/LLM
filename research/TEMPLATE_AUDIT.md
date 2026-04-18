# Template Catalog Audit — 2026-04-15

**Total templates**: 163 active (4 retired at weight=0)
**Audit criteria**: Presence of attention ops, FFN, residual count, parallel mixing

## Category Summary

| Category | Count | Action |
|----------|-------|--------|
| STRONG | ~48 | Keep, boost weights for top performers |
| DECENT | ~36 | Could upgrade to STRONG by adding parallel mixing |
| WEAK | ~61 | Need structural fixes (missing FFN/attention) |
| EXOTIC | ~23 | Keep for novelty, low weight |
| REFERENCE | 5 | Keep as baselines |

## STRONG Templates (attention + FFN + 2+ residuals + parallel mixing)

These follow the winning pattern and need no changes:

| Template | Attention Op | FFN | Parallel | Notes |
|----------|-------------|-----|----------|-------|
| latent_attn_ssm_hybrid | latent_attention_compressor | motif-picked | yes (SSM) | **BEST** 47% S1 |
| local_attn_ssm_hybrid | local_window_attn | motif-picked | yes (SSM) | 44% S1 |
| attn_ssm_hybrid | motif-picked | motif-picked | yes (SSM) | Generic hybrid |
| attn_conv_hybrid | motif-picked | motif-picked | yes (conv) | Conv complement |
| latent_attn_conv_hybrid | latent_attention_compressor | motif-picked | yes (conv) | |
| diff_attn_conv_hybrid | diff_attention | motif-picked | yes (conv) | |
| attn_state_space_hybrid | motif-picked | motif-picked | yes (SSM) | |
| attn_rwkv_hybrid | motif-picked | swiglu_mlp | yes (RWKV) | |
| transformer_block | motif-picked | motif-picked | sequential | Classic |
| latent_attn_ffn_block | latent_attention_compressor | motif-picked | sequential | 41.5% S1 |
| graph_attn_ffn_block | graph_attention | motif-picked | sequential | 40% S1 |
| local_attn_ffn_block | local_window_attn | motif-picked | sequential | |
| diff_attn_ffn_block | diff_attention | motif-picked | sequential | |
| diff_attn_gated_ffn | diff_attention | gated motif | sequential | |
| local_attn_swiglu | local_window_attn | swiglu_mlp | sequential | |
| latent_attn_moe | latent_attention_compressor | MoE | sequential | |
| local_attn_moe | local_window_attn | MoE | sequential | |
| diff_attn_moe | diff_attention | MoE | sequential | |
| graph_attn_moe | graph_attention | MoE | sequential | |
| attn_sparse_moe | motif-picked | sparse MoE | sequential | |
| attn_routing_block | softmax_attention | swiglu_mlp | sequential | +routing |
| attn_three_way_split | motif-picked | motif-picked | 3-way split | 86.4% S1 parent |
| attn_conditional_compute | motif-picked | motif-picked | gated | |
| attn_cross_dim | motif-picked | motif-picked | sequential | Cross-dim mixing |
| attn_spectral_filter | motif-picked | motif-picked | sequential | spectral 0.329 S1 |
| recursive_attn_ssm_hybrid | latent_attention_compressor | sparse/FFN | sequential | +recursion |
| recursive_moe_attn | motif-picked | MoE | +recursion | |
| induction_matmul_block | motif-picked | matmul+FFN | sequential | |

### New templates added this session (all STRONG):
| Template | Pattern | Weight |
|----------|---------|--------|
| recursive_attn_ssm_depth | latent_attn \|\| SSM + adaptive_recursion + FFN | 5.5 |
| latent_attn_padic_hybrid | latent_attn \|\| padic_expand + FFN | 5.0 |
| graph_attn_ssm_recursive | graph_attn \|\| SSM + FFN | 4.5 |

## WEAK Templates — Priority Fix List

### Batch 1: Missing FFN (easiest fix — just add FFN sub-block)

| Template | Current Pattern | Fix | Priority |
|----------|----------------|-----|----------|
| bottleneck | down → motif → up → residual | Add norm → FFN → residual after up | HIGH |
| sparse_ffn | attn → sparse → residual | Add norm → swiglu_mlp → residual | HIGH |
| moe | norm → MoE → residual | Add attention path + FFN | HIGH |
| gated_maximum | proj → max → proj → residual | Add attention + FFN | MED |
| attn_exp_gated | attn → exp → residual | Add FFN block | MED |
| attn_gated_product | attn → outer_product → residual | Add FFN block | MED |
| attn_gated_minimum | attn → minimum → residual | Add FFN block | MED |
| attn_gated_maximum | attn → maximum → residual | Add FFN block | MED |
| attn_hyperbolic | attn → hyp paths → residual | Add FFN block | LOW |
| attn_safe_division | attn → div_safe → residual | Add FFN block | LOW |
| attn_spiking_hybrid | attn \|\| spiking → residual | Add FFN block | MED |
| multi_head_mix_block | multi_head_mix → mixer → residual | Add FFN block | MED |
| spiking_stdp_block | lif → stdp_attention → residual | Add FFN block | LOW |

### Batch 2: Missing Attention (need to add attention path)

| Template | Current Pattern | Fix | Priority |
|----------|----------------|-----|----------|
| residual_block | norm → motif → residual | Force attention motif | HIGH |
| cumulative_sequence | cumsum → proj → residual | Add parallel attention | MED |
| sqrt_gated_ffn | sqrt → gate → residual | Add attention path | MED |
| reduce_attend | FFN → reduce → residual | Add attention path | MED |
| fused_gelu_ffn | fused_gelu → gate → residual | Add attention path | MED |
| exp_gated_residual | exp → gate → proj → residual | Add attention path | MED |
| conv_residual_block | conv → FFN → residual | Add parallel attention | MED |
| cross_dim_mixer | transpose → mixer → transpose → residual | Add attention | LOW |
| dual_axis_block | split → mixer/FFN → concat → residual | Force attention | LOW |

### Batch 3: Gradient Issues (need structural repair)

| Template | Issue | Fix | Priority |
|----------|-------|-----|----------|
| reciprocal_gated | reciprocal kills gradients | Retired (weight=0) | DONE |
| log_gated | log(sigmoid) saturates | Add parallel attention + FFN, bound log | LOW |
| sign_ste_gated | sign_ste zeros gradients | Add attention bypass | LOW |

### Batch 4: Sequential → Parallel Upgrade

| Template | Current | Upgrade | Priority |
|----------|---------|---------|----------|
| attn_chebyshev_hybrid | attn → chebyshev → residual | attn \|\| chebyshev → FFN | LOW |
| attn_kronecker_hybrid | attn → kronecker → residual | attn \|\| kronecker → FFN | LOW |
| attn_log_gated | attn → log → residual | attn \|\| log_gate → FFN | LOW |
| attn_decay_sequence | attn → decay → residual | Keep (decay is meaningful sequential) | — |

## Routing Templates — Special Category

These are structurally complex with difficulty-based routing. Some work (recursive_depth_router 37.7% S1), some don't (multiscale variants 0% S1).

| Template | S1 Rate | Status | Notes |
|----------|---------|--------|-------|
| recursive_depth_router | 37.7% | KEEP (w=6.0) | Strong routing pattern |
| difficulty_routed_block | good | KEEP (w=5.0) | Established |
| intelligent_multilane_router | — | KEEP (w=5.5) | Controlled compute |
| multiscale_difficulty_router | — | KEEP (w=5.5) | Base variant |
| multiscale_difficulty_router_adaptive_attn_ssm | 50% | KEEP (w=0.25) | Best routing variant |
| multiscale_difficulty_router_easy_attn_ssm | 0% | RETIRED (w=0) | Over-complex |
| multiscale_difficulty_router_blocksparse_attn_ssm | 0% | RETIRED (w=0) | Over-complex |
| three_lane_adaptive | — | KEEP (w=5.0) | |
| cascaded_early_exit | — | KEEP (w=4.5) | |
| hybrid_sparse_triplet_router | — | KEEP (w=6.0) | |

## Exotic Templates — Keep for Novelty

These use non-standard math (tropical, hyperbolic, spiking) and may discover novel architectures:

| Template | Math Domain | S1 Rate | Notes |
|----------|------------|---------|-------|
| tropical_center_block | min-plus algebra | some | Proven S1 with lr=0.079 |
| spiking_moe_block | spiking + tropical + MoE | 38.5% | Strong combo |
| hyperbolic_bridge_block | Poincare ball | — | Geometric learning |
| poincare_add_bridge | Hyperbolic addition | — | Geometric learning |
| geometric_product_block | Clifford algebra | — | Rotor transforms |
| tropical_residual | min-plus addition | — | |
| tropical_matmul_block | tropical matmul | — | |

## Exercise List for Future Chat Sessions

### Exercise 1: Fix Missing-FFN Templates (Batch 1)
**Time**: ~30 min per template
**Steps**:
1. Read the template function
2. Add `norm → FFN → residual` after the current output
3. Compile check, build check, gradient check, 25-seed validation
4. Run 1000-step GPU eval
5. Update weight in templates.py

### Exercise 2: Add Attention to Non-Attention Templates (Batch 2)  
**Time**: ~45 min per template
**Steps**:
1. Read the template function
2. Add parallel attention path: `{attention || existing_path} → merge → ...`
3. Use `state_space` or direct attention op to avoid context rule violations
4. Full validation suite
5. Run 1000-step GPU eval with binding probes

### Exercise 3: Upgrade Sequential to Parallel (Batch 4)
**Time**: ~20 min per template
**Steps**:
1. Identify the sequential attn → exotic_op chain
2. Split into parallel: `{attn || exotic_op} → merge`
3. Ensure FFN follows
4. Validate and eval

### Exercise 4: Deep Evaluation Cycle
**Time**: 2+ hours
**Steps**:
1. Pick 5 templates from the WEAK list
2. Fix them all using the playbook
3. Run `python -m research.tools.eval_templates --steps 5000`
4. Compare to GPT-2 reference
5. Iterate on any that don't beat GPT-2
6. Store results in notebook DB
