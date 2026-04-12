# ML Hardening Plan

Generated: 2026-04-08

## ML Influence Status

- `screening_ensemble`: quality=strong requested=False allowed=False reason=meets_screening_thresholds
- `gbm_gate`: quality=usable requested=None allowed=None reason=None
- `graph_predictor`: quality=usable requested=None allowed=None reason=None
- `learned_candidate_weights`: quality=unproven requested=False allowed=False reason=requires_manual_override_until_validated
- `screening_signal_weights`: quality=unproven requested=False allowed=False reason=requires_manual_override_until_validated
- `learned_grammar_weights`: quality=unproven requested=False allowed=False reason=requires_manual_override_until_validated
- `investigation_predictor`: quality=monitor_only requested=True allowed=False reason=heldout_metrics_below_threshold

## Backfill Priority Templates

- `attn_safe_division`: evidence=insufficient n_used=2 s1_rate=1.0 actions=Backfill this template before changing weights.; Bias backfills toward longer-range token-interaction motifs.; Probe slot choices that preserve non-local token access.
- `attn_softmax_normalized_matmul_fixed_tail_norm`: evidence=sparse n_used=3 s1_rate=0.0 actions=Downweight until slot and motif evidence improves.
- `depth_gated_block_matmul_stable`: evidence=sparse n_used=4 s1_rate=0.0 actions=Downweight until slot and motif evidence improves.
- `attn_softmax_normalized_matmul_v2`: evidence=sparse n_used=4 s1_rate=0.0 actions=Downweight until slot and motif evidence improves.
- `attn_softmax_normalized_matmul_compact_ffn`: evidence=sparse n_used=5 s1_rate=0.0 actions=Downweight until slot and motif evidence improves.
- `depth_gated_block_matmul_norm`: evidence=sparse n_used=6 s1_rate=1.0 actions=Use as a reference family for nearby sparse templates.; Probe slot choices that preserve non-local token access.; Do not trust perplexity-only wins from this family.
- `attn_linear_no_matmul_ffn_direct_recovery`: evidence=sparse n_used=6 s1_rate=0.0 actions=Downweight until slot and motif evidence improves.
- `attn_linear_no_matmul_ffn_dense_tail`: evidence=sparse n_used=7 s1_rate=0.14285714285714285 actions=Downweight until slot and motif evidence improves.; Do not trust perplexity-only wins from this family.
- `attn_linear_no_matmul_ffn_v2`: evidence=sparse n_used=8 s1_rate=0.0 actions=Downweight until slot and motif evidence improves.
- `depth_gated_block_matmul`: evidence=building n_used=10 s1_rate=0.2 actions=Bias backfills toward longer-range token-interaction motifs.; Probe slot choices that preserve non-local token access.
- `attn_linear_softmax_recovery_control`: evidence=building n_used=10 s1_rate=0.0 actions=Downweight until slot and motif evidence improves.
- `multiscale_difficulty_router`: evidence=building n_used=12 s1_rate=0.25 actions=Bias backfills toward longer-range token-interaction motifs.; Probe slot choices that preserve non-local token access.; Do not trust perplexity-only wins from this family.
- `attn_softmax_router_sidecar`: evidence=building n_used=14 s1_rate=0.0 actions=Downweight until slot and motif evidence improves.
- `attn_normalized_matmul`: evidence=building n_used=15 s1_rate=0.06666666666666667 actions=Downweight until slot and motif evidence improves.; Do not trust perplexity-only wins from this family.
- `attn_bottleneck_hybrid`: evidence=building n_used=17 s1_rate=0.058823529411764705 actions=Downweight until slot and motif evidence improves.; Do not trust perplexity-only wins from this family.
- `attn_reciprocal_gated`: evidence=building n_used=18 s1_rate=0.0 actions=Downweight until slot and motif evidence improves.; Do not trust perplexity-only wins from this family.
- `gpt2_reference`: evidence=building n_used=19 s1_rate=0.3684210526315789 actions=Bias backfills toward longer-range token-interaction motifs.; Probe slot choices that preserve non-local token access.; Do not trust perplexity-only wins from this family.
- `poincare_add_bridge`: evidence=building n_used=23 s1_rate=0.0 actions=Downweight until slot and motif evidence improves.
- `graph_attn_sparse_ffn`: evidence=building n_used=23 s1_rate=0.0 actions=Downweight until slot and motif evidence improves.; Treat it as a slow starter and extend targeted backfills.
- `attn_residual_block`: evidence=building n_used=24 s1_rate=0.041666666666666664 actions=Downweight until slot and motif evidence improves.

## Reference Families

- `depth_gated_block_matmul_norm`: n_used=6 s1_rate=1.0 best_loss=0.4359215342848298
- `attn_safe_division`: n_used=2 s1_rate=1.0 best_loss=0.6152439264817225
- `three_way_split`: n_used=941 s1_rate=0.8671625929861849 best_loss=0.10975049325262459
- `hyperbolic_bridge_block`: n_used=145 s1_rate=0.6068965517241379 best_loss=0.10069491086551921
- `routed_bottleneck`: n_used=562 s1_rate=0.5871886120996441 best_loss=0.07223220493212536
- `fused_gelu_ffn`: n_used=173 s1_rate=0.5722543352601156 best_loss=0.07549545546550689
- `spiking_moe_block`: n_used=483 s1_rate=0.5548654244306418 best_loss=0.0842747084945671
- `rwkv_block`: n_used=229 s1_rate=0.5502183406113537 best_loss=0.11128325179657164

## Data Recommendations

- Collect more slot-level evidence for sparse/building attention templates before any positive weighting.
- Expand induction and binding coverage for families that pass S1 but remain weak on non-perplexity probes.
- Prefer backfills with complete screening provenance so candidate/signal weighting can be validated on trusted rows.
- Use reference families with repeated low-loss survivors to anchor nearby template backfills rather than globally raising weak families.
