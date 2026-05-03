# Construction Prior Snapshot v20260501-200917

## Summary
- n_avoid: 3
- n_local_edit_observations: 1102
- n_local_exact_rule_priors: 224
- n_local_slot_motif_multipliers: 15
- n_mixed: 73
- n_op_weights: 14
- n_rules: 169
- n_slot_motif_denylist: 0
- n_slot_motif_multipliers: 27
- n_use: 93
- version: v20260501-200917

## Policy
- advisory only; gates: false
- positive local score means the ablation child beat its parent across the weighted full S1 metric vector; negative means the original component was protective

## Top Local Op Contexts
- rope_rotate: n=12 parents=1 direction=original_better avg=-0.4145 edit/original/mixed=0/12/0 metric wins/losses=23/60
- padic_gate: n=6 parents=1 direction=original_better avg=-0.3959 edit/original/mixed=0/6/0 metric wins/losses=5/37
- semi_structured_2_4_linear: n=12 parents=1 direction=original_better avg=-0.3393 edit/original/mixed=0/12/0 metric wins/losses=18/66
- relu: n=12 parents=3 direction=original_better avg=-0.3324 edit/original/mixed=0/8/4 metric wins/losses=20/61
- mul: n=11 parents=6 direction=original_better avg=-0.3027 edit/original/mixed=0/8/3 metric wins/losses=24/51
- spectral_filter: n=20 parents=3 direction=original_better avg=-0.2476 edit/original/mixed=1/15/4 metric wins/losses=40/98
- token_type_classifier: n=10 parents=2 direction=original_better avg=-0.2324 edit/original/mixed=0/7/3 metric wins/losses=22/48
- add: n=164 parents=22 direction=original_better avg=-0.2302 edit/original/mixed=4/121/39 metric wins/losses=382/730
- softmax_attention: n=26 parents=3 direction=original_better avg=-0.2205 edit/original/mixed=3/16/7 metric wins/losses=58/123
- adjacent_token_merge: n=49 parents=9 direction=original_better avg=-0.1702 edit/original/mixed=1/35/13 metric wins/losses=134/194
- linear_proj: n=189 parents=15 direction=original_better avg=-0.165 edit/original/mixed=4/111/74 metric wins/losses=502/788
- matmul: n=15 parents=5 direction=original_better avg=-0.1635 edit/original/mixed=0/12/3 metric wins/losses=36/65

## Component Contexts
- binary_merge: n=133 parents=13 direction=original_better avg=-0.2765 edit/original/mixed=1/108/24 metric wins/losses=274/624
- routing_signal: n=22 parents=5 direction=original_better avg=-0.1817 edit/original/mixed=2/13/7 metric wins/losses=56/97
- sequence_mixer: n=203 parents=20 direction=original_better avg=-0.1714 edit/original/mixed=16/115/72 metric wins/losses=553/832
- normalization: n=88 parents=19 direction=original_better avg=-0.1571 edit/original/mixed=2/52/34 metric wins/losses=237/350
- projection: n=295 parents=19 direction=original_better avg=-0.1437 edit/original/mixed=14/165/116 metric wins/losses=820/1192
- activation: n=113 parents=20 direction=original_better avg=-0.1257 edit/original/mixed=3/55/55 metric wins/losses=321/438

## Top Slot-Motif Adjustments
- latent_attn_sparse_ffn.slot2:sparse_ternary: n=26 direction=original_better avg=-0.4681 multiplier=1.328
- feature_sparse_block.slot1:routed_ternary: n=13 direction=original_better avg=-0.3461 multiplier=1.242
- feature_sparse_block.slot0:norm_layer: n=7 direction=original_better avg=-0.3339 multiplier=1.234
- attn_linear_no_matmul_ffn_direct_recovery.slot0:norm_rms: n=4 direction=original_better avg=-0.2898 multiplier=1.203
- latent_attn_sparse_ffn.slot0:norm_layer: n=32 direction=original_better avg=-0.2244 multiplier=1.157
- conditional_compute.slot1:routed_ternary: n=19 direction=original_better avg=-0.2175 multiplier=1.152
- normalized_matmul.slot0:norm_layer: n=46 direction=original_better avg=-0.195 multiplier=1.136
- attn_softmax_normalized_matmul[0].slot2:norm_rms: n=9 direction=original_better avg=-0.1878 multiplier=1.131
- attn_softmax_normalized_matmul[0].slot1:norm_layer: n=16 direction=original_better avg=-0.1868 multiplier=1.131
- n_way_moe_block.slot1:norm_rms: n=4 direction=original_better avg=-0.1835 multiplier=1.128
- attn_softmax_normalized_matmul[0].slot0:norm_layer: n=15 direction=original_better avg=-0.1639 multiplier=1.115
- n_way_moe_block.slot0:norm_rms: n=8 direction=original_better avg=-0.1461 multiplier=1.102
