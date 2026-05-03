# High-Induction Motif Analysis

- Parents: 10
- Observations analyzed: 72 of 132 total
- Exclude attention-like edits: True
- Exclude local neighborhoods containing attention: True
- Outcomes: {'inconclusive': 27, 'refuted_ablation_improved': 33, 'supported': 12}
- Rule types: {'node_delete': 34, 'component_replace': 38}
- Mean induction delta: -0.488
- Mean binding delta: -0.169
- Mean loss advantage: 0.007
- Loss-better with induction drop > 0.50: 20

## Worst Non-Attention Edit Patterns
- node_delete node_delete matmul: n=1 parents=1 avg_ind=-0.978 avg_bind=-0.370 avg_loss_adv=0.052 loss_better_bad_ind=1 outcomes={'refuted_ablation_improved': 1}
- node_delete node_delete adjacent_token_merge: n=1 parents=1 avg_ind=-0.972 avg_bind=-0.383 avg_loss_adv=0.059 loss_better_bad_ind=1 outcomes={'refuted_ablation_improved': 1}
- node_delete node_delete gather_topk: n=1 parents=1 avg_ind=-0.972 avg_bind=-0.357 avg_loss_adv=0.167 loss_better_bad_ind=1 outcomes={'refuted_ablation_improved': 1}
- component_replace activation swiglu_mlp -> relu: n=1 parents=1 avg_ind=-0.968 avg_bind=-0.375 avg_loss_adv=0.014 loss_better_bad_ind=1 outcomes={'inconclusive': 1}
- component_replace projection nm_sparse_linear -> block_sparse_linear: n=1 parents=1 avg_ind=-0.962 avg_bind=-0.368 avg_loss_adv=0.006 loss_better_bad_ind=1 outcomes={'refuted_ablation_improved': 1}
- node_delete node_delete rmsnorm: n=3 parents=1 avg_ind=-0.960 avg_bind=-0.368 avg_loss_adv=0.020 loss_better_bad_ind=3 outcomes={'inconclusive': 1, 'refuted_ablation_improved': 2}
- component_replace projection nm_sparse_linear -> grouped_linear: n=1 parents=1 avg_ind=-0.950 avg_bind=-0.360 avg_loss_adv=0.045 loss_better_bad_ind=1 outcomes={'refuted_ablation_improved': 1}
- component_replace sequence_mixer conv1d_seq -> spectral_filter: n=1 parents=1 avg_ind=-0.932 avg_bind=-0.350 avg_loss_adv=0.026 loss_better_bad_ind=1 outcomes={'refuted_ablation_improved': 1}
- component_replace sequence_mixer conv1d_seq -> rope_rotate: n=1 parents=1 avg_ind=-0.924 avg_bind=-0.350 avg_loss_adv=-0.008 loss_better_bad_ind=0 outcomes={'refuted_ablation_improved': 1}
- node_delete node_delete conv1d_seq: n=1 parents=1 avg_ind=-0.920 avg_bind=-0.358 avg_loss_adv=0.007 loss_better_bad_ind=1 outcomes={'inconclusive': 1}
- component_replace projection nm_sparse_linear -> tied_proj: n=1 parents=1 avg_ind=-0.906 avg_bind=-0.355 avg_loss_adv=0.023 loss_better_bad_ind=1 outcomes={'refuted_ablation_improved': 1}
- node_delete node_delete nm_sparse_linear: n=1 parents=1 avg_ind=-0.890 avg_bind=-0.336 avg_loss_adv=-0.014 loss_better_bad_ind=0 outcomes={'inconclusive': 1}
- node_delete node_delete add: n=20 parents=9 avg_ind=-0.852 avg_bind=-0.269 avg_loss_adv=0.004 loss_better_bad_ind=8 outcomes={'inconclusive': 7, 'supported': 5, 'refuted_ablation_improved': 8}
- component_replace activation swiglu_mlp -> tanh: n=1 parents=1 avg_ind=-0.574 avg_bind=-0.258 avg_loss_adv=-0.020 loss_better_bad_ind=0 outcomes={'inconclusive': 1}
- node_delete node_delete relu: n=1 parents=1 avg_ind=-0.232 avg_bind=-0.071 avg_loss_adv=0.016 loss_better_bad_ind=0 outcomes={'inconclusive': 1}
- node_delete node_delete token_type_classifier: n=1 parents=1 avg_ind=-0.230 avg_bind=-0.069 avg_loss_adv=-0.045 loss_better_bad_ind=0 outcomes={'supported': 1}
- component_replace projection linear_proj -> semi_structured_2_4_linear: n=10 parents=9 avg_ind=-0.146 avg_bind=-0.043 avg_loss_adv=-0.013 loss_better_bad_ind=0 outcomes={'refuted_ablation_improved': 4, 'inconclusive': 4, 'supported': 2}
- component_replace projection linear_proj -> block_sparse_linear: n=10 parents=9 avg_ind=-0.143 avg_bind=-0.043 avg_loss_adv=0.005 loss_better_bad_ind=0 outcomes={'refuted_ablation_improved': 4, 'inconclusive': 4, 'supported': 2}

## Repeated Role Signatures
- projection|pos=middle|in=activation|out=merge|up_norm=rmsnorm|up_route=none|down_merge=add: n=4 parents=1 avg_ind=-0.927 avg_bind=-0.355 loss_better_bad_ind=3
- sequence|pos=early|in=normalization|out=activation|up_norm=rmsnorm|up_route=none|down_merge=add: n=3 parents=1 avg_ind=-0.925 avg_bind=-0.353 loss_better_bad_ind=2
- merge|pos=late|in=merge+merge|out=merge+normalization|up_norm=layernorm|up_route=token_type_classifier|down_merge=add: n=7 parents=7 avg_ind=-0.922 avg_bind=-0.278 loss_better_bad_ind=2
- merge|pos=late|in=projection+merge|out=merge|up_norm=layernorm|up_route=token_type_classifier|down_merge=add: n=6 parents=6 avg_ind=-0.914 avg_bind=-0.275 loss_better_bad_ind=3
- activation|pos=late|in=routing|out=merge|up_norm=rmsnorm|up_route=gather_topk|down_merge=add: n=3 parents=1 avg_ind=-0.529 avg_bind=-0.242 loss_better_bad_ind=1
- projection|pos=middle|in=normalization|out=merge|up_norm=layernorm|up_route=none|down_merge=add: n=19 parents=8 avg_ind=-0.149 avg_bind=-0.044 loss_better_bad_ind=0
- routing|pos=middle|in=normalization|out=other|up_norm=layernorm|up_route=none|down_merge=mul: n=6 parents=5 avg_ind=-0.094 avg_bind=-0.028 loss_better_bad_ind=0
- projection|pos=early|in=normalization|out=merge|up_norm=layernorm|up_route=none|down_merge=add: n=5 parents=2 avg_ind=-0.067 avg_bind=-0.021 loss_better_bad_ind=0

## Parent Critical Backbones
### 956a97dd-a05
- node 17 add node_delete ind=-0.984 bind=-0.297 loss_adv=-0.008; linear_proj+mul+relu>add+add>add>add+rmsnorm>add+spectral_filter
- node 11 linear_proj -> semi_structured_2_4_linear component_replace ind=-0.580 bind=-0.174 loss_adv=-0.032; add>layernorm>linear_proj>add>add
### 4b69e623-3ea
- node 16 add node_delete ind=-0.988 bind=-0.297 loss_adv=-0.020; entropy_score+layernorm+linear_proj>linear_proj+mul>add>add>add+rmsnorm
- node 17 add node_delete ind=-0.984 bind=-0.296 loss_adv=-0.014; linear_proj+mul+relu>add+add>add>add+rmsnorm>add+spectral_filter
### 5193b8ff-51a
- node 17 add node_delete ind=-0.984 bind=-0.297 loss_adv=-0.085; linear_proj+mul>add>add>add+rmsnorm>add+spectral_filter
- node 16 add node_delete ind=-0.974 bind=-0.294 loss_adv=-0.082; entropy_score+layernorm+linear_proj>linear_proj+mul>add>add>add+rmsnorm
### 0d212637-8f9
- node 16 add node_delete ind=-0.796 bind=-0.238 loss_adv=0.052; entropy_score+layernorm+linear_proj>linear_proj+mul>add>add>add+rmsnorm
- node 17 add node_delete ind=-0.796 bind=-0.238 loss_adv=0.058; linear_proj+mul+relu>add+add>add>add+rmsnorm>add+spectral_filter
### bb63f4e1-fd3
- node 17 add node_delete ind=-0.932 bind=-0.282 loss_adv=-0.003; linear_proj+mul+relu>add+add>add>add+rmsnorm>add+spectral_filter
- node 16 add node_delete ind=-0.926 bind=-0.280 loss_adv=-0.039; entropy_score+layernorm+linear_proj>linear_proj+mul>add>add>add+rmsnorm
### fb08228a-848
- node 16 add node_delete ind=-0.984 bind=-0.296 loss_adv=0.053; entropy_score+layernorm+linear_proj>linear_proj+mul>add>add>add+rmsnorm
- node 17 add node_delete ind=-0.968 bind=-0.291 loss_adv=0.023; linear_proj+mul+semi_structured_2_4_linear>add+add>add>add+rmsnorm>add+spectral_filter
### 176f5529-38f
- node 16 add node_delete ind=-0.956 bind=-0.288 loss_adv=-0.029; entropy_score+layernorm+linear_proj>linear_proj+mul>add>add>add+rmsnorm
- node 17 add node_delete ind=-0.952 bind=-0.287 loss_adv=-0.031; linear_proj+mul+relu>add+add>add>add+rmsnorm>add+add
### 665a093d-329
- node 9 rmsnorm node_delete ind=-0.978 bind=-0.372 loss_adv=0.028; add+nm_sparse_linear>add>rmsnorm>gather_topk+linear_proj>matmul+matmul+swiglu_mlp
- node 11 matmul node_delete ind=-0.978 bind=-0.370 loss_adv=0.052; rmsnorm+rmsnorm>linear_proj+linear_proj>matmul>linear_proj>gather_topk
- node 2 adjacent_token_merge node_delete ind=-0.972 bind=-0.383 loss_adv=0.059; input>rmsnorm>adjacent_token_merge>add+rmsnorm>add+conv1d_seq
- node 13 gather_topk node_delete ind=-0.972 bind=-0.357 loss_adv=0.167; add+matmul>linear_proj+rmsnorm>gather_topk>swiglu_mlp>add
- node 17 rmsnorm node_delete ind=-0.972 bind=-0.371 loss_adv=0.020; add>add>rmsnorm>output>output
- node 14 swiglu_mlp -> relu component_replace ind=-0.968 bind=-0.375 loss_adv=0.014; linear_proj+rmsnorm>gather_topk>swiglu_mlp>add>add
- node 15 add node_delete ind=-0.966 bind=-0.365 loss_adv=0.037; add+gather_topk+nm_sparse_linear>add+swiglu_mlp>add>add>rmsnorm
- node 7 nm_sparse_linear -> block_sparse_linear component_replace ind=-0.962 bind=-0.368 loss_adv=0.006; conv1d_seq>gelu>nm_sparse_linear>add>add+rmsnorm
- node 7 nm_sparse_linear -> grouped_linear component_replace ind=-0.950 bind=-0.360 loss_adv=0.045; conv1d_seq>gelu>nm_sparse_linear>add>add+rmsnorm
- node 5 conv1d_seq -> spectral_filter component_replace ind=-0.932 bind=-0.350 loss_adv=0.026; adjacent_token_merge>rmsnorm>conv1d_seq>gelu>nm_sparse_linear
### bd1799ce-f3f
- node 17 add node_delete ind=-0.838 bind=-0.253 loss_adv=-0.011; linear_proj+mul+relu>add+add>add>add+rmsnorm>add+spectral_filter
- node 16 add node_delete ind=-0.834 bind=-0.252 loss_adv=0.053; entropy_score+layernorm+linear_proj>linear_proj+mul>add>add>add+rmsnorm

## Induction-Preserving Non-Attention Alternatives
- node_delete node_delete gelu: n=1 parents=1 avg_ind=-0.018 avg_bind=-0.078 avg_loss_adv=0.006
- node_delete node_delete entropy_score: n=1 parents=1 avg_ind=-0.028 avg_bind=-0.009 avg_loss_adv=-0.013
- component_replace activation swiglu_mlp -> gelu: n=1 parents=1 avg_ind=-0.044 avg_bind=-0.092 avg_loss_adv=-0.013
- node_delete node_delete linear_proj: n=1 parents=1 avg_ind=-0.050 avg_bind=-0.015 avg_loss_adv=0.040
- component_replace routing_signal token_type_classifier -> topk_gate: n=5 parents=5 avg_ind=-0.067 avg_bind=-0.020 avg_loss_adv=-0.002
- component_replace projection linear_proj -> nm_sparse_linear: n=3 parents=3 avg_ind=-0.075 avg_bind=-0.022 avg_loss_adv=0.005
- component_replace projection semi_structured_2_4_linear -> block_sparse_linear: n=2 parents=2 avg_ind=-0.075 avg_bind=-0.023 avg_loss_adv=0.052
- node_delete node_delete spectral_filter: n=1 parents=1 avg_ind=-0.092 avg_bind=-0.028 avg_loss_adv=-0.006
