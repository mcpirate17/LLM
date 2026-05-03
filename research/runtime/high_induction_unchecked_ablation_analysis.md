# High-Induction Unchecked Ablation Analysis

- Targets: 10 previously unchecked high-induction parents
- Child observations: 132 across 98 evidence rows
- By rule type: {'node_delete': 52, 'component_replace': 80}
- By outcome: {'inconclusive': 60, 'refuted_ablation_improved': 50, 'supported': 22}
- Missing required S1 metrics: 0 (checked separately against program_results)

## Mean Child Minus Parent Deltas
- loss_advantage_child_better_positive: -0.0002
- ppl_advantage_pct_child_better_positive: -2.5858
- induction_delta_child_minus_parent: -0.4514
- binding_delta_child_minus_parent: -0.1476
- hellaswag_delta_child_minus_parent: 0.0208
- blimp_delta_child_minus_parent: -0.0081
- ar_delta_child_minus_parent: -0.0018

## Important Counts
- induction_drop_gt_0_25: 63
- induction_drop_gt_0_50: 54
- binding_drop_gt_0_10: 56
- child_loss_better: 70
- child_ppl_better: 14

## Strongest Induction Damage
- node_delete add parent=956a97dd-a05 outcome=supported ind_delta=-1.000 loss_adv=-0.202 ppl_adv=-0.104
- node_delete add parent=956a97dd-a05 outcome=supported ind_delta=-1.000 loss_adv=-0.035 ppl_adv=-0.104
- node_delete softmax_attention parent=4b69e623-3ea outcome=inconclusive ind_delta=-1.000 loss_adv=-0.010 ppl_adv=-0.858
- component_replace softmax_attention -> latent_attention_compressor parent=4b69e623-3ea outcome=inconclusive ind_delta=-1.000 loss_adv=-0.010 ppl_adv=-0.827
- node_delete softmax_attention parent=956a97dd-a05 outcome=inconclusive ind_delta=-0.990 loss_adv=-0.007 ppl_adv=-0.862
- node_delete add parent=4b69e623-3ea outcome=inconclusive ind_delta=-0.988 loss_adv=-0.020 ppl_adv=-0.767
- node_delete add parent=956a97dd-a05 outcome=inconclusive ind_delta=-0.984 loss_adv=-0.008 ppl_adv=-1.029
- node_delete add parent=4b69e623-3ea outcome=inconclusive ind_delta=-0.984 loss_adv=-0.014 ppl_adv=-0.860
- node_delete add parent=5193b8ff-51a outcome=supported ind_delta=-0.984 loss_adv=-0.085 ppl_adv=-17.296
- node_delete softmax_attention parent=fb08228a-848 outcome=inconclusive ind_delta=-0.984 loss_adv=0.002 ppl_adv=-0.873

## Loss Looks Better But Induction Breaks
- node_delete softmax_attention parent=fb08228a-848 outcome=inconclusive ind_delta=-0.984 loss_adv=0.002 child_ind=0.000
- node_delete add parent=fb08228a-848 outcome=refuted_ablation_improved ind_delta=-0.984 loss_adv=0.053 child_ind=0.000
- component_replace softmax_attention -> latent_attention_compressor parent=fb08228a-848 outcome=inconclusive ind_delta=-0.984 loss_adv=0.006 child_ind=0.000
- node_delete rope_rotate parent=956a97dd-a05 outcome=refuted_ablation_improved ind_delta=-0.978 loss_adv=0.024 child_ind=0.022
- node_delete rmsnorm parent=665a093d-329 outcome=refuted_ablation_improved ind_delta=-0.978 loss_adv=0.028 child_ind=0.002
- node_delete matmul parent=665a093d-329 outcome=refuted_ablation_improved ind_delta=-0.978 loss_adv=0.052 child_ind=0.002
- node_delete adjacent_token_merge parent=665a093d-329 outcome=refuted_ablation_improved ind_delta=-0.972 loss_adv=0.059 child_ind=0.008
- node_delete gather_topk parent=665a093d-329 outcome=refuted_ablation_improved ind_delta=-0.972 loss_adv=0.167 child_ind=0.008
- node_delete rmsnorm parent=665a093d-329 outcome=refuted_ablation_improved ind_delta=-0.972 loss_adv=0.020 child_ind=0.008
- component_replace softmax_attention -> latent_attention_compressor parent=50c6fd82-f68 outcome=inconclusive ind_delta=-0.972 loss_adv=0.009 child_ind=0.000

## Aggregate Signals By Edit
- node_delete node_delete rope_rotate: n=1 parents=1 avg_ind=-0.978 avg_loss_adv=0.024 outcomes={'refuted_ablation_improved': 1}
- node_delete node_delete matmul: n=1 parents=1 avg_ind=-0.978 avg_loss_adv=0.052 outcomes={'refuted_ablation_improved': 1}
- node_delete node_delete adjacent_token_merge: n=1 parents=1 avg_ind=-0.972 avg_loss_adv=0.059 outcomes={'refuted_ablation_improved': 1}
- node_delete node_delete gather_topk: n=1 parents=1 avg_ind=-0.972 avg_loss_adv=0.167 outcomes={'refuted_ablation_improved': 1}
- component_replace activation swiglu_mlp -> relu: n=1 parents=1 avg_ind=-0.968 avg_loss_adv=0.014 outcomes={'inconclusive': 1}
- component_replace projection nm_sparse_linear -> block_sparse_linear: n=1 parents=1 avg_ind=-0.962 avg_loss_adv=0.006 outcomes={'refuted_ablation_improved': 1}
- node_delete node_delete rmsnorm: n=3 parents=1 avg_ind=-0.960 avg_loss_adv=0.020 outcomes={'inconclusive': 1, 'refuted_ablation_improved': 2}
- component_replace projection nm_sparse_linear -> grouped_linear: n=1 parents=1 avg_ind=-0.950 avg_loss_adv=0.045 outcomes={'refuted_ablation_improved': 1}
- component_replace sequence_mixer softmax_attention -> latent_attention_compressor: n=9 parents=9 avg_ind=-0.941 avg_loss_adv=-0.011 outcomes={'inconclusive': 6, 'supported': 3}
- node_delete node_delete softmax_attention: n=8 parents=8 avg_ind=-0.936 avg_loss_adv=-0.007 outcomes={'inconclusive': 5, 'refuted_ablation_improved': 2, 'supported': 1}
- component_replace sequence_mixer conv1d_seq -> spectral_filter: n=1 parents=1 avg_ind=-0.932 avg_loss_adv=0.026 outcomes={'refuted_ablation_improved': 1}
- component_replace sequence_mixer conv1d_seq -> rope_rotate: n=1 parents=1 avg_ind=-0.924 avg_loss_adv=-0.008 outcomes={'refuted_ablation_improved': 1}
- node_delete node_delete conv1d_seq: n=1 parents=1 avg_ind=-0.920 avg_loss_adv=0.007 outcomes={'inconclusive': 1}
- component_replace projection nm_sparse_linear -> tied_proj: n=1 parents=1 avg_ind=-0.906 avg_loss_adv=0.023 outcomes={'refuted_ablation_improved': 1}
- node_delete node_delete nm_sparse_linear: n=1 parents=1 avg_ind=-0.890 avg_loss_adv=-0.014 outcomes={'inconclusive': 1}
- node_delete node_delete add: n=23 parents=9 avg_ind=-0.836 avg_loss_adv=-0.006 outcomes={'inconclusive': 8, 'supported': 7, 'refuted_ablation_improved': 8}
- component_replace activation swiglu_mlp -> tanh: n=1 parents=1 avg_ind=-0.574 avg_loss_adv=-0.020 outcomes={'inconclusive': 1}
- node_delete node_delete mul: n=1 parents=1 avg_ind=-0.278 avg_loss_adv=-0.058 outcomes={'supported': 1}
- node_delete node_delete latent_attention_compressor: n=1 parents=1 avg_ind=-0.236 avg_loss_adv=0.010 outcomes={'inconclusive': 1}
- node_delete node_delete semi_structured_2_4_linear: n=1 parents=1 avg_ind=-0.232 avg_loss_adv=0.026 outcomes={'refuted_ablation_improved': 1}
