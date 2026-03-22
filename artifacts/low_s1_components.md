# Low-S1 Components

| component | obs | compile_rate | s1_rate | status | contaminated | structural | stale | selectable | action |
|---|---:|---:|---:|---|---|---|---|---|---|
| graph_attention | 96 | 99.0% | 0.0% | works_now | yes | no | yes | yes | rerun with corrected conditions |
| local_window_attn | 75 | 0.0% | 0.0% | fixed_now | no | no | no | yes | code fix |
| state_space | 68 | 97.1% | 0.0% | works_now | yes | no | yes | yes | rerun with corrected conditions |
| sliding_window_mask | 66 | 100.0% | 0.0% | structural | yes | yes | yes | yes | reclassification as structural / restricted-use op |
| causal_mask | 63 | 95.2% | 0.0% | structural | yes | yes | yes | yes | reclassification as structural / restricted-use op |
| tropical_center | 62 | 96.8% | 0.0% | works_now | yes | no | yes | yes | rerun with corrected conditions |
| early_exit | 60 | 68.3% | 0.0% | works_now | no | no | no | yes | context-rule fix |
| gated_delta | 55 | 87.3% | 0.0% | works_now | yes | no | no | yes | restriction |
| diff_attention | 54 | 96.3% | 0.0% | works_now | yes | no | no | yes | restriction |
| fused_linear_gelu | 54 | 66.7% | 0.0% | works_now | yes | no | no | yes | restriction |
| integral_kernel | 50 | 92.0% | 0.0% | works_now | yes | no | no | yes | restriction |
| split3 | 50 | 62.0% | 0.0% | structural | yes | yes | no | no | reclassification as structural / restricted-use op |
| exp | 47 | 91.5% | 0.0% | works_now | yes | no | no | yes | restriction |
| cumprod_safe | 40 | 97.5% | 0.0% | works_now | no | no | no | yes | context-rule fix |
| lif_neuron | 40 | 85.0% | 0.0% | works_now | yes | no | no | yes | context-rule fix |
| sparse_threshold | 36 | 63.9% | 0.0% | works_now | yes | no | no | yes | context-rule fix |
| stdp_attention | 36 | 63.9% | 0.0% | works_now | yes | no | no | yes | context-rule fix |
| cumsum | 31 | 80.7% | 0.0% | works_now | no | no | no | yes | restriction |
| sub | 31 | 96.8% | 0.0% | works_now | yes | no | no | yes | restriction |
| chebyshev_spectral_mix | 30 | 93.3% | 0.0% | works_now | no | no | no | yes | restriction |
| minimum | 24 | 91.7% | 0.0% | works_now | yes | no | no | yes | restriction |
| embedding_lookup | 23 | 78.3% | 0.0% | works_now | yes | no | no | yes | context-rule fix |
| geometric_product | 23 | 95.7% | 0.0% | works_now | yes | no | no | yes | context-rule fix |
| n_way_sparse_router | 22 | 54.5% | 0.0% | works_now | yes | no | no | yes | context-rule fix |
| sign_ste | 22 | 86.4% | 0.0% | works_now | yes | no | no | yes | restriction |
| tropical_matmul | 22 | 95.5% | 0.0% | works_now | yes | no | no | yes | context-rule fix |
| norm_last | 21 | 76.2% | 0.0% | structural | yes | yes | no | yes | reclassification as structural / restricted-use op |
| sqrt | 20 | 100.0% | 0.0% | works_now | no | no | no | yes | restriction |
| log | 19 | 100.0% | 0.0% | works_now | no | no | no | yes | restriction |
| mod_topk | 19 | 89.5% | 0.0% | works_now | yes | no | no | yes | context-rule fix |
| sum_last | 17 | 94.1% | 0.0% | structural | yes | yes | no | yes | reclassification as structural / restricted-use op |
| max_last | 15 | 93.3% | 0.0% | structural | yes | yes | no | yes | reclassification as structural / restricted-use op |
| mean_last | 15 | 80.0% | 0.0% | structural | yes | yes | no | yes | reclassification as structural / restricted-use op |
| reciprocal | 10 | 100.0% | 0.0% | works_now | no | no | no | yes | context-rule fix |
| linear_attention | 158 | 98.1% | 0.6% | works_now | yes | no | yes | yes | rerun with corrected conditions |
| identity | 306 | 94.8% | 0.7% | structural | yes | yes | yes | yes | reclassification as structural / restricted-use op |
| progressive_compression_gate | 149 | 97.3% | 0.7% | works_now | yes | no | yes | yes | rerun with corrected conditions |
| softmax_attention | 240 | 97.1% | 0.8% | works_now | yes | no | yes | yes | rerun with corrected conditions |
| linear_proj_down | 979 | 94.6% | 1.0% | works_now | yes | no | yes | yes | rerun with corrected conditions |
| linear_proj_up | 990 | 95.8% | 1.3% | works_now | yes | no | no | yes | restricted-use |
| concat | 489 | 82.8% | 1.4% | structural | yes | yes | yes | no | reclassification as structural / restricted-use op |
| hyp_linear | 64 | 95.3% | 1.6% | works_now | yes | no | yes | yes | rerun with corrected conditions |
| hyp_tangent_nonlinear | 64 | 95.3% | 1.6% | works_now | yes | no | yes | yes | rerun with corrected conditions |
| rwkv_time_mixing | 63 | 98.4% | 1.6% | works_now | yes | no | yes | yes | restriction |
| rope_rotate | 57 | 98.2% | 1.8% | works_now | yes | no | yes | yes | rerun with corrected conditions |
| split2 | 389 | 88.2% | 1.8% | structural | yes | yes | yes | no | reclassification as structural / restricted-use op |
| fixed_point_iter | 55 | 100.0% | 1.8% | works_now | yes | no | yes | yes | rerun with corrected conditions |
| transpose_sd | 55 | 94.5% | 1.8% | works_now | yes | no | no | yes | restricted-use |
| grouped_linear | 52 | 96.2% | 1.9% | works_now | yes | no | yes | yes | rerun with corrected conditions |
