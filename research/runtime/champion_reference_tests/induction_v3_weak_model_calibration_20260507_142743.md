# Induction V3 Weak Model Calibration

- generated_at: `2026-05-07T14:39:56-0400`
- device: `cuda`
- protocol: `induction_v3_5k` default budget, no DB writes

| case | kind | train_ckpt_steps | status | auc | max_gap | gap_cv | gap_accs | elapsed_ms | wall_s |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| untrained_gpt2_6L | untrained_reference |  | ok | 0.0140 | 0.0400 | 1.1384 | 4:0.040, 8:0.025, 16:0.000, 32:0.005, 64:0.000 | 87963.4 | 88.0 |
| gpt2_6L_step10000 | checkpoint | 10000 | ok | 0.9850 | 0.9900 | 0.0045 | 4:0.980, 8:0.990, 16:0.980, 32:0.985, 64:0.990 | 87222.7 | 87.3 |
| gpt2_6L_step20000 | checkpoint | 20000 | ok | 0.9870 | 1.0000 | 0.0109 | 4:0.995, 8:0.970, 16:1.000, 32:0.990, 64:0.980 | 87353.0 | 87.5 |
| gpt2_6L_step40000 | checkpoint | 40000 | ok | 0.8350 | 0.8750 | 0.0283 | 4:0.825, 8:0.825, 16:0.875, 32:0.805, 64:0.845 | 87403.5 | 87.5 |
| untrained_mamba_6L | untrained_reference |  | ok | 0.8000 | 1.0000 | 0.5000 | 4:1.000, 8:1.000, 16:1.000, 32:1.000, 64:0.000 | 102309.1 | 102.4 |
| untrained_rwkv_6L | untrained_reference |  | ok | 0.9920 | 1.0000 | 0.0051 | 4:0.990, 8:0.985, 16:0.995, 32:1.000, 64:0.990 | 149101.8 | 149.2 |
| untrained_retrieval_augmented_6L | untrained_reference |  | ok | 0.0070 | 0.0150 | 0.7284 | 4:0.000, 8:0.005, 16:0.010, 32:0.005, 64:0.015 | 130900.9 | 131.0 |
