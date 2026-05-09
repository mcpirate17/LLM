# AR probe comparison — v1

archs=['gpt2', 'mamba', 'rwkv', 'retrieval_augmented'] seeds=[0, 1] probes=['intermediate', 'validation', 'curriculum']

## Headline (mean ± std across seeds)

| arch | ar_intermediate (held_pair_lift) | ar_validation (held_pair_acc) | ar_curriculum (curriculum_learning) |
|---|---:|---:|---:|
| GPT-2 | 0.088±0.009 | 0.062±0.005 | 0.017±0.011 |
| Mamba | 0.164±0.003 | 0.086±0.000 | 0.176±0.021 |
| RWKV | 0.186±0.023 | 0.066±0.022 | 0.095±0.009 |
| Retrieval-Augmented | 0.045±0.008 | 0.014±0.008 | -0.072±0.033 |

## Wall time (mean across seeds, seconds)

| arch | ar_intermediate | ar_validation | ar_curriculum (cum + frozen) | total |
|---|---:|---:|---:|---:|
| GPT-2 | 5.6 | 17.4 | 65.8 | 88.8 |
| Mamba | 8.2 | 26.5 | 96.9 | 131.6 |
| RWKV | 11.3 | 37.0 | 151.8 | 200.1 |
| Retrieval-Augmented | 8.4 | 27.6 | 102.8 | 138.8 |

## Probe rank-correlation (Spearman) — does the probe rank archs the same way?

Spearman ρ ≈ 1: probes are duplicates (same ranking). ρ ≈ 0: orthogonal (complementary signals). ρ ≈ -1: anti-correlated (one probe's winner is the other's loser).

| pair | spearman | pearson |
|---|---:|---:|
| ar_intermediate ↔ ar_validation | +0.800 | +0.823 |
| ar_curriculum ↔ ar_intermediate | +0.800 | +0.896 |
| ar_curriculum ↔ ar_validation | +1.000 | +0.937 |

## Per-arch / per-seed details

- **GPT-2** seed=0 **ar_intermediate**: held_pair_lift=0.082 wall=5.9s status=ok
- **GPT-2** seed=0 **ar_validation**: held_pair_acc=0.066 wall=17.5s status=ok
- **GPT-2** seed=0 **ar_curriculum**: curriculum_learning (cum_AUC - frozen_S0_AUC)=0.009 wall=65.8s status=ok
- **GPT-2** seed=1 **ar_intermediate**: held_pair_lift=0.094 wall=5.3s status=ok
- **GPT-2** seed=1 **ar_validation**: held_pair_acc=0.059 wall=17.4s status=ok
- **GPT-2** seed=1 **ar_curriculum**: curriculum_learning (cum_AUC - frozen_S0_AUC)=0.024 wall=65.7s status=ok
- **Mamba** seed=0 **ar_intermediate**: held_pair_lift=0.166 wall=8.3s status=ok
- **Mamba** seed=0 **ar_validation**: held_pair_acc=0.086 wall=26.3s status=ok
- **Mamba** seed=0 **ar_curriculum**: curriculum_learning (cum_AUC - frozen_S0_AUC)=0.161 wall=96.6s status=ok
- **Mamba** seed=1 **ar_intermediate**: held_pair_lift=0.162 wall=8.1s status=ok
- **Mamba** seed=1 **ar_validation**: held_pair_acc=0.086 wall=26.7s status=ok
- **Mamba** seed=1 **ar_curriculum**: curriculum_learning (cum_AUC - frozen_S0_AUC)=0.191 wall=97.2s status=ok
- **RWKV** seed=0 **ar_intermediate**: held_pair_lift=0.170 wall=11.3s status=ok
- **RWKV** seed=0 **ar_validation**: held_pair_acc=0.082 wall=36.8s status=ok
- **RWKV** seed=0 **ar_curriculum**: curriculum_learning (cum_AUC - frozen_S0_AUC)=0.088 wall=135.5s status=ok
- **RWKV** seed=1 **ar_intermediate**: held_pair_lift=0.202 wall=11.3s status=ok
- **RWKV** seed=1 **ar_validation**: held_pair_acc=0.051 wall=37.1s status=ok
- **RWKV** seed=1 **ar_curriculum**: curriculum_learning (cum_AUC - frozen_S0_AUC)=0.102 wall=168.2s status=ok
- **Retrieval-Augmented** seed=0 **ar_intermediate**: held_pair_lift=0.051 wall=8.3s status=ok
- **Retrieval-Augmented** seed=0 **ar_validation**: held_pair_acc=0.008 wall=27.3s status=ok
- **Retrieval-Augmented** seed=0 **ar_curriculum**: curriculum_learning (cum_AUC - frozen_S0_AUC)=-0.049 wall=104.2s status=ok
- **Retrieval-Augmented** seed=1 **ar_intermediate**: held_pair_lift=0.039 wall=8.6s status=ok
- **Retrieval-Augmented** seed=1 **ar_validation**: held_pair_acc=0.019 wall=28.0s status=ok
- **Retrieval-Augmented** seed=1 **ar_curriculum**: curriculum_learning (cum_AUC - frozen_S0_AUC)=-0.096 wall=101.3s status=ok
