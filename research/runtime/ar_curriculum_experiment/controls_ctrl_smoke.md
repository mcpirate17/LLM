# AR Curriculum controls — ctrl_smoke

stage_set=default steps_per_stage=10 eval_examples=16 archs=['gpt2'] seeds=[0]

**Chance per stage** (1/n_values): S0=0.125, S1=0.083, S2=0.062, S3=0.042, S4=0.031, S5=0.021

## AUC pair final (mean ± std across seeds)

| arch | cumulative | frozen_s0 (matched compute) | untrained (empirical chance) | Δ cum-vs-frozen |
|---|---:|---:|---:|---:|
| GPT-2 | 0.042±0.000 | 0.062±0.000 | 0.021±0.000 | -0.020 |

_Theoretical random AUC (mean of 1/n_values): 0.061_

## Stage-0 forgetting test — strict

If cumulative S0 acc < frozen_s0 S0 acc, the model FORGOT what it could have learned with the same compute. If cumulative ≥ frozen_s0, no forgetting (joint training generalizes as well or better).

| arch | cumulative S0 acc | frozen_s0 S0 acc | Δ (forgetting) |
|---|---:|---:|---:|
| GPT-2 | 0.125 | 0.312 | -0.187 (forgetting) |

## Per-stage acc — cumulative vs frozen_s0 vs untrained

### GPT-2

| mode | S0 | S1 | S2 | S3 | S4 | S5 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| chance | 0.125 | 0.083 | 0.062 | 0.042 | 0.031 | 0.021 | 0.061 |
| cumulative | 0.125 | 0.000 | 0.000 | 0.062 | 0.062 | 0.000 | 0.042 |
| frozen_s0 | 0.312 | 0.062 | 0.000 | 0.000 | 0.000 | 0.000 | 0.062 |
| untrained | 0.062 | 0.062 | 0.000 | 0.000 | 0.000 | 0.000 | 0.021 |

Z-scores (≥2 = significantly above chance):

| mode | S0 | S1 | S2 | S3 | S4 | S5 | mean Z |
|---|---:|---:|---:|---:|---:|---:|---:|
| cumulative | +0.0 | -1.2 | -1.0 | +0.4 | +0.7 | -0.6 | -0.3 |
| frozen_s0 | +2.3 | -0.3 | -1.0 | -0.8 | -0.7 | -0.6 | -0.2 |
| untrained | -0.8 | -0.3 | -1.0 | -0.8 | -0.7 | -0.6 | -0.7 |

