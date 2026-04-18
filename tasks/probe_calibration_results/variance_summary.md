# Probe v2 variance check (Phase 1)

Seeds: [11, 23, 47, 89, 131]
Induction v2: 500 mixed-gap steps
Binding:      2400 steps

## Gate criteria

- Attention family CoV < 0.05 at high-AUC archs (attn_2l, attn_4l)
- Family ordering holds: attention ≥ hybrid > conv > ssm/rwkv
- No arch flips family rank across seeds

### Induction v2 — per-architecture variance

| arch | family | n | mean | median | std | CoV | min | max |
|---|---|---|---|---|---|---|---|---|
| `attn_1l` | attention | 5 | 0.7118 | 0.9930 | 0.4250 | 0.597 | 0.0440 | 0.9990 |
| `attn_2l` | attention | 5 | 0.9978 | 0.9990 | 0.0018 | 0.002 | 0.9950 | 0.9990 |
| `attn_4l` | attention | 5 | 0.9986 | 0.9990 | 0.0011 | 0.001 | 0.9970 | 1.0000 |
| `conv3_2l` | conv | 5 | 0.0032 | 0.0030 | 0.0013 | 0.407 | 0.0020 | 0.0050 |
| `conv7_2l` | conv | 5 | 0.0174 | 0.0180 | 0.0029 | 0.166 | 0.0140 | 0.0210 |
| `conv7_4l` | conv | 5 | 0.2040 | 0.2080 | 0.0313 | 0.153 | 0.1520 | 0.2360 |
| `ssm_2l` | ssm | 5 | 0.0038 | 0.0030 | 0.0025 | 0.655 | 0.0020 | 0.0080 |
| `ssm_4l` | ssm | 5 | 0.0032 | 0.0030 | 0.0008 | 0.261 | 0.0020 | 0.0040 |
| `rwkv_2l` | rwkv | 5 | 0.0052 | 0.0050 | 0.0004 | 0.086 | 0.0050 | 0.0060 |
| `hybrid_2l` | hybrid | 5 | 0.9858 | 0.9950 | 0.0214 | 0.022 | 0.9480 | 0.9980 |
| `hybrid_4l` | hybrid | 5 | 0.9982 | 0.9990 | 0.0025 | 0.002 | 0.9940 | 1.0000 |

### Induction v2 — per-family separation

| family | n | min | median | max |
|---|---|---|---|---|
| attention | 15 | 0.0440 | 0.9980 | 1.0000 |
| conv | 15 | 0.0020 | 0.0180 | 0.2360 |
| hybrid | 10 | 0.9480 | 0.9980 | 1.0000 |
| rwkv | 5 | 0.0050 | 0.0050 | 0.0060 |
| ssm | 10 | 0.0020 | 0.0030 | 0.0080 |

### Binding — per-architecture variance

| arch | family | n | mean | median | std | CoV | min | max |
|---|---|---|---|---|---|---|---|---|
| `attn_1l` | attention | 5 | 0.9716 | 0.9711 | 0.0063 | 0.006 | 0.9660 | 0.9819 |
| `attn_2l` | attention | 5 | 0.7963 | 0.9939 | 0.4424 | 0.556 | 0.0050 | 0.9950 |
| `attn_4l` | attention | 5 | 0.9941 | 0.9940 | 0.0008 | 0.001 | 0.9933 | 0.9954 |
| `conv3_2l` | conv | 5 | 0.0049 | 0.0050 | 0.0016 | 0.336 | 0.0030 | 0.0073 |
| `conv7_2l` | conv | 5 | 0.4478 | 0.4502 | 0.0068 | 0.015 | 0.4391 | 0.4560 |
| `conv7_4l` | conv | 5 | 0.7127 | 0.7129 | 0.0030 | 0.004 | 0.7087 | 0.7160 |
| `ssm_2l` | ssm | 5 | 0.0026 | 0.0023 | 0.0007 | 0.278 | 0.0018 | 0.0037 |
| `ssm_4l` | ssm | 5 | 0.0022 | 0.0024 | 0.0007 | 0.294 | 0.0013 | 0.0030 |
| `rwkv_2l` | rwkv | 5 | 0.0016 | 0.0011 | 0.0011 | 0.664 | 0.0005 | 0.0032 |
| `hybrid_2l` | hybrid | 5 | 0.9898 | 0.9947 | 0.0148 | 0.015 | 0.9635 | 0.9987 |
| `hybrid_4l` | hybrid | 5 | 0.9995 | 0.9996 | 0.0001 | 0.000 | 0.9994 | 0.9997 |

### Binding — per-family separation

| family | n | min | median | max |
|---|---|---|---|---|
| attention | 15 | 0.0050 | 0.9933 | 0.9954 |
| conv | 15 | 0.0030 | 0.4502 | 0.7160 |
| hybrid | 10 | 0.9635 | 0.9990 | 0.9997 |
| rwkv | 5 | 0.0005 | 0.0011 | 0.0032 |
| ssm | 10 | 0.0013 | 0.0023 | 0.0037 |