---
status: active
created: 2026-03-30
author: claude-opus
---

# TODO — Next Session

## ACTIVE: Scale test running
Champion c9c7075e at 177M params on FineWeb-Edu + UltraChat with Muon optimizer.
- **Status**: Step 33K/50K (66%), avg loss 2.70, best 2.14, 0.41B tokens
- **Restartable**: `python -m research.tools.scale_test` resumes from `research/artifacts/scale_test/latest.pt`
- **Loss curve**: `research/artifacts/scale_test/loss_curve.png` + `loss_curve.json`
- **ETA**: ~2 more hours to finish

### Loss trajectory:
```
Step   5K:  3.15
Step  10K:  2.95
Step  20K:  2.78
Step  30K:  2.75
Step  33K:  2.70  (current)
Best:       2.14
```

## 1. Fix scaling reference cache key (HIGH PRIORITY)
Validation takes 60+ minutes because reference architectures are retrained every run.
Cache key includes n_steps/vocab_size/data_tag — should just be family+d_model+n_layers.
File: `research/eval/scaling_reference.py` line 357

## 2. Add step-level progress to validation
Show: `scaling reference comparison (d512) (14/15) — gpt2 8-layer: step 1200/10000 loss=1.83`
Files: `research/eval/scaling_reference.py`, `research/scientist/runner/execution_validation.py`

## 3. Make validation writes incremental
Currently writes all results at the end. Crash at step 14/15 = total loss.
Write each result as it completes.

## 4. Compare validation vs investigation loss
Fingerprint 606dacc0 was fixed this session — investigation_loss was garbage (0.576) but validation_loss was 0.009. Check other entries for same issue: investigation_loss_ratio that's much worse than screening or validation.

## 5. Scoring reality check
`scaling_param_efficiency` drives ~375 of ~750 points. At small scale this rewards compression tricks. Should loss/PPL be weighted higher since MoE/MoD/MoR can be bolted on later? The core architecture's learning ability is the hard part.

## 6. Context rules (prompt for other Claude)
- `tasks/fix_context_rules_prompt.md` — selective_scan, softmax_attention, transpose_sd
- token_merge + adjacent_token_merge rules already added this session

## 7. Low-confidence op backfills
```bash
python -m research.tools.backfill_templates --templates sparse_moe_block routed_bottleneck --target 100
```

## Session Summary (March 29-30)
### Built
- 5 ML modules: interaction_analysis, temporal_bayesian, op_embeddings, interaction_model, gnn_predictor
- Ensemble predictor with calibrated logistic regression (93% precision)
- Bayesian tracker with temporal decay + code-fix detection
- Op interaction heatmaps (7,754 pairs analyzed)
- Targeted backfill tool (`research/tools/targeted_backfill.py`)
- Scale test script (`research/tools/scale_test.py`) — restartable, Muon optimizer, live loss curve

### Fixed
- Cleared 5 false-positive failure signatures, adjusted 20 more
- Added context rules for token_merge, adjacent_token_merge, and 7 routing ops
- Fixed token_merge_block backfill (was failing due to routing_mandatory=True)
- Fixed code-fix detection cascade (was resetting core op posteriors to 0)
- Removed replication dampening from scoring (was penalizing tested entries)
- Converted outlier penalty to needs_extended_training flag
- Fixed investigation_robustness NULL for 14 investigation entries
- Fixed wikitext_score JSON blob corruption
- Backfilled n_routing_ops/n_sparse_ops/n_moe_ops for 1145 entries
- Fixed 606dacc0 investigation_loss_ratio (0.576 → 0.009)
- Rescored entire leaderboard 3 times with corrected data

### Mutations tested
- Champion baseline: PPL 4.4 (matched original)
- Fix activation order: PPL 4.5 (no improvement)
- Learned gate: PPL 6.7 (promising, still converging)
- Full hybrid (sparse FFN + Mamba + sparse channel): PPL 12.1 (needs more steps)
