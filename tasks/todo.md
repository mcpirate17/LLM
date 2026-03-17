# Gate & Scoring Fix (2026-03-16)

## Problem 1: Corpus-aware Stage 1 gate
- [ ] Add `stage1_learning_gate()` to `runner/_helpers.py`
- [ ] Add `get_reference_losses()` DB query helper
- [ ] Wire into `execution_training.py` replacing line 857 threshold
- [ ] Add corpus_type derivation from config
- [ ] Test gate with 7 test cases

## Problem 2: Open-ended scoring v6
- [ ] Add `compute_composite_v6()` to `leaderboard_scoring.py`
- [ ] Update `rescore_leaderboard.py` for `--version v6`
- [ ] Run dry-run rescore, verify GPT-2=100, Var H 103-108
- [ ] Update dashboard `scoreColor` for new scale
- [ ] Update leaderboard header text
- [ ] Add score color tiers (gold/green/white/grey)

## Output
- [ ] Write `tasks/audit/GATE_AND_SCORING_FIX.md`
