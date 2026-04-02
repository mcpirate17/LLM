# Changes

## 2026-03-31 — Associative Recall probe rewrite

**Problem**: The AR probe had no key collision prevention (keys could share tokens,
creating ambiguous retrieval), used token 0 as separator (conflicts with padding),
didn't use a fixed eval seed, had no step-0 baseline eval, and returned a simple
mean AUC instead of proper trapezoidal integration. The return format also lacked
`timed_out`, `above_chance`, and `final_acc` fields needed by the scoring pipeline.

**Changes**:

1. **`eval/associative_recall.py`** — Full rewrite:
   - Restricted vocab IDs 100-355 (avoids special tokens at both ends)
   - Keys: 2-token sequences drawn WITHOUT replacement per sequence
   - Values: single tokens guaranteed distinct from all key tokens
   - SEP/ANSWER tokens: 50256/50257 (falls back to last two vocab IDs)
   - Fixed seed (42) for eval set, variable seeds for training batches
   - Step 0 eval (pre-training baseline) included in learning curve
   - Trapezoidal AUC normalized by max possible area
   - Returns: `auc`, `final_acc`, `learning_curve` (step, acc tuples),
     `timed_out`, `above_chance`, `steps_trained`, `status`, `elapsed_ms`
   - Default n_pairs=20 (seq_len=64), batch_size=16

2. **`scientist/runner/_helpers.py`** — Updated callers:
   - Investigation + validation: n_pairs=20, batch_size=16 (was 10/32)
   - Stores `ar_final_acc`, `ar_timed_out`, `ar_above_chance` in results
   - Both investigation benchmark and validation paths store new columns
   - Fixed local_only gate: 3-signal AND (ar + induction + binding_auc)
     instead of 2-signal AND — prevents false penalty on Mamba/RWKV
   - Added DISCOVERY logging when ar_auc > 0.15 without attention ops

3. **`scientist/notebook/_shared.py`** — Added DB columns:
   `ar_final_acc` (REAL), `ar_timed_out` (INTEGER), `ar_above_chance` (INTEGER)
   to `_PROGRAM_RESULTS_NEW_COLUMNS` — auto-migrated on next notebook open.

4. **`scientist/leaderboard_scoring.py`** — Full scoring integration:
   - Added `ar_timed_out` and `ar_above_chance` as scoring function params
   - Timed-out AR treated as missing data (None), not zero — a timeout is a
     measurement failure, not evidence the model lacks retrieval capability
   - `ar_above_chance` exempts from soft penalty — a model with real retrieval
     signal (>10x random chance) should not trigger local_only even if AUC is low
   - Wired through `_pr_dict_to_score_kwargs` and `_PR_SELECT_COLS`

5. **`scientist/api_routes/leaderboard_bp.py`** — Added `ar_final_acc`,
   `ar_timed_out`, `ar_above_chance` to compact leaderboard entry response.

6. **`dashboard/src/components/leaderboard/LeaderboardRow.js`** — Fixed AR
   color thresholds: red < 0.05, yellow 0.05-0.20, green > 0.20 (was 0.10/0.25).

## 2026-03-31 — Binding capacity as first-class screening objective

**Problem**: Local-only architectures (conv1d k=3, token_merge) achieve competitive
WikiText perplexity but zero HellaSwag across 160K steps. The binding probes run at
screening and investigation, but the 50pt scoring component was too weak (~7% of total)
to prevent local-only winners from consuming investigation budget.

**Changes**:

1. **`synthesis/primitives.py`** — Added `binding_range_class` field to `PrimitiveOp`
   (`"full"`, `"medium"`, `"local"`, `"none"`). Annotated 13 mixer/sequence ops.
   Added `graph_binding_range_class()` helper to classify a graph's binding reach.

2. **`scientist/leaderboard_scoring.py`** — Increased binding component from 50pt to
   120pt max (~25% of screening budget). Changed composite formula from
   `0.6*ar + 0.4*induction` to `0.4*ar + 0.3*induction + 0.3*binding_auc` to include
   all three binding signals. Added `binding_auc` parameter and plumbing.

3. **`scientist/leaderboard_scoring.py`** — Fixed soft penalty gate from 2-signal AND
   (ar + induction) to 3-signal AND (ar + induction + binding_auc). This prevents
   penalizing Mamba/SSM/RWKV which score ~0 on induction (exact retrieval) but have
   real non-local capability via binding_auc. The penalty now only fires for true
   local-only architectures (conv-3 case) where ALL signals are near zero.

4. **`scientist/thresholds.py`** — Added `BINDING_BINDING_AUC_SOFT_GATE = 0.10`,
   updated AR/induction thresholds to 0.05, expanded calibration comments explaining
   why Mamba/RWKV correctly fail induction without being penalized.

5. **`search/evolution.py`** — Added binding-aware mutation bias: when a parent graph
   has only local/none binding range, mutation and crossover grammars boost full-range
   mixer ops (attention, SSM, etc.) by 3x and relevant templates by 2.5x.

6. **`scientist/runner/execution_training.py`** — Updated binding composite formula
   at screening. Added HIGH PRIORITY DISCOVERY logging: any candidate scoring
   induction_auc > 0.20 without standard attention ops is logged at WARNING level
   for immediate investigation (novel retrieval mechanism).

**Key design decision**: The induction probe measures exact token retrieval across gaps.
Only full attention reliably passes. Mamba/SSM/RWKV failing induction is CORRECT — their
failure mechanism (state compression) is fundamentally different from conv-3's (zero
receptive field). The 3-signal AND ensures they are not unfairly penalized.
