# Leaderboard Replication Backfill — Action Plan

## Problem

760 screening entries on the leaderboard have `replication_n = NULL`. Stream A added
replication dampening to `compute_composite_score()` but it only activates when
`replication_n` is populated. The else branch at `leaderboard_scoring.py:603` sets
`repl_confidence = 1.0` (no dampening) when data is missing — so every score on the
leaderboard is currently undampened single-run scores.

Result: composite scores are inflated. Two of the top 3 screening entries failed
investigation with loss ratios >0.5 (14x worse than screening). The leaderboard is
unreliable.

## Root Cause

`upsert_leaderboard()` calls `get_fingerprint_aggregates()` and populates
`replication_n` / `replication_loss_mean` — but only on new upserts. The 760 existing
entries were never backfilled.

## Plan

### Step 1: Backfill replication aggregates on all existing leaderboard entries

For each leaderboard entry:
1. Get its `graph_fingerprint` from `program_results`
2. Call `get_fingerprint_aggregates(fingerprint)` to count all runs with that fingerprint
3. UPDATE `replication_n` and `replication_loss_mean` on the leaderboard row

This is a single SQL join + UPDATE — no training, no GPU, ~5 seconds.

### Step 2: Rescore all leaderboard entries with dampened composites

After replication data is populated, re-run `compute_composite_score()` on every entry
to apply the dampening. Single-run entries will get `sqrt(1/3) = 0.577` multiplier,
cutting their composite by ~42%.

### Step 3: Verify the leaderboard makes sense

- Top screening entries should have lower composites
- Any entry with replication_n >= 3 should keep its full score
- Investigation/validation entries should be unaffected (they have real multi-run data)

### Step 4: Add backfill to continuous run startup

So this never happens again — on continuous run init, run the backfill once.

## Files to modify

- `research/scientist/notebook/notebook_leaderboard.py` — add `backfill_replication_aggregates()` method
- `research/scientist/runner/core.py` or `continuous_loop.py` — call backfill on startup
- New: `research/tools/backfill_replication.py` — one-shot CLI script

## Risk

Low. This is a data-only change — no model code, no grammar changes. Composite scores
will drop for under-replicated entries, which is the correct behavior.
